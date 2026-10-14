#!/usr/bin/env python3
"""MCP adapter for the existing zimage_server_lora.py TCP server.

No third-party MCP package is required.  The server supports:
  - stdio (default): for MCP clients that launch a command;
  - streamable HTTP-ish JSON-RPC: --transport http --port 8765.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import socket
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DEFAULT_BACKEND_HOST = os.environ.get("ZIMAGE_BACKEND_HOST", "127.0.0.1")
DEFAULT_BACKEND_PORT = int(os.environ.get("ZIMAGE_BACKEND_PORT", "9877"))
# Generations and slow clients may legitimately take a long time: 30 minutes.
IO_TIMEOUT = float(os.environ.get("ZIMAGE_MCP_TIMEOUT", "1800"))
MAX_REQUEST_BYTES = 1024 ** 3
DEFAULT_OUTPUT_DIR = os.environ.get("ZIMAGE_OUTPUT_DIR") or os.path.expanduser("~/zimage_outputs")


def log(message: str) -> None:
    print(f"[zimage-mcp] {message}", file=sys.stderr, flush=True)


def backend_request(payload: dict[str, Any], host: str, port: int) -> dict[str, Any]:
    """Call the existing 4-byte-length-prefixed JSON TCP API."""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    with socket.create_connection((host, port), timeout=IO_TIMEOUT) as conn:
        conn.sendall(struct.pack("!I", len(data)) + data)
        header = _recv_exact(conn, 4)
        if len(header) != 4:
            raise RuntimeError("El backend Z-Image no devolvió una respuesta")
        length = struct.unpack("!I", header)[0]
        response = _recv_exact(conn, length)
    result = json.loads(response.decode("utf-8"))
    if not isinstance(result, dict):
        raise RuntimeError("Respuesta inválida del backend Z-Image")
    return result


def _recv_exact(conn: socket.socket, length: int) -> bytes:
    chunks = []
    remaining = length
    while remaining:
        chunk = conn.recv(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def image_content(source: str) -> dict[str, Any] | None:
    """Convert a local path or image URL into an MCP Base64 image block."""
    try:
        parsed = urllib.parse.urlparse(source)
        if parsed.scheme == "data":
            header, encoded = source.split(",", 1)
            mime = header[5:].split(";", 1)[0] or "image/png"
            raw = base64.b64decode(encoded) if ";base64" in header else urllib.parse.unquote_to_bytes(encoded)
        elif parsed.scheme in ("http", "https"):
            request = urllib.request.Request(source, headers={"User-Agent": "zimage-mcp/1.0"})
            with urllib.request.urlopen(request, timeout=IO_TIMEOUT) as response:
                raw = response.read()
                mime = response.headers.get_content_type()
        else:
            path = Path(urllib.request.url2pathname(parsed.path)) if parsed.scheme == "file" else Path(source)
            raw = path.read_bytes()
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
        if not mime.startswith("image/"):
            return None
        return {"type": "image", "data": base64.b64encode(raw).decode("ascii"), "mimeType": mime}
    except (OSError, ValueError, urllib.error.URLError, UnicodeError):
        return None


def tool_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") != "ok":
        return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}], "isError": True}
    # The current TCP backend returns `path`; support URL-shaped responses
    # as well so the adapter works with backends returning `url` or
    # `image_url`.
    source = next((result.get(key) for key in ("url", "image_url", "path") if isinstance(result.get(key), str)), None)
    image = image_content(source) if source else None
    if source and image is None:
        return {
            "content": [{"type": "text", "text": json.dumps({**result, "image_error": f"No se pudo leer la imagen desde: {source}"}, ensure_ascii=False, indent=2)}],
            "isError": True,
        }
    structured = dict(result)
    content = []
    if image:
        # Keep the standard MCP binary content block and also expose the
        # exact block in structuredContent for clients that only render the
        # structured tool result (as llama.cpp's UI currently does).
        content.append(image)
        structured["image"] = image
        log(f"Imagen convertida a MCP image: source={source!r}, mime={image['mimeType']}, base64_chars={len(image['data'])}")
    content.append({"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)})
    return {"content": content, "structuredContent": structured}


def tool_definitions() -> list[dict[str, Any]]:
    number = {"type": "number"}
    integer = {"type": "integer"}
    return [
        {
            "name": "generate_image",
            "description": "Genera una imagen con Z-Image usando exactamente los parámetros del backend zimage_server_lora.py.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Descripción de la imagen; preferiblemente en inglés."},
                    "request_id": {"type":"string", "description":"Identificador opcional para consultar generation_progress."},
                    "negative_prompt": {"type": "string", "default": ""},
                    "width": {**integer, "default": 1024, "minimum": 64},
                    "height": {**integer, "default": 1024, "minimum": 64},
                    "steps": {**integer, "default": 8, "minimum": 1},
                    "cfg": {**number, "default": 6, "minimum": 0},
                    "seed": {**integer, "default": 0},
                    "lora": {"type": "string", "description": "Nombre[@peso], o varios separados por ';', por ejemplo estilo:0.8;color:0.5."},
                    "filename": {"type": "string"},
                    "output_dir": {"type": "string", "default": DEFAULT_OUTPUT_DIR},
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
        },
        {"name": "generation_progress", "description": "Progreso de una generación identificada por request_id.", "inputSchema": {"type":"object", "properties":{"request_id":{"type":"string"}}, "required":["request_id"]}},
        {"name": "list_loras", "description": "Lista los LoRA disponibles en Z-Image.", "inputSchema": {"type": "object", "properties": {}}},
        {"name": "zimage_status", "description": "Comprueba el estado del backend Z-Image y si el modelo está cargado.", "inputSchema": {"type": "object", "properties": {}}},
    ]


class ZImageMCP:
    def __init__(self, backend_host: str, backend_port: int):
        self.backend_host = backend_host
        self.backend_port = backend_port

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == 'generation_progress':
            return tool_result(backend_request({'cmd': 'progress', 'request_id': arguments.get('request_id', '')}, self.backend_host, self.backend_port))
        if name == "generate_image":
            args = dict(arguments)
            args["cmd"] = "generate"
            if not args.get("prompt"):
                raise ValueError("prompt es obligatorio")
            return tool_result(backend_request(args, self.backend_host, self.backend_port))
        if name == "list_loras":
            return tool_result(backend_request({"cmd": "list_loras"}, self.backend_host, self.backend_port))
        if name == "zimage_status":
            return tool_result(backend_request({"cmd": "status"}, self.backend_host, self.backend_port))
        raise ValueError(f"Herramienta desconocida: {name}")

    def rpc(self, message: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(message, dict):
            return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Petición JSON-RPC inválida"}}
        method = message.get("method")
        request_id = message.get("id")
        if not isinstance(method, str):
            if "result" in message or "error" in message:
                return None  # a client response; this server sends no requests
            return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32600, "message": "Falta el campo method"}}
        if request_id is None and method.startswith("notifications/"):
            return None
        if method == "initialize":
            return {"jsonrpc": "2.0", "id": request_id, "result": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "zimage", "version": "1.0.0"}}}
        if method == "ping":
            return {"jsonrpc": "2.0", "id": request_id, "result": {}}
        try:
            if method == "tools/list":
                result = {"tools": tool_definitions()}
            elif method == "tools/call":
                params = message.get("params") or {}
                result = self.call(params.get("name", ""), params.get("arguments") or {})
            else:
                return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": f"Método no encontrado: {method}"}}
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except Exception as exc:
            log(f"error en {method}: {exc}")
            return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32000, "message": str(exc)}}


def run_stdio(server: ZImageMCP) -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            response = server.rpc(json.loads(line))
            if response is not None:
                print(json.dumps(response, ensure_ascii=False), flush=True)
        except Exception as exc:
            log(f"JSON-RPC inválido: {exc}")


class HTTPHandler(BaseHTTPRequestHandler):
    server_version = "zimage-mcp/1.0"
    # Socket timeout for reads/writes; a stalled peer is released after 30 min
    # instead of holding its thread forever. Active streams are unaffected.
    timeout = IO_TIMEOUT

    def _cors_headers(self) -> None:
        # The llama.cpp Web UI runs on localhost:8085 and calls this local
        # server from the browser, so this is a required cross-origin bridge.
        # origin = self.headers.get("Origin", "*")
        origin = "*"
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Access-Control-Allow-Origin, Access-Control-Allow-Methods, Content-Type, Accept, Mcp-Session-Id, Last-Event-ID, Mcp-Protocol-Version")
        self.send_header("Access-Control-Expose-Headers", "Mcp-Session-Id, Content-Type")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        """Open the optional server-to-client SSE stream used by MCP HTTP."""
        if self.path.split("?", 1)[0].rstrip("/") not in ("", "/mcp"):
            self.send_error(404)
            return
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(b": zimage-mcp connected\n\n")
            self.wfile.flush()
            # Keep the stream available for server notifications.  The
            # current adapter has none, but MCP clients still expect GET to
            # be a valid transport operation after initialization.
            while True:
                time.sleep(15)
                self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def do_POST(self) -> None:  # noqa: N802
        # llama.cpp stores a bare URL and posts to `/`; also accept `/mcp`
        # for clients that use the conventional MCP endpoint.
        if self.path.rstrip("/") not in ("", "/mcp"):
            self.send_error(404)
            return
        try:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if length < 0:
                raise ValueError("Content-Length inválido")
            if length > MAX_REQUEST_BYTES:
                self.send_response(413)
                self._cors_headers()
                body = json.dumps({"error": "Petición superior a 1 GiB"}).encode("utf-8")
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            message = json.loads(self.rfile.read(length))
            response = self.server.zimage.rpc(message)  # type: ignore[attr-defined]
            if response is None:
                self.send_response(202)
                self._cors_headers()
                self.end_headers()
                return
            body = json.dumps(response, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if message.get("method") == "initialize":
                self.send_header("Mcp-Session-Id", uuid.uuid4().hex)
            self._cors_headers()
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            self.send_response(400)
            self._cors_headers()
            self.send_header("Content-Type", "application/json")
            body = json.dumps({"error": str(exc)}).encode("utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        log(fmt % args)


def run_http(server: ZImageMCP, host: str, port: int) -> None:
    httpd = ThreadingHTTPServer((host, port), HTTPHandler)
    httpd.zimage = server  # type: ignore[attr-defined]
    log(f"MCP HTTP escuchando en http://{host}:{port}/mcp")
    httpd.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=("stdio", "http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--backend-host", default=DEFAULT_BACKEND_HOST)
    parser.add_argument("--backend-port", type=int, default=DEFAULT_BACKEND_PORT)
    args = parser.parse_args()
    server = ZImageMCP(args.backend_host, args.backend_port)
    if args.transport == "http":
        run_http(server, args.host, args.port)
    else:
        run_stdio(server)


if __name__ == "__main__":
    main()
