#!/usr/bin/env python3
"""Local web panel/proxy for the Z-Image TCP server.

Run: python3 zimage_panel.py --backend-port 9877 --web-port 8765
Then open http://127.0.0.1:8765
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse, json, mimetypes, os, socket, struct
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
HTML = ROOT / "zimage_panel.html"

def backend_request(payload, host, port):
    data = json.dumps(payload).encode()
    with socket.create_connection((host, port), timeout=900) as conn:
        conn.sendall(struct.pack("!I", len(data)) + data)
        header = conn.recv(4)
        if len(header) != 4:
            raise RuntimeError("El backend cerró la conexión")
        length = struct.unpack("!I", header)[0]
        result = bytearray()
        while len(result) < length:
            chunk = conn.recv(min(1024 * 1024, length - len(result)))
            if not chunk:
                raise RuntimeError("Respuesta incompleta del backend")
            result.extend(chunk)
    return json.loads(result)

class Handler(BaseHTTPRequestHandler):
    server_version = "ZImagePanel/1.0"
    def send_json(self, value, code=200):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                body = HTML.read_bytes(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
            if parsed.path == "/api/status":
                self.send_json(backend_request({"cmd": "status"}, self.server.backend_host, self.server.backend_port)); return
            if parsed.path == "/api/loras":
                self.send_json(backend_request({"cmd": "list_loras"}, self.server.backend_host, self.server.backend_port)); return
            if parsed.path == "/api/image":
                raw = parse_qs(parsed.query).get("path", [""])[0]
                path = Path(raw).resolve()
                allowed = [Path("~/zimage_outputs").resolve(), Path("~/zimage_outputs_gguf").resolve()]
                if not any(path == base or base in path.parents for base in allowed): raise PermissionError("Ruta no permitida")
                body = path.read_bytes(); self.send_response(200); self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
            self.send_error(404)
        except Exception as exc: self.send_json({"status": "error", "error": str(exc)}, 502)
    def do_POST(self):
        if urlparse(self.path).path != "/api/generate": self.send_error(404); return
        try:
            length = int(self.headers.get("Content-Length", "0")); payload = json.loads(self.rfile.read(length))
            payload["cmd"] = "generate"
            self.send_json(backend_request(payload, self.server.backend_host, self.server.backend_port))
        except Exception as exc: self.send_json({"status": "error", "error": str(exc)}, 502)
    def log_message(self, fmt, *args): print("[panel] " + fmt % args, flush=True)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--backend-host", default="127.0.0.1"); parser.add_argument("--backend-port", type=int, default=9877); parser.add_argument("--web-host", default="127.0.0.1"); parser.add_argument("--web-port", type=int, default=8765)
    args = parser.parse_args(); httpd = ThreadingHTTPServer((args.web_host, args.web_port), Handler); httpd.backend_host=args.backend_host; httpd.backend_port=args.backend_port
    print(f"Panel: http://{args.web_host}:{args.web_port}  Backend: {args.backend_host}:{args.backend_port}", flush=True); httpd.serve_forever()
if __name__ == "__main__": main()
