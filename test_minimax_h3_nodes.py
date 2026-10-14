"""CPU-only smoke check for the separate native ComfyUI checkout."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path("/home/ignacia")
COMFY = ROOT / "ComfyUI-minimax-h3"
PORT = 19879
BASE = f"http://127.0.0.1:{PORT}"


class NativeNodesTests(unittest.TestCase):
    def test_model_choices_without_loading_weights(self):
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", PORT)) == 0:
                self.fail(f"Puerto de prueba {PORT} ocupado")
        command = [
            sys.executable, str(COMFY / "main.py"), "--cpu", "--disable-auto-launch",
            "--disable-comfy-compiler", "--listen", "127.0.0.1", "--port", str(PORT),
            "--extra-model-paths-config", str(ROOT / "minimax_h3_model_paths.yaml"),
        ]
        with tempfile.TemporaryFile(mode="w+t") as log:
            process = subprocess.Popen(command, cwd=COMFY, stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        break
                    try:
                        with urlopen(BASE + "/object_info", timeout=2) as response:
                            nodes = json.load(response)
                        break
                    except (URLError, TimeoutError):
                        time.sleep(1)
                else:
                    nodes = None
                if not nodes:
                    log.seek(0)
                    self.fail("ComfyUI no registró nodos: " + log.read()[-2500:])
                required = (
                    "MiniMaxH3ImageToVideo", "MiniMaxH3SigmaShift",
                    "UnetLoaderGGUFAdvanced", "ImageFromBatch", "SaveVideo", "VAEDecodeAudio",
                )
                for name in required:
                    self.assertIn(name, nodes)
                loader = nodes["UnetLoaderGGUFAdvanced"]["input"]["required"]["unet_name"][0]
                self.assertIn("MiniMax-H3-FL2VA-Pruned-Q4_K_M.gguf", loader)
                clip_types = nodes["CLIPLoader"]["input"]["required"]["type"][0]
                self.assertIn("minimax", clip_types)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    unittest.main()
