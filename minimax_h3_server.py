#!/usr/bin/env python3
"""Persistent ComfyUI launcher for MiniMax-H3 FL2VA Q4_K_M."""
import os
import runpy
import sys

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MIOPEN_FIND_MODE", "2")
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "0"
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
# ROCm 10 reports the 8060S as gfx1151. Do not inherit the old gfx1150
# workaround used by the previous ROCm installation.
os.environ.pop("HSA_OVERRIDE_GFX_VERSION", None)

INSTALL_ROOT = os.environ.get("MINIMAX_H3_INSTALL_ROOT", "/home/ignacia")
COMFYUI_DIR = os.environ.get("MINIMAX_H3_COMFYUI_DIR", os.path.join(INSTALL_ROOT, "ComfyUI-minimax-h3"))
MODEL_PATHS = os.environ.get("MINIMAX_H3_MODEL_PATHS", os.path.join(os.path.dirname(__file__), "minimax_h3_model_paths.yaml"))
port = int(os.environ.get("MINIMAX_H3_PORT", "9879"))
if not 1024 <= port <= 65535:
    raise ValueError("MINIMAX_H3_PORT debe estar entre 1024 y 65535")
if port == 9877:
    raise ValueError("El puerto 9877 está reservado a Z-Image")
if COMFYUI_DIR not in sys.path:
    sys.path.insert(0, COMFYUI_DIR)

sys.argv = [
    "main.py", "--listen", "127.0.0.1", "--port", str(port),
    # Force CLIP, DiT and both VAEs to execute/reside on the ROCm device.
    # This machine exposes the AMD GPU through PyTorch as cuda:0.
    "--disable-auto-launch", "--gpu-only", "--disable-async-offload",
    "--disable-comfy-compiler",
    "--extra-model-paths-config", MODEL_PATHS,
]
os.chdir(COMFYUI_DIR)
runpy.run_path("main.py", run_name="__main__")
