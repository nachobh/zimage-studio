# Z-Image Studio

A local, multi-service multimodal panel: text-to-image diffusion (Z-Image,
GGUF quantized) with LoRA training/conversion, an OpenAI-compatible chat
backend (llama.cpp), TTS/STT (Qwen3-TTS, Kokoro, Whisper), an MCP bridge, and
a browser panel with SQLite-backed conversations. Everything runs locally on
consumer/workstation GPU (ROCm or CUDA).

**Scope note:** this repository is a personal homelab project with a lot of
exploratory material (long-form planning docs, status notes, half-finished
experiments). This README documents the project as a whole; see "Suggested
publication scope" below for what is actually clean enough to publish.

## What it is

- **Diffusion**: Z-Image-Turbo served from GGUF, LoRA weights applied at
  generation time; `convert_lora_zimage.py` / `convert_loras_to_peft.py` /
  `repair_zimage_lora_keys.py` handle LoRA format conversion and key repair
  between training checkpoints and the inference format.
- **LoRA training**: in-panel character/style trainer
  (`zimage_character_train.py`) with configurable epochs/LR/rank, a
  `training_report.json` with a LoRA-strength heuristic (median ‖B·A‖/‖W‖
  across adapted modules), and `ckpt2safetensors.sh` for checkpoint
  conversion.
- **Chat backend**: llama.cpp `llama-server`, OpenAI-compatible, serving a
  Qwen3.5-9B-class model (`run_qwen35_9b.sh`) for the text/character chat.
- **Audio**: Qwen3-TTS voice cloning server, Kokoro/Whisper STT bridge
  (`zimage_audio_server.py`, `zimage_kokoclone_server.py`,
  `qwen3_tts_trial.py`).
- **Video**: MiniMax-H3/ComfyUI integration for short video generation from
  chat images (`comfyui_video.py`, `gen_minimax_h3.py`).
- **MCP bridge**: `zimage_mcp_server.py` exposes panel actions as MCP tools.
- **Persistence**: `zimage_conversations.py` stores chats in SQLite with a
  private (browser IndexedDB)/public (server SQLite) split per conversation,
  media stored content-addressed by SHA-256 and deduplicated
  (`zimage_media_audit.py` verifies hashes/sizes before any deletion, with a
  dry-run mode and a lock-protected delete path).
- **Panel**: `zimage_panel.py` + `zimage_panel_grid.py` serve the HTML/JS
  frontend (character chat, group chat, gallery, LoRA admin, storage
  auditing).
- Test suite: ~15 `test_*.py` files covering auth, conversations, media
  audit, bootstrap, character training/caption, panel grid behavior, and a
  few "live" tests that require running services and real inference.

## Architecture

```
                     Browser (panel, PWA)
                             |
                     zimage_panel.py (HTTP)
          +------------------+-------------------+
          |                  |                   |
   llama-server :8086   Z-Image GGUF        zimage_mcp_server.py
   (chat, OpenAI API)   diffusion engine     (MCP bridge)
          |                  |                   |
   zimage_conversations  LoRA adapters      external MCP clients
   (SQLite, dedup media) (PEFT/safetensors)
          |
   Qwen3-TTS :8770 / Kokoro+Whisper :8767  (voice clone, STT)
          |
   MiniMax-H3 / ComfyUI :9879  (image -> short video)
```

All services bind to `127.0.0.1` by default; LAN exposure requires an
explicit panel password and, for the "existing install" variant, HTTPS with a
locally-generated CA (`setup_zimage_https.py`).

## How to run

```bash
./start.sh                   # fresh portable install: venv, GPU check, models, services
./start.sh --dry-run         # describe steps only, no installs/downloads
./start.sh --models-only     # prepare env + models, don't launch services
./start.sh --status
./start.sh --stop

# Existing/local install variant (audio + HTTPS by default):
./start_instalacion_existente.sh
./start_instalacion_existente.sh --restart
./start_instalacion_existente.sh --restart-qwen3-tts

# Tests
python3 -m unittest -v test_zimage_panel_grid test_zimage_conversations
python3 test_zimage_media_audit.py
```

Data/models/LoRAs/outputs live under `$ZIMAGE_DATA_DIR` (default `data/`),
overridable with `ZIMAGE_DATA_DIR`, `ZIMAGE_LORA_DIR`, `ZIMAGE_TRANSFORMER_GGUF`,
etc. `requirements.txt`: diffusers, transformers, peft, safetensors, gguf,
accelerate, huggingface_hub, Pillow.

## What was measured / learned

Hard quantitative data in this repo is thin — this is primarily an
engineering/integration project, not a benchmarked one. The few concrete
numbers found:

- LoRA training step count is derived directly from configuration:
  `steps = epochs x photos_used` (documented default 40 epochs; e.g. 5 epochs
  x 27 photos = 135 steps, described in the README as producing a
  practically invisible LoRA; 40 epochs x 27 photos = 1,080 steps).
- LoRA strength is reported as the median of ‖B·A‖/‖W‖ over adapted modules;
  the README documents an empirical, non-authoritative "healthy" band of
  0.03-0.10, explicitly stating this does not replace visual evaluation, and
  notes the computation was buggy in earlier reports (`modules: 0`) and has
  since been fixed.
- One recorded video-generation latency figure in internal status notes:
  ~5.17 seconds for a short MiniMax-H3 clip (single data point from a status
  log, not a benchmark suite — treat as anecdotal).
- Voice sample constraints are fixed by design/testing: 1-30s, up to 20MB,
  3-10s recommended for voice cloning quality.

No systematic throughput/latency benchmark (images/sec, training
steps/sec, TTS realtime factor) was found in the repository. If this project
is used in an application context, that is the first gap worth closing before
claiming "measured performance."

## Relevance

This project demonstrates integrating several independent local inference
services (diffusion, LLM chat, TTS, STT, video) behind one coordinating panel
and an MCP bridge, with real engineering practices: content-addressed/hashed
media storage with a dry-run-first deletion path, a non-trivial automated
test suite including "live" integration tests against running model servers,
and documented (if informal) numerical sanity checks for LoRA training
(strength heuristic, step-count derivation). It does **not** demonstrate GPU
kernel-level work or inference-engine internals — all heavy inference is
delegated to existing engines (diffusers, llama.cpp, ComfyUI); the value here
is multi-service orchestration and operational rigor (hashing, dedup,
lock-protected deletes, auth, HTTPS).

## Suggested publication scope

This repository mixes clean, testable code with a large volume of personal
operational notebooks (`PLAN_*.md`, `GUIA_*.md`, `AUDITORIA_*.md`,
`ESTADO_*.md` — ~15 long Spanish planning/status documents) that reference
local absolute paths and a home-network IP address. Recommended: publish the
Python/shell source, `requirements.txt`, the test suite, and a curated
README; drop or rewrite the planning/status markdown files before making the
repository public (see the parent report for the full file-level list).
