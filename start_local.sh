#!/usr/bin/env bash
# Arranque de ESTA máquina: reutiliza el venv, modelos, LoRA y datos ya instalados en $HOME.
# Acepta las opciones de start.sh más --restart. Para una instalación nueva usa start.sh.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HOME_DIR="${ZIMAGE_HOME:-$HOME}"
MODELS="$HOME_DIR/models"
QWEN_SNAPSHOT="$MODELS/models--DavidAU--Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF/snapshots/def4849b209b3ad226e4761585b74b04d8da743c"

export ZIMAGE_DATA_DIR="${ZIMAGE_DATA_DIR:-$HOME_DIR/data}"
export ZIMAGE_VENV_DIR="${ZIMAGE_VENV_DIR:-$HOME_DIR/ai-rocm-venv}"
export ZIMAGE_BASE_PIPELINE="${ZIMAGE_BASE_PIPELINE:-$MODELS/zimage-official}"
export ZIMAGE_TRANSFORMER_GGUF="${ZIMAGE_TRANSFORMER_GGUF:-$MODELS/Z-Image-Turbo-GGUF/z_image_turbo-${ZIMAGE_TRANSFORMER_QUANT:-Q3_K_M}.gguf}"
export ZIMAGE_TEXT_ENCODER_DIR="${ZIMAGE_TEXT_ENCODER_DIR:-$MODELS/Qwen3-4b-Z-Image-Turbo-AbliteratedV1}"
export ZIMAGE_LORA_DIR="${ZIMAGE_LORA_DIR:-$HOME_DIR/lora_models_peft}"
export ZIMAGE_OUTPUT_DIR="${ZIMAGE_OUTPUT_DIR:-$HOME_DIR/zimage_outputs_gguf}"
export LLAMA_BIN="${LLAMA_BIN:-$HOME_DIR/llama-qwen38-rocm10-rocmfpx-hip27841/build-hip27841-gfx1150-gfx1151/bin/llama-server}"
export MODEL_FILE="${MODEL_FILE:-$QWEN_SNAPSHOT/Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-MTP-Q4_K_M.gguf}"
export LLAMA_MMPROJ="${LLAMA_MMPROJ:-$QWEN_SNAPSHOT/mmproj-BF16.gguf}"
export QWEN_SCRIPT="${QWEN_SCRIPT:-$PROJECT_ROOT/run_qwen35_9b.local.sh}"
# El venv existente tiene un conflicto datasets/fsspec preexistente que haría fallar pip check.
export ZIMAGE_PIP_CHECK_WARN_ONLY="${ZIMAGE_PIP_CHECK_WARN_ONLY:-1}"
export MINIMAX_H3_STATE_DIR="${MINIMAX_H3_STATE_DIR:-$HOME_DIR/.service-state/minimax-h3}"

if [ "$#" -eq 1 ] && [ "$1" = --restart-qwen3-tts ]; then
  exec bash "$PROJECT_ROOT/start.sh" --restart-qwen3-tts
fi
RESTART=0
ARGS=()
for argument in "$@"; do
  if [ "$argument" = --restart ]; then RESTART=1; else ARGS+=("$argument"); fi
done
if [ "$RESTART" -eq 1 ]; then
  bash "$PROJECT_ROOT/start.sh" --no-install --no-download --stop
fi
exec bash "$PROJECT_ROOT/start.sh" --no-install --no-download "${ARGS[@]}"
