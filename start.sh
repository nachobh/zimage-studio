#!/usr/bin/env bash
# Bootstrap portable del estudio Z-Image.
# Primera ejecución: crea el venv, instala dependencias y descarga modelos.
# Ejecuciones siguientes: comprueba lo que falta y ejecuta zimage_stack.sh.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${ZIMAGE_DATA_DIR:-$PROJECT_ROOT/data}"
export ZIMAGE_TRAIN_MODEL_DIR="${ZIMAGE_TRAIN_MODEL_DIR:-~/models/zimage-official}"
VENV_DIR="${ZIMAGE_VENV_DIR:-$PROJECT_ROOT/.venv}"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"
PYTHON="$VENV_DIR/bin/python3"
INSTALL=1
DOWNLOAD=1
MODE="start"
STOP=0
OPTIONAL_ARGS=()
HTTPS_IP=""
WITH_COMFYUI=1
COMFYUI_EXPLICIT=0
if [ "$#" -eq 1 ] && [ "$1" = --restart-qwen3-tts ]; then
  exec "$PYTHON_BOOTSTRAP" "$PROJECT_ROOT/zimage_optional_services.py" --restart-qwen3-tts --no-install
fi

usage() {
  cat <<EOF
Uso: bash start.sh [opciones]

  --with-audio    Instala y arranca Kokoro/Whisper (GPU ROCm 10 si se detecta).
  --with-qwen3-tts Añade Qwen3-TTS; incluye el puente de audio Kokoro/Whisper.
  --with-kokoclone Añade voz por WAV; incluye audio.
  --with-comfyui  Exige MiniMax-H3/ComfyUI local (lanzador obligatorio).
  --no-comfyui    No inicia MiniMax-H3/ComfyUI (viene activado por defecto).
  --https-ip IP   Crea/verifica certificado local; requiere confianza en el móvil.
  --extras-only   Prepara solo audio/HTTPS elegidos, sin arrancar el stack base.
  --stop          Detiene audio y stack gestionados, sin instalar ni descargar.
  --restart-qwen3-tts Reinicia solo Qwen3-TTS de esta instalación; no toca el stack.
  --status        No instala ni arranca; muestra el estado del stack.
  --dry-run       Muestra los pasos sin instalar, descargar ni arrancar.
  --models-only   Prepara dependencias y modelos, sin arrancar servicios.
  --no-install    No modifica el entorno Python.
  --no-download   No descarga modelos; falla si faltan.
  -h, --help      Muestra esta ayuda.

Variables útiles:
  ZIMAGE_DATA_DIR       Datos/modelos/LoRA/salidas (por defecto ./data).
  ZIMAGE_VENV_DIR       Entorno Python (por defecto ./.venv).
  LLAMA_BIN             Ruta a llama-server compatible con tu GPU.
  MODEL_FILE            GGUF de chat Qwen alternativo.
  LLAMA_MMPROJ          mmproj alternativo.
  ZIMAGE_TORCH_PACKAGE  Paquete torch; por defecto torch.
  ZIMAGE_TORCH_INDEX_URL Índice opcional para una build ROCm de torch.
  ZIMAGE_AUDIO_VENV      Entorno compartido de Kokoro/KokoClone.
  ZIMAGE_AUDIO_DEVICE    auto, cuda o cpu; por defecto auto.
  COMFYUI_START_SCRIPT  Lanzador MiniMax-H3 configurado; por defecto ./levantar_minimax_h3.sh.
  MINIMAX_H3_INSTALL_ROOT Ruta de ComfyUI/venv/modelos (por defecto directorio padre del proyecto).
  ZIMAGE_PANEL_HOST     Host del panel; por defecto 0.0.0.0.
  ZIMAGE_PANEL_PASSWORD Contraseña del panel en LAN (si falta, se genera en data/conversations/panel-password).
  ZIMAGE_PANEL_REQUIRE_AUTH=1 Fuerza inicio de sesión también en loopback.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --with-audio) OPTIONAL_ARGS+=(--audio); shift ;;
    --with-qwen3-tts) OPTIONAL_ARGS+=(--audio --qwen3-tts-trial); shift ;;
    --with-kokoclone) OPTIONAL_ARGS+=(--kokoclone); shift ;;
    --with-comfyui) WITH_COMFYUI=1; COMFYUI_EXPLICIT=1; shift ;;
    --no-comfyui) WITH_COMFYUI=0; shift ;;
    --https-ip) HTTPS_IP="${2:?falta IP}"; OPTIONAL_ARGS+=(--https-ip "$HTTPS_IP"); shift 2 ;;
    --extras-only) MODE=extras-only; shift ;;
    --status) MODE=status; shift ;;
    --stop) STOP=1; shift ;;
    --restart-qwen3-tts) echo '--restart-qwen3-tts se usa como única opción.' >&2; exit 2 ;;
    --dry-run) MODE=dry-run; shift ;;
    --models-only) MODE=models-only; shift ;;
    --no-install) INSTALL=0; shift ;;
    --no-download) DOWNLOAD=0; shift ;;
    *) echo "Opción desconocida: $1" >&2; usage >&2; exit 2 ;;
  esac
done

export COMFYUI_START_SCRIPT="${COMFYUI_START_SCRIPT:-$PROJECT_ROOT/levantar_minimax_h3.sh}"
export MINIMAX_H3_STATE_DIR="${MINIMAX_H3_STATE_DIR:-$PROJECT_ROOT/.service-state/minimax-h3}"
if [ "$WITH_COMFYUI" -eq 1 ] && [ ! -f "$COMFYUI_START_SCRIPT" ]; then
  if [ "$COMFYUI_EXPLICIT" -eq 1 ]; then
    echo "--with-comfyui requiere un lanzador existente: $COMFYUI_START_SCRIPT" >&2; exit 2
  fi
  echo "Aviso: sin lanzador MiniMax-H3 ($COMFYUI_START_SCRIPT); se continúa sin ComfyUI." >&2
  WITH_COMFYUI=0
fi
if [ "$WITH_COMFYUI" -eq 1 ]; then
  export MINIMAX_H3_INSTALL_ROOT="${MINIMAX_H3_INSTALL_ROOT:-$(cd -- "$(dirname -- "$COMFYUI_START_SCRIPT")/.." && pwd)}"
  OPTIONAL_ARGS+=(--comfyui)
  if [ "$COMFYUI_EXPLICIT" -eq 1 ]; then export ZIMAGE_COMFYUI_REQUIRED=1; fi
fi

if [ "$STOP" -eq 1 ]; then
  if [ "$MODE" = dry-run ]; then
    echo 'Parada: servicios de voz y stack de esta instalación; sin arranque ni instalación.'
    exit 0
  fi
  if [ "$MODE" != start ]; then echo '--stop no se combina con otros modos salvo --dry-run.' >&2; exit 2; fi
  OPTIONAL_STOP_STATUS=0
  "$PYTHON_BOOTSTRAP" "$PROJECT_ROOT/zimage_optional_services.py" --stop || OPTIONAL_STOP_STATUS=$?
  export ZIMAGE_ROOT="$PROJECT_ROOT" ZIMAGE_VENV_PY="$PYTHON"
  export ZIMAGE_LOG_DIR="${ZIMAGE_LOG_DIR:-$DATA_DIR/logs}"
  bash "$PROJECT_ROOT/zimage_stack.sh" --stop
  exit "$OPTIONAL_STOP_STATUS"
fi

source "$PROJECT_ROOT/bootstrap_host.sh"
if [ "$MODE" = dry-run ]; then
  printf '%s\n' "1. Python y venv: $VENV_DIR" "2. PyTorch y cálculo GPU ROCm/CUDA" "3. Dependencias Diffusers/GGUF" "4. llama-server existente o compilación" "5. Modelos: $DATA_DIR/models" "6. Arranque del stack"
  printf "Extras solicitados: %s\n" "${OPTIONAL_ARGS[*]:-ninguno}"
  if [ "$WITH_COMFYUI" -eq 1 ]; then printf 'ComfyUI/MiniMax: %s (URL http://127.0.0.1:%s)\n' "$COMFYUI_START_SCRIPT" "${MINIMAX_H3_PORT:-9879}"; fi
  exit 0
fi
cd "$PROJECT_ROOT"
export ZIMAGE_ROOT="$PROJECT_ROOT"
export ZIMAGE_VENV_PY="$PYTHON"
export ZIMAGE_MODELS_DIR="$DATA_DIR/models"
export ZIMAGE_BASE_PIPELINE="${ZIMAGE_BASE_PIPELINE:-$DATA_DIR/models/zimage-official}"
export ZIMAGE_TRANSFORMER_GGUF="${ZIMAGE_TRANSFORMER_GGUF:-$DATA_DIR/models/Z-Image-Turbo-GGUF/z_image_turbo-${ZIMAGE_TRANSFORMER_QUANT:-Q3_K_M}.gguf}"
export ZIMAGE_TEXT_ENCODER_DIR="${ZIMAGE_TEXT_ENCODER_DIR:-$DATA_DIR/models/Qwen3-4b-Z-Image-Turbo-AbliteratedV1}"
export ZIMAGE_LORA_DIR="${ZIMAGE_LORA_DIR:-$DATA_DIR/lora_models_peft}"
export LORA_DIR="$ZIMAGE_LORA_DIR"
export ZIMAGE_OUTPUT_DIR="${ZIMAGE_OUTPUT_DIR:-$DATA_DIR/outputs}"
export MODEL_FILE="${MODEL_FILE:-$DATA_DIR/models/qwen/Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-MTP-Q4_K_M.gguf}"
export LLAMA_MMPROJ="${LLAMA_MMPROJ:-$DATA_DIR/models/qwen/mmproj-BF16.gguf}"

export ZIMAGE_DATA_DIR="$DATA_DIR"
export SETTINGS_FILE="${SETTINGS_FILE:-$DATA_DIR/panel-settings.json}"
export ZIMAGE_LOG_DIR="${ZIMAGE_LOG_DIR:-$DATA_DIR/logs}"
export PANEL_HOST="${ZIMAGE_PANEL_HOST:-0.0.0.0}"
export MCP_BIND="${ZIMAGE_MCP_BIND:-127.0.0.1}"
if [ "$MODE" = status ]; then
  "$PYTHON_BOOTSTRAP" "$PROJECT_ROOT/zimage_optional_services.py" --status
  exec bash "$PROJECT_ROOT/zimage_stack.sh" --status
fi
if [ "$MODE" = extras-only ]; then
  if [ "$INSTALL" -eq 0 ]; then OPTIONAL_ARGS+=(--no-install); fi
  "$PYTHON_BOOTSTRAP" "$PROJECT_ROOT/zimage_optional_services.py" "${OPTIONAL_ARGS[@]}"
  echo '[start] Si creaste HTTPS con el panel ya activo, reinicia solo el panel.'
  exit 0
fi
trap 'echo "[start] Falló la línea $LINENO. Corrige el error indicado y vuelve a ejecutar ./start.sh; se reutiliza lo completado." >&2' ERR
mkdir -p "$DATA_DIR" "$ZIMAGE_LORA_DIR" "$ZIMAGE_OUTPUT_DIR" "$ZIMAGE_LOG_DIR"
prepare_python
prepare_dependencies
if [ "$MODE" != models-only ]; then
  for command in ss pgrep curl; do
    if ! command -v "$command" >/dev/null; then system_packages iproute2 procps curl; break; fi
  done
  prepare_llama
  export LLAMA_BIN
fi

if [ "$DOWNLOAD" -eq 1 ]; then
  "$PYTHON" "$PROJECT_ROOT/bootstrap_models.py" --data "$DATA_DIR"
else
  for required in "$ZIMAGE_BASE_PIPELINE/model_index.json" "$ZIMAGE_TRANSFORMER_GGUF" "$ZIMAGE_TEXT_ENCODER_DIR/Z-Image-AbliteratedV1.${ZIMAGE_TEXT_ENCODER_QUANT:-Q3_K_M}.gguf" "$MODEL_FILE" "$LLAMA_MMPROJ"; do
    [ -f "$required" ] || { echo "Falta modelo con --no-download: $required" >&2; exit 4; }
  done
fi

if [ "$MODE" = models-only ]; then
  echo "[start] Dependencias y modelos preparados. LoRAs: $ZIMAGE_LORA_DIR"
  exit 0
fi
if [ "$INSTALL" -eq 0 ]; then OPTIONAL_ARGS+=(--no-install); fi
if [ "${#OPTIONAL_ARGS[@]}" -gt 0 ]; then
  "$PYTHON" "$PROJECT_ROOT/zimage_optional_services.py" "${OPTIONAL_ARGS[@]}"
fi
if [ -n "$HTTPS_IP" ]; then echo '[start] HTTPS necesita reiniciar el panel si ya estaba activo; instala ca.crt en el móvil.'; fi
exec bash "$PROJECT_ROOT/zimage_stack.sh" --wait-timeout "${ZIMAGE_WAIT_TIMEOUT:-300}"
