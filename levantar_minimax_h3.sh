#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${MINIMAX_H3_INSTALL_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
VENV="$ROOT/minimax-h3-venv"
SERVER="$SCRIPT_DIR/minimax_h3_server.py"
COMFYUI_DIR="${MINIMAX_H3_COMFYUI_DIR:-$ROOT/ComfyUI-minimax-h3}"
MODEL_PATHS="${MINIMAX_H3_MODEL_PATHS:-$SCRIPT_DIR/minimax_h3_model_paths.yaml}"
STATE="${MINIMAX_H3_STATE_DIR:-$SCRIPT_DIR/.service-state/minimax-h3}"
PIDFILE="$STATE/server.pid"
LOG="$STATE/server.log"
PORT="${MINIMAX_H3_PORT:-9879}"

[[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT >= 1024 && PORT <= 65535 )) || {
  echo 'MINIMAX_H3_PORT debe estar entre 1024 y 65535.' >&2; exit 2;
}
[[ "$PORT" != 9877 ]] || { echo 'El puerto 9877 está reservado a Z-Image.' >&2; exit 2; }
API="http://127.0.0.1:$PORT"

process_start() {
  local stat
  stat=$(<"/proc/$1/stat") || return 1
  stat="${stat##*) }"
  set -- $stat
  echo "${20}"
}

owned_process() {
  local pid="$1" expected="$2" arg
  [[ "$pid" =~ ^[0-9]+$ ]] && (( pid > 1 )) || return 1
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  [[ "$(process_start "$pid")" == "$expected" ]] || return 1
  while IFS= read -r -d '' arg; do
    [[ "$arg" == "$SERVER" ]] && return 0
  done <"/proc/$pid/cmdline"
  return 1
}

read_pid() {
  [[ -f "$PIDFILE" ]] || return 1
  read -r SAVED_PID SAVED_START <"$PIDFILE" || return 1
  [[ "$SAVED_PID" =~ ^[0-9]+$ && "$SAVED_START" =~ ^[0-9]+$ ]]
}

healthy() {
  curl -fsS --max-time 2 "$API/system_stats" >/dev/null 2>&1
}

listening() {
  ss -ltnH "( sport = :$PORT )" 2>/dev/null | rg -q LISTEN
}

case "${1:-}" in
  --status|status)
    if read_pid && owned_process "$SAVED_PID" "$SAVED_START"; then
      echo "MiniMax-H3 gestionado: PID $SAVED_PID; $(healthy && echo "API $API sana" || echo 'API no disponible')."
    elif healthy || listening; then
      echo "Puerto $PORT ocupado por un servicio no identificado como propio."
    else
      echo "MiniMax-H3 parado; puerto $PORT libre."
    fi
    exit 0 ;;
  --stop|stop)
    if ! read_pid; then
      if [[ -f "$PIDFILE" ]]; then echo "Pidfile inválido: $PIDFILE" >&2; exit 1; fi
      if healthy || listening; then
        echo "Puerto $PORT ocupado sin PID gestionado; no se detiene ese proceso." >&2; exit 1
      fi
      echo 'MiniMax-H3 no tiene proceso gestionado registrado.'; exit 0
    fi
    if [[ ! -e "/proc/$SAVED_PID" ]]; then
      unlink "$PIDFILE"
      if healthy || listening; then
        echo "PID obsoleto retirado; puerto $PORT sigue ocupado por otro proceso." >&2; exit 1
      fi
      echo 'Pidfile obsoleto retirado; MiniMax-H3 ya estaba parado.'; exit 0
    fi
    if ! owned_process "$SAVED_PID" "$SAVED_START"; then
      echo "PID $SAVED_PID no coincide con el servidor MiniMax-H3 registrado; no se detiene." >&2; exit 1
    fi
    kill -TERM "$SAVED_PID"
    for _ in {1..50}; do
      if ! owned_process "$SAVED_PID" "$SAVED_START"; then
        unlink "$PIDFILE"
        if healthy || listening; then
          echo "PID propio detenido; el puerto $PORT sigue ocupado por otro proceso." >&2; exit 1
        fi
        echo 'MiniMax-H3 detenido.'; exit 0
      fi
      sleep .2
    done
    echo "PID $SAVED_PID sigue activo; revisa $LOG." >&2; exit 1 ;;
  '') ;;
  *) echo 'Uso: levantar_minimax_h3.sh [--status|--stop]' >&2; exit 2 ;;
esac

if read_pid; then
  if owned_process "$SAVED_PID" "$SAVED_START"; then
    if healthy; then echo "MiniMax-H3 ya activo: PID $SAVED_PID, $API."; exit 0; fi
    echo "El proceso MiniMax-H3 $SAVED_PID aún existe, pero la API no está lista; revisa $LOG." >&2; exit 1
  fi
  if [[ -e "/proc/$SAVED_PID" ]]; then
    echo "PID $SAVED_PID pertenece a otro proceso; revisa $PIDFILE antes de arrancar." >&2; exit 1
  fi
  unlink "$PIDFILE"
fi
if healthy || listening; then
  echo "Puerto $PORT ocupado por un servicio ajeno o sin PID verificable; no se inicia otro." >&2
  exit 1
fi

for file in \
  "$SERVER" "$MODEL_PATHS" \
  "$ROOT/models/minimax-h3-q4/diffusion_models/MiniMax-H3-FL2VA-Pruned-Q4_K_M.gguf" \
  "$ROOT/models/minimax-h3-q4/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
  "$ROOT/models/minimax-h3-q4/vae/minimax_h3_video_vae_fp16.safetensors" \
  "$ROOT/models/minimax-h3-q4/vae/minimax_h3_audio_vae_fp32.safetensors" \
  "$VENV/bin/python" "$COMFYUI_DIR/main.py" \
  "$COMFYUI_DIR/custom_nodes/ComfyUI-GGUF/nodes.py"; do
  [[ -f "$file" ]] || { echo "Falta $file" >&2; exit 1; }
done
[[ -e /dev/kfd ]] || { echo 'ROCm no expone /dev/kfd.' >&2; exit 1; }
"$VENV/bin/python" -c 'import torch; assert torch.cuda.is_available()' >/dev/null || {
  echo 'PyTorch ROCm no detecta GPU.' >&2; exit 1;
}

install -d -m 700 "$STATE"
echo "Iniciando MiniMax-H3 en $API..."
export MINIMAX_H3_PORT="$PORT"
export MINIMAX_H3_INSTALL_ROOT="$ROOT" MINIMAX_H3_COMFYUI_DIR="$COMFYUI_DIR" MINIMAX_H3_MODEL_PATHS="$MODEL_PATHS"
setsid -f bash -c '
  pidfile=$1; python=$2; server=$3
  pid=$BASHPID
  stat=$(<"/proc/$pid/stat")
  stat="${stat##*) }"
  set -- $stat
  printf "%s %s\n" "$pid" "${20}" >"$pidfile"
  exec "$python" "$server"
' _ "$PIDFILE" "$VENV/bin/python" "$SERVER" </dev/null >"$LOG" 2>&1
for _ in {1..20}; do
  if read_pid; then break; fi
  sleep .1
done
read_pid || { echo "No se pudo registrar el PID; revisa $LOG." >&2; exit 1; }
pid="$SAVED_PID"
start="$SAVED_START"
for _ in {1..60}; do
  if healthy; then echo "Servidor listo: $API (PID $pid; log: $LOG)"; exit 0; fi
  if ! owned_process "$pid" "$start"; then
    unlink "$PIDFILE"
    echo "Servidor terminó antes de responder; revisa $LOG." >&2; exit 1
  fi
  sleep 1
done
echo "El servidor sigue arrancando tras 60 s (PID $pid); revisa $LOG antes de otro intento." >&2
exit 1
