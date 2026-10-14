#!/usr/bin/env bash
# Acceso explícito a la instalación existente de esta máquina (start_local.sh), sin sustituir start.sh.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXISTING_START="${ZIMAGE_EXISTING_START:-$PROJECT_ROOT/start_local.sh}"
export ZIMAGE_TRAIN_MODEL_DIR="${ZIMAGE_TRAIN_MODEL_DIR:-~/models/zimage-official}"

if [ ! -f "$EXISTING_START" ]; then
  echo "No existe el lanzador de instalación existente: $EXISTING_START" >&2
  echo "Usa ./start.sh para el flujo portable o define ZIMAGE_EXISTING_START." >&2
  exit 2
fi
if [ "$(cd -- "$(dirname -- "$EXISTING_START")" && pwd)/$(basename -- "$EXISTING_START")" = "$PROJECT_ROOT/start.sh" ]; then
  echo 'El lanzador existente no puede ser el start.sh portable de este proyecto.' >&2
  exit 2
fi
if [ "$#" -eq 1 ] && [ "$1" = --restart-qwen3-tts ]; then
  exec bash "$EXISTING_START" --restart-qwen3-tts
fi

# Los argumentos explícitos van al final: --https-ip prevalece sobre el valor
# local predeterminado. El lanzador conserva --status/--dry-run sin cambios.
# El acceso a la instalación existente publica HTTP y HTTPS en todas las
# interfaces; el usuario puede limitarlo con ZIMAGE_PANEL_HOST.
export ZIMAGE_PANEL_HOST="${ZIMAGE_PANEL_HOST:-0.0.0.0}"
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      echo 'Instalación existente: Qwen3-TTS + puente de audio y HTTPS activados por defecto.'
      echo 'IP del certificado: ZIMAGE_HTTPS_IP (<PANEL_IP> por defecto), o --https-ip IP.'
      echo '--with-comfyui exige el lanzador local de MiniMax-H3 (API loopback :9879); --no-comfyui lo desactiva (viene activado por defecto).'
      echo '--stop detiene los servicios gestionados; --stop --dry-run muestra la parada.'
      echo '--restart-qwen3-tts reinicia solo Qwen3-TTS sin tocar el panel ni el stack.'
      echo 'Panel HTTP/HTTPS: 0.0.0.0 por defecto; ZIMAGE_PANEL_HOST permite cambiarlo.'
      echo 'La bandeja en / requiere contraseña en LAN: ZIMAGE_PANEL_PASSWORD o data/conversations/panel-password.'
      break ;;
  esac
done
exec bash "$EXISTING_START" --with-qwen3-tts \
  --https-ip "${ZIMAGE_HTTPS_IP:-<PANEL_IP>}" "$@"
