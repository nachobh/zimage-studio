#!/usr/bin/env bash
# zimage_stack_doc.sh — documenta y demuestra zimage_stack.sh.
# Uso: bash zimage_stack_doc.sh [--status] [--dry-run-demo] [--check-files]
set -euo pipefail

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ZIMAGE_ROOT:-$SCRIPT_ROOT}"
STACK="$ROOT/zimage_stack.sh"

section() { printf '\n=== %s ===\n' "$*"; }

section "1. Qué es zimage_stack.sh"
cat <<EOF
Gestiona el stack base (por defecto solo arranca lo que falta):
Audio/HTTPS/KokoClone: bash start.sh --with-kokoclone --https-ip <PANEL_IP>
Guía: ZIMAGE_CAMBIOS_MULTIMEDIA.md. HTTPS comparte proceso de panel en :8768. 8766 es HTTP; nunca abras HTTPS en 8766.
Audio :8767 y KokoClone :8769 son extras; --stop/--restart del stack base no los gestionan.

  Qwen :8086 + Z-Image TCP :9877 → MCP :8765 → panel :8766

- Sin flags ............ detecta cada elemento y arranca el que falte.
- --restart ............ para lo arrancado y lo vuelve a arrancar.
- --stop / --status .... para / solo informa.
- --only / --skip ...... filtra componentes (qwen,zimage,mcp,panel).
- Puertos/parámetros ... --qwen-port, --zimage-port/--backend-port,
  --mcp-port, --panel-port/--web-port, --chat-url, --mcp-url,
  --no-external-mcp, --lora-dir, --settings-file, --venv, --root,
  --log-dir, --wait-timeout, --dry-run. Sin indicarlos usa el defecto.
- Detección ............ pgrep -f + ss -ltn + health-check
  (Qwen: curl /props; Z-Image: TCP {"cmd":"status"};
   MCP: POST /mcp initialize; panel: curl /api/loras).
- Parada segura ........ verifica la línea de comando del PID antes
  de matar; espera 15 s y solo entonces usa kill -9.
- Orden ................ arranque Qwen→Z-Image→MCP→panel;
  parada en orden inverso.
- OJO .................. el MCP usa backend 9877 por defecto (ZIMAGE_BACKEND_PORT);
  el script lo pasa explícito (--backend-port 9877 por defecto).
  9877 es TCP (4 bytes big-endian + JSON), no HTTP.
EOF

section "2. Ayuda real del script"
bash "$STACK" --help

section "3. Ejemplos comentados"
cat <<'EOF'
# Arrancar lo que falte (defectos 8086/9877/8765/8766):
bash zimage_stack.sh

# Solo ver estado:
bash zimage_stack.sh --status

# Reiniciar todo:
bash zimage_stack.sh --restart

# HTML/JS: basta recargar el navegador. Si cambia Python del panel:
bash zimage_stack.sh --restart --only panel

# Ver qué haría sin hacerlo:
bash zimage_stack.sh --restart --only zimage --dry-run

# Puertos no estándar:
bash zimage_stack.sh --zimage-port 9878 --mcp-port 8770 --panel-port 8771
# Las URLs internas se ajustan automáticamente; usa --chat-url/--mcp-url
# si Qwen o MCP están en otra máquina.

# Panel con dispatcher integrado (sin proceso MCP externo):
bash zimage_stack.sh --no-external-mcp

# Parar solo el panel:
bash zimage_stack.sh --stop --only panel

# Misma selección vía entorno:
QWEN_PORT=8086 ZIMAGE_SERVER_PORT=9877 MCP_PORT=8765 PANEL_PORT=8766 bash zimage_stack.sh --status
EOF

DO_STATUS=0; DO_DEMO=0; DO_FILES=0
for a in "$@"; do case "$a" in
  --status) DO_STATUS=1;; --dry-run-demo) DO_DEMO=1;;
  --check-files) DO_FILES=1;; -h|--help) sed -n '1,4p' "$0"; exit 0 ;;
  *) echo "Flag doc desconocida: $a (uso: --status --dry-run-demo --check-files)" >&2; exit 2;;
esac; done
[ $# -eq 0 ] && { DO_STATUS=1; DO_DEMO=1; DO_FILES=1; }

if [ "$DO_FILES" -eq 1 ]; then
  section "4. Ficheros esperados en $ROOT"
  for f in start.sh zimage_optional_services.py setup_zimage_https.py zimage_panel_audio.js zimage_panel_video.js comfyui_video.py zimage_audio_server.py zimage_kokoclone_server.py run_zimage_audio.sh run_zimage_kokoclone.sh ZIMAGE_CAMBIOS_MULTIMEDIA.md ZIMAGE_AUDIO.md ZIMAGE_HTTPS.md check_zimage_copies.py run_qwen35_9b.sh zimage_server_gguf.py zimage_mcp_server.py zimage_panel_grid.py \
           zimage_progress.py zimage_panel_progress.js zimage_panel_group.html zimage_panel_group.js \
           gen_minimax_h3.py gen_minimax_h3.sh levantar_minimax_h3.sh minimax_h3_server.py \
           minimax_h3_model_paths.yaml minimax_h3_prompt_speech_example.txt \
           start.sh bootstrap_models.py requirements.txt zimage_stack.sh \
           GUIA_ARRANQUE_REINICIO_ZIMAGE.md GUIA_INSTALACION_CERO_ZIMAGE.md; do
    if [ -e "$ROOT/$f" ]; then echo "OK  $f"; else echo "FALTA  $f"; fi
  done
  if [ -x "${ZIMAGE_VENV_PY:-$ROOT/.venv/bin/python3}" ]; then
    echo "OK  ${ZIMAGE_VENV_PY:-$ROOT/.venv/bin/python3}"
  else
    echo "INFO  venv aún no creado (ejecuta ./start.sh)"
  fi
  if [ -d "${LORA_DIR:-$ROOT/data/lora_models_peft}" ]; then
    echo "OK  ${LORA_DIR:-$ROOT/data/lora_models_peft}"
  else
    echo "INFO  carpeta de LoRA aún no creada (ejecuta ./start.sh)"
  fi
fi
if [ "$DO_STATUS" -eq 1 ]; then
  section "5. Estado actual (--status, solo lectura)"
  bash "$STACK" --status || true
fi
if [ "$DO_DEMO" -eq 1 ]; then
  section "6. Demo sin cambios (--dry-run)"
  bash "$STACK" --dry-run || true
  bash "$STACK" --restart --only panel --dry-run || true
fi

section "7. Puertos, logs y verificación manual"
cat <<EOF
Puertos: Qwen 8086 (HTTP), Z-Image 9877 (TCP), MCP 8765 (HTTP /mcp), panel 8766 (HTTP) / 8768 (HTTPS con certificado), audio 8767 y KokoClone 8769 (loopback).
Logs/PIDs del stack portable: $ROOT/data/logs (o ZIMAGE_LOG_DIR).
Abrir: http://127.0.0.1:8766 (panel), http://127.0.0.1:8766/group (grupo), http://127.0.0.1:8086 (Qwen).
Audio gestionado por start.sh: $ROOT/.service-state (si se reutiliza un servicio, conserva sus logs previos).
HTTPS: https://<PANEL_IP>:8768 tras instalar la CA; ver ZIMAGE_HTTPS.md.
Docs: ZIMAGE_CAMBIOS_MULTIMEDIA.md, ZIMAGE_AUDIO.md, ZIMAGE_HTTPS.md, GUIA_ARRANQUE_REINICIO_ZIMAGE.md, GUIA_INSTALACION_CERO_ZIMAGE.md, README.md, ZIMAGE_PANEL_PERSONAJES.md, AGENTS.md
EOF
