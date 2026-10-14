#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -eq 0 ]]; then set -- --guided; fi
if [[ "${1:-}" == --guided && ! -t 0 ]]; then
  echo 'El modo guiado requiere un terminal interactivo.' >&2; exit 2
fi
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  INSTALL_ROOT="${MINIMAX_H3_INSTALL_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
  exec "$INSTALL_ROOT/minimax-h3-venv/bin/python" "$SCRIPT_DIR/gen_minimax_h3.py" --help
fi
if ! curl -fsS --max-time 2 "http://127.0.0.1:${MINIMAX_H3_PORT:-9879}/system_stats" >/dev/null 2>&1; then
  echo "ComfyUI no está activo; iniciándolo..."
  "$SCRIPT_DIR/levantar_minimax_h3.sh"
fi
INSTALL_ROOT="${MINIMAX_H3_INSTALL_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
exec "$INSTALL_ROOT/minimax-h3-venv/bin/python" "$SCRIPT_DIR/gen_minimax_h3.py" "$@"
