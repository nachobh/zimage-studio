#!/usr/bin/env bash
set -euo pipefail
TRIAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRIAL_VENV="${QWEN3_TTS_VENV:-$TRIAL_ROOT/.venv-qwen3-tts}"
if [ ! -x "$TRIAL_VENV/bin/python" ] && [ -x "$TRIAL_ROOT/../.venv-qwen3-tts/bin/python" ]; then
  TRIAL_VENV="$TRIAL_ROOT/../.venv-qwen3-tts"
fi
if [ "${1:-}" = "--install-only" ]; then
  if [ ! -x "$TRIAL_VENV/bin/python" ]; then python3 -m venv "$TRIAL_VENV"; fi
  if ! "$TRIAL_VENV/bin/python" -c 'import qwen_tts, torch, torchaudio; assert torch.cuda.is_available()' >/dev/null 2>&1; then
    "$TRIAL_VENV/bin/python" -m ensurepip --upgrade
    ROCM_VERSION="$(sed -n '1p' /opt/rocm/core-10.0/.info/version 2>/dev/null || true)"
    ROCM_ARCH="${ZIMAGE_ROCM_ARCH:-$(rocminfo 2>/dev/null | sed -n 's/.*Name:[[:space:]]*\(gfx[0-9]*\).*/\1/p' | head -n 1 || true)}"
    if [[ "$ROCM_VERSION" == 10.* && "$ROCM_ARCH" == gfx* ]]; then
      "$TRIAL_VENV/bin/python" -m pip install "torch[device-$ROCM_ARCH]==2.13.0+rocm10.0.0" 'torchaudio==2.11.0.2+rocm10.0.0' --index-url https://stable.repo.amd.com/rocm/whl-next/
    elif [ -n "${QWEN3_TTS_TORCH_INDEX_URL:-}" ]; then
      "$TRIAL_VENV/bin/python" -m pip install torch torchaudio --index-url "$QWEN3_TTS_TORCH_INDEX_URL"
    else
      echo 'No se detectó ROCm 10. Define QWEN3_TTS_TORCH_INDEX_URL con el índice GPU compatible.' >&2
      exit 1
    fi
    "$TRIAL_VENV/bin/python" -m pip install 'qwen-tts==0.1.1'
  fi
  "$TRIAL_VENV/bin/python" -c 'import torch; assert torch.cuda.is_available(), "Qwen3-TTS requiere GPU"; print("Qwen3-TTS GPU:", torch.cuda.get_device_name(0), torch.__version__)'
  exit 0
fi
[ -x "$TRIAL_VENV/bin/python" ] || { echo "Falta $TRIAL_VENV. Ejecuta $0 --install-only." >&2; exit 1; }
exec "$TRIAL_VENV/bin/python" "$TRIAL_ROOT/qwen3_tts_trial.py" "$@"
