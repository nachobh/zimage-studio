#!/usr/bin/env bash
set -euo pipefail
AUDIO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AUDIO_VENV="${ZIMAGE_AUDIO_VENV:-$AUDIO_ROOT/.venv-audio-rocm10}"
if ! python3 -c "import ctypes.util; assert ctypes.util.find_library('espeak-ng')" >/dev/null 2>&1; then
  echo 'Kokoro necesita espeak-ng. Instálalo con: sudo apt-get install espeak-ng' >&2
  exit 1
fi
if [ ! -x "$AUDIO_VENV/bin/python" ]; then
  # Kokoro 0.9.4 requires Python >=3.10,<3.13. Keep system Python untouched.
  if python3 -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,13)' 2>/dev/null; then
    python3 -m venv "$AUDIO_VENV"
  else
    if [ ! -x "$AUDIO_ROOT/.audio-tools/bin/uv" ]; then
      python3 -m venv "$AUDIO_ROOT/.audio-tools"
      "$AUDIO_ROOT/.audio-tools/bin/python" -m pip install uv
    fi
    "$AUDIO_ROOT/.audio-tools/bin/uv" venv --seed --python 3.12 "$AUDIO_VENV"
  fi
fi
if ! "$AUDIO_VENV/bin/python" -c 'import kokoro, faster_whisper, soundfile, torchaudio' >/dev/null 2>&1; then
  ROCM_VERSION="$(sed -n '1p' /opt/rocm/core-10.0/.info/version 2>/dev/null || true)"
  ROCM_ARCH="${ZIMAGE_ROCM_ARCH:-$(rocminfo 2>/dev/null | sed -n 's/.*Name:[[:space:]]*\(gfx[0-9]*\).*/\1/p' | head -n 1 || true)}"
  if [[ "$ROCM_VERSION" == 10.* && "$ROCM_ARCH" == gfx* ]]; then
    TORCH_INDEX_URL="${ZIMAGE_TORCH_INDEX_URL:-https://stable.repo.amd.com/rocm/whl-next/}"
    TORCH_PACKAGE="${ZIMAGE_AUDIO_TORCH_PACKAGE:-torch[device-$ROCM_ARCH]==2.13.0+rocm10.0.0}"
    TORCHAUDIO_PACKAGE="${ZIMAGE_AUDIO_TORCHAUDIO_PACKAGE:-torchaudio==2.11.0.2+rocm10.0.0}"
  else
    TORCH_INDEX_URL="${ZIMAGE_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
    TORCH_PACKAGE="${ZIMAGE_AUDIO_TORCH_PACKAGE:-torch}"
    TORCHAUDIO_PACKAGE="${ZIMAGE_AUDIO_TORCHAUDIO_PACKAGE:-torchaudio}"
  fi
  echo "Instalando PyTorch para audio desde $TORCH_INDEX_URL"
  "$AUDIO_VENV/bin/python" -m pip install "$TORCH_PACKAGE" "$TORCHAUDIO_PACKAGE" --index-url "$TORCH_INDEX_URL"
  "$AUDIO_VENV/bin/python" -m pip install 'kokoro==0.9.4' 'faster-whisper>=1.1,<2' soundfile
fi
if [ "${1:-}" = "--install-only" ]; then exit 0; fi
exec "$AUDIO_VENV/bin/python" "$AUDIO_ROOT/zimage_audio_server.py" "$@"
