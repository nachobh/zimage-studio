#!/usr/bin/env bash
set -euo pipefail
CLONE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CLONE_REPO="${ZIMAGE_KOKOCLONE_REPO:-$CLONE_ROOT/vendor/kokoclone}"
CLONE_VENV="${ZIMAGE_KOKOCLONE_VENV:-${ZIMAGE_AUDIO_VENV:-$CLONE_ROOT/.venv-audio-rocm10}}"
CLONE_REV=dd6bd3acd3010dc223978839761db74957195f98
if [ ! -d "$CLONE_REPO/.git" ]; then
  git clone https://github.com/Ashish-Patnaik/kokoclone.git "$CLONE_REPO"
  git -C "$CLONE_REPO" checkout --detach "$CLONE_REV"
fi
if [ ! -x "$CLONE_VENV/bin/python" ]; then
  if [ ! -x "$CLONE_ROOT/.audio-tools/bin/uv" ]; then
    python3 -m venv "$CLONE_ROOT/.audio-tools"
    "$CLONE_ROOT/.audio-tools/bin/python" -m pip install uv
  fi
  "$CLONE_ROOT/.audio-tools/bin/uv" venv --seed --python 3.12 "$CLONE_VENV"
fi
if ! "$CLONE_VENV/bin/python" -c 'import kanade_tokenizer, kokoro_onnx, misaki, soundfile' >/dev/null 2>&1; then
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
  echo "Instalando PyTorch para KokoClone desde $TORCH_INDEX_URL"
  "$CLONE_VENV/bin/python" -m pip install "$TORCH_PACKAGE" "$TORCHAUDIO_PACKAGE" --index-url "$TORCH_INDEX_URL"
  "$CLONE_VENV/bin/python" -m pip install kokoro-onnx 'misaki==0.9.4' soundfile 'git+https://github.com/frothywater/kanade-tokenizer@961f20bf892c59f391d0b6c5f7b88e70ed919b99'
fi
if [ "${1:-}" = "--install-only" ]; then exit 0; fi
export ZIMAGE_KOKOCLONE_REPO="$CLONE_REPO"
exec "$CLONE_VENV/bin/python" "$CLONE_ROOT/zimage_kokoclone_server.py" "$@"
