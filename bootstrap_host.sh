#!/usr/bin/env bash
# Funciones del instalador. No ejecuta cambios al ser importado.
fail() { echo "[start] ERROR: $*" >&2; return 1; }
system_packages() {
  [ "$INSTALL" -eq 1 ] || { fail "Faltan paquetes: $* (--no-install)."; return 1; }
  command -v apt-get >/dev/null || { fail "Instala estos paquetes con el gestor de tu distribución: $*"; return 1; }
  local elevate=()
  if [ "$(id -u)" != 0 ]; then
    command -v sudo >/dev/null || { fail "Falta sudo; un administrador debe instalar: $*"; return 1; }
    elevate=(sudo)
  fi
  echo "[start] Instalando paquetes del sistema: $* (puede solicitar contraseña sudo)"
  "${elevate[@]}" apt-get update
  "${elevate[@]}" apt-get install -y --no-install-recommends "$@"
}

prepare_python() {
  if ! command -v "$PYTHON_BOOTSTRAP" >/dev/null; then
    [ "$PYTHON_BOOTSTRAP" = python3 ] || { fail "Intérprete personalizado inexistente: $PYTHON_BOOTSTRAP"; return 1; }
    system_packages python3 python3-venv
  fi
  "$PYTHON_BOOTSTRAP" -c 'import sys; assert sys.version_info >= (3,10), "Se necesita Python >= 3.10"'
  if [ ! -x "$PYTHON" ]; then
    [ "$INSTALL" -eq 1 ] || { fail "Falta el venv $VENV_DIR"; return 1; }
    if ! "$PYTHON_BOOTSTRAP" -c 'import ensurepip' >/dev/null 2>&1; then
      system_packages python3-venv
    fi
    "$PYTHON_BOOTSTRAP" -m venv "$VENV_DIR"
  fi
}

gpu_probe() {
  "$PYTHON" "$PROJECT_ROOT/check_gpu.py"
}

prepare_dependencies() {
  if ! "$PYTHON" -c 'import torch' >/dev/null 2>&1; then
    [ "$INSTALL" -eq 1 ] || { fail 'Falta PyTorch (--no-install).'; return 1; }
    local index="${ZIMAGE_TORCH_INDEX_URL:-}"
    if [ -z "$index" ]; then
      if [ -e /dev/kfd ]; then
        index=https://download.pytorch.org/whl/rocm7.2
      elif command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
        index=https://download.pytorch.org/whl/cu128
      else
        fail 'No hay GPU ROCm/CUDA accesible. Instala el controlador compatible y reinicia. Vulkan solo sirve al chat, no a este servidor Diffusers.'
        return 1
      fi
    fi
    echo "[start] Instalando PyTorch desde $index"
    "$PYTHON" -m pip install "${ZIMAGE_TORCH_PACKAGE:-torch}" --index-url "$index"
  fi
  # Una prueba de cálculo evita descargar modelos si solo funcionan los imports.
  gpu_probe || { fail 'PyTorch no puede calcular en GPU. Revisa el diagnóstico anterior, permisos /dev/kfd y /dev/dri, y la build de torch. No se reemplazará tu instalación automáticamente.'; return 1; }
  if [ "$INSTALL" -eq 1 ]; then
    "$PYTHON" -m pip install -r "$PROJECT_ROOT/requirements.txt"
  fi
  "$PYTHON" -c 'from diffusers import ZImagePipeline, GGUFQuantizationConfig; from transformers import Qwen3Model; import peft, gguf, accelerate, safetensors, PIL, huggingface_hub'
  if [ "${ZIMAGE_PIP_CHECK_WARN_ONLY:-0}" = 1 ]; then
    "$PYTHON" -m pip check || echo '[start] Aviso: pip check reporta conflictos en el entorno; se continúa (ZIMAGE_PIP_CHECK_WARN_ONLY=1).' >&2
  else
    "$PYTHON" -m pip check
  fi
}

prepare_llama() {
  if [ -n "${LLAMA_BIN:-}" ]; then
    [ -x "$LLAMA_BIN" ] || { fail "LLAMA_BIN no es ejecutable: $LLAMA_BIN"; return 1; }
    return
  fi
  LLAMA_BIN="$(command -v llama-server || true)"
  [ -n "$LLAMA_BIN" ] && return
  local source="$PROJECT_ROOT/vendor/llama.cpp"
  LLAMA_BIN="$source/build/bin/llama-server"
  [ -x "$LLAMA_BIN" ] && return
  [ "$INSTALL" -eq 1 ] || { fail 'Falta llama-server (--no-install).'; return 1; }
  # Vulkan evita depender de un compilador HIP distinto del runtime de torch.
  # Se puede elegir cpu para el chat si no hay un ICD Vulkan compatible.
  local backend="${LLAMA_BUILD_BACKEND:-vulkan}" flags=()
  case "$backend" in
    vulkan) system_packages git cmake build-essential libcurl4-openssl-dev libvulkan-dev glslc vulkan-tools
            vulkaninfo --summary || { fail 'Vulkan no funciona; instala el ICD de tu GPU o usa LLAMA_BUILD_BACKEND=cpu.'; return 1; }
            flags=(-DGGML_VULKAN=ON) ;;
    cpu) system_packages git cmake build-essential libcurl4-openssl-dev ;;
    *) fail 'LLAMA_BUILD_BACKEND admite vulkan o cpu; usa LLAMA_BIN para una compilación propia.'; return 1 ;;
  esac
  if [ ! -d "$source/.git" ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$source"
  fi
  if [ -n "${LLAMA_CPP_REF:-}" ]; then
    git -C "$source" fetch --depth 1 origin "$LLAMA_CPP_REF"
    git -C "$source" checkout --detach FETCH_HEAD
  fi
  cmake -S "$source" -B "$source/build" -DCMAKE_BUILD_TYPE=Release "${flags[@]}"
  cmake --build "$source/build" --target llama-server -j "${BUILD_JOBS:-4}"
  git -C "$source" rev-parse HEAD > "$DATA_DIR/llama-build-revision.txt"
}
