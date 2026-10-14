#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${ZIMAGE_DATA_DIR:-$SCRIPT_ROOT/data}"
LLAMA_BIN="${LLAMA_BIN:-$(command -v llama-server || true)}"
MODEL_FILE="${MODEL_FILE:-$DATA_DIR/models/qwen/Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-MTP-Q4_K_M.gguf}"
LLAMA_MMPROJ="${LLAMA_MMPROJ:-$DATA_DIR/models/qwen/mmproj-BF16.gguf}"
[ -x "$LLAMA_BIN" ] || { echo 'Falta llama-server: ejecuta ./start.sh' >&2; exit 1; }
for file in "$MODEL_FILE" "$LLAMA_MMPROJ"; do
  [ -s "$file" ] || { echo "Falta modelo: $file" >&2; exit 1; }
done
if [ -n "${QWEN35_ROCM_PATH:-}" ]; then
  export LD_LIBRARY_PATH="$QWEN35_ROCM_PATH/lib:$QWEN35_ROCM_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
# Selección de dispositivo automática: no fijar ROCm0 en una build Vulkan.
args=(--model "$MODEL_FILE" --mmproj "$LLAMA_MMPROJ" --jinja
      -c "${QWEN_CONTEXT:-32768}" -ngl "${QWEN_GPU_LAYERS:-999}"
      --parallel 1 --host "${QWEN_BIND:-127.0.0.1}" --port "${QWEN_PORT:-8086}"
      --timeout 900)
# El chat del panel usa su propio puente MCP y no necesita el proxy WebUI.
if [ "${QWEN_WEBUI_MCP_PROXY:-0}" = 1 ]; then args+=(--webui-mcp-proxy); fi
exec "$LLAMA_BIN" "${args[@]}"
