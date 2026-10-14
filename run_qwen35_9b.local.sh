#!/usr/bin/env bash
set -euo pipefail
# Variante de ESTA máquina (ROCm 10, llama-server compilado, contexto 100K). La genérica es run_qwen35_9b.sh.
# HSA/HIP solo viven en ~/.bashrc -> exportarlas aquí para no depender de la shell.
ROCM_ROOT="${QWEN35_ROCM_PATH:-/opt/rocm/core-10.0}"
export ROCM_PATH="$ROCM_ROOT"
export HIP_PATH="$ROCM_ROOT"
export PATH="$ROCM_ROOT/bin:$ROCM_ROOT/llvm/bin:/usr/local/bin:$PATH"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.0}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"


# Usar el llama-server ROCmFPX recompilado con el WebUI embebido actualizado.
# LLAMA_BIN="${LLAMA_BIN:-~/llama-qwen38-rocm10-prs/build-rocm10-gfx1150-gfx1151/bin/llama-server}"
LLAMA_BIN="${LLAMA_BIN:-~/llama-qwen38-rocm10-rocmfpx-hip27841/build-hip27841-gfx1150-gfx1151/bin/llama-server}"
# El build genérico tiene GGML_HIP_GRAPHS=ON y VMM activo; con ROCm 10 se
# bloquea al ejecutar el primer batch en esta GPU.
# LLAMA_BIN="${LLAMA_BIN:-~/llama.cpp/build/bin/llama-server}"
# LLAMA_BIN="${LLAMA_BIN:-~/llama-rocmfpx/build-strix-rocmfp4/bin/llama-server}"
# LLAMA_BIN="${LLAMA_BIN:-~/llama-qwen38-rocm10-prs/build-rocm10-gfx1150-gfx1151/bin/llama-server}"
export LD_LIBRARY_PATH="$(dirname -- "$LLAMA_BIN"):$ROCM_ROOT/lib:$ROCM_ROOT/lib64:$ROCM_ROOT/lib/llvm/lib:$ROCM_ROOT/lib/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ ! -x "$LLAMA_BIN" ]]; then
  echo "No encuentro un llama-server ejecutable en: $LLAMA_BIN" >&2
  exit 1
fi

# Qwen3.5-9B-The-Defiant-Fable (Heretic NEO IMATRIX MAX MTP), el mismo GGUF que usa ~/sabios/launch_sabios.sh
MODEL_FILE="${MODEL_FILE:-~/models/models--DavidAU--Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF/snapshots/def4849b209b3ad226e4761585b74b04d8da743c/Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-MTP-Q4_K_M.gguf}"

if [[ ! -f "$MODEL_FILE" ]]; then
  echo "No encuentro el GGUF local: $MODEL_FILE"
  echo "Configura MODEL_FILE con una ruta local existente. Este script no descargará nada."
  exit 1
fi

# Una sola instancia del 9B. El gestor de stack puede inyectar QWEN_PORT.
# Flags idénticos a los que usa launch_sabios.sh para este modelo (probados):
# contexto 100K, batch 2048/1024, KV q8_0.
COMMON_ARGS=(
  -c 100000
  --fit off
  -ngl 999
  --device ROCm0
  -b 2048
  --ubatch-size 1024
  --jinja
  --parallel 1
  --metrics
  --flash-attn on
  --temp 1.0
  --top-p 0.95
  --min-p 0.0
  --top-k 64
  --repeat-penalty 1.0
  --presence-penalty 0.0
  --cache-type-k q8_0
  --cache-type-v q8_0
  # El modelo es multimodal: cache-reuse no está soportado y puede dejar
  # la primera petición bloqueada después de actualizar ROCm.
  # ROCm 10 / llama.cpp reciente: estos dos flags han sido fuente de cuelgues
  # en otros lanzadores Qwen/Qwen-MTP. La advertencia nueva también marca
  # --no-mmap como deprecated, así que dejamos el load mode por defecto.
  --load-mode mmap
  --host 0.0.0.0
  --port "${QWEN_PORT:-8086}"
  --cors-origins "*"
  --webui-mcp-proxy
  --ui-config '{"mcpRequestTimeoutSeconds":900}'
  --timeout 900
  --mmproj ~/models/models--DavidAU--Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF/snapshots/def4849b209b3ad226e4761585b74b04d8da743c/mmproj-BF16.gguf

)

# Keep argument boundaries intact; the previous ${COMMON_ARGS[*]} form was
# re-parsed by a shell and made paths/options fragile.
if id -nG | tr ' ' '\n' | grep -qx render; then
  exec "$LLAMA_BIN" --model "$MODEL_FILE" "${COMMON_ARGS[@]}"
fi

CMD=()
for arg in "$LLAMA_BIN" --model "$MODEL_FILE" "${COMMON_ARGS[@]}"; do
  printf -v quoted '%q' "$arg"
  CMD+=("$quoted")
done
printf -v command_line '%s ' "${CMD[@]}"
exec sg render -c "exec $command_line"
