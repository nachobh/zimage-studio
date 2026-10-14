#!/bin/bash
# gen_zimage_gguf.sh - Generador interactivo de imágenes Z-Image (GGUF)
# Servidor: zimage_server_gguf.py en puerto configurable via ZIMAGE_SERVER_PORT
# Modelo: transformer GGUF (jayn7/Z-Image-Turbo-GGUF) + text encoder GGUF (Qwen3-4B)

set -e
SERVER_HOST="127.0.0.1"
SERVER_PORT="${ZIMAGE_SERVER_PORT:-9877}"
PYTHON="~/ai-rocm-venv/bin/python3"
CLIENT_SCRIPT="/tmp/zimage_client_gguf_$(date +%s).py"

echo ""
echo "  Z-IMAGE GENERATOR (GGUF)"
echo "  Transformer GGUF (jayn7/Z-Image-Turbo-GGUF) + text encoder GGUF (Qwen3-4B)"
echo "  Servidor: ${SERVER_HOST}:${SERVER_PORT}"
echo ""

# Función para pedir entrada con valor por defecto
# IMPRIME EL PROMPT A STDERR para que no se capture en $(...)
ask() {
    local prompt="$1"
    local default="$2"
    local value
    if [ -n "$default" ]; then
        printf "  %s [%s]: " "$prompt" "$default" >&2
    else
        printf "  %s: " "$prompt" >&2
    fi
    read -r value
    if [ -z "$value" ]; then
        echo "$default"
    else
        echo "$value"
    fi
}

# 1. Verificar servidor
echo "  [1/8] Verificando servidor..."
if $PYTHON -c "import socket, json, struct;
try:
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM); conn.settimeout(300); conn.connect(('127.0.0.1', ${SERVER_PORT})); data = json.dumps({'cmd': 'status'}).encode('utf-8'); conn.sendall(struct.pack('!I', len(data)) + data); header = conn.recv(4); length = struct.unpack('!I', header)[0]; response = b'';
    while len(response) < length:
        chunk = conn.recv(length - len(response));
        if not chunk: break
        response += chunk
    result = json.loads(response.decode('utf-8'));
    if result.get('status') == 'ok': print('OK')
    else: print('FAIL')
    conn.close()
except: print('FAIL')
" 2>/dev/null | grep -q "OK"; then
    echo "  OK - Servidor conectado"
    LORAS_JSON=$($PYTHON -c "import socket, json, struct;
conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM); conn.connect(('127.0.0.1', ${SERVER_PORT}));
data = json.dumps({'cmd': 'list_loras'}).encode('utf-8'); conn.sendall(struct.pack('!I', len(data)) + data);
header = conn.recv(4); length = struct.unpack('!I', header)[0]; response = b'';
while len(response) < length:
    chunk = conn.recv(length - len(response));
    if not chunk: break
    response += chunk
print(response.decode('utf-8')); conn.close()
" 2>/dev/null)
    echo ""
else
    echo "  ADVERTENCIA: El servidor no está en 127.0.0.1:${SERVER_PORT}"
    echo "  ¿Iniciar el servidor automáticamente? [S/n]: "
    if [ "$(ask "" "s")" != "n" ]; then
        echo "  Iniciando zimage_server_gguf.py ..."
        "$PYTHON" ~/zimage_server_gguf.py >/tmp/zimage_server_gguf.log 2>&1 &
        SRV_PID=$!
        echo "  Servidor PID $SRV_PID - esperando a que cargue el modelo..."
        for i in $(seq 1 90); do
            if $PYTHON -c "import socket, json, struct;
try:
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM); conn.settimeout(5); conn.connect(('127.0.0.1', ${SERVER_PORT})); data = json.dumps({'cmd': 'status'}).encode('utf-8'); conn.sendall(struct.pack('!I', len(data)) + data); header = conn.recv(4); length = struct.unpack('!I', header)[0]; response = b'';
    while len(response) < length:
        chunk = conn.recv(length - len(response));
        if not chunk: break
        response += chunk
    result = json.loads(response.decode('utf-8'));
    if result.get('status') == 'ok' and result.get('model_loaded'): print('OK')
    else: print('WAIT')
    conn.close()
except: print('WAIT')
" 2>/dev/null | grep -q "OK"; then
                echo "  OK - Servidor listo (esperó ${i}s)"
                break
            fi
            sleep 1
        done
        LORAS_JSON=""
    fi
    echo ""
fi

# 2. Prompt
echo "  [2/8] PROMPT"
echo "  Descripción de la imagen a generar"
echo ""
PROMPT=$(ask "  Prompt" "")
if [ -z "$PROMPT" ]; then
    echo "  Usando prompt por defecto..."
    PROMPT="a beautiful mountain landscape at sunset, ultra detailed"
fi

# 3. Negative prompt
echo ""
echo "  [3/8] NEGATIVE PROMPT (opcional)"
NEGATIVE=$(ask "  Negative prompt" "")
if [ -z "$NEGATIVE" ]; then
    NEGATIVE=""
fi

# 4. Dimensiones
echo ""
echo "  [4/8] DIMENSIONES"
echo "  Opciones rápidas:"
echo "    1 - 512x512   (rápido, prueba)"
echo "    2 - 768x768   (medio)"
echo "    3 - 1024x1024 (alta calidad, recomendado)"
echo ""
echo -n "  Elige (1/2/3) o escribe ANCHOxALTO: "
read -r dim_choice
case "$dim_choice" in
    1) WIDTH=512; HEIGHT=512 ;;
    2) WIDTH=768; HEIGHT=768 ;;
    3) WIDTH=1024; HEIGHT=1024 ;;
    *)
        echo -n "  Formato: ANCHOxALTO (ej: 1024x1024): "
        read -r dim_input
        WIDTH=$(echo "$dim_input" | cut -dx -f1)
        HEIGHT=$(echo "$dim_input" | cut -dx -f2)
        if ! echo "$WIDTH" | grep -qE '^[0-9]+$'; then
            echo "  Ancho inválido, usando 1024"
            WIDTH=1024
        fi
        if ! echo "$HEIGHT" | grep -qE '^[0-9]+$'; then
            echo "  Alto inválido, usando 1024"
            HEIGHT=1024
        fi
        ;;
esac
echo "  Dimensiones: ${WIDTH}x${HEIGHT}"

# 5. Pasos y CFG
echo ""
echo "  [5/8] CONFIGURACIÓN DE GENERACIÓN"
STEPS=$(ask "  Pasos de generación" "8")
CFG=$(ask "  CFG / guidance_scale (Turbo usa 0.0)" "0.0")

# 6. Seed
echo ""
echo "  [6/8] SEED"
echo "  0 = aleatorio, otro valor = reproducibilidad"
SEED=$(ask "  Seed" "0")
if [ "$SEED" = "0" ] || [ -z "$SEED" ]; then
    SEED=$RANDOM
fi

# 7. LoRA
echo ""
echo "  [7/8] LoRA"
echo "  Sin LoRA = imagen base del modelo"
echo ""
if [ -n "$LORAS_JSON" ]; then
    echo "  LoRAs disponibles en el servidor:"
    echo "$LORAS_JSON" | $PYTHON -c "
import sys, json
try:
    data = json.loads(sys.stdin.read())
    for lora in data.get('lora_list', []):
        print('    {} ({} MB)'.format(lora['name'], lora['size_mb']))
except:
    pass
" 2>/dev/null || echo "  (no se pudo parsear la lista)"
fi
echo ""

# Bucle para múltiples LoRAs
LOLA_LIST=""
while true; do
    LORA_NAME=$(ask "  Nombre del LoRA (o 'none' para terminar)")
    if [ "$LORA_NAME" = "none" ] || [ -z "$LORA_NAME" ] || [ "$LORA_NAME" = "None" ]; then
        break
    fi
    LORA_WEIGHT=$(ask "  Peso del LoRA (0.0-1.0, 1.0 = máximo)" "0.8")
    if [ -z "$LORA_WEIGHT" ] || [ "$LORA_WEIGHT" = "1" ]; then
        if [ -n "$LOLA_LIST" ]; then
            LOLA_LIST="${LOLA_LIST};${LORA_NAME}"
        else
            LOLA_LIST="${LOLA_LIST}${LORA_NAME}"
        fi
    else
        if [ -n "$LOLA_LIST" ]; then
            LOLA_LIST="${LOLA_LIST};${LORA_NAME}:${LORA_WEIGHT}"
        else
            LOLA_LIST="${LORA_NAME}:${LORA_WEIGHT}"
        fi
    fi
    echo "    -> Añadido: ${LORA_NAME} @ ${LORA_WEIGHT}"
    ADD_MORE=$(ask "  ¿Añadir otro LoRA?" "n")
    if [ "$ADD_MORE" != "s" ] && [ "$ADD_MORE" != "S" ] && [ "$ADD_MORE" != "si" ] && [ "$ADD_MORE" != "Si" ]; then
        break
    fi
done
LORA="$LOLA_LIST"

# 8. Resumen
echo ""
echo "  === RESUMEN ==="
echo "  Servidor:    ${SERVER_HOST}:${SERVER_PORT}"
echo "  Prompt:       ${PROMPT}"
if [ -n "$NEGATIVE" ]; then
    echo "  Negative:     ${NEGATIVE}"
fi
echo "  Dimensiones:  ${WIDTH}x${HEIGHT}"
echo "  Pasos:        ${STEPS}"
echo "  CFG:          ${CFG}"
echo "  Seed:         ${SEED}"
if [ -n "$LORA" ]; then
    echo "  LoRA:         ${LORA}"
else
    echo "  LoRA:         (ninguno)"
fi
echo ""

CONFIRM=$(ask "  ¿Generar imagen?" "s")
if [ "$CONFIRM" != "s" ] && [ "$CONFIRM" != "S" ] && [ "$CONFIRM" != "si" ] && [ "$CONFIRM" != "Si" ]; then
    echo "  Cancelado."
    exit 0
fi

# Generar imagen vía el cliente Python (que habla con el servidor TCP)
echo ""
echo "  === INICIANDO GENERACIÓN ==="
echo ""

$PYTHON - "$SERVER_HOST" "$SERVER_PORT" "$PROMPT" "$NEGATIVE" "$WIDTH" "$HEIGHT" "$STEPS" "$CFG" "$SEED" "$LORA" << 'PYCLIENT'
import socket, json, struct, sys, random
host, port = sys.argv[1], int(sys.argv[2])
prompt = sys.argv[3]; negative = sys.argv[4]
width = int(sys.argv[5]); height = int(sys.argv[6])
steps = int(sys.argv[7]); cfg = float(sys.argv[8]); seed = int(sys.argv[9])
lora = sys.argv[10]
conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
conn.settimeout(600)
conn.connect((host, port))
payload = {"cmd": "generate", "prompt": prompt, "width": width,
           "height": height, "steps": steps, "cfg": cfg, "seed": seed}
if negative:
    payload["negative_prompt"] = negative
if lora:
    payload["lora"] = lora
data = json.dumps(payload).encode("utf-8")
conn.sendall(struct.pack("!I", len(data)) + data)
header = conn.recv(4)
length = struct.unpack("!I", header)[0]
resp = b""
while len(resp) < length:
    chunk = conn.recv(length - len(resp))
    if not chunk: break
    resp += chunk
result = json.loads(resp.decode("utf-8"))
conn.close()
print(json.dumps(result, indent=2))
if result.get("status") == "ok":
    print()
    print("  IMAGEN GENERADA: " + result.get("path", "N/A"))
    print("  TIEMPO: " + result.get("duration", "N/A"))
    print("  VRAM: " + str(result.get("vram_gb", "N/A")) + " GB")
else:
    print()
    print("  ERROR: " + result.get("error", "Desconocido"))
PYCLIENT

echo ""
echo "  Fin."
