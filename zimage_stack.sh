#!/usr/bin/env bash
# zimage_stack.sh — gestor del stack Z-Image + Qwen + MCP + Panel.
# Sin flags: detecta cada elemento y arranca solo lo que falta.
# Con --restart: para lo arrancado y lo vuelve a arrancar.
# Ver GUIA_ARRANQUE_REINICIO_ZIMAGE.md para el orden y puertos.
set -euo pipefail

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ZIMAGE_ROOT:-$SCRIPT_ROOT}"
ROOT_INITIAL="$ROOT"
VENV_PY="${ZIMAGE_VENV_PY:-$ROOT/.venv/bin/python3}"
LOG_DIR="${ZIMAGE_LOG_DIR:-$ROOT/data/logs}"
# La carga del transformer, text encoder y todos los LoRA puede superar dos
# minutos en esta máquina; 300 s evita declarar fallos durante una carga sana.
WAIT_TIMEOUT="${ZIMAGE_WAIT_TIMEOUT:-300}"

QWEN_PORT="${QWEN_PORT:-8086}"
QWEN_HOST="${QWEN_HOST:-127.0.0.1}"
QWEN_SCRIPT="${QWEN_SCRIPT:-$ROOT/run_qwen35_9b.sh}"

ZIMAGE_PORT="${ZIMAGE_SERVER_PORT:-9877}"
ZIMAGE_HOST="${ZIMAGE_BACKEND_HOST:-127.0.0.1}"
ZIMAGE_SCRIPT="${ZIMAGE_SCRIPT:-$ROOT/zimage_server_gguf.py}"

MCP_PORT="${MCP_PORT:-8765}"
MCP_HOST="${MCP_HOST:-127.0.0.1}"
MCP_BIND="${MCP_BIND:-0.0.0.0}"
MCP_SCRIPT="${MCP_SCRIPT:-$ROOT/zimage_mcp_server.py}"

PANEL_PORT="${PANEL_PORT:-8766}"
PANEL_HOST="${PANEL_HOST:-0.0.0.0}"
PANEL_SCRIPT="${PANEL_SCRIPT:-$ROOT/zimage_panel_grid.py}"
CHAT_URL_FROM_ENV="${CHAT_URL+x}"
MCP_URL_FROM_ENV="${MCP_URL+x}"
CHAT_URL="${CHAT_URL:-http://127.0.0.1:8086/v1}"
MCP_URL="${MCP_URL:-http://127.0.0.1:8765/mcp}"
USE_EXTERNAL_MCP=1
LORA_DIR="${LORA_DIR:-$ROOT/lora_models_peft}"
SETTINGS_FILE="${SETTINGS_FILE:-$ROOT/zimage_panel_settings.json}"

MODE="ensure"   # ensure | status | stop | restart
ONLY="qwen,zimage,mcp,panel"
SKIP=""
DRY_RUN=0

usage() {
  cat <<EOF
Uso: $(basename "$0") [opciones]

Modos (uno solo):
  (sin modo)   Detecta cada elemento y arranca solo lo que falta (por defecto).
  --restart    Si está arrancado, lo para y lo vuelve a arrancar.
  --stop       Para los componentes seleccionados.
  --status     Solo informa del estado, no arranca ni para nada.

Selección:
  --only LISTA        Subconjunto a gestionar (coma-separado entre: qwen,zimage,mcp,panel).
  --skip LISTA        Excluye componentes de la selección.
  --dry-run           Muestra lo que haría sin ejecutarlo.

Puertos y parámetros (si no se indican, valores por defecto del stack):
  --qwen-port P       Puerto llama-server Qwen (def. $QWEN_PORT).
  --zimage-port P     Puerto TCP Z-Image (def. $ZIMAGE_PORT). También: --backend-port.
  --mcp-port P        Puerto HTTP MCP (def. $MCP_PORT).
  --panel-port P      Puerto HTTP panel (def. $PANEL_PORT). También: --web-port.
  --qwen-host H       Host para probar Qwen (def. $QWEN_HOST).
  --backend-host H    Host para probar Z-Image TCP (def. $ZIMAGE_HOST).
  --mcp-host H        Host para probar MCP (def. $MCP_HOST).
  --mcp-bind H        Bind al arrancar MCP (def. $MCP_BIND).
  --panel-host H      Bind/host del panel (def. $PANEL_HOST).
  --chat-url URL      URL OpenAI del chat para el panel (def. $CHAT_URL).
  --mcp-url URL       URL MCP para el panel (def. $MCP_URL).
                      Con --no-external-mcp el panel usa su dispatcher integrado.
  --no-external-mcp   No arranca MCP externo y fuerza el dispatcher integrado,
                      incluso si el JSON persistido tiene una URL MCP.
  --lora-dir DIR      Carpeta de LoRAs del panel (def. $LORA_DIR).
  --settings-file F   JSON de ajustes del panel (def. $SETTINGS_FILE).
  --venv PY           Python del venv ROCm (def. $VENV_PY).
  --root DIR          Raíz del sistema (def. $ROOT).
  --log-dir DIR       Logs y pidfiles (def. $LOG_DIR).
  --wait-timeout S    Espera máx. por dependencia (def. $WAIT_TIMEOUT).
  -h, --help          Esta ayuda.

Variables de entorno equivalentes: QWEN_PORT, ZIMAGE_SERVER_PORT, MCP_PORT,
PANEL_PORT, CHAT_URL, MCP_URL, ZIMAGE_VENV_PY, ZIMAGE_ROOT, ZIMAGE_LOG_DIR.

Ejemplos:
  $(basename "$0")
  $(basename "$0") --status
  $(basename "$0") --restart
  $(basename "$0") --restart --only panel,mcp
  $(basename "$0") --only zimage --zimage-port 9878 --dry-run
  $(basename "$0") --stop --only panel
  QWEN_PORT=8086 ZIMAGE_SERVER_PORT=9877 MCP_PORT=8765 PANEL_PORT=8766 $(basename "$0") --status
EOF
}

log() { printf '[zimage-stack] %s\n' "$*"; }
dry() { if [ "$DRY_RUN" -eq 1 ]; then printf '[dry-run] %s\n' "$*"; return 0; fi; return 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --restart) MODE="restart"; shift ;;
    --stop) MODE="stop"; shift ;;
    --status) MODE="status"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --only) ONLY="${2:?falta valor para --only}"; shift 2 ;;
    --skip) SKIP="${2:?falta valor para --skip}"; shift 2 ;;
    --qwen-port) QWEN_PORT="$2"; shift 2 ;;
    --zimage-port|--backend-port) ZIMAGE_PORT="$2"; shift 2 ;;
    --mcp-port) MCP_PORT="$2"; shift 2 ;;
    --panel-port|--web-port) PANEL_PORT="$2"; shift 2 ;;
    --qwen-host) QWEN_HOST="$2"; shift 2 ;;
    --backend-host) ZIMAGE_HOST="$2"; shift 2 ;;
    --mcp-host) MCP_HOST="$2"; shift 2 ;;
    --mcp-bind) MCP_BIND="$2"; shift 2 ;;
    --panel-host) PANEL_HOST="$2"; shift 2 ;;
    --chat-url) CHAT_URL="$2"; CHAT_URL_FROM_ENV=1; shift 2 ;;
    --mcp-url) MCP_URL="$2"; MCP_URL_FROM_ENV=1; shift 2 ;;
    --no-external-mcp) USE_EXTERNAL_MCP=0; shift ;;
    --lora-dir) LORA_DIR="$2"; shift 2 ;;
    --settings-file) SETTINGS_FILE="$2"; shift 2 ;;
    --venv) VENV_PY="$2"; shift 2 ;;
    --root) ROOT="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --wait-timeout) WAIT_TIMEOUT="$2"; shift 2 ;;
    --qwen-script) QWEN_SCRIPT="$2"; shift 2 ;;
    *) echo "Flag desconocida: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# --root debe trasladar las rutas derivadas cuando no se personalizaron.
if [ "$ROOT" != "$ROOT_INITIAL" ]; then
  [ "$VENV_PY" = "$ROOT_INITIAL/.venv/bin/python3" ] && VENV_PY="$ROOT/.venv/bin/python3"
  [ "$QWEN_SCRIPT" = "$ROOT_INITIAL/run_qwen35_9b.sh" ] && QWEN_SCRIPT="$ROOT/run_qwen35_9b.sh"
  [ "$ZIMAGE_SCRIPT" = "$ROOT_INITIAL/zimage_server_gguf.py" ] && ZIMAGE_SCRIPT="$ROOT/zimage_server_gguf.py"
  [ "$MCP_SCRIPT" = "$ROOT_INITIAL/zimage_mcp_server.py" ] && MCP_SCRIPT="$ROOT/zimage_mcp_server.py"
  [ "$PANEL_SCRIPT" = "$ROOT_INITIAL/zimage_panel_grid.py" ] && PANEL_SCRIPT="$ROOT/zimage_panel_grid.py"
  [ "$LORA_DIR" = "$ROOT_INITIAL/lora_models_peft" ] && LORA_DIR="$ROOT/lora_models_peft"
  [ "$SETTINGS_FILE" = "$ROOT_INITIAL/zimage_panel_settings.json" ] && SETTINGS_FILE="$ROOT/zimage_panel_settings.json"
fi

# Al cambiar puertos, actualiza las URLs por defecto; conserva las URLs
# proporcionadas explícitamente por entorno o por línea de comandos.
[ -z "$CHAT_URL_FROM_ENV" ] && CHAT_URL="http://$QWEN_HOST:$QWEN_PORT/v1"
[ -z "$MCP_URL_FROM_ENV" ] && MCP_URL="http://$MCP_HOST:$MCP_PORT/mcp"

# Aplica --skip sobre --only.
if [ -n "$SKIP" ]; then
  _new=""
  IFS=',' read -ra _want <<< "$ONLY"
  IFS=',' read -ra _skip <<< "$SKIP"
  for c in "${_want[@]}"; do
    _drop=0
    for s in "${_skip[@]}"; do [ "$c" = "$s" ] && _drop=1; done
    [ "$_drop" -eq 0 ] && _new="${_new:+$_new,}$c"
  done
  ONLY="$_new"
fi
wanted() { case ",$ONLY," in *",$1,"*) return 0;; *) return 1;; esac; }
[ "$USE_EXTERNAL_MCP" -eq 0 ] && ONLY="$(echo ",$ONLY," | sed 's/,mcp,/,/g; s/^,//; s/,$//; s/,,/,/g')"
[ -z "$ONLY" ] && { echo "Selección vacía (--only/--skip)." >&2; exit 2; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "Falta comando: $1" >&2; exit 3; }; }
need_cmd ss; need_cmd pgrep; need_cmd curl; need_cmd nohup

QWEN_LOG="$LOG_DIR/zimage_qwen.log"
ZIMAGE_LOG="$LOG_DIR/zimage_server_gguf.log"
MCP_LOG="$LOG_DIR/zimage_mcp.log"
PANEL_LOG="$LOG_DIR/zimage_panel_grid.log"
QWEN_PIDF="$LOG_DIR/zimage_stack_qwen.pid"
ZIMAGE_PIDF="$LOG_DIR/zimage_stack_zimage.pid"
MCP_PIDF="$LOG_DIR/zimage_stack_mcp.pid"
PANEL_PIDF="$LOG_DIR/zimage_stack_panel.pid"

port_listening() {
  # Evita `ss | grep -q` bajo pipefail: grep puede cerrar la tubería antes de
  # que ss termine y producir un falso negativo por SIGPIPE.
  ss -ltnH 2>/dev/null | awk -v port="$2" '$4 ~ (":" port "$") {found=1} END {exit !found}'
}
proc_pattern() { pgrep -f "$1" >/dev/null 2>&1; }

owned_process() {
  local pidf="$1" pattern="$2" pid
  [ -f "$pidf" ] || return 1
  read -r pid < "$pidf"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  ps -p "$pid" -o args= 2>/dev/null | grep -E "$pattern" >/dev/null
}
qwen_running() {
  owned_process "$QWEN_PIDF" 'llama-server' && port_listening "$QWEN_HOST" "$QWEN_PORT"
}
qwen_healthy() {
  curl -sf --max-time 10 "http://$QWEN_HOST:$QWEN_PORT/props" >/dev/null 2>&1
}
zimage_running() {
  owned_process "$ZIMAGE_PIDF" "zimage_server_gguf\.py" || return 1
  port_listening "$ZIMAGE_HOST" "$ZIMAGE_PORT" || return 1
  return 0
}
zimage_healthy() {
  "$VENV_PY" - "$ZIMAGE_HOST" "$ZIMAGE_PORT" <<'PY' >/dev/null 2>&1
import socket, json, struct, sys
def recv_exact(c, size):
    result = b''
    while len(result) < size:
        chunk = c.recv(size-len(result))
        if not chunk:
            raise ConnectionError('Respuesta incompleta')
        result += chunk
    return result
host, port = sys.argv[1], int(sys.argv[2])
data = json.dumps({"cmd": "status"}).encode()
with socket.create_connection((host, port), timeout=10) as c:
    c.sendall(struct.pack('!I', len(data)) + data)
    n = struct.unpack('!I', recv_exact(c, 4))[0]
    result = json.loads(recv_exact(c, n))
    if not result.get('model_loaded'):
        raise SystemExit(1)
PY
}
mcp_running() {
  [ "$USE_EXTERNAL_MCP" -eq 1 ] || return 1
  owned_process "$MCP_PIDF" "zimage_mcp_server\.py" || return 1
  port_listening "$MCP_HOST" "$MCP_PORT" || return 1
  return 0
}
mcp_healthy() {
  curl -sf --max-time 10 -X POST "http://$MCP_HOST:$MCP_PORT/mcp" \
    -H 'Content-Type: application/json' \
    -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"zimage-stack","version":"1.0"}}}' >/dev/null 2>&1
}
panel_running() {
  owned_process "$PANEL_PIDF" "zimage_panel_grid\.py" || return 1
  port_listening "$PANEL_HOST" "$PANEL_PORT" || return 1
  return 0
}
panel_healthy() {
  # Con acceso por LAN el panel exige sesión: 401 también significa que está vivo.
  local code
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "http://$PANEL_HOST:$PANEL_PORT/api/loras" 2>/dev/null || true)"
  [ "$code" = 200 ] || [ "$code" = 401 ]
}

wait_for() {
  local desc="$1" timeout="$2"; shift 2
  local t=0
  until "$@" >/dev/null 2>&1; do
    sleep 2; t=$((t + 2))
    if [ "$t" -ge "$timeout" ]; then echo "TIMEOUT esperando: $desc" >&2; return 1; fi
  done
}

stop_one() {
  local name="$1" pidf="$2" pattern="$3"
  local pid=""
  [ -f "$pidf" ] && pid="$(cat "$pidf" 2>/dev/null || true)"
  if [ -z "$pid" ]; then
    log "$name: sin PID gestionado; no se detiene otro proceso."
    return 0
  fi
  if ! [[ "$pid" =~ ^[0-9]+$ ]] || [ "$pid" -lt 2 ]; then
    echo "$name: PID inválido en $pidf." >&2; return 1
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    log "$name: PID $pid obsoleto; no hay proceso que detener."
    if [ "$DRY_RUN" -eq 0 ]; then rm -f "$pidf"; fi
    return 0
  fi
  # Verifica línea de comando antes de matar (no matar otro servicio).
  if ! ps -ww -p "$pid" -o args= 2>/dev/null | grep -qF -- "$pattern"; then
    log "$name: PID $pid no coincide con '$pattern', no lo mato."
    return 1
  fi
  if [ "$name" = qwen ] && ! ps -ww -p "$pid" -o args= 2>/dev/null | grep -qE -- "[[:space:]]--port[[:space:]]+$QWEN_PORT([[:space:]]|$)"; then
    echo "qwen: PID $pid no usa el puerto $QWEN_PORT; no se detiene." >&2
    return 1
  fi
  if dry "kill $pid ($name)"; then return 0; fi
  log "$name: parando PID $pid..."
  kill "$pid" 2>/dev/null || true
  local t=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 1; t=$((t + 1))
    if [ "$t" -ge 15 ]; then log "$name: forzando kill -9 $pid"; kill -9 "$pid" 2>/dev/null || true; break; fi
  done
  rm -f "$pidf"
}

start_qwen() {
  if dry "bash $QWEN_SCRIPT (puerto $QWEN_PORT)"; then return 0; fi
  [ -x "$QWEN_SCRIPT" ] || { echo "No ejecutable: $QWEN_SCRIPT" >&2; return 1; }
  log "Arrancando Qwen :$QWEN_PORT (1-3 min)..."
  QWEN_PORT="$QWEN_PORT" nohup bash "$QWEN_SCRIPT" >>"$QWEN_LOG" 2>&1 &
  echo $! >"$QWEN_PIDF"
  wait_for "Qwen :$QWEN_PORT" "$WAIT_TIMEOUT" port_listening "$QWEN_HOST" "$QWEN_PORT"
}
start_zimage() {
  if dry "$VENV_PY $ZIMAGE_SCRIPT (ZIMAGE_SERVER_PORT=$ZIMAGE_PORT)"; then return 0; fi
  [ -f "$ZIMAGE_SCRIPT" ] || { echo "Falta: $ZIMAGE_SCRIPT" >&2; return 1; }
  [ -x "$VENV_PY" ] || { echo "Falta venv python: $VENV_PY" >&2; return 1; }
  log "Arrancando Z-Image TCP :$ZIMAGE_PORT (puede tardar varios minutos)..."
  ZIMAGE_SERVER_PORT="$ZIMAGE_PORT" nohup "$VENV_PY" "$ZIMAGE_SCRIPT" >>"$ZIMAGE_LOG" 2>&1 &
  echo $! >"$ZIMAGE_PIDF"
  wait_for "Z-Image :$ZIMAGE_PORT" "$WAIT_TIMEOUT" port_listening "$ZIMAGE_HOST" "$ZIMAGE_PORT"
}
start_mcp() {
  if [ "$USE_EXTERNAL_MCP" -eq 0 ]; then log "MCP externo desactivado (--no-external-mcp)."; return 0; fi
  if dry "$VENV_PY $MCP_SCRIPT --transport http --host $MCP_BIND --port $MCP_PORT --backend-port $ZIMAGE_PORT"; then return 0; fi
  log "Arrancando MCP :$MCP_PORT -> backend :$ZIMAGE_PORT..."
  # El MCP usa 9877 por defecto; se pasa explícito por si ZIMAGE_SERVER_PORT cambia.
  nohup "$VENV_PY" "$MCP_SCRIPT" --transport http --host "$MCP_BIND" \
    --port "$MCP_PORT" --backend-host "$ZIMAGE_HOST" --backend-port "$ZIMAGE_PORT" >>"$MCP_LOG" 2>&1 &
  echo $! >"$MCP_PIDF"
  wait_for "MCP :$MCP_PORT" "$WAIT_TIMEOUT" port_listening "$MCP_HOST" "$MCP_PORT"
}
start_panel() {
  local args=(--backend-host "$ZIMAGE_HOST" --backend-port "$ZIMAGE_PORT" --web-host "$PANEL_HOST" --web-port "$PANEL_PORT" --chat-url "$CHAT_URL" --lora-dir "$LORA_DIR" --settings-file "$SETTINGS_FILE")
  if [ "$USE_EXTERNAL_MCP" -eq 1 ]; then args+=(--mcp-url "$MCP_URL"); fi
  if [ "$USE_EXTERNAL_MCP" -eq 0 ]; then args+=(--integrated-mcp); fi
  if dry "$VENV_PY $PANEL_SCRIPT ${args[*]}"; then return 0; fi
  log "Arrancando panel :$PANEL_PORT..."
  nohup "$VENV_PY" "$PANEL_SCRIPT" "${args[@]}" >>"$PANEL_LOG" 2>&1 &
  echo $! >"$PANEL_PIDF"
  wait_for "panel :$PANEL_PORT" "$WAIT_TIMEOUT" port_listening "$PANEL_HOST" "$PANEL_PORT"
}

report() {
  local name="$1" comp="$2" port="$3"; local st="PARADO" hc=""
  if "${comp}_running"; then st="ARRANCADO"; "${comp}_healthy" && hc=" (sano)" || hc=" (arrancado, health-check falla)"; fi
  printf '%-8s %-10s :%-5s %s\n' "$name" "$st" "$port" "$hc"
}
do_status() {
  echo "Stack Z-Image (ROOT=$ROOT):"
  if wanted qwen; then report qwen qwen "$QWEN_PORT"; fi
  if wanted zimage; then report zimage zimage "$ZIMAGE_PORT"; fi
  if wanted mcp && [ "$USE_EXTERNAL_MCP" -eq 1 ]; then report mcp mcp "$MCP_PORT"; fi
  if wanted panel; then report panel panel "$PANEL_PORT"; fi
  echo "Chat: $CHAT_URL"
  if [ "$USE_EXTERNAL_MCP" -eq 1 ]; then echo "MCP URL panel: $MCP_URL"; else echo "MCP: integrado en panel (sin proceso externo)"; fi
}

do_stop() {
  # Orden inverso al arranque.
  if wanted panel; then stop_one panel "$PANEL_PIDF" "$PANEL_SCRIPT"; fi
  if wanted mcp && [ "$USE_EXTERNAL_MCP" -eq 1 ]; then stop_one mcp "$MCP_PIDF" "$MCP_SCRIPT"; fi
  if wanted zimage; then stop_one zimage "$ZIMAGE_PIDF" "$ZIMAGE_SCRIPT"; fi
  if wanted qwen; then stop_one qwen "$QWEN_PIDF" "llama-server"; fi
  if [ "$DRY_RUN" -eq 0 ]; then
    if wanted panel && port_listening "$PANEL_HOST" "$PANEL_PORT"; then echo "panel :$PANEL_PORT sigue ocupado; no se arranca un duplicado." >&2; return 1; fi
    if wanted mcp && [ "$USE_EXTERNAL_MCP" -eq 1 ] && port_listening "$MCP_HOST" "$MCP_PORT"; then echo "MCP :$MCP_PORT sigue ocupado; no se arranca un duplicado." >&2; return 1; fi
    if wanted zimage && port_listening "$ZIMAGE_HOST" "$ZIMAGE_PORT"; then echo "Z-Image :$ZIMAGE_PORT sigue ocupado; no se arranca un duplicado." >&2; return 1; fi
    if wanted qwen && port_listening "$QWEN_HOST" "$QWEN_PORT"; then echo "Qwen :$QWEN_PORT sigue ocupado; no se arranca un duplicado." >&2; return 1; fi
  fi
}

do_ensure() {
  if [ "$DRY_RUN" -eq 0 ]; then mkdir -p "$LOG_DIR"; fi
  local component prefix port host pidf pattern
  for component in qwen zimage mcp panel; do
    wanted "$component" || continue
    prefix="${component^^}"
    local portvar="${prefix}_PORT" hostvar="${prefix}_HOST" pidvar="${prefix}_PIDF"
    port="${!portvar}"; host="${!hostvar}"; pidf="${!pidvar}"
    case "$component" in
      qwen) pattern='llama-server|run_qwen35_9b' ;;
      zimage) pattern='zimage_server_gguf\.py' ;;
      mcp) pattern='zimage_mcp_server\.py' ;;
      panel) pattern='zimage_panel_grid\.py' ;;
    esac
    if [ "$DRY_RUN" -eq 1 ]; then
      "start_$component"
      continue
    fi
    if port_listening "$host" "$port" && ! owned_process "$pidf" "$pattern"; then
      echo "Puerto $port ocupado por otro proceso. No se adoptará ni detendrá. Consulta ss -ltnp." >&2
      return 1
    fi
    if owned_process "$pidf" "$pattern"; then
      log "$component: proceso existente; esperando disponibilidad..."
      wait_for "$component :$port" "$WAIT_TIMEOUT" port_listening "$host" "$port"
    else
      "start_$component"
    fi
    "${component}_healthy" || { echo "$component escucha pero su comprobación de salud falla." >&2; return 1; }
  done
}

case "$MODE" in
  status) do_status ;;
  stop) do_stop; do_status ;;
  restart) do_stop; do_ensure; do_status ;;
  ensure) do_ensure; do_status ;;
esac
