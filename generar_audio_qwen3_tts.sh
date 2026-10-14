#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  cat <<'EOF'
Uso: ./generar_audio_qwen3_tts.sh

Solicita WAV de referencia, idioma de salida, transcripción opcional de la
referencia, texto y archivo de destino. Requiere Qwen3-TTS activo en GPU en
http://127.0.0.1:8770. Puedes cambiar la URL base con QWEN3_TTS_URL.
EOF
  exit 0
elif [[ $# -gt 0 ]]; then
  echo 'Este asistente no acepta argumentos; usa --help para ver los requisitos.' >&2
  exit 2
fi

API_BASE="${QWEN3_TTS_URL:-${ZIMAGE_QWEN3_TTS_URL:-http://127.0.0.1:8770}}"
API_BASE="${API_BASE%/}"

trim_input() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

clean_path() {
  local value
  value="$(trim_input "$1")"
  if [[ ${#value} -ge 2 ]]; then
    if [[ "$value" == \"*\" ]]; then value="${value:1:${#value}-2}"
    elif [[ "$value" == \'*\' ]]; then value="${value:1:${#value}-2}"
    fi
  fi
  printf '%s' "$value"
}

read -r -p 'Ruta del WAV de referencia: ' reference
reference="$(clean_path "$reference")"
if [[ ! -f "$reference" || -L "$reference" ]]; then
  echo 'Error: la referencia debe ser un archivo WAV regular; no se aceptan enlaces simbólicos.' >&2
  exit 2
fi
reference_size="$(wc -c < "$reference")"
if (( reference_size == 0 || reference_size > 20 * 1024 * 1024 )); then
  echo 'Error: el WAV debe ocupar entre 1 byte y 20 MiB.' >&2
  exit 2
fi
if ! python3 - "$reference" <<'PY'
import pathlib, sys
with open(sys.argv[1], 'rb') as audio:
    header = audio.read(12)
raise SystemExit(0 if len(header) == 12 and header[:4] == b'RIFF' and header[8:12] == b'WAVE' else 1)
PY
then
  echo 'Error: el archivo de referencia no tiene cabecera WAV (RIFF/WAVE).' >&2
  exit 2
fi

echo 'Idioma de salida:'
echo '  1) Español de España (predeterminado)'
echo '  2) Inglés'
echo '  3) Chino'
echo '  4) Japonés'
echo '  5) Coreano'
echo '  6) Alemán'
echo '  7) Francés'
echo '  8) Ruso'
echo '  9) Portugués'
echo ' 10) Italiano'
read -r -p 'Elige [1]: ' language_choice
language_choice="$(trim_input "${language_choice:-1}")"
case "$language_choice" in
  1) language='Spanish' ;;
  2) language='English' ;;
  3) language='Chinese' ;;
  4) language='Japanese' ;;
  5) language='Korean' ;;
  6) language='German' ;;
  7) language='French' ;;
  8) language='Russian' ;;
  9) language='Portuguese' ;;
  10) language='Italian' ;;
  *) echo 'Elige un número del 1 al 10.' >&2; exit 2 ;;
esac
if [[ "$language" == Spanish ]]; then
  echo 'Nota: Qwen3-TTS recibe el idioma «Spanish»; el acento peninsular depende del WAV de referencia.'
fi

read -r -p 'Transcripción exacta del WAV de referencia (opcional; Intro para omitir): ' reference_text
reference_text="$(trim_input "$reference_text")"
if (( ${#reference_text} > 10000 )); then
  echo 'Error: la transcripción de referencia no puede superar 10000 caracteres.' >&2
  exit 2
fi
read -r -p 'Texto que debe decir (máximo 5000 caracteres): ' speech_text
speech_text="$(trim_input "$speech_text")"
if [[ -z "$speech_text" || ${#speech_text} -gt 5000 ]]; then
  echo 'Error: el texto debe tener entre 1 y 5000 caracteres.' >&2
  exit 2
fi

output_default="qwen3tts_$(date +%Y%m%d_%H%M%S).wav"
read -r -p "Archivo WAV de salida [$output_default]: " output
output="$(clean_path "$output")"
if [[ -z "$output" ]]; then output="$output_default"; fi
output_dir="$(dirname -- "$output")"
output_name="$(basename -- "$output")"
[[ "$output_name" != '.' && "$output_name" != '/' ]] || { echo 'Ruta de salida inválida.' >&2; exit 2; }
if [[ "${output_name,,}" != *.wav ]]; then output_name="$output_name.wav"; fi
mkdir -p -- "$output_dir"
output_dir="$(cd -- "$output_dir" && pwd)"
output="$output_dir/$output_name"
if [[ -e "$output" || -L "$output" ]]; then
  echo "Error: el archivo ya existe y no se sobrescribirá: $output" >&2
  exit 2
fi

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/qwen3-tts-cli.XXXXXX")"
response_file="$(mktemp "$output_dir/.qwen3tts-response.XXXXXX")"
cleanup() { rm -rf -- "$work_dir"; [[ ! -e "$response_file" ]] || rm -f -- "$response_file"; }
trap cleanup EXIT
request_file="$work_dir/request.json"
health_file="$work_dir/health.json"
printf '%s' "$reference_text" > "$work_dir/reference.txt"
printf '%s' "$speech_text" > "$work_dir/text.txt"

if ! curl -fsS --max-time 3 "$API_BASE/health" -o "$health_file"; then
  echo "Qwen3-TTS no responde en $API_BASE. Arráncalo con ./start_instalacion_existente.sh --with-qwen3-tts o ./start.sh --with-qwen3-tts y vuelve a intentarlo." >&2
  exit 3
fi
if ! python3 - "$health_file" <<'PY'
import json, sys
try:
    status = json.load(open(sys.argv[1], encoding='utf-8'))
    assert status.get('status') == 'ok'
    assert status.get('device') == 'cuda' and status.get('runtime', {}).get('gpu')
except Exception:
    raise SystemExit(1)
PY
then
  echo 'Error: el servidor Qwen3-TTS no confirma una GPU activa; revisa su endpoint /health.' >&2
  exit 3
fi

python3 - "$reference" "$language" "$request_file" "$work_dir/reference.txt" "$work_dir/text.txt" <<'PY'
import base64, json, pathlib, sys
reference, language, destination, ref_text_path, text_path = sys.argv[1:]
body = {
    'input': pathlib.Path(text_path).read_text(encoding='utf-8'),
    'reference_audio': base64.b64encode(pathlib.Path(reference).read_bytes()).decode('ascii'),
    'reference_text': pathlib.Path(ref_text_path).read_text(encoding='utf-8'),
    'language': language,
}
pathlib.Path(destination).write_text(json.dumps(body, ensure_ascii=False), encoding='utf-8')
PY

echo 'Generando audio con Qwen3-TTS en GPU…'
http_status="$(curl -sS --max-time "${QWEN3_TTS_TIMEOUT:-900}" \
  -o "$response_file" -w '%{http_code}' \
  -H 'Content-Type: application/json' --data-binary "@$request_file" "$API_BASE/tts")" || {
    echo 'Error de conexión o timeout durante la generación.' >&2; exit 4;
  }
if [[ "$http_status" != 200 ]]; then
  python3 - "$response_file" "$http_status" <<'PY'
import json, pathlib, sys
try:
    body = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))
    print(f"Error HTTP {sys.argv[2]}: {body.get('error', body)}", file=sys.stderr)
except Exception:
    print(f"Error HTTP {sys.argv[2]} al generar audio.", file=sys.stderr)
PY
  exit 4
fi
if ! python3 - "$response_file" <<'PY'
import pathlib, sys
data = pathlib.Path(sys.argv[1]).read_bytes()
raise SystemExit(0 if len(data) >= 44 and data[:4] == b'RIFF' and data[8:12] == b'WAVE' else 1)
PY
then
  echo 'Error: la respuesta no es un WAV válido.' >&2
  exit 4
fi
if ! ln -- "$response_file" "$output"; then
  echo "Error: no se pudo crear la salida (¿ya existe?): $output" >&2
  exit 4
fi
rm -f -- "$response_file"
echo "Audio guardado: $output"
