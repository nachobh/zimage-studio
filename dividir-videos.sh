#!/usr/bin/env bash
# Divide un vídeo en partes por segundos (con decimales) o por fotogramas.
# Contraparte de unir-videos.sh; usa los mismos códecs (x264 CRF 20, AAC 128k).
set -euo pipefail
export LC_ALL=C

uso() {
    cat <<'EOF'
Uso: dividir-videos.sh [opciones] VIDEO

Puntos de corte (se pueden repetir y combinar; cada uno crea una parte más):
  --en SEGUNDOS            corta en ese instante (admite decimales: 3.5)
  --en-fotograma N         corta antes del fotograma N (el primero es el 0)

Trozos regulares (no se combinan con los puntos de corte):
  --cada SEGUNDOS          partes de esa duración (la última puede ser menor)
  --cada-fotogramas N      partes de N fotogramas

Opciones:
  -o, --salida DIR         carpeta de salida (por defecto, la del vídeo)
  --copiar                 sin recodificar: rápido, pero corta en el fotograma
                           clave anterior (no es exacto) y conserva el formato
  --sobrescribir           reemplaza partes que ya existan
  -h, --help               esta ayuda

Salida: NOMBRE_parte01.mp4, NOMBRE_parte02.mp4, ...
Por defecto se recodifica (exacto al fotograma): las partes encajan sin perder
ni repetir ninguno. Un corte en el instante t empieza la parte siguiente en el
primer fotograma con marca >= t. La salida tiene tasa constante, la del vídeo.

Ejemplos:
  dividir-videos.sh --en 3.5 clip.mp4                  # 2 partes: 0-3.5 y 3.5-fin
  dividir-videos.sh --en 2 --en 5.25 clip.mp4          # 3 partes
  dividir-videos.sh --en-fotograma 90 clip.mp4         # corta en el fotograma 90
  dividir-videos.sh --cada 2.5 clip.mp4                # trozos de 2.5 s
  dividir-videos.sh --cada-fotogramas 120 -o partes clip.mp4
EOF
}

fallo() { echo "Error: $*" >&2; exit 1; }
es_decimal() { [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]; }
es_entero() { [[ "$1" =~ ^[0-9]+$ ]]; }

PUNTOS_S=()      # instantes de corte en segundos
PUNTOS_F=()      # fotogramas de corte
CADA_S=""
CADA_F=""
COPIAR=0
SOBRESCRIBIR=0
DIR_SALIDA=""
ENTRADA=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --en)
            [[ $# -ge 2 ]] || fallo "--en necesita un valor"
            es_decimal "$2" || fallo "--en espera segundos (p. ej. 3.5), recibí '$2'"
            PUNTOS_S+=("$2"); shift 2 ;;
        --en-fotograma)
            [[ $# -ge 2 ]] || fallo "--en-fotograma necesita un valor"
            es_entero "$2" || fallo "--en-fotograma espera un entero, recibí '$2'"
            PUNTOS_F+=("$2"); shift 2 ;;
        --cada)
            [[ $# -ge 2 ]] || fallo "--cada necesita un valor"
            es_decimal "$2" || fallo "--cada espera segundos (p. ej. 2.5), recibí '$2'"
            CADA_S="$2"; shift 2 ;;
        --cada-fotogramas)
            [[ $# -ge 2 ]] || fallo "--cada-fotogramas necesita un valor"
            es_entero "$2" || fallo "--cada-fotogramas espera un entero, recibí '$2'"
            CADA_F="$2"; shift 2 ;;
        -o|--salida)
            [[ $# -ge 2 ]] || fallo "$1 necesita una carpeta"
            DIR_SALIDA="$2"; shift 2 ;;
        --copiar) COPIAR=1; shift ;;
        --sobrescribir) SOBRESCRIBIR=1; shift ;;
        -h|--help) uso; exit 0 ;;
        -*) uso >&2; fallo "opción desconocida: $1" ;;
        *)
            [[ -z "$ENTRADA" ]] || fallo "solo se admite un vídeo (ya hay '$ENTRADA')"
            ENTRADA="$1"; shift ;;
    esac
done

[[ -n "$ENTRADA" ]] || { uso >&2; exit 1; }
[[ -f "$ENTRADA" ]] || fallo "no existe el vídeo: $ENTRADA"
command -v ffmpeg >/dev/null && command -v ffprobe >/dev/null || fallo "faltan ffmpeg/ffprobe"

HAY_PUNTOS=$(( ${#PUNTOS_S[@]} + ${#PUNTOS_F[@]} ))
HAY_CADA=0; [[ -n "$CADA_S" || -n "$CADA_F" ]] && HAY_CADA=1
[[ $HAY_PUNTOS -gt 0 || $HAY_CADA -eq 1 ]] || { uso >&2; fallo "indica --en, --en-fotograma, --cada o --cada-fotogramas"; }
[[ $HAY_PUNTOS -eq 0 || $HAY_CADA -eq 0 ]] || fallo "los puntos de corte no se combinan con --cada/--cada-fotogramas"
[[ -z "$CADA_S" || -z "$CADA_F" ]] || fallo "usa solo uno entre --cada y --cada-fotogramas"

# Duración, tasa de imagen y audio del vídeo.
DURACION="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$ENTRADA" | head -n1)"
FPS_RAT="$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$ENTRADA" | head -n1)"
es_decimal "${DURACION:-}" || fallo "no pude leer la duración de '$ENTRADA'"
[[ "${FPS_RAT:-}" =~ ^[0-9]+(/[0-9]+)?$ ]] || fallo "no pude leer los fotogramas por segundo de '$ENTRADA'"
HAY_AUDIO=0
[[ -n "$(ffprobe -v error -select_streams a -show_entries stream=index -of csv=p=0 "$ENTRADA")" ]] && HAY_AUDIO=1

# Conversión entre tiempo y fotograma con la tasa exacta (p. ej. 30000/1001).
# Un corte en el instante t empieza la parte siguiente en el primer fotograma
# cuya marca es >= t; así las partes encajan sin perder ni repetir fotogramas.
segundos_a_fotograma() {
    awk -v t="$1" -v r="$FPS_RAT" 'BEGIN { split(r, a, "/"); d = (a[2] == "" || a[2] == 0) ? 1 : a[2]; x = t * a[1] / d; i = int(x); if (x - i > 1e-6) i++; print i }'
}
fotograma_a_segundos() {
    awk -v n="$1" -v r="$FPS_RAT" 'BEGIN { split(r, a, "/"); d = (a[2] == "" || a[2] == 0) ? 1 : a[2]; printf "%.6f", n * d / a[1] }'
}
# Fotogramas reales: nb_frames del contenedor; si falta, la duración del flujo
# de vídeo (no la del archivo, que puede incluir audio más largo).
TOTAL_FOTOGRAMAS="$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$ENTRADA" | head -n1)"
if ! es_entero "${TOTAL_FOTOGRAMAS:-}" || [[ "$TOTAL_FOTOGRAMAS" -le 0 ]]; then
    DURACION_VIDEO="$(ffprobe -v error -select_streams v:0 -show_entries stream=duration -of csv=p=0 "$ENTRADA" | head -n1)"
    es_decimal "${DURACION_VIDEO:-}" || DURACION_VIDEO="$DURACION"
    TOTAL_FOTOGRAMAS="$(awk -v t="$DURACION_VIDEO" -v r="$FPS_RAT" 'BEGIN { split(r, a, "/"); d = (a[2] == "" || a[2] == 0) ? 1 : a[2]; printf "%d", t * a[1] / d + 0.5 }')"
fi
[[ "$TOTAL_FOTOGRAMAS" -gt 1 ]] || fallo "el vídeo tiene muy pocos fotogramas para dividirlo"

# Lista de fotogramas donde empieza cada parte nueva.
CORTES=()
for s in "${PUNTOS_S[@]}"; do CORTES+=("$(segundos_a_fotograma "$s")"); done
for f in "${PUNTOS_F[@]}"; do CORTES+=("$f"); done
if [[ $HAY_CADA -eq 1 ]]; then
    if [[ -n "$CADA_F" ]]; then
        [[ "$CADA_F" -gt 0 ]] || fallo "--cada-fotogramas debe ser mayor que 0"
        mapfile -t CORTES < <(awk -v p="$CADA_F" -v n="$TOTAL_FOTOGRAMAS" 'BEGIN { for (k = p; k < n; k += p) print k }')
    else
        awk -v p="$CADA_S" 'BEGIN { exit !(p > 0) }' || fallo "--cada debe ser mayor que 0"
        mapfile -t CORTES < <(awk -v p="$CADA_S" -v r="$FPS_RAT" -v n="$TOTAL_FOTOGRAMAS" 'BEGIN { split(r, a, "/"); d = (a[2] == "" || a[2] == 0) ? 1 : a[2]; for (k = 1; ; k++) { x = k * p * a[1] / d; i = int(x); if (x - i > 1e-6) i++; if (i >= n) break; print i } }')
    fi
fi

# Ordenar, quitar repetidos y comprobar que caen dentro del vídeo.
[[ ${#CORTES[@]} -gt 0 ]] || fallo "no hay ningún punto de corte: el vídeo dura ${DURACION}s ($TOTAL_FOTOGRAMAS fotogramas)"
mapfile -t CORTES < <(printf '%s\n' "${CORTES[@]}" | sort -n -u)
for c in "${CORTES[@]}"; do
    [[ "$c" -gt 0 && "$c" -lt "$TOTAL_FOTOGRAMAS" ]] \
        || fallo "el corte en el fotograma $c queda fuera del vídeo (tiene $TOTAL_FOTOGRAMAS fotogramas, ${DURACION}s)"
done

# Cada parte abarca los fotogramas [LIMITES[i], LIMITES[i+1]).
LIMITES=(0 "${CORTES[@]}")
TOTAL=${#LIMITES[@]}

BASE="$(basename -- "$ENTRADA")"
NOMBRE="${BASE%.*}"
if [[ $COPIAR -eq 1 ]]; then EXT="${BASE##*.}"; else EXT="mp4"; fi
DIR_SALIDA="${DIR_SALIDA:-$(dirname -- "$ENTRADA")}"
mkdir -p -- "$DIR_SALIDA"

SALIDAS=()
for ((i = 0; i < TOTAL; i++)); do
    SALIDAS+=("$(printf '%s/%s_parte%02d.%s' "$DIR_SALIDA" "$NOMBRE" $((i + 1)) "$EXT")")
done
if [[ $SOBRESCRIBIR -eq 0 ]]; then
    for archivo in "${SALIDAS[@]}"; do
        [[ ! -e "$archivo" ]] || fallo "ya existe $archivo (usa --sobrescribir)"
    done
fi

[[ $COPIAR -eq 1 ]] && echo "Aviso: --copiar corta en fotogramas clave; los límites pueden desviarse unos instantes." >&2
echo "Vídeo: $ENTRADA (${DURACION}s, ${FPS_RAT} fps, $TOTAL_FOTOGRAMAS fotogramas) -> $TOTAL partes"

for ((i = 0; i < TOTAL; i++)); do
    A="${LIMITES[$i]}"
    SALIDA="${SALIDAS[$i]}"
    TA="$(fotograma_a_segundos "$A")"
    if (( i + 1 < TOTAL )); then
        B="${LIMITES[$((i + 1))]}"
        TB="$(fotograma_a_segundos "$B")"
        FILTRO_V="trim=start_frame=${A}:end_frame=${B},setpts=N/FRAME_RATE/TB"
        FILTRO_A="atrim=start=${TA}:end=${TB},asetpts=PTS-STARTPTS"
        OPCION_LARGO=(-t "$(awk -v a="$TA" -v b="$TB" 'BEGIN { printf "%.6f", b - a }')")
        TRAMO="fotogramas ${A}-$((B - 1)) ($((B - A)))"
    else
        TB=""
        FILTRO_V="trim=start_frame=${A},setpts=N/FRAME_RATE/TB"
        FILTRO_A="atrim=start=${TA},asetpts=PTS-STARTPTS"
        OPCION_LARGO=()
        TRAMO="fotogramas ${A}-$((TOTAL_FOTOGRAMAS - 1)) ($((TOTAL_FOTOGRAMAS - A)))"
    fi
    if [[ $COPIAR -eq 1 ]]; then
        ffmpeg -y -hide_banner -loglevel error \
            -ss "$TA" -i "$ENTRADA" "${OPCION_LARGO[@]}" \
            -map 0:v:0 -map 0:a? -c copy -avoid_negative_ts make_zero "$SALIDA"
    else
        OPCIONES_AUDIO=()
        [[ $HAY_AUDIO -eq 1 ]] && OPCIONES_AUDIO=(-map 0:a -af "$FILTRO_A" -c:a aac -b:a 128k)
        ffmpeg -y -hide_banner -loglevel error \
            -i "$ENTRADA" \
            -map 0:v:0 -vf "$FILTRO_V" -fps_mode cfr -r "$FPS_RAT" \
            -c:v libx264 -preset medium -crf 20 \
            "${OPCIONES_AUDIO[@]}" \
            -movflags +faststart "$SALIDA"
    fi
    FIN_TXT="fin"; [[ -n "$TB" ]] && FIN_TXT="${TB}s"
    printf 'Parte %02d/%02d  %-30s %ss -> %s  %s\n' $((i + 1)) "$TOTAL" "$TRAMO" "$TA" "$FIN_TXT" "$SALIDA"
done

echo "Hecho: $TOTAL partes en $DIR_SALIDA"
