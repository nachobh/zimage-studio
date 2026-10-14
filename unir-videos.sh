#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

INPUT1="${1:-$DIR/h3-1788212804760.mp4}"
INPUT2="${2:-$DIR/h3-1788111907910.mp4}"
OUTPUT="${3:-$DIR/h3-unido.mp4}"

if [[ ! -f "$INPUT1" || ! -f "$INPUT2" ]]; then
    echo "Uso: $0 [video1.mp4] [video2.mp4] [salida.mp4]" >&2
    exit 1
fi

ffmpeg -y \
    -i "$INPUT1" \
    -i "$INPUT2" \
    -filter_complex \
    "[0:v]scale=640:1152:force_original_aspect_ratio=decrease,pad=640:1152:(ow-iw)/2:(oh-ih)/2,setsar=1[v0];[1:v]scale=640:1152:force_original_aspect_ratio=decrease,pad=640:1152:(ow-iw)/2:(oh-ih)/2,setsar=1[v1];[0:a]aresample=async=1[a0];[1:a]aresample=async=1[a1];[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]" \
    -map "[v]" \
    -map "[a]" \
    -c:v libx264 \
    -preset medium \
    -crf 20 \
    -c:a aac \
    -b:a 128k \
    -movflags +faststart \
    "$OUTPUT"

echo "Vídeo creado: $OUTPUT"
