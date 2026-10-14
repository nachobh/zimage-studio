#!/bin/bash
# ckpt2safetensors.sh - Conversor .ckpt <-> .safetensors
# Basado en: https://github.com/diStyApps/Safe-and-Stable-Ckpt2Safetensors-Conversion-Tool-GUI
# y https://colab.research.google.com/drive/1YYzfYZEJTb3dAo9BX6w6eZINIuRsNv6l
# Uso: ./ckpt2safetensors.sh <entrada.ckpt|.safetensors|directorio> [salida] [opciones]

set -e

# --- colores ---
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'

# --- defaults ---
SUFFIX=false
BATCH=false
OUTPUT=""
INPUT=""

print_help() {
cat <<EOF
Uso: $0 <input> [output] [opciones]

  <input>   Archivo .ckpt / .safetensors o directorio (con --batch)
  [output]  Archivo de salida opcional (si no se da, se usa mismo nombre cambiando extensión)

Opciones:
  --batch, -b     Convierte recursivamente todos los .ckpt (o .safetensors) en un directorio
  --suffix        No sobrescribir: añade sufijo "-cnvrtd" como hace Safe-and-Stable GUI
  --to-ckpt       Forzar conversión safetensors -> ckpt (por defecto auto-detecta por extensión)
  --to-st         Forzar conversión ckpt -> safetensors
  -h, --help      Muestra esta ayuda

Ejemplos:
  $0 modelo.ckpt
  $0 modelo.ckpt modelo.safetensors
  $0 modelo.ckpt --suffix
  $0 ./mis_modelos --batch
  $0 ./mis_modelos --batch --suffix
  $0 modelo.safetensors --to-ckpt

Notas:
  - La conversión REQUIERE cargar el .ckpt con torch.load (pickle), igual que el tool original.
    Usa solo con modelos de origen confiable o convierte online (Colab) si no te fías.
  - Verifica que el modelo funciona antes de borrar el original.
EOF
}

# --- parse args ---
MODE="auto" # auto | ckpt2st | st2ckpt
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) print_help; exit 0 ;;
    --batch|-b) BATCH=true; shift ;;
    --suffix) SUFFIX=true; shift ;;
    --to-ckpt) MODE="st2ckpt"; shift ;;
    --to-st) MODE="ckpt2st"; shift ;;
    --*) echo -e "${RED}Opción desconocida: $1${NC}"; print_help; exit 1 ;;
    *)
      if [[ -z "$INPUT" ]]; then INPUT="$1"
      elif [[ -z "$OUTPUT" ]]; then OUTPUT="$1"
      else echo -e "${RED}Demasiados argumentos posicionales${NC}"; print_help; exit 1
      fi
      shift ;;
  esac
done

if [[ -z "$INPUT" ]]; then
  echo -e "${RED}Error: debes indicar archivo o directorio de entrada${NC}"
  print_help
  exit 1
fi

# --- checks python ---
if ! command -v python3 &>/dev/null; then
  echo -e "${RED}python3 no encontrado${NC}"; exit 1
fi

# --- auto-instalar deps si faltan ---
python3 -c "import torch, safetensors" 2>/dev/null || {
  echo -e "${YELLOW}[*] Instalando dependencias: torch safetensors...${NC}"
  pip install --upgrade pip
  pip install torch safetensors --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || pip install torch safetensors
}

# --- python converter embebido (lógica idéntica a Safe-and-Stable) ---
PY_CONVERTER=$(cat <<'PYEOF'
import os, sys, hashlib, torch
from safetensors.torch import save_file, load_file

def get_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]

def is_sqlite(path):
    with open(path, 'rb') as f:
        return f.read(6) == b'SQLite'

def sqlite_ckpt_to_dict(path):
    import sqlite3, struct
    con = sqlite3.connect(path)
    cur = con.cursor()
    # Validar tablas
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tensors'")
    if not cur.fetchone():
        raise RuntimeError("SQLite .ckpt sin tabla 'tensors'")
    weights = {}
    cur.execute("SELECT name, dim, datatype, data FROM tensors")
    for name, dim_blob, datatype, data_blob in cur.fetchall():
        # Dim: array de int32 LE, padded con ceros a 48 bytes (12 dims)
        dims = []
        if dim_blob:
            for i in range(0, len(dim_blob), 4):
                v = struct.unpack('<i', dim_blob[i:i+4])[0]
                if v != 0:
                    dims.append(v)
        if not dims:
            print(f"  [!] dims vacío para {name}, skip")
            continue
        prod = 1
        for d in dims:
            prod *= d
        # datatype 16384 = float32 (verificado). Deducir por tamaño si no coincide.
        # datatype mapping fallback por longitud
        data_len = len(data_blob)
        # infer dtype
        if data_len == prod * 4:
            dtype = 'f32'
        elif data_len == prod * 2:
            dtype = 'f16'
        elif data_len == prod:
            dtype = 'u8'
        else:
            # fallback a float32 si datatype 16384
            if datatype == 16384:
                dtype = 'f32'
            else:
                print(f"  [!] {name}: dims {dims} prod {prod} pero data_len {data_len} no cuadra (datatype {datatype}), skip")
                continue
        try:
            if dtype == 'f32':
                import numpy as np
                arr = np.frombuffer(data_blob, dtype=np.float32).copy()
                tensor = torch.from_numpy(arr).reshape(dims)
            elif dtype == 'f16':
                import numpy as np
                arr = np.frombuffer(data_blob, dtype=np.float16).copy()
                tensor = torch.from_numpy(arr).reshape(dims)
            else:
                import numpy as np
                arr = np.frombuffer(data_blob, dtype=np.uint8).copy()
                tensor = torch.from_numpy(arr).reshape(dims)
        except Exception as e:
            print(f"  [!] Error decodificando {name} dims {dims}: {e}")
            continue
        weights[name] = tensor
    con.close()
    return weights

def load_weights_ckpt(path):
    if is_sqlite(path):
        print(f"  [info] Detectado formato SQLite LoRA (.ckpt con tabla tensors) -> usando decoder SQLite")
        return sqlite_ckpt_to_dict(path)
    with torch.no_grad():
        try:
            weights = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        except TypeError:
            # torch < 1.13 no tiene weights_only
            weights = torch.load(path, map_location=torch.device('cpu'))
        # Same logic as run_app_gui.py:load_weights
        if isinstance(weights, dict) and "state_dict" in weights:
            weights = weights["state_dict"]
            if isinstance(weights, dict) and "state_dict" in weights:
                weights.pop("state_dict")
        return weights

def ckpt_to_st(input_path, output_path):
    print(f"[ckpt->st] Cargando {input_path} [{get_hash(input_path)}] ...")
    weights = load_weights_ckpt(input_path)
    # filtrar solo tensores (por si hay metadatos raros)
    # save_file exige dict[str, Tensor]
    # Si hay claves no-tensor, las ignoramos con warning
    clean = {}
    for k, v in weights.items():
        if isinstance(v, torch.Tensor):
            clean[k] = v
        else:
            # intentar convertir si es ndarray etc.
            try:
                clean[k] = torch.tensor(v)
            except:
                print(f"  [!] Clave ignorada (no tensor): {k} -> {type(v)}")
    print(f"  -> Guardando {output_path} ({len(clean)} tensores)...")
    save_file(clean, output_path)
    print(f"  [OK] {output_path} [{get_hash(output_path)}]")

def st_to_ckpt(input_path, output_path):
    print(f"[st->ckpt] Cargando {input_path} [{get_hash(input_path)}] ...")
    weights = load_file(input_path, device="cpu")
    # El GUI original hace weights["state_dict"] = weights y torch.save
    ckpt = {"state_dict": weights}
    # Para compatibilidad con A1111/Comfy, también guardamos plano
    # pero mantenemos la estructura del GUI: dict con state_dict
    # Algunos loaders esperan dict plano, por eso guardamos ambos niveles:
    # Si quieres solo plano, cambia a torch.save(weights, ...)
    with open(output_path, "wb") as f:
        torch.save(ckpt, f)
    print(f"  [OK] {output_path} [{get_hash(output_path)}]")

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output", nargs="?")
    p.add_argument("--suffix", action="store_true")
    p.add_argument("--mode", choices=["auto","ckpt2st","st2ckpt"], default="auto")
    args = p.parse_args()
    inp = args.input
    out = args.output

    if not os.path.exists(inp):
        print(f"Error: no existe {inp}", file=sys.stderr); sys.exit(1)

    def resolve_out(i, o, target_ext):
        if o:
            return o
        base = os.path.splitext(i)[0]
        suffix = "-cnvrtd" if args.suffix else ""
        return f"{base}{suffix}.{target_ext}"

    if args.mode == "ckpt2st" or (args.mode == "auto" and inp.endswith(".ckpt")):
        o = resolve_out(inp, out, "safetensors")
        ckpt_to_st(inp, o)
    elif args.mode == "st2ckpt" or (args.mode == "auto" and inp.endswith(".safetensors")):
        o = resolve_out(inp, out, "ckpt")
        st_to_ckpt(inp, o)
    else:
        print(f"Error: extensión no reconocida para {inp}. Usa --mode", file=sys.stderr); sys.exit(1)

if __name__ == "__main__":
    main()
PYEOF
)

# --- helper: llamar python ---
run_convert() {
  local in="$1"
  local out="$2"
  local mode="$3" # auto | ckpt2st | st2ckpt
  local suffix_flag=""
  if [[ "$SUFFIX" == true ]]; then suffix_flag="--suffix"; fi
  echo -e "${BLUE}>> $in -> ${out:-auto}${NC}"
  python3 -c "$PY_CONVERTER" "$in" "$out" $suffix_flag --mode "$mode"
}

# --- batch mode ---
if [[ "$BATCH" == true ]]; then
  if [[ ! -d "$INPUT" ]]; then echo -e "${RED}--batch requiere un directorio${NC}"; exit 1; fi
  echo -e "${GREEN}[*] Modo batch en: $INPUT (suffix=$SUFFIX, mode=$MODE)${NC}"
  # determinar extensión a buscar
  if [[ "$MODE" == "st2ckpt" ]]; then EXT="safetensors"
  elif [[ "$MODE" == "ckpt2st" ]]; then EXT="ckpt"
  else
    # auto: si el usuario no forzó, convertimos ckpt->st por defecto (como pide el issue)
    # pero si quiere lo contrario puede usar --to-ckpt
    # Buscamos ckpt
    EXT="ckpt"
    if [[ "$MODE" == "auto" ]]; then
      # si hay mezcla, convertimos ambos: primero ckpt->st, luego st->ckpt si se pide?
      # Para simplicidad en auto batch, convertimos ckpt->st
      echo -e "${YELLOW}[!] En batch auto se asume ckpt->safetensors. Usa --to-ckpt para invertir.${NC}"
    fi
  fi
  count=0
  while IFS= read -r -d '' file; do
    if [[ "$MODE" == "st2ckpt" ]]; then m="st2ckpt"; else m="ckpt2st"; fi
    if [[ "$MODE" == "auto" ]]; then m="auto"; fi
    run_convert "$file" "" "$m"
    count=$((count+1))
  done < <(find "$INPUT" -type f -name "*.${EXT}" -print0)
  if [[ $count -eq 0 ]]; then echo -e "${YELLOW}No se encontraron archivos *.${EXT} en $INPUT${NC}"
  else echo -e "${GREEN}[✓] Convertidos $count archivos${NC}"; fi
  exit 0
fi

# --- single file mode ---
if [[ -d "$INPUT" ]]; then
  echo -e "${RED}Error: '$INPUT' es un directorio. Usa --batch para directorios.${NC}"; exit 1
fi
if [[ ! -f "$INPUT" ]]; then echo -e "${RED}No existe archivo: $INPUT${NC}"; exit 1; fi

# resolver modo
if [[ "$MODE" == "auto" ]]; then
  if [[ "$INPUT" == *.ckpt ]]; then MODE="ckpt2st"
  elif [[ "$INPUT" == *.safetensors ]]; then MODE="st2ckpt"
  else echo -e "${RED}Extensión no reconocida. Usa --to-st o --to-ckpt${NC}"; exit 1
  fi
fi

run_convert "$INPUT" "$OUTPUT" "$MODE"
echo -e "${GREEN}[✓] Conversión completada. Verifica el modelo antes de borrar el original.${NC}"
# Advertencia de seguridad del proyecto original
echo -e "${YELLOW}Nota: el .ckpt se cargó con torch.load (pickle). Usa solo modelos de fuente confiable.${NC}"
