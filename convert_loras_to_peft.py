#!/usr/bin/env python3
"""Normaliza LoRAs Z-Image al formato PEFT que espera Diffusers.

Convierte:
  transformer.layers...lora_down(.weight) -> layers...lora_A.weight
  diffusion_model.layers...lora_up(.weight) -> layers...lora_B.weight

Las claves PEFT se guardan sin el nombre del adaptador; PEFT lo inserta al cargar.
Los tensores .alpha se omiten, pero su escala alpha/rank se incorpora en
lora_B para conservar la intensidad original del LoRA.
"""
from pathlib import Path
import shutil
import sys

from safetensors.torch import load_file, save_file


SRC = Path("~/lora_models")
DST = Path("~/lora_models_peft")


def normalize_key(key: str):
    # Quitar namespaces del formato de entrenamiento.
    for prefix in ("transformer.", "diffusion_model."):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break

    # El checkpoint no debe incluir el nombre de instancia del adaptador.
    key = key.replace(".lora_A.default.weight", ".lora_A.weight")
    key = key.replace(".lora_B.default.weight", ".lora_B.weight")

    # Ya convertido / formato PEFT.
    if ".lora_A." in key or ".lora_B." in key:
        return key

    if key.endswith(".lora_down.weight"):
        return key[:-len(".lora_down.weight")] + ".lora_A.weight"
    if key.endswith(".lora_up.weight"):
        return key[:-len(".lora_up.weight")] + ".lora_B.weight"
    if key.endswith(".lora_down"):
        return key[:-len(".lora_down")] + ".lora_A.weight"
    if key.endswith(".lora_up"):
        return key[:-len(".lora_up")] + ".lora_B.weight"

    # alpha no es un peso PEFT; se omite en convert_file().
    if key.endswith(".alpha"):
        return None
    return key


def convert_file(src: Path, dst: Path):
    data = load_file(str(src), device="cpu")
    converted = {}
    skipped_alpha = 0

    # Kohya guarda alpha como un escalar por módulo. Lo convertimos en una
    # escala sobre B, porque el formato PEFT que cargará Diffusers no recibe
    # network_alphas cuando se usa prefix=None.
    alphas = {}
    for key, tensor in data.items():
        if key.endswith(".alpha"):
            base = key[:-len(".alpha")]
            for prefix in ("transformer.", "diffusion_model."):
                if base.startswith(prefix):
                    base = base[len(prefix):]
                    break
            alphas[base] = float(tensor.item())

    down_ranks = {}
    for key, tensor in data.items():
        new_key = normalize_key(key)
        if new_key and ".lora_A.weight" in new_key:
            base = new_key[:-len(".lora_A.weight")]
            down_ranks[base] = tensor.shape[0]

    for key, tensor in data.items():
        new_key = normalize_key(key)
        if new_key is None:
            skipped_alpha += 1
            continue
        if ".lora_B.weight" in new_key:
            base = new_key[:-len(".lora_B.weight")]
            if base in alphas and base in down_ranks:
                tensor = tensor * (alphas[base] / down_ranks[base])
        if new_key in converted:
            raise ValueError(f"colisión de claves: {new_key}")
        converted[new_key] = tensor

    if not converted:
        raise ValueError("archivo vacío después de normalizar")

    bad = [k for k in converted if not (".lora_A." in k or ".lora_B." in k)]
    if bad:
        raise ValueError(f"quedan claves no PEFT, ejemplo: {bad[0]}")

    save_file(converted, str(dst), metadata={"format": "pt", "converted_to": "diffusers-peft"})
    return len(data), len(converted), skipped_alpha


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else SRC
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else DST
    dst.mkdir(parents=True, exist_ok=True)

    ok = failed = 0
    for path in sorted(src.glob("*.safetensors")):
        out = dst / path.name
        try:
            total, kept, alpha = convert_file(path, out)
            print(f"OK {path.name}: {total} -> {kept} tensores PEFT; alpha omitidos={alpha}")
            ok += 1
        except Exception as exc:
            print(f"ERROR {path.name}: {exc}")
            failed += 1

    print(f"Resultado: {ok} convertidos, {failed} errores; salida: {dst}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
