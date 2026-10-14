#!/usr/bin/env python3
"""Download the public model files used by the portable Z-Image stack.

The downloader is resumable: Hugging Face reuses files already present in the
target directories. LoRAs are deliberately not downloaded because they are
user-specific and may contain copyrighted or private training material.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


QWEN_REPO = "DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF"
QWEN_FILE = "Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-MTP-Q4_K_M.gguf"
QWEN_MMPROJ = "mmproj-BF16.gguf"
TEXT_REPO = "BennyDaBall/Qwen3-4b-Z-Image-Turbo-AbliteratedV1"
TEXT_FILES = ("config.json", "tokenizer.json", "chat_template.jinja", "tokenizer_config.json")


def existing(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--transformer-quant", default=os.environ.get("ZIMAGE_TRANSFORMER_QUANT", "Q3_K_M"))
    parser.add_argument("--text-quant", default=os.environ.get("ZIMAGE_TEXT_ENCODER_QUANT", "Q3_K_M"))
    parser.add_argument("--skip-qwen", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download, snapshot_download

    data = args.data.expanduser().resolve()
    models = data / "models"
    base = Path(os.environ.get('ZIMAGE_BASE_PIPELINE', str(models / 'zimage-official')))
    transformer = models / "Z-Image-Turbo-GGUF"
    text = Path(os.environ.get('ZIMAGE_TEXT_ENCODER_DIR', str(models / 'Qwen3-4b-Z-Image-Turbo-AbliteratedV1')))
    qwen = models / "qwen"
    for directory in (base, transformer, text, qwen, data / "lora_models_peft", data / "outputs"):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    marker = base / '.bootstrap-complete'
    if not marker.is_file():
        print(f"[models] Descargando pipeline base Tongyi-MAI/Z-Image-Turbo en {base}")
        snapshot_download("Tongyi-MAI/Z-Image-Turbo", local_dir=str(base), max_workers=8)
        marker.touch()
    else:
        print(f"[models] Pipeline base ya presente: {base}")

    transformer_file = Path(os.environ.get('ZIMAGE_TRANSFORMER_GGUF', str(transformer / f"z_image_turbo-{args.transformer_quant}.gguf")))
    if not existing(transformer_file):
        print(f"[models] Descargando transformer GGUF ({args.transformer_quant})")
        hf_hub_download("jayn7/Z-Image-Turbo-GGUF", transformer_file.name, local_dir=str(transformer_file.parent))

    text_file = text / f"Z-Image-AbliteratedV1.{args.text_quant}.gguf"
    if not existing(text_file):
        print(f"[models] Descargando text encoder GGUF ({args.text_quant})")
        hf_hub_download(TEXT_REPO, text_file.name, local_dir=str(text))
    for filename in TEXT_FILES:
        if not existing(text / filename):
            print(f"[models] Descargando metadato del text encoder: {filename}")
            hf_hub_download(TEXT_REPO, filename, local_dir=str(text))

    if not args.skip_qwen:
        for variable, filename in (('MODEL_FILE', QWEN_FILE), ('LLAMA_MMPROJ', QWEN_MMPROJ)):
            target = Path(os.environ.get(variable, str(qwen / filename)))
            if not existing(target):
                if target.name != filename:
                    raise FileNotFoundError(f'{variable}: falta modelo personalizado {target}; no se conoce su repositorio')
                print(f"[models] Descargando Qwen: {filename}")
                hf_hub_download(QWEN_REPO, filename, local_dir=str(target.parent))

    print("[models] Modelos públicos comprobados.")
    print("[models] Los LoRA se colocan manualmente en:", data / "lora_models_peft")


if __name__ == "__main__":
    main()
