#!/usr/bin/env python3
"""Normalize and audit LoRAs on CPU; default is read-only, --apply backs up and replaces.

Uses the real ZImage architecture and Diffusers loader, without base checkpoint
weights or GPU allocation. This verifies loading, not image quality.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from datetime import datetime

import torch
from accelerate import init_empty_weights
from diffusers import ZImageTransformer2DModel
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from convert_loras_to_peft import convert_file, normalize_key
from verify_zimage_lora import verify_loaded_lora


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(path, config):
    with init_empty_weights():
        model = ZImageTransformer2DModel.from_config(config)
    # Allocate only target base matrices on CPU, without initializing or using
    # them. PEFT places adapter tensors on the corresponding base device.
    with safe_open(str(path), framework='pt', device='cpu') as source:
        for key in source.keys():
            if key.endswith('.lora_A.weight'):
                module = model.get_submodule(key[:-len('.lora_A.weight')])
                shape = source.get_slice(key).get_shape()
                if shape[1] != module.in_features:
                    raise ValueError(f'Dimensión de entrada incorrecta: {key}')
                module.to_empty(device='cpu')
    model.load_lora_adapter(str(path.parent), weight_name=path.name,
                            adapter_name='key_audit', prefix=None,
                            low_cpu_mem_usage=True)
    result = verify_loaded_lora(model, path, 'key_audit')
    del model
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('~/lora_models_peft'))
    parser.add_argument('--config', default='~/models/zimage-official/transformer')
    parser.add_argument('--backup-dir', type=Path, default=Path('~/lora_key_repair_backups'))
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--pattern', default='*.safetensors')
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = ZImageTransformer2DModel.load_config(args.config)
    run_dir = args.backup_dir / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir.mkdir(parents=True)
    report = {'apply': args.apply, 'files': []}
    paths = sorted(args.directory.glob(args.pattern))
    if not paths:
        raise ValueError('No se encontraron archivos')
    for path in paths:
        entry = {'file': str(path)}
        temporary = None
        try:
            with safe_open(str(path), framework='pt', device='cpu') as source:
                keys, metadata = source.keys(), source.metadata()
            needs_fix = any(normalize_key(k) != k for k in keys)
            entry['sha256_before'] = sha256(path)
            candidate = path
            if needs_fix:
                fd, name = tempfile.mkstemp(prefix='.lora-repair-', suffix='.safetensors', dir=path.parent)
                os.close(fd)
                temporary = Path(name)
                convert_file(path, temporary)
                # Keep original metadata. Conversion only changes key names,
                # except legacy alpha/rank, folded into B by convert_file.
                converted = load_file(str(temporary))
                with safe_open(str(path), framework='pt', device='cpu') as source:
                    if not any(k.endswith('.alpha') for k in keys):
                        for k in keys:
                            if not torch.equal(source.get_tensor(k), converted[normalize_key(k)]):
                                raise ValueError(f'La conversión cambió valores: {k}')
                save_file(converted, str(temporary), metadata=metadata)
                del converted
                candidate = temporary
            entry.update(audit(candidate, config))
            entry['sha256_after'] = sha256(candidate)
            if needs_fix and args.apply:
                backup = run_dir / path.name
                shutil.copy2(path, backup)
                if sha256(backup) != entry['sha256_before'] or sha256(path) != entry['sha256_before']:
                    raise ValueError('Backup distinto o archivo modificado durante la auditoría')
                shutil.copymode(path, candidate)
                os.replace(candidate, path)
                temporary = None
                entry['backup'] = str(backup)
            entry['status'] = ('repaired' if args.apply else 'would_repair') if needs_fix else 'verified'
            print(f"OK {path.name}: {entry['status']}, {entry['tensors_verified']} tensores", flush=True)
        except Exception as exc:
            entry.update(status='error', error=str(exc))
            print(f'ERROR {path.name}: {exc}', flush=True)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        report['files'].append(entry)
        (run_dir / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'Informe: {run_dir / "report.json"}', flush=True)
    return int(any(e['status'] == 'error' for e in report['files']))


if __name__ == '__main__':
    raise SystemExit(main())
