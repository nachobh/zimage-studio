#!/usr/bin/env python3
"""Offline, single-image-at-a-time Z-Image PEFT training from reviewed captions.

Requires a complete Diffusers Z-Image checkpoint. A GGUF inference transformer
cannot provide gradients and is deliberately rejected by the panel preflight.
"""
import argparse
import math
import json
import os
from pathlib import Path
import random
import re
import tempfile

from PIL import Image

REPORT_NAME = 'training_report.json'
TARGET_STEPS = 3000
TRAIN_DEFAULTS = {'epochs': None, 'learning_rate': 5e-5, 'rank': 16, 'resolution': 1024,
                  'centering_y': 0.3, 'min_size': 512, 'skip_small': True,
                  'sample_every': 250, 'sample_seed': 42}
STRENGTH_LOW = 0.01
STRENGTH_BAND = (0.03, 0.10)
TRAIN_MODULE = re.compile(r'^layers\.\d+\..+\.(?:to_q|to_k|to_v|to_out\.0|w1|w2|w3)$')


def training_target_modules(transformer):
    """Match the seven modules per main layer used by the established LoRAs."""
    modules = [name for name, _ in transformer.named_modules() if TRAIN_MODULE.fullmatch(name)]
    counts = {}
    for name in modules:
        layer = name.split('.')[1]
        counts[layer] = counts.get(layer, 0) + 1
    if not counts or any(count != 7 for count in counts.values()):
        raise RuntimeError('No se hallaron los siete módulos LoRA por capa principal del transformer')
    return modules


def prepare_trainable_parameters(transformer):
    """Keep LoRA updates and AdamW moments in FP32, leaving frozen weights intact."""
    import torch
    trainable = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError('PEFT no insertó parámetros entrenables')
    for parameter in trainable:
        parameter.data = parameter.data.to(dtype=torch.float32)
    return trainable


def training_schedule(photo_count, epochs=None):
    """Auto mode stops at 3000 updates, including a partial final epoch."""
    if photo_count < 1:
        raise ValueError('Se necesita al menos una foto')
    steps = TARGET_STEPS if epochs is None else epochs * photo_count
    return math.ceil(steps / photo_count), steps


def training_sample(pipe, prompt, resolution, seed, destination):
    """Diffuse on GPU, then decode on CPU without changing the training VAE."""
    import torch
    was_training = pipe.transformer.training
    vae = pipe.vae
    original_device = vae.device
    original_tiling = (vae.use_tiling, vae.tile_sample_min_size, vae.tile_latent_min_size)
    try:
        pipe.transformer.eval()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
            generator = torch.Generator(device='cuda').manual_seed(seed)
            print('Muestra · iniciando difusión en GPU', flush=True)
            # Returning latents skips the pipeline's full-frame float32 GPU decode.
            latents = pipe(prompt=prompt, height=resolution, width=resolution,
                           num_inference_steps=9, guidance_scale=0.0,
                           generator=generator, output_type='latent').images
            latents = latents.detach().to(device='cpu', dtype=vae.dtype)
            torch.cuda.synchronize()
            print('Muestra · difusión terminada; decodificando VAE en CPU por bloques', flush=True)
            vae.to('cpu')
            vae.enable_tiling()
            vae.tile_sample_min_size = min(512, original_tiling[1])
            vae.tile_latent_min_size = vae.tile_sample_min_size // pipe.vae_scale_factor
            latents = latents / vae.config.scaling_factor + vae.config.shift_factor
            decoded = vae.decode(latents, return_dict=False)[0]
            image = pipe.image_processor.postprocess(decoded, output_type='pil')[0]
            print('Muestra · decodificación terminada; guardando PNG', flush=True)
            image.save(destination)
    finally:
        # Encoding the next training photo must keep its original device and tiling.
        vae.use_tiling, vae.tile_sample_min_size, vae.tile_latent_min_size = original_tiling
        try:
            print('Muestra · restaurando VAE al dispositivo de entrenamiento', flush=True)
            vae.to(original_device)
        finally:
            pipe.transformer.train(was_training)


def usable_photos(folder, min_size, keep_small=False):
    """Split ``*.jpg`` photos by their smaller side, so low-res frames can be skipped."""
    photos = sorted(Path(folder, 'photos').glob('*.jpg'))
    kept, skipped = [], []
    for photo in photos:
        if keep_small or not min_size:
            kept.append(photo)
            continue
        try:
            with Image.open(photo) as image:
                side = min(image.size)
        except OSError:
            skipped.append((photo, 0))
            continue
        if side < min_size:
            skipped.append((photo, side))
        else:
            kept.append(photo)
    return kept, skipped


def _strength(transformer):
    """Median ``||B·A|| / ||W||`` of the installed adapter, the usual LoRA intensity proxy."""
    import torch
    params = dict(transformer.named_parameters())
    relatives = []
    for key, parameter in params.items():
        if '.lora_A.' not in key:
            continue
        up = key.replace('.lora_A.', '.lora_B.')
        if up not in params:
            continue
        base_key = None
        for suffix in ('.lora_A.default.weight', '.lora_A.weight'):
            if key.endswith(suffix):
                base_key = key[:-len(suffix)] + '.base_layer.weight'
                break
        base = params.get(base_key) if base_key else None
        if base is None:
            continue
        # PEFT stores A as (rank, in) and B as (out, rank): ΔW = B @ A.
        delta = (params[up].detach().to('cpu', torch.float32) @
                 parameter.detach().to('cpu', torch.float32))
        denominator = base.detach().to('cpu', torch.float32).norm().item()
        numerator = delta.norm().item()
        relatives.append(numerator / denominator if denominator else 0.0)
    if not relatives:
        return None
    relatives.sort()
    return {'median': relatives[len(relatives) // 2],
            'p90': relatives[int(0.9 * (len(relatives) - 1))],
            'max': relatives[-1],
            'modules': len(relatives)}


def advice_for(report):
    """Plain-language verdicts for the training report shown on the creator page."""
    strength = report.get('rel_median')
    advice = []
    if strength is None:
        advice.append('No se pudo medir la intensidad del LoRA; revisa que el entrenamiento guardó matrices lora_A y lora_B.')
    elif strength < STRENGTH_LOW:
        advice.append(f'Actualización de pesos débil según el umbral orientativo (Δ {strength:.4f} < {STRENGTH_LOW}). '
                      'Este valor no determina el parecido: compáralo con un LoRA de referencia antes de cambiar pasos o learning rate.')
    elif strength > STRENGTH_BAND[1]:
        advice.append(f'LoRA muy intenso (Δ {strength:.4f} > {STRENGTH_BAND[1]}): usa pesos bajos (0.4–0.7) '
                      'para evitar sobreajuste y caras recicladas.')
    elif STRENGTH_BAND[0] <= strength <= STRENGTH_BAND[1]:
        advice.append(f'Intensidad adecuada (Δ {strength:.4f}, banda {STRENGTH_BAND[0]}–{STRENGTH_BAND[1]}).')
    else:
        advice.append(f'Intensidad algo baja respecto a la referencia orientativa (Δ {strength:.4f}); '
                      'compara visualmente con el LoRA anterior usando los mismos ajustes.')
    advice.append('Compara pesos 0.6, 0.8, 1.0 y 1.2 con las mismas semillas y pasos; '
                  'prueba CFG 0, 0.3 y 1 y conserva la combinación que mejor reproduzca la identidad.')
    advice.append('La banda de intensidad es orientativa; confirma la calidad visual con varias semillas y el mismo prompt.')
    return advice


def write_report(folder, report):
    """Atomic write of ``training_report.json`` next to the dataset."""
    folder = Path(folder)
    fd, temp = tempfile.mkstemp(prefix='.training-', suffix='.json', dir=folder)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        os.replace(temp, folder / REPORT_NAME)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def train(args):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from PIL import Image, ImageOps
    from diffusers import ZImagePipeline
    from peft import LoraConfig
    from safetensors.torch import save_file

    if not torch.cuda.is_available():
        raise RuntimeError('Se necesita una GPU CUDA/ROCm para entrenar este LoRA')
    folder = Path(args.dataset)
    captions = json.loads((folder / 'captions.json').read_text(encoding='utf-8'))
    photos = sorted((folder / 'photos').glob('*.jpg'))
    if len(photos) < 5 or any(not captions.get(p.name, '').strip() for p in photos):
        raise ValueError('Se requieren 5 fotos y una caption revisada por foto')
    kept, skipped = usable_photos(folder, args.min_size, args.keep_small)
    for photo, side in skipped:
        print(f'Foto omitida ({side} px < {args.min_size}): {photo.name}', flush=True)
    if len(kept) < 5:
        raise ValueError(f'Quedan {len(kept)} fotos de al menos {args.min_size} px; añade fotos más '
                         'grandes o baja el mínimo de píxeles')
    epochs, steps_total = training_schedule(len(kept), args.epochs)
    print(f'{len(kept)} fotos · {epochs} épocas · {steps_total} pasos · lr {args.learning_rate:g} · '
          f'rango {args.rank} · {args.resolution} px · recorte vertical {args.centering_y:g}', flush=True)
    model = Path(args.model)
    if not (model / 'transformer').is_dir() or not (model / 'model_index.json').is_file():
        raise ValueError('Falta el checkpoint completo de Diffusers Z-Image')
    print('Cargando checkpoint Z-Image completo', flush=True)
    pipe = ZImagePipeline.from_pretrained(model, torch_dtype=torch.bfloat16, local_files_only=True).to('cuda')
    pipe.vae.requires_grad_(False)
    pipe.vae.to(dtype=torch.float32)
    pipe.text_encoder.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    targets = training_target_modules(pipe.transformer)
    print(f'Aplicando LoRA a {len(targets)} módulos de capas principales (sin refinadores)', flush=True)
    pipe.transformer.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank,
                                            target_modules=targets,
                                            lora_dropout=0.0))
    pipe.transformer.enable_gradient_checkpointing()
    trainable = prepare_trainable_parameters(pipe.transformer)
    print(f'Parámetros LoRA entrenables: {sum(p.numel() for p in trainable)} · '
          'pesos y optimizador float32 · forward mixto bfloat16', flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    random.seed(42)
    device = torch.device('cuda')
    step = 0
    sample_paths = []
    sample_prompt = captions[kept[0].name]
    sample_dir = Path(tempfile.mkdtemp(prefix='samples-', dir=folder)) if args.sample_every else None
    pipe.transformer.train()
    for epoch in range(epochs):
        random.shuffle(kept)
        for photo in kept:
            if step >= steps_total:
                break
            with Image.open(photo) as source:
                image = ImageOps.fit(source.convert('RGB'), (args.resolution, args.resolution),
                                     centering=(0.5, args.centering_y))
            import numpy as np
            pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0)
            pixels = (pixels.to(device=device, dtype=torch.float32) / 127.5) - 1
            with torch.no_grad():
                clean = pipe.vae.encode(pixels).latent_dist.sample()
                clean = (clean - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
                embeddings = pipe.encode_prompt(captions[photo.name], device=device,
                                                do_classifier_free_guidance=False, max_sequence_length=256)[0][0]
            clean = clean.to(dtype=torch.bfloat16)
            noise = torch.randn_like(clean)
            t = torch.rand(1, device=device, dtype=torch.float32)
            sample = noise * (1 - t) + clean * t
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                predicted = pipe.transformer([sample[0].unsqueeze(1)], t, [embeddings],
                                             return_dict=False)[0][0].squeeze(1)
                loss = torch.nn.functional.mse_loss(predicted.float(), (clean[0] - noise[0]).float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            print(f'Paso {step}/{steps_total} · pérdida {loss.item():.5f}', flush=True)
            # Release the backward graph before running inference on the same GPU.
            del loss, predicted, pixels, clean, noise, sample, embeddings
            if args.sample_every and (step % args.sample_every == 0 or step == steps_total):
                print(f'Generando muestra del paso {step} · semilla {args.sample_seed}', flush=True)
                destination = sample_dir / f'step-{step:06d}.png'
                training_sample(pipe, sample_prompt, args.resolution, args.sample_seed, destination)
                sample_paths.append(str(destination.relative_to(folder)))
                print(f'Muestra del paso {step} guardada · semilla {args.sample_seed}', flush=True)
    print('Guardando matrices LoRA', flush=True)
    weights = {}
    for key, parameter in pipe.transformer.named_parameters():
        if parameter.requires_grad and ('lora_A.' in key or 'lora_B.' in key):
            key = key.replace('.default.weight', '.weight')
            weights[key] = parameter.detach().cpu().contiguous().float()
    if not weights or not any(k.endswith('.lora_A.weight') for k in weights):
        raise RuntimeError('No se obtuvieron matrices LoRA')
    output = Path(args.output)
    fd, temp = tempfile.mkstemp(prefix='.adapter-', suffix='.safetensors', dir=output.parent)
    os.close(fd)
    try:
        save_file(weights, temp)
        os.replace(temp, output)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    print(f'LoRA guardado: {output.name}', flush=True)
    print('Calculando diagnóstico de intensidad', flush=True)
    strength = _strength(pipe.transformer)
    report = {'photos_used': len(kept), 'photos_skipped': len(skipped),
              'photos_total': len(photos), 'epochs': epochs, 'steps': step,
              'automatic_steps': args.epochs is None, 'steps_per_image': step / len(kept),
              'batch_size': 1, 'sample_every': args.sample_every,
              'sample_seed': args.sample_seed, 'samples': sample_paths,
              'sample_decode_device': 'cpu', 'sample_decode_tile_size': 512,
              'trainable_dtype': 'float32', 'optimizer_moments_dtype': 'float32',
              'learning_rate': args.learning_rate, 'rank': args.rank,
              'resolution': args.resolution, 'min_size': args.min_size,
              'keep_small': bool(args.keep_small), 'centering_y': args.centering_y,
              'output': output.name,
              'rel_median': round(strength['median'], 6) if strength else None,
              'rel_p90': round(strength['p90'], 6) if strength else None,
              'rel_max': round(strength['max'], 6) if strength else None,
              'modules': strength['modules'] if strength else 0}
    report['advice'] = advice_for(report)
    write_report(folder, report)
    if strength:
        print(f"Diagnóstico · Δ mediano {strength['median']:.4f} · p90 {strength['p90']:.4f} · "
              f"objetivo {STRENGTH_BAND[0]}–{STRENGTH_BAND[1]}", flush=True)
    for line in report['advice']:
        print(line, flush=True)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=TRAIN_DEFAULTS['epochs'],
                        help='Pasos por imagen; por defecto 3000 pasos totales / fotos usadas')
    parser.add_argument('--rank', type=int, default=TRAIN_DEFAULTS['rank'])
    parser.add_argument('--resolution', type=int, default=TRAIN_DEFAULTS['resolution'])
    parser.add_argument('--learning-rate', type=float, default=TRAIN_DEFAULTS['learning_rate'])
    parser.add_argument('--min-size', type=int, default=512,
                        help='Lado menor mínimo de una foto para usarla (0 desactiva el filtro)')
    parser.add_argument('--keep-small', action='store_true',
                        help='Entrenar también con fotos por debajo de --min-size')
    parser.add_argument('--centering-y', type=float, default=0.3,
                        help='Centrado vertical del recorte cuadrado (0 arriba, 1 abajo)')
    parser.add_argument('--sample-every', type=int, default=250, help='Muestra cada N pasos (0 desactiva)')
    parser.add_argument('--sample-seed', type=int, default=42)
    return parser


def check_args(parser, args):
    if args.epochs is not None and not 1 <= args.epochs <= 3000:
        parser.error('epochs: 1–3000')
    if not 4 <= args.rank <= 128:
        parser.error('rank: 4–128')
    if not 256 <= args.resolution <= 1024 or args.resolution % 64:
        parser.error('resolution: múltiplos de 64 entre 256 y 1024')
    if not 1e-6 <= args.learning_rate <= 1e-3:
        parser.error('learning-rate: 1e-6–1e-3')
    if not 0 <= args.min_size <= 2048:
        parser.error('min-size: 0–2048')
    if not 0 <= args.centering_y <= 1:
        parser.error('centering-y: 0–1')
    if not 0 <= args.sample_every <= 3000:
        parser.error('sample-every: 0–3000')
    if not 0 <= args.sample_seed <= 2**32 - 1:
        parser.error('sample-seed: 0–4294967295')
    return args


if __name__ == '__main__':
    parser = build_parser()
    train(check_args(parser, parser.parse_args()))
