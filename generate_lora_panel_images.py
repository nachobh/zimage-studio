#!/usr/bin/env python3
"""Genera secuencialmente las tarjetas PNG del panel usando Z-Image TCP.

python3 generate_lora_panel_images.py --dry-run
python3 generate_lora_panel_images.py
python3 generate_lora_panel_images.py --only 'Aimee_Lou_Wood_(vrtlAimeeLouWood)'
python3 generate_lora_panel_images.py --vertical --profile-prompts

Solo biblioteca estándar. No carga modelos: usa el servidor ya iniciado.
Omite imágenes existentes y LoRAs no cargados; --overwrite permite regenerar.
Los prompts se construyen desde los nombres, sin cambiar los perfiles JSON.
--vertical busca exclusivamente PNG `_1_2` ausentes y los publica a 512×1024,
16 pasos; no altera las tarjetas cuadradas ni verticales ya existentes.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import socket
import struct
import tempfile
import time

ROOT = Path(__file__).resolve().parent
PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


def recv_exact(conn, length):
    data = bytearray()
    while len(data) < length:
        chunk = conn.recv(min(length - len(data), 1024 * 1024))
        if not chunk:
            raise ConnectionError('Respuesta TCP incompleta')
        data.extend(chunk)
    return data


def request(payload, host, port, timeout):
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    with socket.create_connection((host, port), timeout=timeout) as conn:
        conn.sendall(struct.pack('!I', len(data)) + data)
        length = struct.unpack('!I', recv_exact(conn, 4))[0]
        if not 0 < length <= 16 * 1024**2:
            raise ValueError('Longitud de respuesta inválida')
        result = json.loads(recv_exact(conn, length))
    if not isinstance(result, dict) or result.get('status') != 'ok':
        raise RuntimeError(f'Z-Image: {result}')
    return result


def identity(name):
    match = re.search(r'\((vrtl[^()]*)\)', name, re.IGNORECASE)
    if match:
        return match.group(1), name[:match.start()].replace('_', ' ').strip()
    match = re.search(r'(vrtl[a-zA-Z]+)', name, re.IGNORECASE)
    if match:
        return match.group(1), name[:match.start()].replace('_', ' ').strip()
    return '', name.replace('_', ' ').strip()


def prompt_for(name, names):
    trigger, label = identity(name)
    if trigger and ' ' not in label:
        # Reuse the readable spelling from a matching LoRA, e.g.
        # magalivazquez_vrtlmaguiansuz -> Magali Vazquez (vrtlMaguiAnsuz).
        # The trigger's spelling remains exactly the one in this filename.
        for other in names:
            other_trigger, other_label = identity(other)
            if other_trigger.casefold() == trigger.casefold() and ' ' in other_label:
                label = other_label
                break
    return ' '.join(part for part in (trigger, label, 'mirando a cámara') if part)


def profile_prompt(model, names):
    """Use the optional profile image prompt without accepting arbitrary paths."""
    profile = model.with_suffix('.json')
    try:
        value = json.loads(profile.read_text(encoding='utf-8'))
        prompt = value.get('image_prompt') if isinstance(value, dict) else None
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()
    except (OSError, ValueError, UnicodeDecodeError):
        pass
    return prompt_for(model.stem, names)


def verify_image(path, width, height):
    with path.open('rb') as stream:
        header = stream.read(24)
    if len(header) != 24 or not header.startswith(PNG_SIGNATURE) or header[12:16] != b'IHDR':
        raise ValueError('El servidor no ha producido un PNG válido')
    if struct.unpack('!II', header[16:24]) != (width, height):
        raise ValueError('La imagen devuelta tiene dimensiones distintas de las solicitadas')


def generate_one(args, model, prompt, call=request):
    name = model.stem
    target = model.parent / (name + args.suffix + '.png')
    # A temporary filename prevents incomplete/failed generations appearing as
    # finished cards. It also protects an old PNG during --overwrite.
    with tempfile.TemporaryDirectory(prefix='.panel-image-', dir=model.parent) as stage:
        candidate = Path(stage) / 'portrait.png'
        payload = {'cmd': 'generate', 'prompt': prompt, 'negative_prompt': '',
                   'width': args.width, 'height': args.height, 'steps': args.steps,
                   'cfg': args.cfg, 'seed': args.seed, 'lora': f'{name}:{args.weight}',
                   'output_dir': stage, 'filename': 'portrait.png'}
        result = call(payload, args.host, args.port, args.timeout)
        if result.get('status') != 'ok':
            raise RuntimeError(str(result.get('error', 'Generación fallida')))
        active = result.get('lora', [])
        if len(active) != 1 or active[0][0] != name or not math.isclose(float(active[0][1]), args.weight):
            raise RuntimeError(f'El servidor no confirmó el LoRA solicitado: {active!r}')
        if Path(result.get('path', '')).resolve() != candidate.resolve():
            raise RuntimeError('El servidor devolvió una ruta distinta de la solicitada')
        verify_image(candidate, args.width, args.height)
        if args.overwrite:
            os.replace(candidate, target)
        else:
            # Atomic publish without overwriting a card added by another process.
            os.link(candidate, target)
        return {'path': str(target), 'duration': result.get('duration'), 'lora': active}


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--host', '--backend-host', default='127.0.0.1')
    p.add_argument('--port', '--backend-port', type=int, default=9877)
    p.add_argument('--lora-dir', type=Path, default=ROOT / 'lora_models_peft')
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--height', type=int, default=512)
    p.add_argument('--steps', type=int, default=9)
    p.add_argument('--cfg', type=float, default=0.0)
    p.add_argument('--weight', type=float, default=0.8)
    p.add_argument('--seed', type=int, default=42, help='Misma semilla para todas las tarjetas (por defecto: 42)')
    p.add_argument('--timeout', type=float, default=900, help='Tiempo máximo por petición, en segundos')
    p.add_argument('--only', action='append', default=[], metavar='NOMBRE', help='Nombre exacto sin extensión; se puede repetir')
    p.add_argument('--limit', type=int, help='Máximo de imágenes nuevas que intentar generar')
    p.add_argument('--overwrite', action='store_true', help='Regenerar incluso si existe el PNG')
    p.add_argument('--suffix', default='', help='Sufijo seguro del PNG, por ejemplo _1_2 para retratos verticales')
    p.add_argument('--vertical', action='store_true', help='Tarjetas verticales faltantes: _1_2.png, 512×1024 y 16 pasos')
    p.add_argument('--profile-prompts', action='store_true', help='Usar image_prompt de cada JSON si existe')
    p.add_argument('--dry-run', action='store_true', help='Mostrar los prompts locales sin llamar al servidor ni escribir archivos')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.vertical:
        if args.suffix:
            p.error('--vertical no se combina con --suffix')
        args.suffix = '_1_2'
        args.width, args.height, args.steps = 512, 1024, 16
    if any(value < 64 or value > 4096 or value % 16 for value in (args.width, args.height)):
        p.error('ancho y alto: múltiplos de 16 entre 64 y 4096')
    if not 1 <= args.steps <= 100 or not 0 <= args.seed < 2**32:
        p.error('steps: 1–100; seed: 0–4294967295')
    if not math.isfinite(args.cfg) or not 0 <= args.cfg <= 20 or not math.isfinite(args.weight) or not 0 < args.weight <= 3:
        p.error('CFG: 0–20; peso: mayor que 0 y hasta 3')
    if not math.isfinite(args.timeout) or args.timeout <= 0 or (args.limit is not None and args.limit < 1):
        p.error('timeout y limit deben ser positivos')
    if args.suffix and not re.fullmatch(r'[A-Za-z0-9_]+', args.suffix):
        p.error('suffix: solo letras, números y guion bajo')
    directory = args.lora_dir.resolve()
    models = sorted(directory.glob('*.safetensors'))
    if not models:
        p.error(f'No se encontraron LoRAs en {directory}')
    names = [m.stem for m in models]
    missing = set(args.only) - set(names)
    if missing:
        p.error('LoRAs no encontrados: ' + ', '.join(sorted(missing)))
    if args.only:
        models = [m for m in models if m.stem in args.only]
    log_path = directory / 'panel_image_generation.jsonl'
    skipped = unavailable = successes = failures = 0
    loaded = None
    if not args.dry_run:
        status = request({'cmd': 'status'}, args.host, args.port, args.timeout)
        if not status.get('model_loaded'):
            raise RuntimeError('El modelo no está cargado en el servidor')
        response = request({'cmd': 'list_loras'}, args.host, args.port, args.timeout)
        loaded = {item['name'] for item in response['lora_list']}
    jobs = []
    for model in models:
        if not args.overwrite and (model.parent / (model.stem + args.suffix + '.png')).exists():
            skipped += 1
            print(f'OMITIDO (PNG existente): {model.stem}', flush=True)
        elif loaded is not None and model.stem not in loaded:
            unavailable += 1
            print(f'OMITIDO (LoRA no cargado en el servidor): {model.stem}', flush=True)
        else:
            jobs.append(model)
    if args.limit:
        jobs = jobs[:args.limit]
    kind = ' verticales' if args.vertical else ''
    print(f'{len(jobs)} imágenes{kind} · {args.width}×{args.height} · steps={args.steps} · CFG={args.cfg} · peso={args.weight} · seed={args.seed}', flush=True)
    if args.dry_run:
        print('Simulación: no se comprueba aquí qué LoRAs tiene cargados el servidor.')
    for index, model in enumerate(jobs, 1):
        prompt = profile_prompt(model, names) if args.profile_prompts else prompt_for(model.stem, names)
        print(f'[{index}/{len(jobs)}] {model.stem}\n  Prompt: {prompt}', flush=True)
        if args.dry_run:
            continue
        entry = {'time': datetime.now(timezone.utc).isoformat(), 'name': model.stem, 'prompt': prompt,
                 'width': args.width, 'height': args.height, 'steps': args.steps, 'cfg': args.cfg,
                 'weight': args.weight, 'seed': args.seed}
        start = time.monotonic()
        stop = False
        try:
            result = generate_one(args, model, prompt)
            entry.update(status='ok', **result)
            successes += 1
            print(f'  OK: {result["path"]} ({time.monotonic() - start:.1f}s)', flush=True)
        except (TimeoutError, ConnectionError, socket.timeout) as exc:
            # The server might still be generating; don't enqueue retries.
            entry.update(status='error', error=str(exc)); failures += 1
            stop = True
            print(f'  ERROR de conexión: {exc}. Se detiene el lote; puedes reanudar ejecutándolo de nuevo.', flush=True)
        except Exception as exc:
            entry.update(status='error', error=str(exc)); failures += 1
            print(f'  ERROR: {exc}', flush=True)
        with log_path.open('a', encoding='utf-8') as out:
            out.write(json.dumps(entry, ensure_ascii=False) + '\n')
        if stop:
            break
    print(f'Final: {successes} generadas, {failures} errores, {skipped} existentes, {unavailable} no cargadas.', flush=True)
    if not args.dry_run:
        print(f'Registro: {log_path}', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nInterrumpido. Las tarjetas completadas se conservan; vuelve a ejecutar para reanudar.')
        raise SystemExit(130)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f'ERROR: {exc}')
        raise SystemExit(1)
