"""Private character datasets and offline caption/training jobs for the panel."""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile
import threading
import wave

from PIL import Image, ImageOps

from zimage_character_train import REPORT_NAME, usable_photos
from zimage_conversations import image_errors

ROOT = Path(__file__).resolve().parent
NAME_RE = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z')
TRIGGER_RE = re.compile(r'vrtl[A-Za-z]+')
IMAGE_RE = re.compile(r'data:image/(png|jpeg|webp);base64,([A-Za-z0-9+/]+={0,2})\Z')
MAX_PHOTOS = 80
MAX_BYTES = 16 * 1024 * 1024
MAX_PIXELS = 24_000_000
MAX_VOICE_BYTES = 20 * 1024 * 1024
MAX_VOICE_SECONDS = 30
VOICE_RE = re.compile(r'data:audio/(?:wav|x-wav);base64,([A-Za-z0-9+/]+={0,2})\Z')
TRAIN_DEFAULTS = {'epochs': 40, 'learning_rate': 2e-4, 'rank': 16, 'resolution': 512,
                  'centering_y': 0.3, 'min_size': 512, 'skip_small': True}


def training_options(options):
    """Validate the training knobs sent by the creator page and fill the defaults."""
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError('Opciones de entrenamiento inválidas')

    def bounded(key, low, high, integer=False):
        value = options.get(key, TRAIN_DEFAULTS[key])
        if value is None or value == '':
            value = TRAIN_DEFAULTS[key]
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f'Opción inválida: {key}') from None
        if integer and not value.is_integer():
            raise ValueError(f'{key} debe ser un número entero')
        if not low <= value <= high:
            raise ValueError(f'{key} debe estar entre {low:g} y {high:g}')
        return int(value) if integer else value

    settings = {'epochs': bounded('epochs', 1, 200, integer=True),
                'learning_rate': bounded('learning_rate', 1e-6, 1e-3),
                'rank': bounded('rank', 4, 128, integer=True),
                'resolution': bounded('resolution', 256, 1024, integer=True),
                'centering_y': bounded('centering_y', 0, 1),
                'min_size': bounded('min_size', 0, 2048, integer=True)}
    if settings['resolution'] % 64:
        raise ValueError('resolution debe ser múltiplo de 64')
    skip_small = options.get('skip_small', TRAIN_DEFAULTS['skip_small'])
    if skip_small is None:
        skip_small = TRAIN_DEFAULTS['skip_small']
    elif isinstance(skip_small, str):
        skip_small = skip_small.strip().casefold() not in ('', '0', 'false', 'no')
    settings['skip_small'] = bool(skip_small)
    return settings


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.character-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


class CharacterCreator:
    def __init__(self, data_dir, lora_dir, python=None, model_dir=None):
        self.root = Path(data_dir).resolve() / 'character_training'
        self.lora_dir = Path(lora_dir).resolve()
        self.python = python or os.environ.get('ZIMAGE_TRAIN_PYTHON', sys.executable)
        self.model_dir = Path(model_dir or os.environ.get('ZIMAGE_TRAIN_MODEL_DIR',
                              str(Path(data_dir).resolve() / 'models' / 'zimage-training'))).resolve()
        self.lock = threading.RLock()
        self.jobs = {}

    def directory(self, name):
        if not isinstance(name, str) or not NAME_RE.fullmatch(name):
            raise ValueError('ID de personaje inválido: usa letras, números, guiones o guion bajo')
        if self.root.is_symlink():
            raise ValueError('Directorio de personajes no permitido')
        path = self.root / name
        if path.is_symlink() or (path.exists() and path.resolve().parent != self.root):
            raise ValueError('Ruta de personaje no permitida')
        return path

    def _metadata(self, name):
        path = self.directory(name) / 'character.json'
        if not path.is_file() or path.is_symlink():
            raise ValueError('Personaje no encontrado')
        return json.loads(path.read_text(encoding='utf-8'))

    def _job_active(self, name):
        return name in self.jobs and not self.jobs[name]['finished']

    def create(self, body):
        if not isinstance(body, dict):
            raise ValueError('Datos inválidos')
        name = body.get('name')
        label = body.get('display_name', '')
        if not isinstance(label, str) or not 1 <= len(label.strip()) <= 100:
            raise ValueError('Escribe un nombre de personaje (1–100 caracteres)')
        with self.lock:
            directory = self.directory(name)
            if any((self.lora_dir / (name + suffix)).exists() or
                   (self.lora_dir / (name + suffix)).is_symlink()
                   for suffix in ('.safetensors', '.json', '.png', '.wav')):
                raise ValueError('Ya hay archivos de LoRA o perfil con ese nombre')
            if directory.exists():
                raise ValueError('Ya existe un borrador con ese ID')
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.root, 0o700)
            directory.mkdir(mode=0o700)
            (directory / 'photos').mkdir(mode=0o700)
            trigger_match = TRIGGER_RE.search(name)
            trigger = trigger_match.group() if trigger_match else 'vrtl' + name.replace('-', '').replace('_', '')
            profile = {'system_prompt': f'Interpreta un personaje ficticio llamado {label.strip()}. Responde en español.',
                       'image_prompt': f'{trigger} {label.strip()} mirando a cámara'}
            metadata = {'name': name, 'display_name': label.strip(), 'trigger': trigger,
                        'profile': profile, 'captions': {}, 'state': 'draft'}
            atomic_json(directory / 'character.json', metadata)
            return self.get(name)

    def get(self, name):
        with self.lock:
            data = self._metadata(name)
            photos = sorted(p.name for p in (self.directory(name) / 'photos').glob('*.jpg') if not p.is_symlink())
            job = self.jobs.get(name)
            status = job['status'] if job else None
            voice = self._voice_info(name)
            return {**data, 'photos': photos, 'voice': voice, 'job': status,
                    'sizes': self.photo_sizes(name), 'report': self.training_report(name),
                    'model_ready': self.model_ready()}

    def photo_sizes(self, name):
        """Pixel size of every photo, so the page can warn about frames below the minimum."""
        sizes = {}
        for path in sorted((self.directory(name) / 'photos').glob('*.jpg')):
            if path.is_symlink():
                continue
            try:
                with Image.open(path) as image:
                    sizes[path.name] = [image.width, image.height]
            except OSError:
                continue
        return sizes

    def upscale_photos(self, name, target):
        """Enlarge the photos below ``target`` px with LANCZOS, keeping the original."""
        try:
            target = int(target)
        except (TypeError, ValueError):
            raise ValueError('Tamaño objetivo inválido')
        if not 128 <= target <= 2048:
            raise ValueError('El tamaño objetivo debe estar entre 128 y 2048 px')
        with self.lock:
            data = self._metadata(name)
            if data['state'] != 'draft' or self._job_active(name):
                raise ValueError('No puedes reescalar fotos durante el trabajo o después de publicar')
            photos_dir = self.directory(name) / 'photos'
            if photos_dir.is_symlink() or photos_dir.resolve().parent != self.directory(name):
                raise ValueError('Directorio de fotos no permitido')
            backup_dir = self.directory(name) / 'photos_original'
            if backup_dir.is_symlink():
                raise ValueError('Directorio de fotos no permitido')
            if not backup_dir.exists():
                backup_dir.mkdir(mode=0o700)
            result = {'status': 'ok', 'target': target, 'upscaled': [], 'skipped': [], 'errors': []}
            for photo in sorted(photos_dir.glob('*.jpg')):
                if photo.is_symlink():
                    continue
                try:
                    with Image.open(photo) as opened:
                        if opened.format != 'JPEG':
                            raise ValueError('Formato de foto no admitido')
                        image = ImageOps.exif_transpose(opened).convert('RGB')
                    before = list(image.size)
                    if min(before) >= target:
                        result['skipped'].append({'photo': photo.name, 'size': before})
                        continue
                    scale = target / min(before)
                    width = min(2048, int(round(before[0] * scale)))
                    height = min(2048, int(round(before[1] * scale)))
                    if width * height > MAX_PIXELS:
                        raise ValueError('Dimensiones resultantes no admitidas')
                    image = image.resize((width, height), Image.Resampling.LANCZOS)
                    # The first backup wins: it is the true original and is never overwritten.
                    backup = backup_dir / photo.name
                    if not backup.exists() and not backup.is_symlink():
                        fd, temp = tempfile.mkstemp(prefix='.backup-', dir=backup_dir)
                        try:
                            with os.fdopen(fd, 'wb') as stream:
                                stream.write(photo.read_bytes())
                            os.replace(temp, backup)
                        finally:
                            if os.path.exists(temp):
                                os.unlink(temp)
                    fd, temp = tempfile.mkstemp(prefix='.photo-', dir=photos_dir)
                    try:
                        with os.fdopen(fd, 'wb') as stream:
                            image.save(stream, 'JPEG', quality=93)
                        os.replace(temp, photo)
                    finally:
                        if os.path.exists(temp):
                            os.unlink(temp)
                    result['upscaled'].append({'photo': photo.name, 'before': before,
                                               'after': [width, height]})
                except Exception as exc:
                    # One broken photo must not stop the batch; the run is resumable.
                    result['errors'].append({'photo': photo.name, 'error': str(exc)})
            return result

    def restore_photos(self, name):
        """Undo the batch: move the originals kept in ``photos_original/`` back."""
        with self.lock:
            data = self._metadata(name)
            if data['state'] != 'draft' or self._job_active(name):
                raise ValueError('No puedes restaurar fotos durante el trabajo o después de publicar')
            photos_dir = self.directory(name) / 'photos'
            backup_dir = self.directory(name) / 'photos_original'
            for path in (photos_dir, backup_dir):
                if path.is_symlink() or path.resolve().parent != self.directory(name):
                    raise ValueError('Directorio de fotos no permitido')
            result = {'status': 'ok', 'restored': [], 'errors': []}
            if backup_dir.is_dir():
                for backup in sorted(backup_dir.glob('*.jpg')):
                    if backup.is_symlink():
                        continue
                    try:
                        with Image.open(backup) as opened:
                            if opened.format != 'JPEG':
                                raise ValueError('Formato de foto no admitido')
                            opened.verify()
                        os.replace(backup, photos_dir / backup.name)
                        result['restored'].append(backup.name)
                    except Exception as exc:
                        result['errors'].append({'photo': backup.name, 'error': str(exc)})
                if not any(backup_dir.iterdir()):
                    backup_dir.rmdir()
            return result

    def training_report(self, name):
        path = self.directory(name) / REPORT_NAME
        if path.is_symlink() or not path.is_file():
            return None
        try:
            report = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None
        return report if isinstance(report, dict) else None

    def _voice_info(self, name):
        path = self.directory(name) / 'voice.wav'
        if path.is_symlink() or not path.is_file():
            return {'present': False}
        try:
            with wave.open(str(path), 'rb') as audio:
                return {'present': True, 'size_bytes': path.stat().st_size,
                        'duration': round(audio.getnframes() / audio.getframerate(), 2)}
        except (wave.Error, EOFError, OSError):
            return {'present': False}

    def voice(self, name):
        self._metadata(name)
        path = self.directory(name) / 'voice.wav'
        if path.is_symlink() or not path.is_file() or path.resolve().parent != self.directory(name):
            raise ValueError('Audio no encontrado')
        return path.read_bytes()

    def add_voice(self, name, data_url):
        match = VOICE_RE.fullmatch(data_url) if isinstance(data_url, str) else None
        if not match or len(data_url) > MAX_VOICE_BYTES * 4 // 3 + 256:
            raise ValueError('Envía un audio WAV PCM de hasta 20 MB')
        try:
            raw = base64.b64decode(match.group(1), validate=True)
            if not 0 < len(raw) <= MAX_VOICE_BYTES:
                raise ValueError('El audio supera el límite de 20 MB')
            with wave.open(io.BytesIO(raw), 'rb') as audio:
                frames = audio.getnframes()
                duration = frames / audio.getframerate()
                if (audio.getcomptype() != 'NONE' or audio.getnchannels() not in (1, 2) or
                        audio.getsampwidth() not in (1, 2, 3, 4) or
                        not 8000 <= audio.getframerate() <= 96000 or
                        not 1 <= duration <= MAX_VOICE_SECONDS):
                    raise ValueError('El WAV debe ser PCM mono/estéreo, 8–96 kHz y durar entre 1 y 30 segundos')
                if len(audio.readframes(frames)) != frames * audio.getnchannels() * audio.getsampwidth():
                    raise ValueError('El WAV está truncado o incompleto')
        except (wave.Error, EOFError, OSError) as exc:
            raise ValueError('El archivo no es un WAV PCM válido') from exc
        with self.lock:
            metadata = self._metadata(name)
            if metadata['state'] != 'draft' or self._job_active(name):
                raise ValueError('No puedes cambiar el audio durante el trabajo o después de publicar')
            directory = self.directory(name)
            path = directory / 'voice.wav'
            fd, temp = tempfile.mkstemp(prefix='.voice-', dir=directory)
            try:
                with os.fdopen(fd, 'wb') as stream:
                    stream.write(raw)
                os.chmod(temp, 0o600)
                os.replace(temp, path)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
            return self.get(name)

    def remove_voice(self, name):
        with self.lock:
            metadata = self._metadata(name)
            if metadata['state'] != 'draft' or self._job_active(name):
                raise ValueError('No puedes cambiar el audio durante el trabajo o después de publicar')
            path = self.directory(name) / 'voice.wav'
            if path.is_symlink():
                raise ValueError('Ruta de audio no permitida')
            path.unlink(missing_ok=True)
            return self.get(name)

    def model_ready(self):
        def weights_complete(folder, index_name):
            if not folder.is_dir():
                return False
            index = folder / index_name
            if index.is_file():
                try:
                    files = set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values())
                except (ValueError, KeyError, TypeError):
                    return False
                return bool(files) and all((folder / filename).is_file() and
                                           (folder / filename).stat().st_size > 0 for filename in files)
            return any(path.stat().st_size > 0 for path in folder.glob('*.safetensors') if path.is_file())

        return ((self.model_dir / 'model_index.json').is_file() and
                weights_complete(self.model_dir / 'transformer', 'diffusion_pytorch_model.safetensors.index.json') and
                weights_complete(self.model_dir / 'text_encoder', 'model.safetensors.index.json') and
                weights_complete(self.model_dir / 'vae', 'diffusion_pytorch_model.safetensors.index.json'))

    def list(self):
        if not self.root.exists():
            return []
        return [self.get(p.name) for p in sorted(self.root.iterdir())
                if p.is_dir() and not p.is_symlink() and (p / 'character.json').is_file()]

    def add_photo(self, name, data_url):
        match = IMAGE_RE.fullmatch(data_url) if isinstance(data_url, str) else None
        if not match or len(data_url) > MAX_BYTES * 2:
            raise ValueError('Envía una foto PNG, JPEG o WebP de hasta 16 MB')
        raw = base64.b64decode(match.group(2), validate=True)
        if not 0 < len(raw) <= MAX_BYTES:
            raise ValueError('Foto demasiado grande')
        with self.lock:
            data = self._metadata(name)
            if data['state'] != 'draft' or self._job_active(name):
                raise ValueError('No puedes añadir fotos durante el trabajo o después de publicar')
            photos_dir = self.directory(name) / 'photos'
            if photos_dir.is_symlink() or photos_dir.resolve().parent != self.directory(name):
                raise ValueError('Directorio de fotos no permitido')
            if len(list(photos_dir.glob('*.jpg'))) >= MAX_PHOTOS:
                raise ValueError('Máximo 80 fotos')
            with image_errors(), Image.open(io.BytesIO(raw)) as image:
                if image.format not in ('PNG', 'JPEG', 'WEBP') or image.width * image.height > MAX_PIXELS:
                    raise ValueError('Formato o dimensiones de foto no admitidos')
                image = ImageOps.exif_transpose(image).convert('RGB')
                image.thumbnail((2048, 2048))
                filename = secrets.token_hex(12) + '.jpg'
                output = photos_dir / filename
                fd, temp = tempfile.mkstemp(prefix='.photo-', dir=photos_dir)
                try:
                    with os.fdopen(fd, 'wb') as stream:
                        image.save(stream, 'JPEG', quality=93)
                    os.replace(temp, output)
                finally:
                    if os.path.exists(temp):
                        os.unlink(temp)
            return self.get(name)

    def photo(self, name, filename):
        if not isinstance(filename, str) or not re.fullmatch(r'[0-9a-f]{24}\.jpg', filename):
            raise ValueError('Nombre de foto inválido')
        self._metadata(name)
        path = self.directory(name) / 'photos' / filename
        if path.parent.is_symlink() or path.is_symlink() or not path.is_file():
            raise ValueError('Foto no encontrada')
        return path.read_bytes()

    def remove_photo(self, name, filename):
        with self.lock:
            if self._metadata(name)['state'] != 'draft' or self._job_active(name):
                raise ValueError('No puedes quitar fotos durante el trabajo o después de publicar')
            self.photo(name, filename)
            (self.directory(name) / 'photos' / filename).unlink()
            data = self._metadata(name)
            data['captions'].pop(filename, None)
            atomic_json(self.directory(name) / 'character.json', data)
            return self.get(name)

    def update(self, name, body):
        with self.lock:
            data = self._metadata(name)
            if self._job_active(name) or data['state'] != 'draft':
                raise ValueError('Espera a que termine el trabajo')
            profile = body.get('profile') if isinstance(body, dict) else None
            captions = body.get('captions') if isinstance(body, dict) else None
            if not isinstance(profile, dict) or any(not isinstance(profile.get(k), str) or
                    len(profile[k]) > 50000 for k in ('system_prompt', 'image_prompt')):
                raise ValueError('Perfil inválido')
            if not isinstance(captions, dict):
                raise ValueError('Captions inválidas')
            photos = set(self.get(name)['photos'])
            if set(captions) - photos or any(not isinstance(v, str) or len(v) > 2000 for v in captions.values()):
                raise ValueError('Caption o foto inválida')
            data['profile'] = {k: profile[k] for k in ('system_prompt', 'image_prompt')}
            data['captions'] = captions
            atomic_json(self.directory(name) / 'character.json', data)
            return self.get(name)

    def _run(self, name, stage, command):
        try:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, env={**os.environ, 'HF_HUB_OFFLINE': '1',
                                                                'TRANSFORMERS_OFFLINE': '1'})
            for line in proc.stdout:
                with self.lock:
                    self.jobs[name]['status']['detail'] = line.strip()[:300]
            code = proc.wait()
            if code:
                raise RuntimeError(f'{stage} terminó con código {code}: {self.jobs[name]["status"]["detail"]}')
            with self.lock:
                data = self._metadata(name)
                if stage == 'captions':
                    captions_path = self.directory(name) / 'captions.json'
                    data['captions'].update(json.loads(captions_path.read_text(encoding='utf-8')))
                    atomic_json(self.directory(name) / 'character.json', data)
                else:
                    self._publish(name)
                self.jobs[name]['status'] = {'stage': stage, 'state': 'done', 'detail': 'Completado'}
        except Exception as exc:
            with self.lock:
                self.jobs[name]['status'] = {'stage': stage, 'state': 'error', 'detail': str(exc)[:500]}
        finally:
            with self.lock:
                self.jobs[name]['finished'] = True

    def _start(self, name, stage, command):
        with self.lock:
            self._metadata(name)
            if any(not job['finished'] for job in self.jobs.values()):
                raise ValueError('Ya hay un trabajo BLIP o LoRA en curso; espera a que termine')
            self.jobs[name] = {'status': {'stage': stage, 'state': 'running', 'detail': 'Iniciando…'}, 'finished': False}
            thread = threading.Thread(target=self._run, args=(name, stage, command), daemon=True)
            thread.start()
            return self.get(name)

    def caption(self, name):
        data = self.get(name)
        if data['state'] != 'draft' or not data['photos']:
            raise ValueError('Añade fotos antes de crear captions')
        return self._start(name, 'captions', [self.python, str(ROOT / 'zimage_character_caption.py'),
                                             str(self.directory(name))])

    def train(self, name, options=None):
        settings = training_options(options)
        data = self.get(name)
        if data['state'] != 'draft' or len(data['photos']) < 5:
            raise ValueError('Se necesitan al menos 5 fotos antes de entrenar')
        kept, skipped = usable_photos(self.directory(name), settings['min_size'],
                                      keep_small=not settings['skip_small'])
        if len(kept) < 5:
            raise ValueError(f'Sólo {len(kept)} fotos llegan a {settings["min_size"]} px y se omitirían '
                             f'{len(skipped)}. Sube fotos más grandes, baja el mínimo o desactiva la omisión')
        missing = [photo.name for photo in kept if not data['captions'].get(photo.name, '').strip()]
        if missing:
            raise ValueError('Revisa y guarda una caption para cada foto usada en el entrenamiento')
        trigger = data['trigger'].casefold()
        without_trigger = [photo.name for photo in kept
                           if trigger not in data['captions'][photo.name].casefold()]
        if without_trigger:
            raise ValueError(f'Incluye el trigger {data["trigger"]} en cada caption')
        if data['trigger'].casefold() not in data['profile']['image_prompt'].casefold():
            raise ValueError('Incluye el trigger en el image prompt')
        if not data['model_ready']:
            raise ValueError('Falta un modelo Z-Image completo con transformer. Configura ZIMAGE_TRAIN_MODEL_DIR; el GGUF de inferencia no sirve para entrenar')
        captions_path = self.directory(name) / 'captions.json'
        atomic_json(captions_path, data['captions'])
        (self.directory(name) / REPORT_NAME).unlink(missing_ok=True)
        command = [self.python, str(ROOT / 'zimage_character_train.py'),
                   '--dataset', str(self.directory(name)), '--model', str(self.model_dir),
                   '--output', str(self.directory(name) / 'adapter.safetensors'),
                   '--epochs', str(settings['epochs']),
                   '--rank', str(settings['rank']),
                   '--resolution', str(settings['resolution']),
                   '--learning-rate', str(settings['learning_rate']),
                   '--min-size', str(settings['min_size']),
                   '--centering-y', str(settings['centering_y'])]
        if not settings['skip_small']:
            command.append('--keep-small')
        return self._start(name, 'training', command)

    def _publish(self, name):
        from safetensors import safe_open
        directory = self.directory(name)
        source = directory / 'adapter.safetensors'
        with safe_open(source, framework='pt', device='cpu') as model:
            keys = model.keys()
            down = {k[:-len('.lora_A.weight')] for k in keys if k.endswith('.lora_A.weight')}
            up = {k[:-len('.lora_B.weight')] for k in keys if k.endswith('.lora_B.weight')}
            if not down or down != up:
                raise ValueError('El entrenamiento no produjo un LoRA PEFT compatible')
        self.lora_dir.mkdir(parents=True, exist_ok=True)
        target = self.lora_dir / (name + '.safetensors')
        if any((self.lora_dir / (name + suffix)).exists() or
               (self.lora_dir / (name + suffix)).is_symlink()
               for suffix in ('.safetensors', '.json', '.png', '.wav')):
            raise ValueError('Ya existen archivos del personaje en el catálogo')
        profile_target = self.lora_dir / (name + '.json')
        thumb_target = self.lora_dir / (name + '.png')
        created = []
        voice_source = directory / 'voice.wav'
        voice_target = self.lora_dir / (name + '.wav')
        try:
            atomic_json(profile_target, self._metadata(name)['profile'])
            created.append(profile_target)
            first = next((directory / 'photos').glob('*.jpg'))
            fd, temp = tempfile.mkstemp(prefix='.new-thumb-', dir=self.lora_dir)
            try:
                with Image.open(first) as image, os.fdopen(fd, 'wb') as out:
                    image.thumbnail((512, 512))
                    image.save(out, 'PNG')
                os.replace(temp, thumb_target)
                created.append(thumb_target)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
            fd, temp = tempfile.mkstemp(prefix='.new-lora-', dir=self.lora_dir)
            try:
                with os.fdopen(fd, 'wb') as out, source.open('rb') as src:
                    while chunk := src.read(1024 * 1024):
                        out.write(chunk)
                os.link(temp, target)
                created.append(target)
            finally:
                os.unlink(temp)
            if voice_source.is_file() and not voice_source.is_symlink():
                fd, temp = tempfile.mkstemp(prefix='.new-voice-', dir=self.lora_dir)
                try:
                    with os.fdopen(fd, 'wb') as out, voice_source.open('rb') as src:
                        while chunk := src.read(1024 * 1024):
                            out.write(chunk)
                    os.link(temp, voice_target)
                    created.append(voice_target)
                finally:
                    if os.path.exists(temp):
                        os.unlink(temp)
        except Exception:
            for path in reversed(created):
                path.unlink(missing_ok=True)
            raise
        data = self._metadata(name)
        data['state'] = 'published'
        atomic_json(directory / 'character.json', data)
