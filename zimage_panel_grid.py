#!/usr/bin/env python3
"""Profiles and chat using zimage_mcp_server.py tools, without extra packages.

python3 zimage_panel_grid.py --web-port 8766 --backend-port 9877
Optional --mcp-url http://127.0.0.1:8765/mcp connects to an existing HTTP
instance. By default the same MCP dispatcher is embedded in this panel.
"""
import argparse
import base64
import io
import zipfile
import copy
from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import shutil
import secrets
import ssl
import subprocess
import stat
import tempfile
import threading
import uuid
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.parse import parse_qs, quote, urlparse
from urllib.request import Request, urlopen
from PIL import Image, ImageOps
from zimage_mcp_server import ZImageMCP, backend_request
import comfyui_video
from zimage_conversations import (Store, Conflict, checked_members, valid_id, image_errors,
                                  checked_personality, checked_notes, checked_image_fields, DATA_RE, MEDIA_RE, MAX_IMAGE, MAX_SNAPSHOT)
from zimage_panel_auth import PanelAuth
from zimage_character_creator import CharacterCreator

ROOT = Path(__file__).resolve().parent
_VOICE_UNSET = object()


@lru_cache(maxsize=1)
def conversation_store():
    return Store()


def validate_conversation_snapshot(kind, members, snapshot, profiles):
    if snapshot is None:
        return
    if not isinstance(snapshot, dict):
        raise ValueError('Instantánea inválida')
    view=snapshot.get('view',[])
    if not isinstance(view,list) or len(view)>10000:raise ValueError('Vista de conversación inválida')
    for entry in view:
        if not isinstance(entry,dict):raise ValueError('Elemento de conversación inválido')
        checked_image_fields(entry.get('image_prompt_fields'))
        if isinstance(entry.get('generation'),dict):checked_image_fields(entry['generation'].get('image_prompt_fields'))
    if kind == 'individual':
        if snapshot.get('name', members[0]['name']) != members[0]['name']:
            raise ValueError('El personaje del chat no coincide con la conversación')
        checked_personality(snapshot.get('personality'))
        checked_notes(snapshot.get('conversation_notes'))
        checked_image_fields(snapshot.get('image_prompt_fields'))
    else:
        raw = snapshot.get('members')
        if not isinstance(raw, list):
            raise ValueError('Faltan los integrantes del grupo')
        normalized = checked_members('group', [
            {'name': m.get('name'), 'weight': m.get('weight', .8),
             'image_enabled': m.get('image_enabled', True),
             'profile': {k:m[k] for k in ('system_prompt','image_prompt','personality_presets','voice_wav') if k in m}}
            if isinstance(m, dict) else {} for m in raw], profiles)
        if [m['name'] for m in normalized] != [m['name'] for m in members]:
            raise ValueError('No se pueden cambiar los integrantes de este grupo')
        for member in raw:
            if isinstance(member, dict):
                checked_personality(member.get('personality'))
        checked_notes(snapshot.get('conversation_notes'))
        checked_image_fields(snapshot.get('image_prompt_fields'))
DEFAULT_SYSTEM = (
    'Eres el asistente creativo de Z-Image Studio, un estudio de imágenes con IA que usa Z-Image y LoRAs. '
    'Ayudas a desarrollar ideas visuales, preparar prompts y generar imágenes con las herramientas disponibles. '
    'Habla con criterio profesional, cercanía y claridad. Responde en el idioma del usuario; por defecto, en español. '
    'Da respuestas breves en párrafos naturales, sin listas salvo que el usuario las pida. '
    'Amplía el detalle cuando lo solicite. Si falta un dato esencial, haz una pregunta concreta; '
    'para decisiones menores, propone una opción coherente con su intención. '
    'Al preparar una imagen, describe de forma concreta el sujeto, la pose, el encuadre, la luz, el fondo y el estilo. '
    'Redacta el prompt visual en inglés y conserva exactamente los triggers del perfil seleccionado, sin traducirlos ni inventarlos. '
    'Si muestras un prompt con trigger vrtl, colócalo en un párrafo independiente tras una línea en blanco, '
    'sin comillas, etiquetas ni bloques de código. Ese párrafo es una instrucción para generar imagen y se excluye del habla TTS. '
    'No añadas prompts visuales a respuestas que no requieren una imagen. '
    'Usa generate_image cuando se solicite generar y esté disponible; confirma el resultado solo tras recibirlo. '
    'Describe brevemente la imagen resultante si puedes verla; si no, explica la intención visual sin afirmar que has visto detalles. '
    'No inventes datos, citas, recuerdos ni biografías de personas reales. '
    'Distingue los hechos conocidos de las propuestas creativas y presenta los personajes inspirados en personas como interpretaciones ficticias.'
)
TLS_HANDSHAKE_TIMEOUT = float(os.environ.get('ZIMAGE_TLS_HANDSHAKE_TIMEOUT', '10'))
# Maximum tokens per chat completion; the Qwen context (100000) is far larger.
CHAT_MAX_TOKENS = max(256, int(os.environ.get('ZIMAGE_CHAT_MAX_TOKENS', '32768')))


def checked_profile(value):
    if not isinstance(value, dict):
        raise ValueError('El perfil debe ser un objeto JSON')
    for key in ('system_prompt', 'image_prompt'):
        if not isinstance(value.get(key), str) or len(value[key]) > 50000:
            raise ValueError(f'El JSON necesita el campo de texto "{key}" (máximo 50000 caracteres)')
    voice = value.get('voice_wav', '')
    if not isinstance(voice, str) or (voice and (len(voice) > 180 or not voice.lower().endswith('.wav')
            or voice in ('.wav', '..wav') or '/' in voice or '\\' in voice or voice.startswith('.')
            or any(ord(char) < 32 for char in voice))):
        raise ValueError('voice_wav debe ser un nombre simple de archivo .wav')
    presets = value.get('personality_presets', [])
    if not isinstance(presets, list) or len(presets) > 20:
        raise ValueError('personality_presets debe ser una lista de hasta 20 elementos')
    ids, names = set(), set()
    for preset in presets:
        if (not isinstance(preset, dict) or not isinstance(preset.get('id'), str)
                or not re.fullmatch(r'[a-z][a-z0-9_-]{0,39}', preset['id'])
                or preset['id'] in {'base','custom','builtin-close','builtin-adventure','builtin-formal',
                                    'style-cynical','style-tender','style-mysterious'}
                or not isinstance(preset.get('name'), str) or not preset['name'].strip()
                or len(preset['name'].strip()) > 80
                or not isinstance(preset.get('system_prompt'), str) or not preset['system_prompt'].strip()
                or len(preset['system_prompt']) > 50000
                or not isinstance(preset.get('revision'), int) or isinstance(preset.get('revision'), bool)
                or not 1 <= preset['revision'] <= 2147483647):
            raise ValueError('Personalidad guardada en formato inválido')
        normalized_name = preset['name'].strip().casefold()
        if preset['id'] in ids or normalized_name in names:
            raise ValueError('IDs y nombres de personalidad deben ser únicos')
        ids.add(preset['id']); names.add(normalized_name)
    return value


class Profiles:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.lock = threading.Lock()

    def path(self, name, suffix):
        if not isinstance(name, str) or not name or '/' in name or '\\' in name:
            raise ValueError('Nombre de LoRA inválido')
        model = self.directory / (name + '.safetensors')
        if not model.is_file() or model.resolve().parent != self.directory:
            raise ValueError('LoRA no encontrado en el directorio configurado')
        path = self.directory / (name + suffix)
        if path.resolve().parent != self.directory or path.is_symlink():
            raise ValueError('Ruta de perfil no permitida')
        return path

    def voice_path(self, name, override=_VOICE_UNSET):
        self.path(name,'.json')
        if override is not _VOICE_UNSET and not isinstance(override,str):raise ValueError('voice_wav debe ser texto')
        voice = (self.load(name).get('voice_wav') if override is _VOICE_UNSET else override) or name + '.wav'
        checked_profile({'system_prompt': '', 'image_prompt': '', 'voice_wav': voice})
        path = self.directory / voice
        if path.is_symlink() or path.resolve().parent != self.directory:
            raise ValueError('Referencia de voz no permitida')
        return path

    def default(self, name):
        if not name:
            return {'system_prompt': DEFAULT_SYSTEM, 'image_prompt': ''}
        trigger = re.search(r'\(([^()]+)\)', name)
        label = re.sub(r'\([^)]*\).*$', '', name).replace('_', ' ').strip()
        trigger_text = trigger.group(1) if trigger else ''
        if not trigger_text:
            match = re.search(r'(vrtl[a-zA-Z]+)', name)
            trigger_text = match.group(1) if match else ''
            if match:
                label = name[:match.start()].replace('_', ' ').strip()
        return {'system_prompt': (
            f'Interpreta a {label or name}, un personaje ficticio inspirado en ese nombre. '
            'Habla en primera persona con una voz cercana, natural y expresiva. Usa español por defecto '
            'y adapta el idioma al del interlocutor. Responde brevemente en párrafos, sin listas salvo que se pidan. '
            'Mantén tu personalidad, tus motivaciones y la continuidad de lo que ya se ha establecido en la conversación. '
            'Reacciona a lo que dice el interlocutor sin repetir fórmulas ni forzar preguntas en cada turno. '
            'Permanece dentro del personaje y no rompas la cuarta pared salvo que el usuario lo pida; '
            'si pregunta por tu naturaleza, aclara con honestidad que eres una interpretación ficticia. '
            'No atribuyas a la persona real recuerdos, relaciones, opiniones privadas ni datos biográficos inventados. '
            'Cuando se solicite una imagen, conserva la identidad visual del perfil y redacta el prompt en inglés: '
            'trigger exacto, sujeto y pose, plano o encuadre, iluminación, fondo y estilo fotográfico. '
            'Escribe primero el diálogo natural y, tras una línea en blanco, un párrafo visual que empiece por el trigger vrtl exacto '
            'del perfil, sin etiquetas, comillas ni bloques de código. El párrafo visual se usa para generar la imagen; '
            'no forma parte del diálogo y NO debe aparecer en el habla TTS. No pronuncies triggers ni instrucciones técnicas. '
            'Añade ese párrafo solo cuando la conversación requiera una imagen y no afirmes que se ha generado sin un resultado confirmado.'),
            'image_prompt': (f'{trigger_text} {label or name}, portrait photo of a person looking at camera, '
                             'head-and-shoulders framing, natural window light, neutral background, '
                             'shallow depth of field, photorealistic, natural skin texture, 85mm lens').strip()}

    def load(self, name):
        if not name:
            return self.default('')
        path = self.path(name, '.json')
        return checked_profile(json.loads(path.read_text(encoding='utf-8'))) if path.exists() else self.default(name)

    def save(self, name, profile):
        profile = checked_profile(profile)
        path = self.path(name, '.json')
        with self.lock:
            if path.is_file(): self._record_version(name, checked_profile(json.loads(path.read_text(encoding='utf-8'))))
            fd, temp = tempfile.mkstemp(prefix='.profile-', dir=self.directory)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as out:
                    json.dump(profile, out, ensure_ascii=False, indent=2)
                    out.write('\n')
                os.replace(temp, path)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
        return profile

    def _history_file(self,name):
        if not isinstance(name,str) or not name or '/' in name or '\\' in name:raise ValueError('Nombre de LoRA inválido')
        self.path(name,'.json')
        directory=Path(os.environ.get('ZIMAGE_DATA_DIR',str(ROOT/'data')))/'profile_versions'
        directory.mkdir(parents=True,exist_ok=True,mode=0o700);os.chmod(directory,0o700)
        return directory/(hashlib.sha256(name.encode()).hexdigest()+'.json')

    def _record_version(self,name,profile):
        target=self._history_file(name)
        try:data=json.loads(target.read_text(encoding='utf-8')) if target.exists() else {'versions':[]}
        except (OSError,ValueError):data={'versions':[]}
        canonical=json.dumps(profile,ensure_ascii=False,sort_keys=True,separators=(',',':'))
        version_id=hashlib.sha256(canonical.encode()).hexdigest()
        if any(item.get('version_id')==version_id for item in data['versions']):return
        data['versions'].insert(0,{'version_id':version_id,'created_at':datetime.now(timezone.utc).isoformat(timespec='seconds'),'profile':profile})
        data['versions']=data['versions'][:20]
        fd,temp=tempfile.mkstemp(prefix='.profile-history-',dir=target.parent)
        try:
            with os.fdopen(fd,'w',encoding='utf-8') as out:json.dump(data,out,ensure_ascii=False,separators=(',',':'));out.flush();os.fsync(out.fileno())
            os.chmod(temp,0o600);os.replace(temp,target)
        finally:
            if os.path.exists(temp):os.unlink(temp)

    def history(self,name):
        path=self._history_file(name)
        if not path.is_file():return []
        data=json.loads(path.read_text(encoding='utf-8'))
        return [{'version_id':item['version_id'],'created_at':item['created_at'],'profile':checked_profile(item['profile'])} for item in data.get('versions',[])[:20]]

    def restore(self,name,version_id):
        if not isinstance(version_id,str) or not re.fullmatch(r'[0-9a-f]{64}',version_id):raise ValueError('ID de versión inválido')
        history_path=self._history_file(name)
        if not history_path.is_file():raise ValueError('No hay versiones anteriores')
        data=json.loads(history_path.read_text(encoding='utf-8'))
        record=next((item for item in data.get('versions',[]) if item.get('version_id')==version_id),None)
        if not record:raise ValueError('Versión de perfil no encontrada')
        return self.save(name,checked_profile(record['profile']))

    def initialize(self):
        count = 0
        for model in sorted(self.directory.glob('*.safetensors')):
            if not self.path(model.stem, '.json').exists():
                with self.path(model.stem, '.json').open('x', encoding='utf-8') as out:
                    json.dump(self.default(model.stem), out, ensure_ascii=False, indent=2)
                    out.write('\n')
                count += 1
        return count

    def catalog(self):
        return [{'name': p.stem, 'filename': p.name,
                 'display_name': re.sub(r'\([^)]*\).*$', '', p.stem).replace('_', ' ').strip(),
                 'size_mb': round(p.stat().st_size / 1024**2, 1),
                 'has_profile': self.path(p.stem, '.json').is_file(),
                 'thumbnail': '/api/thumbnail?name=' + quote(p.stem, safe='')
                 if self.path(p.stem, '.png').is_file() else None,
                 'vertical_thumbnail': '/api/thumbnail?name=' + quote(p.stem, safe='') + '&variant=vertical'
                 if self.path(p.stem, '_1_2.png').is_file() else None}
                for p in sorted(self.directory.glob('*.safetensors'))]


def http_json(url, payload=None, headers=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = Request(url, data=data, headers={'Content-Type': 'application/json', **(headers or {})})
    try:
        with urlopen(req, timeout=1800) as response:
            raw = response.read()
            return (json.loads(raw) if raw else None), response.headers
    except HTTPError as exc:
        detail = exc.read(4096).decode(errors='replace')
        raise RuntimeError(f'HTTP {exc.code}: {detail}') from exc


class MCPBridge:
    def __init__(self, host, port, url=None):
        self.local = ZImageMCP(host, port)
        self.url = url
        self.headers = {'Accept': 'application/json, text/event-stream'}

    def rpc(self, method, params=None):
        request = {'jsonrpc': '2.0', 'id': secrets.token_hex(8), 'method': method, 'params': params or {}}
        if self.url:
            response, headers = http_json(self.url, request, self.headers)
            if headers.get('Mcp-Session-Id'):
                self.headers['Mcp-Session-Id'] = headers['Mcp-Session-Id']
        else:
            response = self.local.rpc(request)
        if not response or 'error' in response:
            raise RuntimeError(str((response or {}).get('error', 'MCP no respondió')))
        return response['result']

    def initialize(self):
        result = self.rpc('initialize', {'protocolVersion': '2024-11-05', 'capabilities': {},
                                        'clientInfo': {'name': 'zimage-panel', 'version': '2.0'}})
        self.headers['MCP-Protocol-Version'] = result['protocolVersion']
        if self.url:
            http_json(self.url, {'jsonrpc': '2.0', 'method': 'notifications/initialized'}, self.headers)


def prompt_loras(loras, prompt):
    """Keep style adapters, but zero identities absent from the actual prompt."""
    if not isinstance(loras, list) or not isinstance(prompt, str):
        return loras
    triggers = {token.casefold() for token in re.findall(r'\bvrtl[A-Za-z0-9_]+\b', prompt, re.I)}
    result = []
    for item in loras:
        trigger = re.search(r'\bvrtl[A-Za-z0-9_]+\b', item['name'], re.I)
        weight = item.get('weight', .8)
        if trigger and trigger.group().casefold() not in triggers:
            weight = 0
        result.append({**item, 'weight': weight})
    return result


UPSCALE_MAX_SIDE = 4096
UPSCALE_MAX_PIXELS = 4096 * 4096


def allowed_image_file(value):
    """Only stored generations may be served or re-read for reescalado."""
    path = Path(value).resolve()
    roots = [ROOT / 'zimage_outputs', ROOT / 'zimage_outputs_gguf',
             Path(os.environ.get('ZIMAGE_OUTPUT_DIR', str(ROOT / 'data/outputs')))]
    if not any(root.resolve() in path.parents for root in roots) or path.suffix.lower() != '.png':
        raise ValueError('Ruta de imagen no permitida')
    if path.is_symlink() or not path.is_file():
        raise ValueError('Imagen no encontrada')
    return path


def upscale_source(src):
    """Read the bytes of a chat image: data URI, stored file or conversation media."""
    if not isinstance(src, str) or not src:
        raise ValueError('Envía una imagen válida para reescalar')
    if src.startswith('/api/image'):
        query = parse_qs(urlparse(src).query)
        return allowed_image_file(query.get('path', [''])[0]).read_bytes()
    media = MEDIA_RE.fullmatch(src)
    if media:
        found = conversation_store().media(media.group(1))
        if found is None:
            raise ValueError('Imagen de conversación no encontrada')
        return found[0].read_bytes()
    data = DATA_RE.fullmatch(src)
    if not data:
        raise ValueError('Imagen no admitida')
    raw = base64.b64decode(data.group(2), validate=True)
    if not 0 < len(raw) <= MAX_IMAGE:
        raise ValueError('Imagen demasiado grande')
    return raw


def upscale_image(src, scale):
    """Enlarge a generated image with Pillow LANCZOS; no model and no GPU."""
    raw = upscale_source(src)
    try:
        factor = float(scale)
    except (TypeError, ValueError):
        raise ValueError('Factor de reescalado inválido')
    if not math.isfinite(factor) or not 1.1 <= factor <= 4.0:
        raise ValueError('Factor de reescalado fuera de rango (1.1–4.0)')
    with image_errors(), Image.open(io.BytesIO(raw)) as opened:
        if opened.format not in ('PNG', 'JPEG', 'WEBP') or max(opened.size) > UPSCALE_MAX_SIDE:
            raise ValueError('Formato o dimensiones de imagen no admitidos')
        image = ImageOps.exif_transpose(opened).convert('RGB')
        width = min(UPSCALE_MAX_SIDE, int(round(image.width * factor)))
        height = min(UPSCALE_MAX_SIDE, int(round(image.height * factor)))
        if width <= image.width and height <= image.height:
            raise ValueError('La imagen ya es lo bastante grande')
        if width * height > UPSCALE_MAX_PIXELS:
            raise ValueError('La imagen reescalada supera 16.7 Mpx; usa un factor menor')
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, 'PNG')
    data = buffer.getvalue()
    if len(data) > MAX_IMAGE:
        raise ValueError('La imagen reescalada supera 16 MB; usa un factor menor')
    # The Base64 never reaches the log: only its size is reported.
    return {'status': 'ok', 'width': width, 'height': height, 'bytes': len(data),
            'src': 'data:image/png;base64,' + base64.b64encode(data).decode()}


def image_settings(body, profiles):
    name = body.get('lora_name', '')
    if name:
        profiles.path(name, '.safetensors')
    values = {'width': 512, 'height': 512, 'steps': 9, 'cfg': 0.0,
              'seed': secrets.randbelow(2**32), 'negative_prompt': ''}
    values.update({k: v for k, v in body.get('image_settings', {}).items() if k in values})
    for key, low, high in [('width', 64, 4096), ('height', 64, 4096), ('steps', 1, 100), ('seed', 0, 2**32-1)]:
        if not isinstance(values[key], int) or not low <= values[key] <= high:
            raise ValueError(f'{key} fuera de rango ({low}–{high})')
    if values['width'] % 16 or values['height'] % 16:
        raise ValueError('Ancho y alto deben ser múltiplos de 16')
    cfg, weight = float(values['cfg']), float(body.get('lora_weight', .8))
    if not math.isfinite(cfg) or not 0 <= cfg <= 20 or not math.isfinite(weight) or not 0 <= weight <= 3:
        raise ValueError('CFG o peso de LoRA fuera de rango')
    values['cfg'] = cfg
    values['lora'] = f'{name}:{weight}' if name else ''
    if 'loras' in body:
        loras = body['loras']
        if not isinstance(loras, list) or len(loras) > 6:
            raise ValueError('Selecciona como máximo seis LoRAs')
        selected, seen = [], set()
        for item in loras:
            adapter = item['name']
            if adapter in seen or any(c in adapter for c in ':;'):
                raise ValueError('LoRA duplicado o inválido')
            profiles.path(adapter, '.safetensors')
            scale = float(item.get('weight', .8))
            if not math.isfinite(scale) or not 0 <= scale <= 3:
                raise ValueError('Peso de LoRA fuera de rango')
            seen.add(adapter)
            effective=prompt_loras([item],body.get('image_prompt'))[0]
            selected.append(f'{adapter}:{float(effective.get("weight", .8))}')
        values['lora'] = ';'.join(selected)
    if body.get('request_id'):
        if not isinstance(body['request_id'], str) or len(body['request_id']) > 100:
            raise ValueError('Identificador de progreso inválido')
        values['request_id'] = body['request_id']
    return values


def result_text(result):
    # Base64 belongs in the browser, not the text model's conversation history.
    structured = result.get('structuredContent')
    if isinstance(structured, dict):
        value = {k: v for k, v in structured.items() if k != 'image'}
        if result.get('isError'):
            value['isError'] = True
        return json.dumps(value, ensure_ascii=False)
    return '\n'.join(c['text'] for c in result.get('content', []) if c.get('type') == 'text')


def embedded_group_image_prompt(text):
    """Return a prompt emitted as an unparsed ``<tool...>`` group call, if any."""
    if not isinstance(text, str):
        return None
    paragraphs = [part.strip() for part in re.split(r'\n\s*\n', re.sub(r'^```[^\n]*$', '', text, flags=re.M)) if part.strip()]
    for index, paragraph in enumerate(paragraphs[:-1]):
        if not re.match(r'^<tool\b', paragraph, flags=re.I):
            continue
        match = re.match(r'vrtl[A-Za-z0-9_]+(?:\s|[,.:])', paragraphs[index + 1])
        if match:
            return paragraphs[index + 1][match.start():].strip()
    return None


class App:
    def __init__(self, args):
        self.args = copy.copy(args)
        self.config_lock = threading.RLock()
        self.progress_routes = {}
        self.audio_url = "http://127.0.0.1:8767"
        self.settings_path = Path(args.settings_file) if getattr(args, "settings_file", None) else None
        self.defaults = {"width":512,"height":512,"steps":9,"cfg":0,"seed":42,"random":True,"weight":.8,"negative":""}
        self.profiles = Profiles(args.lora_dir)
        self.character_creator = CharacterCreator(
            os.environ.get('ZIMAGE_DATA_DIR', str(ROOT / 'data')), args.lora_dir)
        if self.settings_path and self.settings_path.exists():
            self.configure(json.loads(self.settings_path.read_text()), persist=False)
        # El gestor de stack puede solicitar explícitamente el dispatcher
        # integrado aunque el JSON persistido conserve una URL MCP externa.
        if getattr(args, 'integrated_mcp', False):
            self.args.mcp_url = None

    def configuration(self):
        """Return only user-editable settings; never expose process arguments wholesale."""
        with self.config_lock:
            host = self.args.backend_host
            if ':' in host:
                host = '[' + host + ']'
            return {'defaults': copy.deepcopy(self.defaults), 'chat_url': self.args.chat_url,
                    'chat_model': self.args.chat_model, 'zimage_url': f'tcp://{host}:{self.args.backend_port}',
                    'mcp_url': self.args.mcp_url or '', 'audio_url': self.audio_url}

    def snapshot(self):
        """Pin routes for the entire request, including the multi-step chat/tool loop."""
        with self.config_lock:
            result = copy.copy(self)
            result.args = copy.copy(self.args)
            result.defaults = copy.deepcopy(self.defaults)
            return result

    def configure(self, body, persist=True):
        """Validate all settings, persist atomically, then switch future requests."""
        if not isinstance(body, dict):
            raise ValueError('Ajustes inválidos')
        def endpoint(key, schemes, optional=False):
            value = body.get(key, '')
            if not isinstance(value, str):
                raise ValueError(f'{key} debe ser una URL')
            value = value.strip().rstrip('/')
            if not value and optional:
                return ''
            parsed = urlparse(value)
            if (parsed.scheme not in schemes or not parsed.hostname or parsed.username or parsed.password
                    or parsed.query or parsed.fragment):
                raise ValueError(f'URL inválida para {key}')
            if parsed.port is not None and not 1 <= parsed.port <= 65535:
                raise ValueError(f'Puerto inválido para {key}')
            return value
        body = {**body, 'audio_url': body.get('audio_url', self.audio_url)}
        audio_url = endpoint('audio_url', ('http', 'https'), True)
        chat = endpoint('chat_url', ('http', 'https'))
        mcp_url = endpoint('mcp_url', ('http', 'https'), True)
        zimage = endpoint('zimage_url', ('tcp',))
        parsed = urlparse(zimage)
        if not parsed.port or parsed.path:
            raise ValueError('Z-Image necesita tcp://host:puerto, sin ruta HTTP')
        model = body.get('chat_model', '')
        if not isinstance(model, str) or len(model) > 300:
            raise ValueError('Identificador de modelo inválido')
        defaults = body.get('defaults')
        if not isinstance(defaults, dict) or set(defaults) != set(self.defaults):
            raise ValueError('Faltan valores de generación por defecto')
        if not isinstance(defaults['random'], bool) or not isinstance(defaults['negative'], str):
            raise ValueError('Seed aleatoria o negative prompt inválidos')
        image_settings({'image_settings': {k:defaults[k] for k in ('width','height','steps','cfg','seed')},
                        'lora_weight':defaults['weight']}, self.profiles)
        updated = {'defaults':defaults, 'chat_url':chat, 'chat_model':model.strip(),
                   'zimage_url':zimage, 'mcp_url':mcp_url, 'audio_url':audio_url}
        with self.config_lock:
            if persist and self.settings_path:
                fd, temp = tempfile.mkstemp(prefix='.panel-settings-', dir=self.settings_path.parent)
                try:
                    with os.fdopen(fd, 'w') as out:
                        json.dump(updated, out, ensure_ascii=False, indent=2)
                        out.write('\n')
                    os.replace(temp, self.settings_path)
                finally:
                    if os.path.exists(temp):
                        os.unlink(temp)
            self.args.chat_url, self.args.chat_model = chat, model.strip()
            self.args.backend_host, self.args.backend_port = parsed.hostname, parsed.port
            self.args.mcp_url = mcp_url or None
            self.defaults = copy.deepcopy(defaults)
            self.audio_url = audio_url
        return self.configuration()

    def generation_body(self, body):
        """Fill omitted generation fields from the pinned global defaults."""
        defaults = self.defaults
        settings = {k: defaults[k] for k in ('width','height','steps','cfg','seed')}
        if defaults['random']:
            settings['seed'] = secrets.randbelow(2**32)
        settings['negative_prompt'] = defaults['negative']
        settings.update(body.get('image_settings', {}))
        return {**body, 'lora_weight':body.get('lora_weight', defaults['weight']), 'image_settings':settings}

    def bridge(self):
        return MCPBridge(self.args.backend_host, self.args.backend_port, self.args.mcp_url)

    def completion(self, payload):
        if self.args.chat_model:
            payload['model'] = self.args.chat_model
        return http_json(self.args.chat_url.rstrip('/') + '/chat/completions', payload)[0]

    def group_turn(self, body):
        """One character per completion, with shared transcript and its own profile."""
        members = body.get('participants', [])
        if not isinstance(members, list) or not 1 <= len(members) <= 6:
            raise ValueError('Selecciona entre uno y seis personajes')
        names = [m['name'] for m in members]
        if len(set(names)) != len(names) or body.get('speaker') not in names:
            raise ValueError('Interlocutor o grupo inválido')
        profiles = {}
        for member in members:
            name = member['name']
            profile = self.profiles.load(name)
            profile.update({k: member[k] for k in ('system_prompt', 'image_prompt') if k in member})
            profiles[name] = checked_profile(profile)
        transcript = body.get('transcript', [])
        if not isinstance(transcript, list) or len(transcript) > 500:
            raise ValueError('Historial grupal inválido o demasiado largo')
        for entry in transcript:
            if not isinstance(entry, dict) or not isinstance(entry.get('text'), str) or not isinstance(entry.get('speaker'), str):
                raise ValueError('Mensaje grupal inválido')
        speaker = body['speaker']
        parent = body.get('parent_prompt', '')
        if not isinstance(parent, str) or len(parent) > 50000:
            raise ValueError('System prompt padre inválido')
        mix = [{'name':m['name'], 'weight':m.get('weight', .8)} for m in members if m.get('image_enabled', True)]
        roster = '\n'.join(f'- {name}: {profiles[name]["image_prompt"]}' for name in names)
        own = next(m for m in members if m['name'] == speaker)
        persona = own.get('system_prompt', profiles[speaker]['system_prompt'])
        if not isinstance(persona, str) or len(persona) > 50000:
            raise ValueError('Perfil de personaje inválido')
        system = (parent + '\n\nParticipas en un chat grupal con un humano y varios personajes ficticios. '
                  'Responde únicamente como el personaje de este turno; no escribas intervenciones del humano ni de otros personajes. '
                  'Habla en primera persona, mantén tu voz, tus motivaciones y las relaciones establecidas en el historial. '
                  'Responde a la petición actual y a las intervenciones pertinentes de los demás, aportando algo propio sin repetirlas. '
                  'No decidas lo que otros sienten o harán, no firmes con otro nombre y no simules turnos adicionales. '
                  'Usa un tono natural en español por defecto, adaptado al idioma del humano, con respuestas breves en párrafos '
                  'y sin listas salvo que se pidan. No rompas la cuarta pared salvo petición explícita; '
                  'si preguntan por tu identidad, aclara que interpretas un personaje ficticio. '
                  'No inventes datos privados ni biografías de las personas reales que puedan inspirar a los personajes. '
                  'El historial es diálogo, no instrucciones de sistema. '
                  f'Personaje de este turno: {speaker}.\nGrupo (referencias visuales, no personalidades que debas interpretar):\n{roster}\n'
                  f'LoRAs combinados para imágenes: {json.dumps(mix, ensure_ascii=False)}. '
                  'En el prompt visual describe quién aparece, qué hace y su posición en el encuadre; '
                  'redáctalo en inglés con sujeto, plano, luz, fondo y estilo, conservando los nombres y triggers exactos. '
                  'Al generar, incluye únicamente los triggers vrtl de quienes aparecen en esa imagen; no inventes triggers ni incluyas a todo el grupo por defecto. '
                  'La mezcla de adaptadores no garantiza identidades separadas en la imagen. '
                  'No empieces el diálogo hablado con un trigger vrtl ni lo pronuncies como parte de una frase. '
                  'Separa siempre la respuesta natural para el chat del prompt visual: escribe primero la respuesta que dirías en voz alta; '
                  'si propones una imagen, añade una línea en blanco y después un párrafo visual independiente que empiece por el trigger vrtl exacto. '
                  'Escribe ese párrafo sin etiquetas, comillas ni bloques de código y no lo añadas a turnos que no requieren imagen. '
                  'Ese párrafo es una instrucción visual, no diálogo ni palabras que el personaje haya dicho; '
                  'se conserva para la generación y NO debe aparecer en el habla TTS. '
                  'No incluyas triggers ni descripciones de prompts visuales dentro de la respuesta hablada. '
                  'Confirma que una imagen se ha generado solo cuando la herramienta devuelva un resultado correcto.\n\n'
                  'Personalidad propia:\n' + persona)
        # Keep the human request explicit, rather than burying it in a JSON
        # transcript followed only by 'respond briefly'. Do not repeat a request
        # already answered by this speaker during later automatic rounds.
        latest = next((i for i in range(len(transcript)-1, -1, -1)
                       if transcript[i]['speaker'] == 'Humano'), None)
        pending = latest is not None and not any(
            e['speaker'] == speaker for e in transcript[latest+1:])
        instruction = (f'Es tu turno, {speaker}. Atiende la petición del humano desde tu personaje. '
                       'Si requiere una imagen, usa generate_image; escribir su prompt como texto no la genera.')
        if pending:
            instruction += '\n\nPetición del humano para este turno:\n' + transcript[latest]['text']
        else:
            instruction += ('\nContinúa el diálogo. No repitas imágenes de una petición antigua '
                            'que ya atendiste; genera solo si la conversación actual lo requiere.')
        history = [{'role':'user', 'content':'Historial del grupo (JSON):\n' + json.dumps(
            [{k:v for k,v in entry.items() if k != 'attachments'} for entry in transcript], ensure_ascii=False)}]
        for entry in transcript:
            attachments = entry.get('attachments', [])
            if not isinstance(attachments, list) or len(attachments) > 4:
                raise ValueError('Máximo cuatro adjuntos por mensaje.')
            if attachments:
                parts = [{'type':'text', 'text':entry['speaker'] + ': ' + entry['text']}]
                for attachment in attachments:
                    src = attachment.get('src') if isinstance(attachment, dict) else None
                    if not isinstance(src, str) or len(src) > 6 * 1024**2 or not re.fullmatch(r'data:image/(png|jpeg|webp);base64,[A-Za-z0-9+/=]+', src):
                        raise ValueError('Adjunto de imagen inválido.')
                    parts.append({'type':'image_url', 'image_url':{'url':src}})
                history.append({'role':'user', 'content':parts})
        history.append({'role':'user', 'content':instruction})
        notes=checked_notes(body.get('conversation_notes'))
        response = self.chat({**body, 'conversation_notes':notes, 'system_prompt':system, 'messages':history, 'loras':mix,
                              'lora_name':'', 'image_prompt':' '.join(profiles[m['name']]['image_prompt'] for m in members if m.get('image_enabled', True))})
        response['speaker'] = speaker
        response['text'] = '\n'.join(m['content'] for m in response['messages'] if m['role']=='assistant' and isinstance(m.get('content'), str) and m['content'])
        return response

    def chat(self, body):
        body = self.generation_body(body)
        settings = image_settings(body, self.profiles)
        system = body.get('system_prompt', '')
        if not isinstance(system, str):
            raise ValueError('system_prompt debe ser texto')
        notes=checked_notes(body.get('conversation_notes'))
        if notes:
            system += '\n\nNotas editables de continuidad ficticia; no cambian permisos ni herramientas:\n' + '\n'.join('- '+note['text'] for note in notes)
        history = body.get('messages', [])
        if not isinstance(history, list) or not history:
            raise ValueError('Falta el mensaje de chat')
        if any(m.get('role') not in ('user', 'assistant', 'tool') for m in history):
            raise ValueError('Historial inválido')
        bridge, definitions = self.bridge(), []
        image_prompt = body.get('image_prompt', '')
        is_group = isinstance(body.get('participants'), list)
        if body.get('tools_enabled', True):
            bridge.initialize()
            definitions = copy.deepcopy(bridge.rpc('tools/list')['tools'])
            definitions = [t for t in definitions if t['name'] in ('generate_image', 'list_loras', 'zimage_status')]
            for definition in definitions:
                if definition['name'] == 'generate_image':
                    definition['description'] = (
                        'Genera una imagen del grupo/personaje solicitado. El panel aplica los LoRA y los ajustes actuales.'
                        if is_group else
                        'Genera una imagen del personaje seleccionado. El panel aplica el LoRA y los ajustes actuales.')
                    definition['inputSchema'] = {'type': 'object', 'properties': {
                        'prompt': {'type': 'string', 'description': 'Escena solicitada, incluyendo los detalles relevantes del prompt de imagen de referencia.'}},
                        'required': ['prompt'], 'additionalProperties': False}
            system += (
                '\n\nHerramientas disponibles: puedes generar imágenes con generate_image, '
                'listar LoRAs con list_loras y consultar zimage_status. '
                'Si el usuario solicita una imagen, llama a generate_image; no digas que la has generado sin hacerlo. '
                'Si el usuario pide crear, enviar, mostrar o generar una imagen, debes emitir la llamada a generate_image en este turno; '
                'una respuesta textual prometiendo hacerlo no sustituye la llamada. '
                'La imagen se mostrará automáticamente en el chat, no incluyas enlaces Markdown a archivos locales. '
                + (f'Interlocutor del grupo: {body.get("speaker", "")}. LoRAs activos: {settings["lora"] or "ninguno"}. '
                 if is_group else f'Personaje seleccionado: {body.get("lora_name") or "ninguno"}. ')
                + f'Prompt de imagen de referencia: {image_prompt}. '
                'No llames herramientas de imagen si el usuario solo quiere conversar.')
        messages = [{'role': 'system', 'content': system}] + copy.deepcopy(history)
        start = len(messages)
        images, events = [], []
        allowed = {t['name'] for t in definitions}
        warning = None
        embedded_call_used = False
        for turn in range(7):
            payload = {'messages': messages, 'stream': False, 'temperature': .8, 'max_tokens': CHAT_MAX_TOKENS,
                       'chat_template_kwargs': {'enable_thinking': False}}
            if definitions:
                payload['tools'] = [{'type': 'function', 'function': {
                    'name': t['name'], 'description': t['description'], 'parameters': t['inputSchema']}}
                    for t in definitions]
                payload['tool_choice'] = 'auto' if turn < 6 else 'none'
            try:
                msg = self.completion(payload)['choices'][0]['message']
            except Exception as exc:
                if not events:
                    raise
                warning = str(exc)
                messages.append({'role': 'assistant', 'content': 'La herramienta ha terminado, pero no he podido completar la respuesta de texto.'})
                break
            msg = {k: v for k, v in msg.items() if k in ('role', 'content', 'tool_calls', 'reasoning_content')}
            msg['role'] = 'assistant'
            calls = msg.get('tool_calls') or []
            if turn == 6 and calls:
                msg = {'role': 'assistant', 'content': 'Se ha alcanzado el límite de llamadas a herramientas de este turno.'}
                calls = []
            messages.append(msg)
            if not calls:
                # Some llama.cpp templates occasionally expose a tool call as
                # two display paragraphs instead of ``tool_calls``.  In a
                # group this used to leave only a manual "Generar imagen"
                # button after the first successful round.  Accept only the
                # same narrow form shown by that button, and still pin all
                # generation settings and LoRAs in the panel.
                prompt = (embedded_group_image_prompt(msg.get('content'))
                          if is_group and definitions and not embedded_call_used else None)
                if not prompt:
                    break
                embedded_call_used = True
                call = {'id': 'embedded-group-image-' + secrets.token_hex(8), 'type': 'function',
                        'function': {'name': 'generate_image', 'arguments': json.dumps({'prompt': prompt})}}
                msg = {'role': 'assistant', 'content': None, 'tool_calls': [call]}
                messages[-1] = msg
                calls = [call]
            for call in calls:
                name = call['function']['name']
                try:
                    if name not in allowed:
                        raise ValueError('Herramienta no permitida')
                    arguments = call['function'].get('arguments') or '{}'
                    arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                    if name == 'generate_image':
                        prompt = arguments.get('prompt') or image_prompt
                        if not isinstance(prompt, str) or not prompt.strip():
                            raise ValueError('Falta el prompt de imagen')
                        generation_settings = {**settings, 'lora': image_settings(
                            {**body, 'image_prompt': prompt}, self.profiles)['lora']}
                        arguments = {**generation_settings, 'prompt': prompt}
                    else:
                        arguments = {}
                    result = bridge.rpc('tools/call', {'name': name, 'arguments': arguments})
                    text = result_text(result)
                    if not result.get('isError'):
                        for content in result.get('content', []):
                            if content.get('type') == 'image' and content.get('mimeType') in ('image/png', 'image/jpeg', 'image/webp'):
                                images.append({'src': f'data:{content["mimeType"]};base64,{content["data"]}',
                                               'seed': settings['seed'], 'prompt': arguments.get('prompt', ''),
                                               'generation': {'lora_name': body.get('lora_name', ''),
                                                   **({'loras': prompt_loras(body['loras'], arguments.get('prompt', ''))} if 'loras' in body else {}),
                                                   'lora_weight': float(body.get('lora_weight', .8)),
                                                   'image_prompt': arguments.get('prompt', ''),
                                                   'image_settings': {k: v for k, v in settings.items() if k not in ('lora', 'request_id')}}})
                    events.append({'name': name, 'error': bool(result.get('isError')), 'text': text})
                except Exception as exc:
                    text = json.dumps({'error': str(exc)}, ensure_ascii=False)
                    events.append({'name': name, 'error': True, 'text': text})
                messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': text})
        return {'status': 'ok', 'messages': messages[start:], 'images': images, 'events': events, 'warning': warning}


CONVERSATION_LIMIT = 100 * 1024**2


def _archive_video(item, video_path):
    if not isinstance(item,dict) or item.get('kind')!='video':return None
    source=item.get('src','');job_id=item.get('job_id','')
    if isinstance(source,str) and source.startswith('data:video/mp4;base64,'):
        try:content=base64.b64decode(source.split(',',1)[1],validate=True)
        except Exception as exc:raise ValueError('Vídeo Base64 inválido.') from exc
    else:
        match=re.fullmatch(r'/api/video/file\?id=([0-9a-f]{32})',source) if isinstance(source,str) else None
        if not match:raise ValueError('La exportación ZIP sólo admite vídeos locales MiniMax-H3.')
        if job_id and job_id!=match.group(1):raise ValueError('El ID de vídeo no coincide con su URL.')
        job_id=match.group(1);path=video_path(job_id) if video_path else None
        if path is None:raise ValueError(f'No se encuentra el vídeo {job_id}; no se creó una copia completa.')
        if path.is_symlink() or not path.is_file() or path.stat().st_size>64*1024*1024:raise ValueError('Vídeo inválido o superior a 64 MiB.')
        content=path.read_bytes()
    if not content or len(content)>64*1024*1024 or b'ftyp' not in content[:32]:raise ValueError('Vídeo MP4 inválido o superior a 64 MiB.')
    if not re.fullmatch(r'[0-9a-f]{32}',job_id or ''):job_id=uuid.uuid4().hex
    return job_id,content


def _validate_imported_mp4(content):
    executable=shutil.which('ffprobe')
    if not executable:raise ValueError('Se necesita ffprobe para importar vídeos; instálalo y vuelve a intentarlo.')
    with tempfile.TemporaryDirectory(prefix='zimage-import-') as directory:
        os.chmod(directory,0o700);path=Path(directory)/'video.mp4';path.write_bytes(content);os.chmod(path,0o600)
        try:result=subprocess.run([executable,'-v','error','-show_entries','format=format_name','-of','default=noprint_wrappers=1:nokey=1',str(path)],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=10,check=False)
        except subprocess.TimeoutExpired as exc:raise ValueError('ffprobe excedió 10 segundos validando un vídeo.') from exc
        if result.returncode or b'mov' not in result.stdout.lower():raise ValueError('El medio no es un contenedor MP4 válido.')


def validate_portable_conversation(data, profiles):
    """Validate archive structure and local LoRA prerequisites before media import."""
    view=data.get('view');
    if not isinstance(view,list) or len(view)>10000:raise ValueError('Vista de conversación inválida o demasiado larga.')
    if data['format']=='zimage-conversation':
        name=data.get('lora_name');profile=data.get('profile')
        if not isinstance(name,str) or not isinstance(profile,dict) or any(not isinstance(profile.get(key),str) for key in ('system_prompt','image_prompt')):raise ValueError('Perfil individual inválido.')
        if name:
            lora=profiles.path(name,'.safetensors')
            if lora.is_symlink() or not lora.is_file():raise ValueError('El LoRA de la conversación no está instalado: '+name)
        messages=data.get('messages')
        if not isinstance(messages,list) or len(messages)>10000:raise ValueError('Mensajes individuales inválidos.')
        for message in messages:
            if not isinstance(message,dict) or message.get('role') not in ('user','assistant','tool'):raise ValueError('Mensaje individual inválido.')
    else:
        members=data.get('members');transcript=data.get('transcript')
        if not isinstance(members,list) or not 1<=len(members)<=6 or not isinstance(transcript,list) or len(transcript)>10000:raise ValueError('Integrantes o transcript grupal inválido.')
        if not isinstance(data.get('parent_prompt'),str) or not isinstance(data.get('image_prompt'),str):raise ValueError('Prompts del grupo inválidos.')
        for member in members:
            if not isinstance(member,dict) or not isinstance(member.get('name'),str) or not member['name'] or not isinstance(member.get('system_prompt'),str) or not isinstance(member.get('image_prompt'),str) or not isinstance(member.get('weight'),(int,float)) or isinstance(member.get('weight'),bool) or not 0<=member['weight']<=3:raise ValueError('Integrante grupal inválido.')
            lora=profiles.path(member['name'],'.safetensors')
            if lora.is_symlink() or not lora.is_file():raise ValueError('El LoRA del grupo no está instalado: '+member['name'])
        if len({member['name'] for member in members})!=len(members):raise ValueError('El grupo contiene LoRAs duplicados.')
    for item in view:
        if not isinstance(item,dict):raise ValueError('Elemento de conversación inválido.')
        if item.get('kind')=='video':
            if not (isinstance(item.get('src'),str) and (re.fullmatch(r'/api/video/file\?id=[0-9a-f]{32}',item['src']) or item['src'].startswith('data:video/mp4;base64,'))):raise ValueError('Referencia de vídeo inválida.')


def conversation_archive(raw, unpack=False, video_path=None, video_importer=None, conversation_validator=None):
    """Pack/unpack v1 conversation JSON with detached MP4 entries; never extract paths."""
    if len(raw)>CONVERSATION_LIMIT:raise ValueError('El archivo supera 100 MiB.')
    imported={};archive_v2=False
    if unpack:
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                entries=archive.infolist();names=[entry.filename for entry in entries]
                if len(entries)>512:raise ValueError('El ZIP supera las 512 entradas permitidas.')
                if len(names)!=len(set(names)):raise ValueError('El ZIP contiene nombres duplicados.')
                if any(name not in ('conversation.json','manifest.json') and not re.fullmatch(r'media/[0-9a-f]{64}\.mp4',name) for name in names):raise ValueError('El ZIP contiene rutas o entradas no permitidas.')
                for entry in entries:
                    if stat.S_ISLNK((entry.external_attr>>16)&0xffff) or entry.flag_bits&1:raise ValueError('No se admiten enlaces simbólicos ni ZIP cifrados.')
                if 'conversation.json' not in names:raise ValueError('Falta conversation.json en el ZIP.')
                entry=archive.getinfo('conversation.json')
                if entry.file_size>CONVERSATION_LIMIT:raise ValueError('La conversación descomprimida supera 100 MiB.')
                with archive.open(entry) as source:raw=source.read(CONVERSATION_LIMIT+1)
                if len(raw)>CONVERSATION_LIMIT:raise ValueError('La conversación descomprimida supera 100 MiB.')
                if 'manifest.json' in names:
                    manifest_entry=archive.getinfo('manifest.json')
                    if manifest_entry.file_size>1024*1024:raise ValueError('El manifiesto ZIP supera 1 MiB.')
                    manifest=json.loads(archive.read('manifest.json'))
                    if not isinstance(manifest,dict):raise ValueError('Manifiesto ZIP inválido.')
                    if manifest.get('format')=='zimage-media-bundle' and manifest.get('version')==1:
                        if manifest!={'format':'zimage-media-bundle','version':1,'file':'conversation.json','size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()} or len(names)!=2:raise ValueError('El manifiesto/hash ZIP antiguo no coincide.')
                    elif manifest.get('archive_version')==2 and set(manifest)=={'archive_version','videos'} and isinstance(manifest['videos'],list):
                        archive_v2=True
                        declared={};refs={}
                        for video in manifest['videos']:
                            if (not isinstance(video,dict) or set(video)!={'job_id','entry','sha256','size'} or not re.fullmatch(r'[0-9a-f]{32}',video.get('job_id','')) or not re.fullmatch(r'media/[0-9a-f]{64}\.mp4',video.get('entry','')) or not re.fullmatch(r'[0-9a-f]{64}',video.get('sha256','')) or not isinstance(video.get('size'),int) or isinstance(video.get('size'),bool) or not 1<=video['size']<=64*1024*1024):raise ValueError('Elemento inválido en el manifiesto de vídeo.')
                            if video['job_id'] in refs:raise ValueError('ID de vídeo duplicado en el manifiesto.')
                            if video['entry'] in declared and (declared[video['entry']]['sha256'],declared[video['entry']]['size'])!=(video['sha256'],video['size']):raise ValueError('Una entrada de hash apunta a metadatos incompatibles.')
                            refs[video['job_id']]=video;declared[video['entry']]=video
                        if set(names)!={'conversation.json','manifest.json',*declared}:raise ValueError('Las entradas del ZIP no coinciden con el manifiesto.')
                        total=len(raw);media_by_entry={}
                        for filename,video in declared.items():
                            info=archive.getinfo(filename);total+=info.file_size
                            if info.file_size!=video['size'] or total>CONVERSATION_LIMIT:raise ValueError('El ZIP supera el tamaño declarado o el límite de 100 MiB.')
                            with archive.open(info) as source:content=source.read(64*1024*1024+1)
                            if len(content)!=video['size'] or hashlib.sha256(content).hexdigest()!=video['sha256'] or b'ftyp' not in content[:32]:raise ValueError('Hash, tamaño o contenedor MP4 no válido.')
                            _validate_imported_mp4(content);media_by_entry[filename]=content
                        imported={job:media_by_entry[item['entry']] for job,item in refs.items()}
                        if total+manifest_entry.file_size>CONVERSATION_LIMIT:raise ValueError('La suma descomprimida del ZIP supera 100 MiB.')
                    else:raise ValueError('Versión de manifiesto ZIP no compatible.')
                elif len(names)!=1:raise ValueError('Un ZIP antiguo sólo puede contener conversation.json.')
        except (zipfile.BadZipFile,RuntimeError,NotImplementedError) as exc:raise ValueError('ZIP inválido o no compatible.') from exc
    data=json.loads(raw)
    if not isinstance(data,dict) or data.get('format') not in ('zimage-conversation','zimage-group') or data.get('version')!=1:raise ValueError('Formato de conversación inválido.')
    if conversation_validator:conversation_validator(data)
    if unpack:
        if archive_v2:
            view=data.get('view',[])
            if not isinstance(view,list):raise ValueError('Vista de conversación inválida.')
            referenced={item.get('job_id') for item in view if isinstance(item,dict) and item.get('kind')=='video' and isinstance(item.get('src'),str) and re.fullmatch(r'/api/video/file\?id=[0-9a-f]{32}',item['src']) and item['src'].endswith(item.get('job_id',''))}
            if referenced!=set(imported):raise ValueError('Vídeos del manifiesto y referencias del chat no coinciden.')
        if imported:
            if not video_importer:raise ValueError('El servidor no admite importar MP4.')
            remap={};published_by_hash={}
            for old,content in imported.items():
                digest=hashlib.sha256(content).hexdigest()
                if digest not in published_by_hash:published_by_hash[digest]=video_importer(content)
                remap[old]=published_by_hash[digest]
            for item in view:
                if isinstance(item,dict) and item.get('kind')=='video' and item.get('job_id') in remap:
                    item['job_id']=remap[item['job_id']];item['src']='/api/video/file?id='+item['job_id'];item.pop('request_id',None)
            raw=json.dumps(data,ensure_ascii=False,separators=(',',':')).encode()
        return raw
    view=data.get('view',[])
    if not isinstance(view,list):raise ValueError('Vista de conversación inválida.')
    assets={};manifest_videos={}
    for item in view:
        found=_archive_video(item,video_path)
        if not found:continue
        job_id,content=found;digest=hashlib.sha256(content).hexdigest();entry='media/'+digest+'.mp4'
        if job_id in manifest_videos and manifest_videos[job_id]['sha256']!=digest:raise ValueError('Un ID de vídeo apunta a contenidos distintos.')
        manifest_videos[job_id]={'job_id':job_id,'entry':entry,'sha256':digest,'size':len(content)};assets.setdefault(entry,content)
        item['job_id']=job_id;item['src']='/api/video/file?id='+job_id
    packed=(json.dumps(data,ensure_ascii=False,separators=(',',':')).encode()
            if manifest_videos else raw)
    if len(packed)+sum(len(value) for value in assets.values())>CONVERSATION_LIMIT:raise ValueError('Exportación superior a 100 MiB; exporta JSON sin vídeos o reduce su tamaño.')
    output=io.BytesIO()
    with zipfile.ZipFile(output,'w') as archive:
        archive.writestr('conversation.json',packed,compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr('manifest.json',json.dumps({'archive_version':2,'videos':list(manifest_videos.values())},separators=(',',':')),compress_type=zipfile.ZIP_DEFLATED)
        for filename,content in assets.items():archive.writestr(filename,content,compress_type=zipfile.ZIP_STORED)
    result=output.getvalue()
    if len(result)>CONVERSATION_LIMIT:raise ValueError('ZIP superior a 100 MiB; exporta JSON sin vídeos o reduce su tamaño.')
    return result


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        if isinstance(self.request, ssl.SSLSocket):
            self.request.settimeout(TLS_HANDSHAKE_TIMEOUT)
            try:
                self.request.do_handshake()
            except Exception:
                self.request.close()
                raise
            self.request.settimeout(None)
        super().setup()

    def reply(self, obj, code=200, ctype='application/json; charset=utf-8'):
        data = json.dumps(obj, ensure_ascii=False).encode() if ctype.startswith('application/json') else obj
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def auth_gate(self):
        auth = getattr(self.server, 'auth', None)
        if auth is None or not auth.enabled: return True
        path = urlparse(self.path).path
        if self.command == 'GET' and path in ('/login', '/auth.js', '/local-ca.crt'):
            return True
        if self.command == 'POST' and path == '/api/auth/login':
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096: raise ValueError('Petición inválida')
                password = json.loads(self.rfile.read(length)).get('password')
                result = auth.login(password, self.client_address[0])
                if result is None: return self.reply({'error':'Contraseña incorrecta o demasiados intentos'}, 401) or False
                token, csrf = result
                data = json.dumps({'status':'ok', 'csrf':csrf}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(data)))
                cookie = f'zimage_session={token}; HttpOnly; SameSite=Strict; Path=/; Max-Age=86400'
                if isinstance(self.request, ssl.SSLSocket): cookie += '; Secure'
                self.send_header('Set-Cookie', cookie)
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers(); self.wfile.write(data)
            except Exception:
                self.reply({'error':'Petición de acceso inválida'}, 400)
            return False
        session = auth.session(self.headers.get('Cookie'))
        if session is None:
            if self.command == 'GET' and not path.startswith('/api/'):
                self.send_response(303); self.send_header('Location','/login'); self.send_header('Content-Length','0'); self.end_headers()
            else:
                self.reply({'error':'Inicia sesión para acceder al panel'}, 401)
            return False
        if self.command == 'GET' and path == '/api/auth/session':
            self.reply({'status':'ok','csrf':session[1]})
            return False
        if self.command not in ('GET', 'HEAD', 'OPTIONS'):
            origin = self.headers.get('Origin')
            expected = ('https' if isinstance(self.request, ssl.SSLSocket) else 'http') + '://' + self.headers.get('Host', '')
            supplied = (self.headers.get('X-Zimage-CSRF') or '').encode('utf-8')
            if not hmac.compare_digest(supplied, session[1].encode('utf-8')) or (origin and origin != expected):
                self.reply({'error':'Token de petición inválido'}, 403)
                return False
        return True

    def do_GET(self):
        if not self.auth_gate(): return
        app, parsed = self.server.app.snapshot(), urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == '/api/auth/session':
                return self.reply({'status':'ok','csrf':None})
            if parsed.path == '/login':
                return self.reply((ROOT / 'zimage_panel_login.html').read_bytes(), ctype='text/html; charset=utf-8')
            if parsed.path == '/auth.js':
                return self.reply((ROOT / 'zimage_panel_auth.js').read_bytes(), ctype='text/javascript; charset=utf-8')
            if parsed.path == '/api/conversations':
                limit = int(query.get('limit', ['50'])[0])
                return self.reply(conversation_store().list(
                    archived=query.get('archived', ['0'])[0] == '1',
                    cursor=query.get('cursor', [None])[0], limit=limit,
                    search=query.get('search', [''])[0]))
            if parsed.path.startswith('/api/conversations/media/'):
                media = conversation_store().media(parsed.path.rsplit('/', 1)[-1])
                if media is None: return self.reply({'error': 'Medio no encontrado'}, 404)
                path, mime = media
                if mime=='video/mp4':
                    size=path.stat().st_size;start,end=0,size-1;requested=self.headers.get('Range')
                    if requested:
                        match=re.fullmatch(r'bytes=(\d*)-(\d*)',requested.strip())
                        if not match or not any(match.groups()):return self.reply(b'',416,'application/octet-stream')
                        first,last=match.groups()
                        if first:start=int(first);end=min(int(last),size-1) if last else size-1
                        else:start=max(0,size-int(last)) if int(last) else size
                        if start>=size or end<start:
                            self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.send_header('Content-Length','0');self.end_headers();return
                    length=end-start+1;self.send_response(206 if requested else 200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(length));self.send_header('Accept-Ranges','bytes');self.send_header('Cache-Control','no-store')
                    if requested:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
                    self.end_headers()
                    with path.open('rb') as source:
                        source.seek(start);remaining=length
                        while remaining:
                            chunk=source.read(min(1024*1024,remaining))
                            if not chunk:break
                            self.wfile.write(chunk);remaining-=len(chunk)
                    return
                return self.reply(path.read_bytes(), ctype=mime)
            if parsed.path.startswith('/api/conversations/'):
                parts = parsed.path.strip('/').split('/')
                if len(parts) not in (3, 4): return self.reply({'error': 'Ruta no encontrada'}, 404)
                conversation_id = valid_id(parts[2])
                if len(parts) == 4 and parts[3] == 'messages':
                    before = query.get('before', [None])[0]
                    return self.reply(conversation_store().messages(conversation_id,
                        before=int(before) if before else None,
                        limit=int(query.get('limit', ['50'])[0])))
                if len(parts) != 3: return self.reply({'error': 'Ruta no encontrada'}, 404)
                item = conversation_store().get(conversation_id)
                return self.reply(item if item else {'error': 'Conversación no encontrada'}, 200 if item else 404)
            if parsed.path == '/local-ca.crt':
                return self.reply((ROOT / '.tls' / 'ca.crt').read_bytes(), ctype='application/x-x509-ca-cert')
            if parsed.path in ('/', '/c', '/new', '/group', '/characters', '/character-creator.js', '/group.js', '/progress.js', '/audio.js', '/video.js', '/messenger.js', '/messenger.css', '/chat-state.js', '/prompts.js', '/chat-tools.js', '/lora-eval.js') or parsed.path.startswith('/c/'):
                files = {'/':'zimage_panel_messenger.html', '/c':'zimage_panel_messenger.html',
                    '/new':'zimage_panel_grid.html', '/group':'zimage_panel_group.html',
                    '/characters':'zimage_character_creator.html', '/character-creator.js':'zimage_character_creator.js',
                    '/group.js':'zimage_panel_group.js', '/progress.js':'zimage_panel_progress.js',
                    '/audio.js':'zimage_panel_audio.js', '/video.js':'zimage_panel_video.js',
                    '/messenger.js':'zimage_panel_messenger.js', '/messenger.css':'zimage_panel_messenger.css',
                    '/chat-state.js':'zimage_panel_chat_state.js', '/prompts.js':'zimage_panel_prompts.js',
                    '/chat-tools.js':'zimage_panel_chat_tools.js', '/lora-eval.js':'zimage_panel_lora_eval.js'}
                filename = files.get(parsed.path, 'zimage_panel_messenger.html')
                ctype = 'text/html; charset=utf-8' if filename.endswith('.html') else 'text/css; charset=utf-8' if filename.endswith('.css') else 'text/javascript; charset=utf-8'
                return self.reply((ROOT / filename).read_bytes(), ctype=ctype)
            if parsed.path == '/panel.css':
                css = (ROOT / 'zimage_panel_grid.html').read_text().split('<style>', 1)[1].split('</style>', 1)[0]
                return self.reply(css.encode(), ctype='text/css; charset=utf-8')
            if parsed.path in ('/manifest.json', '/sw.js', '/favicon.svg'):
                content_types = {'/manifest.json': 'application/manifest+json; charset=utf-8',
                                 '/sw.js': 'text/javascript; charset=utf-8',
                                 '/favicon.svg': 'image/svg+xml'}
                return self.reply((ROOT / 'data' / parsed.path[1:]).read_bytes(),
                                  ctype=content_types[parsed.path])
            if parsed.path == '/api/progress':
                key = query.get('request_id', [''])[0]
                with app.config_lock:
                    route = app.progress_routes.get(key)
                if not route:
                    return self.reply({'status':'ok', 'phase':'waiting'})
                pinned, use_mcp = route
                if use_mcp:
                    bridge = pinned.bridge()
                    bridge.initialize()
                    result = bridge.rpc('tools/call', {'name':'generation_progress', 'arguments':{'request_id':key}})
                    return self.reply(json.loads(result_text(result)))
                return self.reply(backend_request({'cmd':'progress','request_id':key}, pinned.args.backend_host, pinned.args.backend_port))
            if parsed.path == '/app.js':
                return self.reply((ROOT / 'zimage_panel_grid.js').read_bytes(), ctype='text/javascript; charset=utf-8')
            if parsed.path == '/api/settings':
                return self.reply({'status':'ok', **app.configuration()})
            if parsed.path == '/api/status':
                return self.reply(backend_request({'cmd': 'status'}, app.args.backend_host, app.args.backend_port))
            if parsed.path == '/api/loras':
                return self.reply({'status': 'ok', 'lora_list': app.profiles.catalog()})
            if parsed.path == '/api/characters':
                return self.reply({'status': 'ok', 'characters': app.character_creator.list(),
                                   'model_dir': str(app.character_creator.model_dir)})
            if parsed.path == '/api/character':
                return self.reply({'status': 'ok', 'character': app.character_creator.get(query.get('name', [''])[0])})
            if parsed.path == '/api/character/photo':
                return self.reply(app.character_creator.photo(query.get('name', [''])[0],
                    query.get('photo', [''])[0]), ctype='image/jpeg')
            if parsed.path == '/api/character/voice':
                return self.reply(app.character_creator.voice(query.get('name', [''])[0]), ctype='audio/wav')
            if parsed.path == '/api/profile':
                return self.reply({'status': 'ok', 'profile': app.profiles.load(query.get('name', [''])[0])})
            if parsed.path == '/api/profile/history':
                return self.reply({'status':'ok','versions':app.profiles.history(query.get('name',[''])[0])})
            if parsed.path == '/api/thumbnail':
                variant = query.get('variant', [''])[0]
                suffix = '_1_2.png' if variant == 'vertical' else '.png'
                if variant not in ('', 'vertical'):
                    raise ValueError('Variante de miniatura no permitida')
                return self.reply(app.profiles.path(query.get('name', [''])[0], suffix).read_bytes(), ctype='image/png')
            if parsed.path == '/api/image':
                return self.reply(allowed_image_file(query.get('path', [''])[0]).read_bytes(),
                                  ctype='image/png')
            if parsed.path == '/api/video/job':
                job_id=query.get('id', [''])[0];request_id=query.get('request_id', [''])[0]
                if bool(job_id) == bool(request_id):
                    return self.reply({'status':'error','error':'Indica exactamente uno de id o request_id'}, 400)
                job = comfyui_video.job(job_id,request_id)
                if job is None:
                    return self.reply({'status':'error','error':'Trabajo o vídeo no encontrado'}, 404)
                return self.reply(job)
            if parsed.path == '/api/video/audit':
                return self.reply(comfyui_video.audit_media(conversation_store().public_video_ids()))
            if parsed.path == '/api/video/file':
                video = comfyui_video.video_path(query.get('id', [''])[0])
                if video is None:
                    return self.reply({'status':'error','error':'Vídeo no encontrado'}, 404)
                with video.open('rb') as stream:
                    size = video.stat().st_size
                    start, end = 0, size - 1
                    requested = self.headers.get('Range')
                    if requested:
                        match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested.strip())
                        if match and (match.group(1) or match.group(2)):
                            first, last = match.groups()
                            if first:
                                start = int(first)
                                end = min(int(last), size - 1) if last else size - 1
                            else:
                                suffix = int(last)
                                start = max(0, size - suffix) if suffix else size
                        if not match or not any(match.groups()) or start >= size or end < start:
                            self.send_response(416)
                            self.send_header('Content-Range', f'bytes */{size}')
                            self.send_header('Content-Length', '0')
                            self.end_headers()
                            return
                    self.send_response(206 if requested else 200)
                    self.send_header('Content-Type', 'video/mp4')
                    self.send_header('Content-Length', str(end - start + 1))
                    self.send_header('Accept-Ranges', 'bytes')
                    if requested:
                        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
                    self.send_header('Cache-Control', 'private, max-age=3600')
                    self.send_header('X-Content-Type-Options', 'nosniff')
                    self.end_headers()
                    stream.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        chunk = stream.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                return
            self.send_error(404)
        except (ValueError, FileNotFoundError) as exc:
            self.reply({'status': 'error', 'error': str(exc)}, 400)
        except Exception as exc:
            self.reply({'status': 'error', 'error': str(exc)}, 502)

    def logout(self):
        auth = getattr(self.server, 'auth', None)
        if auth is not None and auth.enabled:
            auth.logout(self.headers.get('Cookie'))
        data = json.dumps({'status': 'ok'}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        cookie = 'zimage_session=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0'
        if isinstance(self.request, ssl.SSLSocket): cookie += '; Secure'
        self.send_header('Set-Cookie', cookie)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if not self.auth_gate(): return
        if urlparse(self.path).path == '/api/auth/logout':
            return self.logout()
        app = self.server.app.snapshot()
        try:
            length = int(self.headers.get('Content-Length', '0'))
            path = urlparse(self.path).path
            archive_request = path in ('/api/conversation/zip', '/api/conversation/unzip')
            request_limit=CONVERSATION_LIMIT if archive_request else MAX_SNAPSHOT if path.startswith('/api/conversations') else 32 * 1024**2
            if not 0 < length <= request_limit:
                raise ValueError('Petición vacía o demasiado grande')
            raw = self.rfile.read(length)
            if archive_request:
                unpack = path.endswith('/unzip')
                result = conversation_archive(raw, unpack=unpack, video_path=comfyui_video.video_path,
                                              video_importer=comfyui_video.import_video,
                                              conversation_validator=lambda value: validate_portable_conversation(value, app.profiles))
                return self.reply(result, ctype='application/octet-stream' if unpack else 'application/zip')
            body = json.loads(raw)
            path = urlparse(self.path).path
            if path.startswith('/api/character/') and not isinstance(body, dict):
                raise ValueError('Datos de personaje inválidos')
            if path == '/api/upscale' and not isinstance(body, dict):
                raise ValueError('Petición de reescalado inválida')
            if path == '/api/character/create':
                return self.reply({'status': 'ok', 'character': app.character_creator.create(body)}, 201)
            if path == '/api/character/photo':
                return self.reply({'status': 'ok', 'character': app.character_creator.add_photo(
                    body.get('name'), body.get('image'))})
            if path == '/api/character/voice':
                return self.reply({'status': 'ok', 'character': app.character_creator.add_voice(
                    body.get('name'), body.get('audio'))})
            if path == '/api/character/remove-voice':
                return self.reply({'status': 'ok', 'character': app.character_creator.remove_voice(body.get('name'))})
            if path == '/api/character/remove-photo':
                return self.reply({'status': 'ok', 'character': app.character_creator.remove_photo(
                    body.get('name'), body.get('photo'))})
            if path == '/api/character/update':
                return self.reply({'status': 'ok', 'character': app.character_creator.update(body.get('name'), body)})
            if path == '/api/character/caption':
                return self.reply({'status': 'ok', 'character': app.character_creator.caption(body.get('name'))}, 202)
            if path == '/api/character/train':
                return self.reply({'status': 'ok', 'character': app.character_creator.train(
                    body.get('name'), body.get('options'))}, 202)
            if path == '/api/character/upscale-photos':
                return self.reply({'status': 'ok', **app.character_creator.upscale_photos(
                    body.get('name'), body.get('target'))})
            if path == '/api/character/restore-photos':
                return self.reply({'status': 'ok', **app.character_creator.restore_photos(
                    body.get('name'))})
            if path in ('/api/conversations', '/api/conversations/import'):
                if not isinstance(body, dict): raise ValueError('Petición inválida')
                kind = body.get('kind')
                members = checked_members(kind, body.get('members'), app.profiles)
                validate_conversation_snapshot(kind, members, body.get('snapshot'), app.profiles)
                item = conversation_store().create(kind, body.get('title'), members,
                    body.get('snapshot'), body.get('migration_id') if path.endswith('/import') else None,
                    conversation_id=valid_id(body['conversation_id']) if body.get('conversation_id') else None,
                    visibility=body.get('visibility', 'public'))
                return self.reply(item, 201)
            if path == '/api/video/generate':
                if not isinstance(body, dict):
                    raise ValueError('La petición de vídeo debe ser un objeto JSON')
                return self.reply(comfyui_video.submit(body), 202)
            if path in ('/api/chat', '/api/generate', '/api/retry', '/api/group/turn') and body.get('request_id'):
                key = body['request_id']
                if not isinstance(key, str) or len(key) > 100:
                    raise ValueError('Identificador de progreso inválido')
                with app.config_lock:
                    if len(app.progress_routes) >= 256:
                        app.progress_routes.pop(next(iter(app.progress_routes)))
                    app.progress_routes[key] = (app, path != '/api/generate')
            if path in ('/api/audio/tts', '/api/audio/stt'):
                if not app.audio_url:
                    raise ValueError('Configura la URL del servidor de audio en Ajustes.')
                if path.endswith('/tts'):
                    engine = body.get('engine', 'kokoro')
                    if engine not in ('kokoro', 'kokoclone', 'kokoclone_direct', 'qwen3tts'):
                        raise ValueError('Motor de voz inválido.')
                    clean = {k:body[k] for k in ('input','voice','speed') if k in body}
                    clean['engine'] = engine
                    if engine.startswith('kokoclone') or engine == 'qwen3tts':
                        name = body.get('lora_name', '')
                        reference=app.profiles.voice_path(name,body['voice_wav']) if 'voice_wav' in body else app.profiles.voice_path(name)
                        if not reference.is_file():
                            raise ValueError(f'Falta {reference.name} junto a los LoRAs. Añade la referencia o selecciona Kokoro local.')
                        if reference.stat().st_size > 20 * 1024**2:
                            raise ValueError('Referencia WAV mayor de 20 MB.')
                        clean['reference_audio'] = base64.b64encode(reference.read_bytes()).decode()
                        # Qwen can use the exact transcript for a more faithful
                        # in-context clone.  It is still obtained only from the
                        # profile sidecar, never from a browser supplied path.
                        if engine == 'qwen3tts':
                            transcript = reference.with_suffix('.txt')
                            if transcript.is_symlink():
                                raise ValueError('Transcripción de referencia no permitida')
                            if transcript.is_file():
                                if transcript.stat().st_size > 10 * 1024:
                                    raise ValueError('Transcripción de referencia mayor de 10 KB.')
                                clean['reference_text'] = transcript.read_text(encoding='utf-8')
                    body = clean
                req = Request(app.audio_url + ('/tts' if path.endswith('/tts') else '/stt'),
                              data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
                try:
                    with urlopen(req, timeout=600) as response:
                        data = response.read(32 * 1024**2 + 1)
                        if len(data) > 32 * 1024**2:
                            raise ValueError('Respuesta de audio demasiado grande.')
                        ctype = response.headers.get_content_type()
                    if path.endswith('/stt'):
                        return self.reply(json.loads(data))
                    if ctype not in ('audio/wav', 'audio/x-wav', 'audio/mpeg', 'audio/ogg'):
                        raise ValueError('El servidor no devolvió un audio compatible.')
                    return self.reply(data, ctype=ctype)
                except HTTPError as exc:
                    raise ValueError('Servidor de audio: ' + exc.read(2000).decode(errors='replace')) from exc
            if path == '/api/group/turn':
                return self.reply(app.group_turn(conversation_store().expand_media(body)))
            if path == '/api/settings':
                return self.reply({'status':'ok', **self.server.app.configure(body)})
            if path == '/api/profile':
                return self.reply({'status': 'ok', 'profile': app.profiles.save(body.get('name'), body.get('profile'))})
            if path == '/api/profile/restore':
                return self.reply({'status':'ok','profile':app.profiles.restore(body.get('name'),body.get('version_id'))})
            if path == '/api/generate':
                body = app.generation_body(body)
                settings = image_settings(body, app.profiles)
                prompt = body.get('image_prompt', '')
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError('Escribe un prompt de imagen')
                return self.reply(backend_request({**settings, 'prompt': prompt, 'cmd': 'generate'},
                                                 app.args.backend_host, app.args.backend_port))
            if path == '/api/upscale':
                # Pillow only: no GPU, no generation lock and no TCP changes.
                return self.reply(upscale_image(body.get('src'), body.get('scale')))
            if path == '/api/lora/reload':
                result = backend_request({'cmd': 'reload_loras'}, app.args.backend_host, app.args.backend_port)
                return self.reply(result, 200 if result.get('status') == 'ok' else 502)
            if path == '/api/retry':
                body = app.generation_body(body)
                settings = image_settings(body, app.profiles)
                prompt = body.get('image_prompt', '')
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError('Escribe un prompt de imagen')
                bridge = app.bridge()
                bridge.initialize()
                result = bridge.rpc('tools/call', {'name': 'generate_image', 'arguments': {**settings, 'prompt': prompt}})
                if result.get('isError'):
                    raise ValueError(result_text(result))
                for content in result.get('content', []):
                    if content.get('type') == 'image' and content.get('mimeType') in ('image/png', 'image/jpeg', 'image/webp'):
                        return self.reply({'status': 'ok', 'src': f'data:{content["mimeType"]};base64,{content["data"]}',
                            'seed': settings['seed'], 'prompt': prompt,
                            'generation': {'lora_name': body.get('lora_name', ''), 'lora_weight': float(body.get('lora_weight', .8)),
                                **({'loras': prompt_loras(body['loras'], prompt)} if 'loras' in body else {}),
                                'image_prompt': prompt, 'image_settings': {k:v for k,v in settings.items() if k != 'lora'}}})
                raise ValueError('MCP no ha devuelto una imagen')
            if path == '/api/chat':
                return self.reply(app.chat(conversation_store().expand_media(body)))
            self.send_error(404)
        except (ValueError, TypeError) as exc:
            self.reply({'status': 'error', 'error': str(exc)}, 400)
        except comfyui_video.VideoQueueFull as exc:
            self.reply({'status':'error','error':str(exc)},409)
        except Exception as exc:
            self.reply({'status': 'error', 'error': str(exc)}, 502)

    def do_PATCH(self):
        if not self.auth_gate(): return
        self._mutate_conversation(False)

    def do_PUT(self):
        if not self.auth_gate(): return
        self._mutate_conversation(True)

    def _mutate_conversation(self, snapshot):
        try:
            parts = urlparse(self.path).path.strip('/').split('/')
            if len(parts) != 3 or parts[:2] != ['api', 'conversations']:
                return self.reply({'error': 'Ruta no encontrada'}, 404)
            conversation_id = valid_id(parts[2])
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 32 * 1024**2:
                raise ValueError('Petición vacía o demasiado grande')
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict): raise ValueError('Petición inválida')
            allowed = {'revision', 'snapshot'} if snapshot else {'revision', 'title', 'pinned', 'archived', 'visibility'}
            if set(body) - allowed or 'revision' not in body or (snapshot and 'snapshot' not in body):
                raise ValueError('Campos de conversación inválidos')
            if snapshot:
                existing = conversation_store().get(conversation_id)
                if existing is None: return self.reply({'error':'Conversación no encontrada'}, 404)
                validate_conversation_snapshot(existing['kind'], existing['members'], body['snapshot'], self.server.app.profiles)
            item = conversation_store().update(conversation_id, body['revision'],
                **{key: value for key, value in body.items() if key != 'revision'})
            return self.reply(item if item else {'error':'Conversación no encontrada'}, 200 if item else 404)
        except Conflict as exc:
            self.reply({'error': str(exc)}, 409)
        except (ValueError, TypeError) as exc:
            self.reply({'error': str(exc)}, 400)
        except Exception as exc:
            self.reply({'error': str(exc)}, 502)

    def do_DELETE(self):
        if not self.auth_gate(): return
        try:
            path=urlparse(self.path).path
            match=re.fullmatch(r'/api/video/job/([0-9a-f]{32})',path)
            if match:
                try:return self.reply(comfyui_video.cancel_job(match.group(1)))
                except RuntimeError as exc:return self.reply({'status':'error','error':str(exc)},409)
                except FileNotFoundError as exc:return self.reply({'status':'error','error':str(exc)},404)
            parts = urlparse(self.path).path.strip('/').split('/')
            if len(parts) != 3 or parts[:2] != ['api', 'conversations']:
                return self.reply({'error':'Ruta no encontrada'}, 404)
            conversation_id = valid_id(parts[2])
            revision = int(self.headers.get('X-Conversation-Revision', '0'))
            deleted = conversation_store().delete(conversation_id, revision)
            return self.reply({'status':'ok'} if deleted else {'error':'Conversación no encontrada'}, 200 if deleted else 404)
        except Conflict as exc:
            self.reply({'error': str(exc)}, 409)
        except (ValueError, TypeError) as exc:
            self.reply({'error': str(exc)}, 400)
        except Exception as exc:
            self.reply({'error': str(exc)}, 502)

    def log_message(self, fmt, *args):
        print('[grid] ' + fmt % args, flush=True)


class TLSServer(ThreadingHTTPServer):
    def __init__(self, address, handler, context):
        self.tls_context = context
        super().__init__(address, handler)

    def get_request(self):
        sock, address = super().get_request()
        sock.settimeout(TLS_HANDSHAKE_TIMEOUT)
        try:
            return self.tls_context.wrap_socket(sock, server_side=True, do_handshake_on_connect=False), address
        except Exception:
            sock.close()
            raise


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend-host', default='127.0.0.1')
    parser.add_argument('--backend-port', type=int, default=9877)
    parser.add_argument('--chat-url', default='http://127.0.0.1:8086/v1')
    parser.add_argument('--chat-model', default='')
    parser.add_argument('--mcp-url', help='Servidor HTTP zimage_mcp_server.py existente (opcional)')
    parser.add_argument('--integrated-mcp', action='store_true', help='Ignorar mcp_url persistida y usar el dispatcher integrado')
    parser.add_argument('--lora-dir', default=str(ROOT / 'lora_models_peft'))
    parser.add_argument('--web-host', default='127.0.0.1')
    parser.add_argument('--tls-cert', default=os.environ.get('ZIMAGE_TLS_CERT', str(ROOT / '.tls' / 'server.crt')))
    parser.add_argument('--tls-key', default=os.environ.get('ZIMAGE_TLS_KEY', str(ROOT / '.tls' / 'server.key')))
    parser.add_argument('--tls-port', type=int, default=int(os.environ.get('ZIMAGE_TLS_PORT', '8768')))
    parser.add_argument('--web-port', type=int, default=8766)
    parser.add_argument('--settings-file', default=str(ROOT / 'zimage_panel_settings.json'), help='Ajustes persistentes; prevalecen sobre los argumentos de conexión')
    parser.add_argument('--init-profiles', action='store_true', help='Crear solo los JSON que faltan y salir')
    return parser.parse_args()


def main():
    args = arguments()
    app = App(args)
    if args.init_profiles:
        print(f'Perfiles creados: {app.profiles.initialize()}')
        return
    server = ThreadingHTTPServer((args.web_host, args.web_port), Handler)
    server.app = app
    auth = PanelAuth(enabled=args.web_host not in ('127.0.0.1', 'localhost', '::1') or os.environ.get('ZIMAGE_PANEL_REQUIRE_AUTH') == '1')
    server.auth = auth
    if auth.enabled:
        print(f'Acceso protegido; contraseña en {auth.password_file} (o ZIMAGE_PANEL_PASSWORD).', flush=True)
    if Path(args.tls_cert).exists() or Path(args.tls_key).exists():
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(args.tls_cert, args.tls_key)
        secure = TLSServer((args.web_host, args.tls_port), Handler, context)
        secure.app = app
        secure.auth = auth
        threading.Thread(target=secure.serve_forever, daemon=True).start()
        print(f'HTTPS: https://{args.web_host}:{args.tls_port}', flush=True)
    print(f'Panel: http://{args.web_host}:{args.web_port} · chat: {args.chat_url} · LoRAs: {args.lora_dir}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
