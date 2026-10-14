"""Durable conversation snapshots and paged messages for the messaging shell."""
from __future__ import annotations

import base64
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
import uuid
import warnings

from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get('ZIMAGE_DATA_DIR', str(ROOT / 'data')))
STORE_DIR = Path(os.environ.get('ZIMAGE_CONVERSATIONS_DIR', str(DATA_DIR / 'conversations')))
UUID_RE = re.compile(r'[0-9a-f]{32}\Z')
MEDIA_RE = re.compile(r'/api/conversations/media/([0-9a-f]{32})\Z')
DATA_RE = re.compile(r'data:image/(png|jpeg|webp);base64,([A-Za-z0-9+/]+={0,2})\Z')
MAX_SNAPSHOT = 100 * 1024 * 1024
MAX_IMAGE = 16 * 1024 * 1024
MAX_VIDEO = 64 * 1024 * 1024


@contextmanager
def image_errors():
    """Turn Pillow decompression bombs and undecodable files into ValueError (HTTP 400)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            yield
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError('Imagen demasiado grande (posible bomba de descompresión)') from exc
    except Image.UnidentifiedImageError as exc:
        raise ValueError('Imagen no válida') from exc


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def valid_id(value):
    if not isinstance(value, str) or not UUID_RE.fullmatch(value):
        raise ValueError('ID de conversación inválido')
    return value


def checked_members(kind, members, profiles):
    if kind not in ('individual', 'group') or not isinstance(members, list):
        raise ValueError('Tipo o integrantes de conversación inválidos')
    if not (1 <= len(members) <= (1 if kind == 'individual' else 6)):
        raise ValueError('Número de integrantes inválido')
    result = []
    for member in members:
        if not isinstance(member, dict) or not isinstance(member.get('name'), str):
            raise ValueError('Integrante inválido')
        name = member['name']
        if name:
            profiles.path(name, '.json')
        elif kind != 'individual':
            raise ValueError('Un grupo necesita personajes con LoRA')
        weight = member.get('weight', .8)
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or not 0 <= weight <= 3:
            raise ValueError('Peso LoRA inválido')
        profile = member.get('profile') or {}
        if not isinstance(profile, dict):
            raise ValueError('Perfil inválido')
        for key in ('system_prompt', 'image_prompt'):
            if key in profile and (not isinstance(profile[key], str) or len(profile[key]) > 50000):
                raise ValueError('Prompt del perfil inválido')
        if 'voice_wav' in profile:
            voice=profile['voice_wav']
            if not isinstance(voice,str) or (voice and (len(voice)>180 or not voice.lower().endswith('.wav') or '/' in voice or '\\' in voice or voice.startswith('.') or any(ord(char)<32 for char in voice))):
                raise ValueError('Referencia WAV del miembro inválida')
        if 'personality_presets' in profile:
            presets = profile['personality_presets']
            if not isinstance(presets, list) or len(presets) > 20:
                raise ValueError('Lista de personalidades inválida')
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
                    raise ValueError('Personalidad del miembro inválida')
                normalized_name = preset['name'].strip().casefold()
                if preset['id'] in ids or normalized_name in names:
                    raise ValueError('IDs y nombres de personalidad deben ser únicos')
                ids.add(preset['id']); names.add(normalized_name)
        result.append({'name': name, 'weight': weight, 'image_enabled': bool(member.get('image_enabled', True)),
                       'profile': {k: profile[k] for k in ('system_prompt', 'image_prompt', 'personality_presets', 'voice_wav') if k in profile}})
    if len({m['name'] for m in result}) != len(result):
        raise ValueError('Integrantes duplicados')
    return result


def checked_personality(value):
    if value is None:
        return None
    if not isinstance(value, dict) or isinstance(value.get('version'), bool) or value.get('version') != 1:
        raise ValueError('Estado de personalidad inválido')
    if set(value) - {'version','base_system_prompt','preset_id','preset_revision','builder_fields'}:
        raise ValueError('El estado de personalidad contiene campos desconocidos')
    base = value.get('base_system_prompt', '')
    if not isinstance(base, str) or len(base) > 50000:
        raise ValueError('Prompt base de personalidad inválido')
    preset_id = value.get('preset_id', 'base')
    if (not isinstance(preset_id, str)
            or not re.fullmatch(r'(?:base|custom|builtin-(?:close|adventure|formal)|[a-z][a-z0-9_-]{0,39})', preset_id)
            or (preset_id.startswith('builtin-') and preset_id not in {'builtin-close','builtin-adventure','builtin-formal'})):
        raise ValueError('ID de personalidad inválido')
    revision = value.get('preset_revision')
    if revision is not None and (not isinstance(revision, int) or isinstance(revision, bool) or not 1 <= revision <= 2147483647):
        raise ValueError('Revisión de personalidad inválida')
    fields = value.get('builder_fields')
    if fields is not None:
        if not isinstance(fields, dict):
            raise ValueError('Campos del constructor de personalidad inválidos')
        maxima = {'name':500, 'tone':500, 'speech_style':500, 'interests':500,
                  'relationship':500, 'role_limits':2000}
        if set(fields) - {*maxima,'response_length'}:
            raise ValueError('Campos desconocidos en el constructor de personalidad')
        for key, maximum in maxima.items():
            item = fields.get(key, '')
            if not isinstance(item, str) or len(item) > maximum:
                raise ValueError('Campo del constructor de personalidad inválido')
        if fields.get('response_length') not in (None, '', 'short', 'normal', 'long'):
            raise ValueError('Longitud de respuesta inválida')
    return value


def checked_notes(value):
    if value is None: return []
    if not isinstance(value, list) or len(value) > 10:
        raise ValueError('La memoria del chat admite hasta 10 notas')
    ids=set();total=0
    for note in value:
        if (not isinstance(note,dict) or not isinstance(note.get('id'),str)
                or not UUID_RE.fullmatch(note['id']) or note['id'] in ids
                or not isinstance(note.get('text'),str) or not note['text'].strip()
                or len(note['text'])>500):
            raise ValueError('Nota de memoria inválida')
        ids.add(note['id']);total+=len(note['text'])
    if total>4000:raise ValueError('La memoria del chat admite hasta 4000 caracteres')
    return value


def checked_image_fields(value):
    if value is None:return None
    limits={'subject':5000,'background':1000,'outfit':1000,'action':1000,
            'framing':1000,'lighting':1000,'style':1000,'negative_prompt':50000}
    if not isinstance(value,dict) or value.get('version',1)!=1 or set(value)-set(limits)-{'version'}:
        raise ValueError('Campos estructurados de imagen inválidos')
    for key,maximum in limits.items():
        item=value.get(key,'')
        if not isinstance(item,str) or len(item)>maximum:raise ValueError('Campo estructurado de imagen inválido')
    return value


class Conflict(Exception):
    pass


class Store:
    def __init__(self, directory=STORE_DIR):
        self.directory = Path(directory)
        self.media_dir = self.directory / 'media'
        self.media_pending_dir = self.media_dir / '.pending'
        self.media_lock_path = self.media_dir / '.media-operation.lock'
        self.db_path = self.directory / 'conversations.sqlite3'
        self.lock = threading.RLock()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.media_dir.is_symlink(): raise ValueError('El directorio de medios no puede ser un symlink')
        self.media_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.directory, 0o700)
        os.chmod(self.media_dir, 0o700)
        self.media_pending_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.media_pending_dir.is_symlink(): raise ValueError('El journal de medios no puede ser un symlink')
        os.chmod(self.media_pending_dir, 0o700)
        if self.media_lock_path.is_symlink(): raise ValueError('El lock de medios no puede ser un symlink')
        self.media_lock_path.touch(mode=0o600, exist_ok=True)
        os.chmod(self.media_lock_path, 0o600)
        with self.connection() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, title TEXT NOT NULL,
                    members TEXT NOT NULL, snapshot TEXT NOT NULL, revision INTEGER NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    archived_at TEXT, pinned INTEGER NOT NULL DEFAULT 0,
                    preview TEXT NOT NULL DEFAULT '', migration_id TEXT UNIQUE);
                CREATE INDEX IF NOT EXISTS conversations_order
                    ON conversations(archived_at, pinned, updated_at DESC);
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL, kind TEXT NOT NULL, speaker TEXT NOT NULL,
                    text TEXT NOT NULL, body TEXT NOT NULL, created_at TEXT NOT NULL,
                    UNIQUE(conversation_id, sequence));
                CREATE TABLE IF NOT EXISTS media (
                    id TEXT PRIMARY KEY, mime TEXT NOT NULL, size INTEGER NOT NULL,
                    sha256 TEXT NOT NULL, refcount INTEGER NOT NULL);
            ''')
            if db.execute('PRAGMA user_version').fetchone()[0] < 2:
                columns = {row['name'] for row in db.execute('PRAGMA table_info(conversations)')}
                if 'visibility' not in columns:
                    db.execute("ALTER TABLE conversations ADD COLUMN visibility TEXT NOT NULL DEFAULT 'public'")
                db.execute('PRAGMA user_version=2')
        os.chmod(self.db_path, 0o600)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.db_path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA busy_timeout=10000')
        db.execute('PRAGMA journal_mode=WAL')
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @contextmanager
    def media_operation(self):
        """Serialize media publication with the audit cleaner across processes."""
        with self.lock, self.media_lock_path.open('a+b') as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _write_media_intent(self, media_id, mime, extension, data, digest):
        marker = self.media_pending_dir / (media_id + '.json')
        temporary = self.media_pending_dir / ('.' + media_id + '.tmp')
        record = {'version': 1, 'id': media_id, 'mime': mime, 'extension': extension,
                  'size': len(data), 'sha256': digest}
        with temporary.open('xb') as stream:
            os.chmod(temporary, 0o600)
            stream.write(json.dumps(record, separators=(',', ':')).encode('utf-8'))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, marker)
        directory_fd = os.open(self.media_pending_dir, os.O_RDONLY)
        try: os.fsync(directory_fd)
        finally: os.close(directory_fd)

    def _finalize_media(self, created):
        for media_id in created:
            (self.media_pending_dir / (media_id + '.json')).unlink(missing_ok=True)

    @staticmethod
    def _read_snapshot(value):
        if not isinstance(value, dict) or value.get('version') != 1:
            raise ValueError('Instantánea de conversación inválida')
        view = value.get('view', [])
        if not isinstance(view, list) or len(view) > 10000:
            raise ValueError('Historial demasiado largo')
        if len(json.dumps(value, ensure_ascii=False).encode()) > MAX_SNAPSHOT:
            raise ValueError('Conversación demasiado grande')
        return value

    def _put_image(self, value, created):
        match = DATA_RE.fullmatch(value)
        if not match:
            raise ValueError('Imagen Base64 no válida')
        data = base64.b64decode(match.group(2), validate=True)
        if not 0 < len(data) <= MAX_IMAGE:
            raise ValueError('Imagen demasiado grande')
        with image_errors(), Image.open(io.BytesIO(data)) as image:
            if image.format not in ('PNG', 'JPEG', 'WEBP') or max(image.size) > 4096:
                raise ValueError('Formato o dimensiones de imagen inválidos')
            image.verify()
        extension = {'png': '.png', 'jpeg': '.jpg', 'webp': '.webp'}[match.group(1)]
        digest = hashlib.sha256(data).hexdigest()
        mime = 'image/' + match.group(1)
        with self.lock, self.connection() as db:
            row = db.execute('SELECT id FROM media WHERE mime=? AND sha256=? AND refcount>0', (mime, digest)).fetchone()
            if row and (self.media_dir / (row['id'] + extension)).is_file():
                return '/api/conversations/media/' + row['id']
        media_id = uuid.uuid4().hex
        target = self.media_dir / (media_id + extension)
        temporary = self.media_dir / ('.' + media_id)
        self._write_media_intent(media_id, mime, extension, data, digest)
        with temporary.open('xb') as stream:
            os.chmod(temporary, 0o600)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(self.media_dir, os.O_RDONLY)
        try: os.fsync(directory_fd)
        finally: os.close(directory_fd)
        created[media_id] = (target, mime, len(data), digest)
        return '/api/conversations/media/' + media_id

    def _put_video(self,value,created):
        match=re.fullmatch(r'data:video/mp4;base64,([A-Za-z0-9+/]+={0,2})',value)
        if not match:raise ValueError('Vídeo de conversación inválido')
        try:data=base64.b64decode(match.group(1),validate=True)
        except Exception as exc:raise ValueError('Vídeo de conversación inválido') from exc
        if not data or len(data)>MAX_VIDEO or b'ftyp' not in data[:32]:raise ValueError('Vídeo MP4 inválido o supera 64 MiB')
        digest=hashlib.sha256(data).hexdigest();mime='video/mp4'
        with self.lock,self.connection() as db:
            row=db.execute('SELECT id FROM media WHERE mime=? AND sha256=? AND refcount>0',(mime,digest)).fetchone()
            if row and (self.media_dir/(row['id']+'.mp4')).is_file():return '/api/conversations/media/'+row['id']
        media_id=uuid.uuid4().hex;target=self.media_dir/(media_id+'.mp4');temporary=self.media_dir/('.'+media_id)
        self._write_media_intent(media_id,mime,'.mp4',data,digest)
        with temporary.open('xb') as stream:
            os.chmod(temporary,0o600);stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,target)
        directory_fd=os.open(self.media_dir,os.O_RDONLY)
        try:os.fsync(directory_fd)
        finally:os.close(directory_fd)
        created[media_id]=(target,mime,len(data),digest)
        return '/api/conversations/media/'+media_id

    def _normalize(self, value, created, dedup):
        if isinstance(value, dict):
            return {key: self._normalize(item, created, dedup) for key, item in value.items()}
        if isinstance(value, list):
            return [self._normalize(item, created, dedup) for item in value]
        if isinstance(value, str) and value.startswith('data:image/'):
            if value not in dedup:
                dedup[value] = self._put_image(value, created)
            return dedup[value]
        if isinstance(value,str) and value.startswith('data:video/mp4;'):
            if value not in dedup:dedup[value]=self._put_video(value,created)
            return dedup[value]
        return value

    @staticmethod
    def _refs(value):
        count = Counter()
        def visit(item):
            if isinstance(item, dict):
                for nested in item.values(): visit(nested)
            elif isinstance(item, list):
                for nested in item: visit(nested)
            elif isinstance(item, str):
                match = MEDIA_RE.fullmatch(item)
                if match: count[match.group(1)] += 1
        visit(value)
        return count

    @staticmethod
    def _preview(snapshot):
        for item in reversed(snapshot.get('view', [])):
            if isinstance(item, dict):
                text = item.get('text') or ('Vídeo' if item.get('kind') == 'video' else 'Imagen' if item.get('src') else '')
                if isinstance(text, str) and text.strip():
                    return text.strip().replace('\n', ' ')[:160]
        return ''

    def _sync_messages(self, db, conversation_id, snapshot, created_at):
        previous = {row['sequence']: (row['body'], row['created_at']) for row in
                    db.execute('SELECT sequence,body,created_at FROM messages WHERE conversation_id=?', (conversation_id,))}
        db.execute('DELETE FROM messages WHERE conversation_id=?', (conversation_id,))
        for sequence, item in enumerate(snapshot.get('view', []), 1):
            if not isinstance(item, dict):
                raise ValueError('Mensaje inválido')
            kind = str(item.get('kind') or item.get('role') or 'assistant')[:30]
            speaker = str(item.get('speaker') or item.get('role') or '')[:300]
            text = item.get('text') or ''
            if not isinstance(text, str) or len(text) > 100000:
                raise ValueError('Texto de mensaje inválido')
            message_id = uuid.uuid5(uuid.UUID(conversation_id), str(sequence)).hex
            body = json.dumps(item, ensure_ascii=False)
            timestamp = previous.get(sequence, (None, created_at))[1] if previous.get(sequence, (None,))[0] == body else created_at
            db.execute('INSERT INTO messages VALUES (?,?,?,?,?,?,?,?)',
                       (message_id, conversation_id, sequence, kind, speaker, text,
                        body, timestamp))

    def _update_refs(self, db, old, new, created):
        old_refs, new_refs = self._refs(old), self._refs(new)
        for media_id, (path, mime, size, digest) in created.items():
            db.execute('INSERT INTO media VALUES (?,?,?,?,0)', (media_id, mime, size, digest))
        removed = []
        for media_id in old_refs.keys() | new_refs.keys():
            delta = new_refs[media_id] - old_refs[media_id]
            if not delta: continue
            result = db.execute('UPDATE media SET refcount=refcount+? WHERE id=?', (delta, media_id))
            if result.rowcount != 1:
                raise ValueError('Medio de conversación no encontrado')
            if new_refs[media_id] == 0:
                row = db.execute('SELECT refcount FROM media WHERE id=?', (media_id,)).fetchone()
                if row['refcount'] == 0:
                    db.execute('DELETE FROM media WHERE id=?', (media_id,))
                    removed.append(media_id)
        return removed

    def _delete_media_files(self, ids):
        for media_id in ids:
            for extension in ('.png', '.jpg', '.webp', '.mp4'):
                (self.media_dir / (media_id + extension)).unlink(missing_ok=True)
            (self.media_dir / ('.' + media_id)).unlink(missing_ok=True)
            (self.media_pending_dir / (media_id + '.json')).unlink(missing_ok=True)

    def create(self, kind, title, members, snapshot=None, migration_id=None, conversation_id=None, visibility='public'):
        with self.media_operation():
            return self._create(kind, title, members, snapshot, migration_id, conversation_id, visibility)

    def _create(self, kind, title, members, snapshot=None, migration_id=None, conversation_id=None, visibility='public'):
        if not isinstance(title, str) or not title.strip() or len(title) > 200:
            raise ValueError('Título inválido')
        if not isinstance(members, list):
            raise ValueError('Integrantes inválidos')
        if conversation_id is not None:
            valid_id(conversation_id)
        if visibility not in ('public', 'private'):
            raise ValueError('Visibilidad inválida')
        snapshot = self._read_snapshot(snapshot or {'version': 1, 'view': []})
        if migration_id is not None and (not isinstance(migration_id, str) or len(migration_id) > 300):
            raise ValueError('ID de migración inválido')
        created, dedup = {}, {}
        try:
            snapshot = self._normalize(snapshot, created, dedup)
            timestamp = now()
            with self.lock, self.connection() as db:
                if migration_id:
                    row = db.execute('SELECT id FROM conversations WHERE migration_id=?', (migration_id,)).fetchone()
                    if row:
                        self._delete_media_files(created)
                        return self.get(row['id'])
                if conversation_id is not None and db.execute('SELECT id FROM conversations WHERE id=?', (conversation_id,)).fetchone():
                    raise Conflict('La conversación ya existe en el servidor')
                conversation_id = conversation_id or uuid.uuid4().hex
                db.execute('INSERT INTO conversations (id,kind,title,members,snapshot,revision,created_at,updated_at,preview,migration_id,visibility) VALUES (?,?,?,?,?,1,?,?,?,?,?)',
                           (conversation_id, kind, title.strip(), json.dumps(members, ensure_ascii=False),
                            json.dumps(snapshot, ensure_ascii=False), timestamp, timestamp, self._preview(snapshot), migration_id, visibility))
                self._sync_messages(db, conversation_id, snapshot, timestamp)
                self._update_refs(db, {}, snapshot, created)
            self._finalize_media(created)
            return self.get(conversation_id)
        except Exception:
            self._delete_media_files(created)
            raise

    def get(self, conversation_id):
        valid_id(conversation_id)
        with self.connection() as db:
            row = db.execute('SELECT * FROM conversations WHERE id=?', (conversation_id,)).fetchone()
            if row is None: return None
            snapshot = json.loads(row['snapshot'])
            return self._record(row, storage_bytes=self._snapshot_storage_bytes(db, snapshot))

    @staticmethod
    def _record(row, include_snapshot=True, storage_bytes=0):
        result = {key: row[key] for key in ('id', 'kind', 'title', 'revision', 'created_at', 'updated_at', 'archived_at', 'pinned', 'preview', 'visibility')}
        result['pinned'] = bool(result['pinned'])
        result['members'] = json.loads(row['members'])
        result['storage_bytes'] = max(0, int(storage_bytes))
        result['storage_scope'] = 'servidor · medios registrados'
        if include_snapshot: result['snapshot'] = json.loads(row['snapshot'])
        return result

    @staticmethod
    def _snapshot_storage_bytes(db, snapshot):
        refs = Store._refs(snapshot)
        if not refs: return 0
        placeholders = ','.join('?' for _ in refs)
        rows = db.execute(f'SELECT id,size FROM media WHERE refcount>0 AND id IN ({placeholders})', tuple(refs)).fetchall()
        return sum(max(0, int(row['size'])) for row in rows)

    def list(self, *, archived=False, cursor=None, limit=50, search=''):
        if not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError('Límite de página inválido')
        if not isinstance(search, str) or len(search) > 100:
            raise ValueError('Búsqueda demasiado larga')
        if cursor is not None:
            valid_id(cursor)
        with self.connection() as db:
            args = [1 if archived else 0]
            where = 'WHERE (archived_at IS NOT NULL)=?'
            if search:
                where += ' AND (title LIKE ? OR members LIKE ? OR EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id=conversations.id AND m.text LIKE ?))'
                args.extend(['%' + search + '%'] * 3)
            if cursor:
                row = db.execute('SELECT pinned,updated_at,id FROM conversations WHERE id=?', (cursor,)).fetchone()
                if row is None: raise ValueError('Cursor inválido')
                where += ' AND (pinned,updated_at,id) < (?,?,?)'
                args.extend([row['pinned'], row['updated_at'], row['id']])
            rows = db.execute('SELECT * FROM conversations ' + where +
                              ' ORDER BY pinned DESC,updated_at DESC,id DESC LIMIT ?', (*args, limit + 1)).fetchall()
            items = [self._record(row, False, self._snapshot_storage_bytes(db, json.loads(row['snapshot']))) for row in rows[:limit]]
            return {'items': items, 'next_cursor': items[-1]['id'] if len(rows) > limit else None}

    def messages(self, conversation_id, before=None, limit=50):
        valid_id(conversation_id)
        if not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError('Límite de página inválido')
        if before is not None and (not isinstance(before, int) or before < 1):
            raise ValueError('Cursor de mensajes inválido')
        with self.connection() as db:
            rows = db.execute('SELECT id,sequence,kind,speaker,text,body,created_at FROM messages WHERE conversation_id=? AND sequence<? ORDER BY sequence DESC LIMIT ?',
                              (conversation_id, before or 2**63-1, limit + 1)).fetchall()
            page = [dict(row) for row in reversed(rows[:limit])]
            for item in page: item['body'] = json.loads(item['body'])
            return {'items': page, 'before': page[0]['sequence'] if len(rows) > limit else None}

    def update(self, conversation_id, revision, *, snapshot=None, title=None, pinned=None, archived=None, visibility=None):
        with self.media_operation():
            return self._update(conversation_id, revision, snapshot=snapshot, title=title, pinned=pinned, archived=archived, visibility=visibility)

    def _update(self, conversation_id, revision, *, snapshot=None, title=None, pinned=None, archived=None, visibility=None):
        valid_id(conversation_id)
        if not isinstance(revision, int) or isinstance(revision, bool):
            raise ValueError('Revisión obligatoria')
        if title is not None and (not isinstance(title, str) or not title.strip() or len(title) > 200):
            raise ValueError('Título inválido')
        if pinned is not None and not isinstance(pinned, bool):
            raise ValueError('Valor de fijado inválido')
        if archived is not None and not isinstance(archived, bool):
            raise ValueError('Valor de archivo inválido')
        if visibility is not None and visibility not in ('public', 'private'):
            raise ValueError('Visibilidad inválida')
        created, dedup = {}, {}
        try:
            if snapshot is not None:
                snapshot = self._normalize(self._read_snapshot(snapshot), created, dedup)
            with self.lock, self.connection() as db:
                row = db.execute('SELECT * FROM conversations WHERE id=?', (conversation_id,)).fetchone()
                if row is None: return None
                if row['revision'] != revision: raise Conflict('La conversación cambió en otra pestaña; recarga antes de guardar')
                old = json.loads(row['snapshot'])
                effective = snapshot if snapshot is not None else old
                timestamp = now()
                activity = timestamp if effective.get('view') != old.get('view') or title is not None or pinned is not None or archived is not None or visibility is not None else row['updated_at']
                db.execute('UPDATE conversations SET title=?, pinned=?, archived_at=?, snapshot=?, preview=?, visibility=?, revision=revision+1, updated_at=? WHERE id=?',
                           (title.strip() if title is not None else row['title'],
                            int(pinned) if pinned is not None else row['pinned'],
                            (timestamp if archived else None) if archived is not None else row['archived_at'],
                            json.dumps(effective, ensure_ascii=False), self._preview(effective),
                            visibility if visibility is not None else row['visibility'], activity, conversation_id))
                removed = self._update_refs(db, old, effective, created)
                if snapshot is not None: self._sync_messages(db, conversation_id, effective, timestamp)
            self._finalize_media(created)
            self._delete_media_files(removed)
            return self.get(conversation_id)
        except Exception:
            self._delete_media_files(created)
            raise

    def delete(self, conversation_id, revision):
        valid_id(conversation_id)
        with self.lock, self.connection() as db:
            row = db.execute('SELECT snapshot,revision FROM conversations WHERE id=?', (conversation_id,)).fetchone()
            if row is None: return False
            if row['revision'] != revision: raise Conflict('La conversación cambió en otra pestaña')
            removed = self._update_refs(db, json.loads(row['snapshot']), {}, {})
            db.execute('DELETE FROM conversations WHERE id=?', (conversation_id,))
        self._delete_media_files(removed)
        return True

    def media(self, media_id):
        valid_id(media_id)
        with self.connection() as db:
            row = db.execute('SELECT mime,size FROM media WHERE id=? AND refcount>0', (media_id,)).fetchone()
        if row is None: return None
        extension = {'image/png': '.png', 'image/jpeg': '.jpg', 'image/webp': '.webp','video/mp4':'.mp4'}.get(row['mime'])
        if extension is None: return None
        path = self.media_dir / (media_id + extension)
        if path.is_symlink() or not path.is_file() or path.resolve().parent != self.media_dir.resolve(): return None
        return path, row['mime']

    def public_video_ids(self):
        """Return public snapshots' explicit MiniMax output IDs only."""
        ids=set()
        with self.connection() as db:
            rows=db.execute("SELECT snapshot FROM conversations WHERE visibility='public'").fetchall()
        pattern=re.compile(r'/api/video/file\?id=([0-9a-f]{32})')
        for row in rows:ids.update(pattern.findall(row['snapshot']))
        return ids

    def expand_media(self, value):
        if isinstance(value, dict): return {key: self.expand_media(item) for key, item in value.items()}
        if isinstance(value, list): return [self.expand_media(item) for item in value]
        if isinstance(value, str):
            match = MEDIA_RE.fullmatch(value)
            if match:
                result = self.media(match.group(1))
                if result is None: raise ValueError('Medio de conversación no encontrado')
                path, mime = result
                return f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode()
        return value
