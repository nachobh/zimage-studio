"""Narrow asynchronous adapter from the panel to the local MiniMax-H3 client."""
from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, asdict
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import json
import hashlib
from urllib.request import Request, urlopen

from PIL import Image

ROOT = Path(__file__).resolve().parent
MAX_IMAGE = 16 * 1024 * 1024
MAX_PROMPT = 16 * 1024
JOB_TIMEOUT = int(os.environ.get("MINIMAX_H3_JOB_TIMEOUT", "7200"))
OUTPUT_DIR = Path(os.environ.get("MINIMAX_H3_OUTPUT_DIR", os.environ.get("ZIMAGE_DATA_DIR", str(ROOT / "data")))) / "videos"
CLIENT = Path(os.environ.get("MINIMAX_H3_CLIENT", str(ROOT / "gen_minimax_h3.py")))
STARTER = Path(os.environ.get("MINIMAX_H3_START_SCRIPT", str(ROOT / "levantar_minimax_h3.sh")))
PORT = int(os.environ.get("MINIMAX_H3_PORT", "9879"))
DEFAULT_INSTALL_ROOT = ROOT if (ROOT / "minimax-h3-venv" / "bin" / "python").is_file() else ROOT.parent
INSTALL_ROOT = Path(os.environ.get("MINIMAX_H3_INSTALL_ROOT", str(DEFAULT_INSTALL_ROOT)))
COMFYUI_DIR = Path(os.environ.get('MINIMAX_H3_COMFYUI_DIR',str(INSTALL_ROOT/'ComfyUI-minimax-h3')))
COMFYUI_OUTPUT_DIR = COMFYUI_DIR/'output'
PYTHON = Path(os.environ.get("MINIMAX_H3_PYTHON", str(INSTALL_ROOT / "minimax-h3-venv" / "bin" / "python")))
_lock = threading.Lock()
_queue_condition=threading.Condition(_lock)
_jobs: dict[str, dict] = {}
_request_jobs: dict[str, str] = {}
_request_fingerprints: dict[str, str] = {}
_active: str | None = None
_waiting: list[str] = []
MAX_WAITING=3
STARTUP_TIMEOUT = int(os.environ.get("MINIMAX_H3_STARTUP_TIMEOUT", "120"))
MIN_FREE_BYTES = int(os.environ.get("MINIMAX_H3_MIN_FREE_GB", "36")) * 1024**3
_TENSOR_MARKERS = ("tensor(", "device='cuda", "device='cpu", "current_inputs", "current_outputs")
JOB_STATE_DIR = OUTPUT_DIR / '.jobs'


class VideoQueueFull(RuntimeError):
    pass


def _persist_job(value: dict) -> None:
    """Persist only opaque IDs, status, timestamps, and output size; never prompts/media/chat IDs."""
    JOB_STATE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(JOB_STATE_DIR, 0o700)
    safe = {key: value[key] for key in ('id', 'request_id', 'prompt_id', 'prefix', 'recovery_required', 'status', 'created', 'updated', 'size') if key in value}
    target = JOB_STATE_DIR / (value['id'] + '.json')
    fd, temporary = tempfile.mkstemp(prefix='.job-', dir=JOB_STATE_DIR)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(safe, stream, separators=(',', ':'))
            stream.flush(); os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _restore_jobs() -> None:
    if not JOB_STATE_DIR.is_dir() or JOB_STATE_DIR.is_symlink(): return
    for path in JOB_STATE_DIR.glob('*.json'):
        if path.is_symlink() or not re.fullmatch(r'[0-9a-f]{32}\.json', path.name): continue
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(value, dict): continue
            job_id = value.get('id')
            if not re.fullmatch(r'[0-9a-f]{32}', job_id or ''): continue
            output = OUTPUT_DIR / (job_id + '.mp4')
            if output.is_file() and not output.is_symlink() and output.stat().st_size:
                value.update(status='done', recovery_required=False, video='/api/video/file?id=' + job_id, size=output.stat().st_size)
            elif value.get('status') in ('queued', 'executing'):
                can_reconcile=(re.fullmatch(r'[0-9a-f]{32}',value.get('prompt_id') or '')
                               and re.fullmatch(r'minimax_h3_[0-9a-f]{32}',value.get('prefix') or ''))
                value.update(status='interrupted', recovery_required=bool(can_reconcile),
                             error='El panel se reinició; se comprobará el ID exacto en ComfyUI antes de reintentar.' if can_reconcile else 'El panel se reinició antes de registrar el ID de ComfyUI; no se reenviará automáticamente.')
            _jobs[job_id] = value
            if re.fullmatch(r'[0-9a-f]{32}', value.get('request_id') or ''):
                _request_jobs[value['request_id']] = job_id
            _persist_job(value)
        except (OSError, ValueError, TypeError): continue


_restore_jobs()


def _api(path: str) -> str:
    return f"http://127.0.0.1:{PORT}{path}"


def _alive() -> bool:
    try:
        with urlopen(_api("/system_stats"), timeout=3) as response:
            return isinstance(json.load(response), dict)
    except Exception:
        return False


def _free_bytes(stats: dict) -> int | None:
    """Return free VRAM; ComfyUI's torch_vram_free is only allocator cache."""
    device_free = []
    torch_free = []
    for device in stats.get("devices") or ():
        if not isinstance(device, dict):
            continue
        value = device.get("vram_free")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            device_free.append(value)
        value = device.get("torch_vram_free")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            torch_free.append(value)
    # ComfyUI defines vram_free as device-free memory plus reclaimable PyTorch
    # allocator cache. torch_vram_free alone is just that cache and may be tiny
    # even when the device has plenty of room (notably ROCm).
    values = device_free or torch_free
    return int(max(values)) if values else None


def _start() -> bool:
    """Ask the launcher to bring ComfyUI up and wait until it answers."""
    if not STARTER.is_file():
        return False
    try:
        result = subprocess.run(["bash", str(STARTER)], cwd=STARTER.parent, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
                                timeout=STARTUP_TIMEOUT)
    except Exception:
        return False
    # A launcher that fails outright only gets a short grace period.
    budget = STARTUP_TIMEOUT if result.returncode == 0 else min(STARTUP_TIMEOUT, 15)
    deadline = time.time() + budget
    while time.time() < deadline:
        if _alive():
            return True
        time.sleep(2)
    return False


def _release_memory() -> None:
    """Drop cached models between jobs so VRAM does not accumulate."""
    request = Request(_api("/free"), data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=30) as response:
            response.read()
    except Exception:
        pass


def _looks_dead(detail: str) -> bool:
    lowered = detail.lower()
    return ("connection refused" in lowered or "errno 111" in lowered
            or "no se puede consultar comfyui" in lowered or "urlopen error" in lowered)


def _failure_text(raw: str) -> str:
    """Keep the useful last line of a traceback and drop tensor dumps."""
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or any(marker in line for marker in _TENSOR_MARKERS):
            continue
        lines.append(line[:300])
    if not lines:
        return ""
    structured = [line for line in lines if re.match(r"^[A-Za-z_.]*(Error|Exception|Timeout)\b", line)]
    return (structured or lines)[-1][:600]


def _require_idle_comfyui() -> None:
    try:
        with urlopen(_api("/queue"), timeout=3) as response:
            queue = json.load(response)
    except Exception as exc:
        raise RuntimeError(f"MiniMax-H3/ComfyUI no responde en 127.0.0.1:{PORT}; inicia con --with-comfyui") from exc
    if queue.get("queue_running") or queue.get("queue_pending"):
        raise RuntimeError("ComfyUI ya tiene un trabajo en curso; espera a que termine antes de generar vídeo")


def _preflight() -> None:
    """Bring ComfyUI up if needed and refuse clearly when there is no room."""
    if not _alive():
        if not _start():
            raise RuntimeError("MiniMax-H3 no está activo; arranca con `start.sh --with-comfyui` "
                               "o revisa el lanzador antes de reintentar")
    # Check first so /free cannot unload a model while another workflow runs.
    _require_idle_comfyui()
    # After an OOM ComfyUI may still hold the workflow's models and allocator
    # cache. Release them before applying the free-memory threshold.
    _release_memory()
    try:
        with urlopen(_api("/system_stats"), timeout=5) as response:
            stats = json.load(response)
    except Exception as exc:
        raise RuntimeError(f"MiniMax-H3 no responde en 127.0.0.1:{PORT}; vuelve a intentarlo en unos segundos") from exc
    free = _free_bytes(stats)
    if free is not None and MIN_FREE_BYTES and free < MIN_FREE_BYTES:
        raise RuntimeError(f"Memoria GPU libre insuficiente para MiniMax-H3: {free / 1024**3:.1f} GiB de "
                           f"{MIN_FREE_BYTES / 1024**3:.0f} GiB recomendados. Espera a que ComfyUI suelte "
                           "memoria, cierra otros usos de la GPU o ajusta MINIMAX_H3_MIN_FREE_GB.")


def _image_data(value: object) -> bytes:
    if not isinstance(value, str) or len(value) > (MAX_IMAGE * 4 // 3 + 128):
        raise ValueError("La imagen debe ser PNG/JPEG/WebP y no superar 16 MiB")
    match = re.fullmatch(r"data:image/(png|jpeg|webp);base64,([A-Za-z0-9+/]+={0,2})", value)
    if not match:
        raise ValueError("Formato de imagen de referencia no válido")
    try:
        data = base64.b64decode(match.group(2), validate=True)
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in ("PNG", "JPEG", "WEBP") or im.width < 1 or im.height < 1 or max(im.size) > 4096:
                raise ValueError("La referencia debe ser PNG/JPEG/WebP de hasta 4096 px")
            im.verify()
    except (binascii.Error, OSError) as exc:
        raise ValueError("La imagen de referencia está dañada") from exc
    if not data or len(data) > MAX_IMAGE:
        raise ValueError("La imagen debe ocupar entre 1 byte y 16 MiB")
    return data


def _resolution(body: dict) -> tuple[int, int]:
    """Read the optional output size; the panel default is 320x480."""
    values = (body.get("width", 320), body.get("height", 480))
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or not 32 <= value <= 1920 or value % 32:
            raise ValueError("width/height deben ser múltiplos de 32 entre 32 y 1920")
    return int(values[0]), int(values[1])


def submit(body: dict) -> dict:
    """Validate a request and enqueue exactly one bounded H3 workflow."""
    global _active
    request_id = body.get('request_id')
    if request_id is not None and (not isinstance(request_id, str) or not re.fullmatch(r'[0-9a-f]{32}', request_id)):
        raise ValueError('request_id debe ser un identificador hexadecimal de 32 caracteres')
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode("utf-8")) > MAX_PROMPT:
        raise ValueError("Escribe un prompt MiniMax-H3 (máximo 16 KiB)")
    audio = body.get("with_audio", True)
    if not isinstance(audio, bool):
        raise ValueError("with_audio debe ser booleano")
    width, height = _resolution(body)
    image = _image_data(body.get("image_data"))
    fingerprint = hashlib.sha256(prompt.encode('utf-8') + b'\0' + image + b'\0' + str((audio, width, height)).encode()).hexdigest()
    if request_id:
        with _lock:
            previous = _request_jobs.get(request_id)
            if previous:
                if request_id in _request_fingerprints and _request_fingerprints[request_id] != fingerprint:
                    raise ValueError('request_id ya se usó con un contenido distinto')
                existing = _jobs.get(previous)
                if existing: return {'status': existing['status'], 'job_id': previous, 'idempotent': True}
    if not CLIENT.is_file() or not STARTER.is_file():
        raise RuntimeError("Faltan los scripts de MiniMax-H3 en esta copia del proyecto")
    if not PYTHON.is_file():
        raise RuntimeError(f"Falta el entorno MiniMax-H3: {PYTHON}. Configura MINIMAX_H3_INSTALL_ROOT o MINIMAX_H3_PYTHON")
    with _queue_condition:
        if len(_waiting)+(1 if _active else 0)>=MAX_WAITING+1:
            raise VideoQueueFull('La cola de vídeo está llena (máximo 3 en espera).')
        job_id = uuid.uuid4().hex
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(OUTPUT_DIR, 0o700)
        job_dir = OUTPUT_DIR / ("." + job_id)
        job_dir.mkdir(mode=0o700)
        image_path = job_dir / "reference.png"
        prompt_path = job_dir / "prompt.txt"
        image_path.write_bytes(image)
        prompt_path.write_text(prompt, encoding="utf-8")
        output_path = OUTPUT_DIR / (job_id + ".mp4")
        now = time.time()
        _jobs[job_id] = {"id": job_id, "request_id":request_id, "status": "queued", "created": now, "updated":now, "error": None, "video": None}
        if request_id:
            _request_jobs[request_id] = job_id
            _request_fingerprints[request_id] = fingerprint
        _persist_job(_jobs[job_id])
        _waiting.append(job_id)
        position=len(_waiting)
        _queue_condition.notify_all()
    threading.Thread(target=_run, args=(job_id, image_path, prompt_path, output_path, audio, width, height), daemon=True).start()
    return {"status": "queued", "job_id": job_id,'queue_position':position}


def _run(job_id: str, image: Path, prompt: Path, output: Path, audio: bool, width: int = 320, height: int = 480) -> None:
    global _active
    with _queue_condition:
        while True:
            record=_jobs.get(job_id)
            if not record or record.get('status')=='cancelled':return
            if _active is None and _waiting and _waiting[0]==job_id:
                _waiting.pop(0);_active=job_id
                record.update(status='executing',updated=time.time());_persist_job(record)
                _queue_condition.notify_all();break
            if job_id not in _waiting:return
            _queue_condition.wait()
    env = os.environ.copy()
    env["MINIMAX_H3_PORT"] = str(PORT)
    env.setdefault("MINIMAX_H3_INSTALL_ROOT", str(INSTALL_ROOT))
    env.setdefault("MINIMAX_H3_COMFYUI_DIR", str(INSTALL_ROOT / "ComfyUI-minimax-h3"))
    status_path=image.parent/'comfyui.json'
    env['MINIMAX_H3_STATUS_FILE']=str(status_path)
    command = [str(PYTHON), str(CLIENT), str(image), "--prompt-file", str(prompt), "--output", str(output),
               "--width", str(width), "--height", str(height), "--steps", "8", "--seconds", "5.0", "--timeout", str(JOB_TIMEOUT)]
    if audio:
        command.append("--with-audio")
    try:
        _preflight()
        # No shell, prompt is read from a private temporary file, never logged.
        detail = ""
        with tempfile.TemporaryFile() as diagnostics:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                     stdout=subprocess.DEVNULL,stderr=diagnostics)
            deadline=time.monotonic()+JOB_TIMEOUT+60
            while process.poll() is None:
                try:
                    if status_path.is_file() and not status_path.is_symlink() and status_path.stat().st_size<=4096:
                        status=json.loads(status_path.read_text(encoding='utf-8'))
                        prompt_id=status.get('prompt_id') if isinstance(status,dict) else None
                        prefix=status.get('prefix') if isinstance(status,dict) else None
                        if re.fullmatch(r'[0-9a-f]{32}',prompt_id or '') and re.fullmatch(r'minimax_h3_[0-9a-f]{32}',prefix or ''):
                            with _lock:
                                _jobs[job_id].update(prompt_id=prompt_id,prefix=prefix,updated=time.time())
                                _persist_job(_jobs[job_id])
                except (OSError,ValueError,TypeError):pass
                if time.monotonic()>=deadline:
                    process.kill();process.wait()
                    raise subprocess.TimeoutExpired(command,JOB_TIMEOUT+60)
                time.sleep(.5)
            if process.returncode != 0:
                # ComfyUI dumps str(tensor) on failure; read only a bounded tail and keep the useful line.
                diagnostics.seek(0, io.SEEK_END)
                size = diagnostics.tell()
                diagnostics.seek(max(0, size - 65536))
                detail = _failure_text(diagnostics.read().decode("utf-8", "replace"))
        if process.returncode != 0:
            if _looks_dead(detail):
                threading.Thread(target=_start, daemon=True).start()
                raise RuntimeError("ComfyUI se detuvo durante la generación (OOM o caída); "
                                   "se relanzará en el siguiente intento.")
            raise RuntimeError(detail or "El generador MiniMax-H3 terminó con error")
        if output.is_symlink() or not output.is_file() or output.stat().st_size <= 0:
            raise RuntimeError("MiniMax-H3 no produjo un vídeo válido")
        with _lock:
            _jobs[job_id].update(status="done", video="/api/video/file?id=" + job_id, size=output.stat().st_size, updated=time.time())
            _persist_job(_jobs[job_id])
    except subprocess.TimeoutExpired:
        with _lock:
            _jobs[job_id].update(status="error", error="La generación excedió el tiempo máximo. Comprueba ComfyUI antes de reintentar.", updated=time.time())
            _persist_job(_jobs[job_id])
    except Exception as exc:
        with _lock:
            _jobs[job_id].update(status="error", error=str(exc)[:1500], updated=time.time())
            _persist_job(_jobs[job_id])
    finally:
        image.unlink(missing_ok=True)
        prompt.unlink(missing_ok=True)
        status_path.unlink(missing_ok=True)
        image.parent.rmdir()
        with _queue_condition:
            if _active == job_id:
                _active = None
            _queue_condition.notify_all()


def job(job_id: str = '', request_id: str = '') -> dict | None:
    if bool(job_id) == bool(request_id): return None
    if request_id:
        if not re.fullmatch(r'[0-9a-f]{32}', request_id): return None
        with _lock: job_id = _request_jobs.get(request_id, '')
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        return None
    with _lock:
        value = _jobs.get(job_id)
        if value is None:
            # MP4 is durable across panel refreshes/restarts, job metadata is not.
            path = OUTPUT_DIR / (job_id + ".mp4")
            if path.is_file() and not path.is_symlink():
                return {"id": job_id, "status": "done", "video": "/api/video/file?id=" + job_id, "size": path.stat().st_size}
            return None
        result=dict(value)
        if result.get('status')=='queued' and job_id in _waiting:result['queue_position']=_waiting.index(job_id)+1
    if result.get('recovery_required') and result.get('prompt_id'):
        _reconcile_comfyui(job_id,result)
        with _lock:return dict(_jobs.get(job_id,result))
    return result


def cancel_job(job_id: str) -> dict:
    if not re.fullmatch(r'[0-9a-f]{32}',job_id or ''):raise ValueError('ID de trabajo inválido.')
    with _queue_condition:
        record=_jobs.get(job_id)
        if record is None:raise FileNotFoundError('Trabajo de vídeo no encontrado.')
        if record.get('status')!='queued' or job_id not in _waiting:
            raise RuntimeError('No se puede cancelar este trabajo en curso; no se enviará un interrupt global a ComfyUI.')
        _waiting.remove(job_id);record.update(status='cancelled',error=None,updated=time.time());_persist_job(record)
        _queue_condition.notify_all()
    job_dir=OUTPUT_DIR/('.job-'+job_id)
    if job_dir.is_dir() and not job_dir.is_symlink() and job_dir.resolve().parent==OUTPUT_DIR.resolve():
        for name in ('reference.png','prompt.txt','comfyui.json'):
            target=job_dir/name
            if target.is_file() and not target.is_symlink():target.unlink()
        try:job_dir.rmdir()
        except OSError:pass
    return dict(record)


def _queue_has_prompt(value, prompt_id: str) -> bool:
    if isinstance(value,str):return value==prompt_id
    if isinstance(value,(list,tuple)):return any(_queue_has_prompt(item,prompt_id) for item in value)
    if isinstance(value,dict):return any(_queue_has_prompt(item,prompt_id) for item in value.values())
    return False


def _publish_recovered_video(job_id: str, source: Path) -> Path:
    if source.is_symlink() or not source.is_file() or source.resolve().parent!=COMFYUI_OUTPUT_DIR.resolve() or source.stat().st_size<=0:
        raise ValueError('ComfyUI no devolvió un MP4 local válido.')
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True,mode=0o700);os.chmod(OUTPUT_DIR,0o700)
    target=OUTPUT_DIR/(job_id+'.mp4')
    fd,temporary=tempfile.mkstemp(prefix='.recover-',dir=OUTPUT_DIR)
    try:
        with os.fdopen(fd,'wb') as out,source.open('rb') as inp:shutil.copyfileobj(inp,out,1024*1024);out.flush();os.fsync(out.fileno())
        os.chmod(temporary,0o600)
        try:os.link(temporary,target)
        except FileExistsError:raise ValueError('El destino del vídeo recuperado ya existe.')
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
    return target


def _reconcile_comfyui(job_id: str, saved: dict) -> None:
    prompt_id=saved.get('prompt_id');prefix=saved.get('prefix')
    if not re.fullmatch(r'[0-9a-f]{32}',prompt_id or '') or not re.fullmatch(r'minimax_h3_[0-9a-f]{32}',prefix or ''):return
    try:
        with urlopen(_api('/history/'+prompt_id),timeout=5) as response:history=json.load(response)
        item=history.get(prompt_id) if isinstance(history,dict) else None
        if item:
            status=item.get('status') or {};status_name=status.get('status_str')
            if status_name=='error':
                error=RuntimeError('ComfyUI marcó el trabajo como fallido')
                with _lock:_jobs[job_id].update(status='error',recovery_required=False,error=str(error),updated=time.time());_persist_job(_jobs[job_id])
                return
            if status.get('completed') or status_name in ('success','completed'):
                candidates=list(COMFYUI_OUTPUT_DIR.glob(prefix+'_*.mp4'))
                if len(candidates)!=1:raise ValueError('ComfyUI terminó pero no hay un único MP4 para el prefijo guardado.')
                target=_publish_recovered_video(job_id,candidates[0])
                with _lock:_jobs[job_id].update(status='done',recovery_required=False,video='/api/video/file?id='+job_id,size=target.stat().st_size,updated=time.time());_persist_job(_jobs[job_id])
                return
        with urlopen(_api('/queue'),timeout=5) as response:queue=json.load(response)
        if _queue_has_prompt(queue,prompt_id):
            with _lock:_jobs[job_id].update(status='executing',recovery_required=True,error=None,updated=time.time());_persist_job(_jobs[job_id])
            return
        with _lock:_jobs[job_id].update(status='interrupted',recovery_required=False,error='ComfyUI no muestra este ID en historial ni en cola; revisa manualmente antes de reintentar.',updated=time.time());_persist_job(_jobs[job_id])
    except Exception:
        # An unavailable ComfyUI is not proof that the job stopped. Keep the
        # interrupted marker and never issue a replacement POST automatically.
        return


def video_path(job_id: str) -> Path | None:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        return None
    path = OUTPUT_DIR / (job_id + ".mp4")
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0 or path.resolve().parent != OUTPUT_DIR.resolve():
        return None
    return path


def import_video(content: bytes) -> str:
    """Publish a validated archive MP4 under a fresh opaque ID; no inference."""
    if not isinstance(content,bytes) or not content or len(content)>64*1024*1024 or b'ftyp' not in content[:32]:
        raise ValueError('Vídeo importado inválido o superior a 64 MiB')
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True,mode=0o700);os.chmod(OUTPUT_DIR,0o700)
    fd,temporary=tempfile.mkstemp(prefix='.import-',dir=OUTPUT_DIR)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(content);stream.flush();os.fsync(stream.fileno())
        os.chmod(temporary,0o600)
        while True:
            job_id=uuid.uuid4().hex;target=OUTPUT_DIR/(job_id+'.mp4')
            try:os.link(temporary,target);break
            except FileExistsError:continue
        os.unlink(temporary)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
    now=time.time();record={'id':job_id,'status':'done','created':now,'updated':now,'size':len(content),'video':'/api/video/file?id='+job_id,'error':None}
    with _lock:_jobs[job_id]=record;_persist_job(record)
    return job_id


def audit_media(public_references: set[str] | None = None) -> dict:
    """Report video files without deleting them; private browser references are unknowable here."""
    known=public_references or set();files=[]
    if OUTPUT_DIR.is_dir() and not OUTPUT_DIR.is_symlink():
        for path in sorted(OUTPUT_DIR.glob('*.mp4')):
            if path.is_symlink() or not path.is_file():continue
            job_id=path.stem
            files.append({'id':job_id,'size':path.stat().st_size,
                          'public_referenced':job_id in known,
                          'classification':'referenced_public' if job_id in known else 'reference_unknown'})
    return {'files':files,'count':len(files),'bytes':sum(item['size'] for item in files),
            'note':'Los MP4 sin referencia pública pueden pertenecer a chats privados de IndexedDB; no se eliminan.'}
