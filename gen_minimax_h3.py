#!/usr/bin/env python3
"""Envía un trabajo MiniMax-H3 local a ComfyUI, con modo guiado opcional."""
import argparse
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib import parse, request
import uuid

try:
    import av
except ImportError:  # CLI remains usable for the no-audio path without PyAV.
    av=None
from PIL import Image


PORT = int(os.environ.get("MINIMAX_H3_PORT", "9879"))
if not 1024 <= PORT <= 65535:
    raise ValueError("MINIMAX_H3_PORT debe estar entre 1024 y 65535")
if PORT == 9877:
    raise ValueError("El puerto 9877 está reservado a Z-Image")
API = f"http://127.0.0.1:{PORT}"
OUTPUT_DIR = Path(os.environ.get("MINIMAX_H3_COMFYUI_DIR", "~/ComfyUI-minimax-h3")) / "output"
MAX_IMAGE_BYTES = 32 * 1024 * 1024


def get_json(path):
    try:
        with request.urlopen(API + path, timeout=15) as response:
            return json.load(response)
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"No se puede consultar ComfyUI en {API}: {exc}") from exc


def post_json(path, payload):
    data = json.dumps(payload).encode()
    req = request.Request(API + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except HTTPError as exc:
        detail = exc.read(2048).decode("utf-8", "replace")
        raise RuntimeError(f"ComfyUI rechazó el grafo HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"No se puede conectar con {API}: {exc}") from exc


def upload(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_IMAGE_BYTES:
        raise ValueError("La imagen debe ser un archivo regular de hasta 32 MiB")
    with Image.open(path) as image:
        if image.format not in ("PNG", "JPEG", "WEBP"):
            raise ValueError("La imagen debe ser PNG, JPEG o WebP")
        extension = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}[image.format]
        image.verify()
    boundary = "----minimax" + uuid.uuid4().hex
    name = "minimax_h3_input_" + uuid.uuid4().hex + "." + extension
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
        'Content-Type: application/octet-stream\r\n\r\n'
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = request.Request(
        API + "/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with request.urlopen(req, timeout=120) as response:
            return json.load(response)["name"]
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Falló la carga de imagen en ComfyUI: {exc}") from exc


def build_prompt(uploaded, args, seed, prefix, last_uploaded=None):
    length = max(5, round(args.seconds * 24))
    length += (5 - length % 17) % 17
    prompt = {
        "1": {"class_type": "UnetLoaderGGUFAdvanced", "inputs": {
            "unet_name": "MiniMax-H3-FL2VA-Pruned-Q4_K_M.gguf",
            "dequant_dtype": "default", "patch_dtype": "default", "patch_on_device": True}},
        "2": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "6": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["2", 0], "vae": ["3", 0], "prompt": args.prompt,
            "width": args.width, "height": args.height, "length": length}},
        "7": {"class_type": "MiniMaxH3SigmaShift", "inputs": {
            "model": ["1", 0], "shift_video": 12.0, "shift_audio": 3.0}},
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "9": {"class_type": "BasicGuider", "inputs": {"model": ["7", 0], "conditioning": ["6", 0]}},
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "11": {"class_type": "BasicScheduler", "inputs": {
            "model": ["7", 0], "scheduler": "simple", "steps": args.steps, "denoise": 1.0}},
        "12": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["8", 0], "guider": ["9", 0], "sampler": ["10", 0],
            "sigmas": ["11", 0], "latent_image": ["6", 1]}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
        "15": {"class_type": "CreateVideo", "inputs": {"images": ["13", 0], "fps": 24.0}},
        "16": {"class_type": "SaveVideo", "inputs": {
            "video": ["15", 0], "filename_prefix": prefix, "format": "auto", "codec": "auto"}},
        "18": {"class_type": "ImageFromBatch", "inputs": {
            "image": ["13", 0], "batch_index": 0, "length": 1}},
        "17": {"class_type": "SaveImage", "inputs": {
            "images": ["18", 0], "filename_prefix": prefix + "_debug"}},
    }
    if uploaded:
        prompt["5"] = {"class_type": "LoadImage", "inputs": {"image": uploaded}}
        prompt["6"]["inputs"]["first_frame"] = ["5", 0]
    if last_uploaded:
        prompt["19"] = {"class_type": "LoadImage", "inputs": {"image": last_uploaded}}
        prompt["6"]["inputs"]["last_frame"] = ["19", 0]
    if args.with_audio:
        prompt["4"] = {"class_type": "VAELoader", "inputs": {
            "vae_name": "minimax_h3_audio_vae_fp32.safetensors"}}
        prompt["14"] = {"class_type": "VAEDecodeAudio", "inputs": {
            "samples": ["12", 0], "vae": ["4", 0]}}
        prompt["15"]["inputs"]["audio"] = ["14", 0]
    return prompt, length


def describe_failure(prompt_id, status):
    """Reduce a ComfyUI execution error to one useful line, without tensor dumps."""
    detail = {}
    for entry in (status.get("messages") or []):
        if isinstance(entry, (list, tuple)) and len(entry) > 1 and entry[0] == "execution_error" and isinstance(entry[1], dict):
            detail = entry[1]
            break
    kind = str(detail.get("exception_type") or "").strip()
    raw = str(detail.get("exception_message") or "")
    first = raw.splitlines()[0].strip()[:300] if raw else ""
    node = str(detail.get("node_id") or "").strip()
    haystack = f"{kind} {first}".lower()
    if "outofmemory" in haystack or "out of memory" in haystack:
        where = f" (nodo {node})" if node else ""
        return (f"Memoria GPU insuficiente para MiniMax-H3{where}: {first or 'sin memoria suficiente'}. "
                "Espera a que ComfyUI suelte los modelos o libera memoria antes de reintentar.")
    parts = [part for part in (kind, first) if part]
    if node:
        parts.append(f"nodo {node}")
    if parts:
        return "ComfyUI falló: " + " · ".join(parts)
    return f"ComfyUI falló en el trabajo {prompt_id}"


def wait_for_result(prompt_id, prefix, timeout):
    started = time.monotonic()
    deadline = started + timeout
    next_notice = started + 30
    while time.monotonic() < deadline:
        history = get_json("/history/" + parse.quote(prompt_id, safe=""))
        job = history.get(prompt_id)
        if job:
            status = job.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(describe_failure(prompt_id, status))
            if status.get("completed") or status.get("status_str") in ("success", "completed"):
                candidates = list(OUTPUT_DIR.glob(prefix + "_*.mp4"))
                if len(candidates) != 1 or candidates[0].is_symlink() or not candidates[0].is_file():
                    raise RuntimeError(f"El trabajo {prompt_id} no produjo un único MP4 con prefijo {prefix}")
                source = candidates[0]
                if source.stat().st_size == 0:
                    raise RuntimeError(f"El MP4 del trabajo {prompt_id} está vacío")
                return source
        if time.monotonic() >= next_notice:
            print(f"Trabajo {prompt_id} sigue en curso ({int(time.monotonic() - started)} s)", flush=True)
            next_notice = time.monotonic() + 30
        time.sleep(2)
    raise TimeoutError(f"Trabajo {prompt_id} excedió {timeout}s; consulta /history/{prompt_id} antes de reintentar")


def copy_new(source, target):
    target = Path(target).expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        shutil.copyfile(source, temporary)
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def write_internal_job_status(prompt_id, prefix):
    """Write only ComfyUI reconciliation identifiers to the panel-owned job dir."""
    raw=os.environ.get('MINIMAX_H3_STATUS_FILE')
    if not raw:return
    target=Path(raw)
    if (target.name!='comfyui.json' or not re.fullmatch(r'\.job-[0-9a-f]{32}',target.parent.name)
        or not re.fullmatch(r'[0-9a-f]{32}',str(prompt_id))
        or not re.fullmatch(r'minimax_h3_[0-9a-f]{32}',str(prefix))):
        raise ValueError('Destino o identificadores internos de estado inválidos')
    target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix='.state-',dir=target.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            json.dump({'prompt_id':prompt_id,'prefix':prefix},stream,separators=(',',':'));stream.flush();os.fsync(stream.fileno())
        os.chmod(temporary,0o600);os.replace(temporary,target)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def report_audio(path):
    if av is None:raise RuntimeError('La inspección de la pista de audio requiere PyAV (módulo av).')
    with av.open(path) as container:
        if not container.streams.audio:
            print("Aviso: el MP4 no contiene pista de audio.")
            return
        peak = 0.0
        for frame in container.decode(audio=0):
            samples = frame.to_ndarray()
            peak = max(peak, abs(float(samples.min())), abs(float(samples.max())))
    if peak < 0.01:
        print("Aviso: la pista de audio es casi silenciosa; prueba un prompt audiovisual más específico.")


def ask(label, default=None):
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{label}{suffix}: ").strip()
    return value if value else default


def ask_yes_no(label):
    while True:
        answer = ask(label + " (s/n)", "n").lower()
        if answer in ("s", "si", "sí"):
            return True
        if answer in ("n", "no"):
            return False
        print("Responde s o n.")


def guided_values():
    """Recoge una escena audiovisual sin enviar nada hasta validar los datos."""
    print("MiniMax-H3 · generación guiada (Intro omite campos opcionales).")
    first = ask("1) Imagen de origen (ruta, o Intro para ninguna)")
    scene = ask("2) Prompt de la escena")
    while not scene:
        print("El prompt es obligatorio.")
        scene = ask("2) Prompt de la escena")
    while True:
        resolution = ask("3) Resolución ancho x alto", "320x480")
        try:
            width, height = (int(part.strip()) for part in resolution.lower().replace("×", "x").split("x"))
        except (ValueError, TypeError):
            print("Usa el formato 320x480.")
            continue
        if all(32 <= d <= 1920 and d % 32 == 0 for d in (width, height)):
            break
        print("Ancho y alto deben ser múltiplos de 32 entre 32 y 1920.")
    while True:
        value = ask("4) Duración en segundos", "5")
        try:
            seconds = float(value.replace(",", "."))
        except ValueError:
            seconds = 0
        if 0.2 <= seconds <= 15:
            break
        print("La duración debe estar entre 0,2 y 15 segundos.")
    voice = ask_yes_no("5) ¿Incluir voz del personaje?")
    speech = None
    if voice:
        while not speech:
            speech = ask("   ¿Qué dirá? (escribe la frase en español)")
            if not speech:
                print("Indica la frase o vuelve a elegir sin voz.")
    music = ask_yes_no("6) ¿Incluir música de fondo?")
    music_description = None
    if music:
        while not music_description:
            music_description = ask("   Describe la música (género, instrumentos, ambiente)")
    last = ask("7) Imagen final (ruta, o Intro para ninguna)")
    # El formato I2VA probado usa secciones explícitas y etiqueta de diálogo.
    picture = "<Picture 1> aparece como primer fotograma. " if first else ""
    final_number = 2 if first else 1
    final = f"<Picture {final_number}> aparece como último fotograma. " if last else ""
    dialogue = f" El personaje dice: <d>[Spanish] {speech}</d>. Sus labios se sincronizan con la frase." if speech else ""
    soundscape = "Voz clara del personaje y sonido ambiente natural." if voice else "Sonido ambiente natural."
    music_line = music_description if music else "N/A"
    prompt = (f"For the target video, {picture}{final}\n\n"
              f"integrated_multimodal_description: [Shot 1] {scene}.{dialogue}\n\n"
              f"overall_soundscape: {soundscape}\n\n"
              f"non_diegetic_music: {music_line}.")
    return argparse.Namespace(image=first, last_image=last, prompt=prompt, prompt_file=None,
                              output=f"minimax_h3_{uuid.uuid4().hex[:12]}.mp4",
                              seed=0, steps=8, width=width, height=height,
                              seconds=seconds, timeout=3600, with_audio=voice or music)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", nargs="?", help="Imagen inicial; omite con --guided")
    ap.add_argument("prompt", nargs="?")
    ap.add_argument("--guided", action="store_true", help="Pregunta escena, resolución, audio e imágenes")
    ap.add_argument("--last-image", type=Path, help="Imagen que debe guiar el fotograma final")
    ap.add_argument("--prompt-file", type=Path, help="Lee un prompt H3 multilínea UTF-8")
    ap.add_argument("-o", "--output", default="minimax_h3.mp4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=544)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--with-audio", action="store_true", help="Decodifica y añade audio nativo; ruta experimental")
    args = ap.parse_args()
    if args.guided:
        if len(sys.argv) > 2:
            ap.error("--guided se usa sin otros argumentos")
        if not sys.stdin.isatty():
            ap.error("--guided requiere un terminal interactivo")
        try:
            args = guided_values()
        except (EOFError, KeyboardInterrupt):
            print("\nGeneración cancelada.")
            return
    elif not args.image:
        ap.error("Indica una imagen inicial o usa --guided")
    if args.prompt_file:
        if args.prompt:
            ap.error("Usa prompt posicional o --prompt-file, no ambos")
        if not args.prompt_file.is_file() or args.prompt_file.stat().st_size > 16 * 1024:
            ap.error("El archivo de prompt debe existir y no superar 16 KiB")
        args.prompt = args.prompt_file.read_text(encoding="utf-8")
    if not args.prompt or not args.prompt.strip():
        ap.error("Indica un prompt o --prompt-file")
    if not 1 <= args.steps <= 50 or not 0.2 <= args.seconds <= 15 or not 30 <= args.timeout <= 86400:
        ap.error("steps 1–50, seconds 0.2–15 y timeout 30–86400 son obligatorios")
    if not all(32 <= dimension <= 1920 and dimension % 32 == 0 for dimension in (args.width, args.height)):
        ap.error("width/height deben ser múltiplos de 32 entre 32 y 1920")
    if Path(args.output).expanduser().exists():
        ap.error(f"La salida ya existe: {args.output}")

    for label, path in (("origen", args.image), ("final", args.last_image)):
        if path:
            path = Path(path).expanduser()
            if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_IMAGE_BYTES:
                ap.error(f"La imagen {label} debe ser un archivo regular de hasta 32 MiB: {path}")

    seed = args.seed or random.randrange(1, 2**63)
    prefix = "minimax_h3_" + uuid.uuid4().hex
    uploaded = upload(Path(args.image).expanduser()) if args.image else None
    last_uploaded = upload(Path(args.last_image).expanduser()) if args.last_image else None
    graph, length = build_prompt(uploaded, args, seed, prefix, last_uploaded)
    response = post_json("/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())})
    prompt_id = response["prompt_id"]
    try:write_internal_job_status(prompt_id,prefix)
    except (OSError,ValueError) as exc:print(f"No se pudo guardar el identificador de reconciliación: {exc}",file=sys.stderr,flush=True)
    print(f"Trabajo {prompt_id}: seed={seed}, {args.width}x{args.height}, {length} frames", flush=True)
    source = wait_for_result(prompt_id, prefix, args.timeout)
    target = copy_new(source, args.output)
    print(f"Guardado: {target}")
    if args.with_audio:
        report_audio(target)
    print(f"Frame de diagnóstico: {OUTPUT_DIR / (prefix + '_debug_00001_.png')}")


if __name__ == "__main__":
    main()
