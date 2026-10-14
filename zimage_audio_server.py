#!/usr/bin/env python3
"""Optional CPU Kokoro + faster-whisper service. Models load on first use."""
import argparse
import base64
import io
import json
import os
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class AudioEngine:
    def __init__(self, whisper_model='small'):
        self.whisper_model = whisper_model
        self.tts = None
        self.stt = None
        self.lock = threading.Lock()

    def torch_device(self):
        """Selecciona GPU si PyTorch la ve; se puede forzar con ZIMAGE_AUDIO_DEVICE."""
        import torch
        requested = os.environ.get('ZIMAGE_AUDIO_DEVICE', 'auto').strip().lower()
        if requested in ('cpu', 'cuda'):
            if requested == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('ZIMAGE_AUDIO_DEVICE=cuda, pero PyTorch no detecta una GPU.')
            return requested
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def speech(self, body):
        engine = body.get('engine', 'kokoro')
        if engine not in ('kokoro', 'kokoclone', 'kokoclone_direct', 'qwen3tts'):
            raise ValueError('Motor de voz inválido.')
        reference = body.get('reference_audio', '')
        if engine.startswith('kokoclone') or engine == 'qwen3tts':
            data = base64.b64decode(reference, validate=True)
            if not 0 < len(data) <= 20 * 1024**2:
                raise ValueError('Referencia WAV vacía o mayor de 20 MB.')
        text = body.get('input', '')
        voice = body.get('voice', 'ef_dora')
        speed = body.get('speed', 1)
        if not isinstance(text, str) or not 0 < len(text.strip()) <= 5000:
            raise ValueError('Usa entre 1 y 5000 caracteres para la voz.')
        if voice not in ('ef_dora', 'em_alex', 'em_santa'):
            raise ValueError('Voz española no válida: ef_dora, em_alex o em_santa.')
        if not isinstance(speed, (int, float)) or not .5 <= speed <= 2:
            raise ValueError('Velocidad entre 0.5 y 2.')
        if engine == 'kokoclone_direct':
            # Compatibility mode: KokoClone chooses its own internal base voice.
            return self.kokoclone('/tts', {'input':text, 'lang':'es', 'reference':reference})
        if engine == 'kokoclone':
            # Keep Kokoro's selected Spanish voice/prosody, then transfer only
            # the target timbre.  KokoClone's internal Spanish base is im_nicola
            # and does not know the panel's Dora/Alex/Santa selection.
            source = self.kokoro_speech(text, voice, speed)
            return self.kokoclone('/convert', {'source':base64.b64encode(source).decode(), 'reference':reference})
        if engine == 'qwen3tts':
            try:
                return self.qwen3tts({'input':text, 'reference_audio':reference,
                                      'reference_text':body.get('reference_text', '')})
            except Exception:
                # Qwen3-TTS is the preferred voice engine.  Keep speech usable
                # when its isolated trial is unavailable by falling back first
                # to the established clone, then to its Kokoro base voice.
                source = self.kokoro_speech(text, voice, speed)
                try:
                    return self.kokoclone('/convert', {
                        'source':base64.b64encode(source).decode(), 'reference':reference})
                except Exception:
                    return source
        return self.kokoro_speech(text, voice, speed)

    def qwen3tts(self, body):
        """Forward only validated profile-derived audio to the isolated trial."""
        url = os.environ.get('ZIMAGE_QWEN3_TTS_URL', 'http://127.0.0.1:8770').rstrip('/')
        request = Request(url + '/tts', data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
        try:
            with urlopen(request, timeout=600) as response:
                result = response.read(32 * 1024**2 + 1)
            if len(result) > 32 * 1024**2 or result[:4] != b'RIFF':
                raise ValueError('Qwen3-TTS no devolvió un WAV válido.')
            return result
        except HTTPError as exc:
            raise ValueError('Qwen3-TTS: ' + exc.read(2000).decode(errors='replace')) from exc

    def kokoclone(self, path, body):
        url = os.environ.get('ZIMAGE_KOKOCLONE_URL', 'http://127.0.0.1:8769').rstrip('/')
        request = Request(url + path, data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
        try:
            with urlopen(request, timeout=600) as response:
                result = response.read(32 * 1024**2 + 1)
            if len(result) > 32 * 1024**2 or result[:4] != b'RIFF':
                raise ValueError('KokoClone no devolvió un WAV válido.')
            return result
        except HTTPError as exc:
            raise ValueError('KokoClone: ' + exc.read(2000).decode(errors='replace')) from exc

    def kokoro_speech(self, text, voice, speed):
        with self.lock:
            import numpy as np
            import soundfile as sf
            if self.tts is None:
                import torch
                torch.set_num_threads(4)
                from kokoro import KPipeline
                device = self.torch_device()
                print(f'Kokoro: usando {device.upper()}', flush=True)
                self.tts = KPipeline(lang_code='e', device=device, repo_id='hexgrad/Kokoro-82M')
            chunks = [audio.numpy() if hasattr(audio, 'numpy') else audio
                      for _, _, audio in self.tts(text, voice=voice, speed=speed)]
            if not chunks:
                raise ValueError('Kokoro no devolvió audio.')
            output = io.BytesIO()
            sf.write(output, np.concatenate(chunks), 24000, format='WAV')
            result = output.getvalue()
            return result

    def transcribe(self, body):
        raw = base64.b64decode(body.get('audio', ''), validate=True)
        if not 0 < len(raw) <= 20 * 1024**2:
            raise ValueError('Audio vacío o mayor de 20 MB.')
        language = body.get('language', 'es')
        if language not in ('es', 'en', 'auto'):
            raise ValueError('Idioma no válido.')
        with self.lock:
            if self.stt is None:
                from faster_whisper import WhisperModel
                # faster-whisper/CTranslate2 no usa ROCm de forma fiable; solo se
                # activa CUDA si se solicita explícitamente y la librería lo admite.
                stt_device = os.environ.get('ZIMAGE_STT_DEVICE', 'cpu').strip().lower()
                compute = 'float16' if stt_device == 'cuda' else 'int8'
                self.stt = WhisperModel(self.whisper_model, device=stt_device,
                                        compute_type=compute, cpu_threads=4)
            # Decode once to enforce a duration limit before transcription.
            from faster_whisper.audio import decode_audio
            audio = decode_audio(io.BytesIO(raw), sampling_rate=16000)
            if len(audio) > 16000 * 180:
                raise ValueError('Máximo 3 minutos de audio.')
            segments, _ = self.stt.transcribe(audio, language=None if language == 'auto' else language,
                                             vad_filter=True, beam_size=1)
            return {'text': ' '.join(segment.text.strip() for segment in segments).strip()}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # Do not log transcripts, text or audio.

    def reply(self, data, content_type='application/json', code=200):
        if content_type == 'application/json':
            data = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/health':
            return self.reply({'status':'ok', 'tts_loaded':self.server.engine.tts is not None,
                               'stt_loaded':self.server.engine.stt is not None})
        self.reply({'error':'Ruta desconocida'}, code=404)

    def do_POST(self):
        try:
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 28 * 1024**2:
                raise ValueError('Petición de audio demasiado grande o vacía.')
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError('Petición inválida.')
            if self.path == '/tts':
                return self.reply(self.server.engine.speech(body), 'audio/wav')
            if self.path == '/stt':
                return self.reply(self.server.engine.transcribe(body))
            self.reply({'error':'Ruta desconocida'}, code=404)
        except Exception as exc:
            self.reply({'error': str(exc)}, code=400)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=8767, type=int)
    parser.add_argument('--whisper-model', default='small')
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.engine = AudioEngine(args.whisper_model)
    print(f'Audio: http://{args.host}:{args.port} (modelos bajo demanda)', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
