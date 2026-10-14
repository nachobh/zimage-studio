#!/usr/bin/env python3
"""Isolated local Qwen3-TTS voice-cloning trial; no panel profiles are written."""
import argparse
from collections import OrderedDict
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Trial:
    LANGUAGES = ('Chinese', 'English', 'Japanese', 'Korean', 'German', 'French',
                 'Russian', 'Portuguese', 'Spanish', 'Italian')

    def __init__(self, model_id, device, cache_dir=None):
        self.model_id, self.device, self.model = model_id, device, None
        self.lock = threading.Lock()
        selected = cache_dir if cache_dir is not None else os.environ.get(
            'QWEN3_TTS_CACHE_DIR', str(Path(__file__).resolve().parent / '.service-state/qwen3_tts_voice_cache'))
        self.cache_dir = Path(selected) if selected else None
        self.prompts = {}
        self.audio_cache = OrderedDict()
        self.audio_cache_bytes = 0
        self.stats = {'prompt_memory_hits':0, 'prompt_disk_hits':0, 'prompt_misses':0, 'audio_hits':0, 'syntheses':0}
        self.timings = {}
        self.runtime = {}
        self.threads = max(1, int(os.environ.get('QWEN3_TTS_THREADS', '8')))

        if self.cache_dir:
            self.cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            self.cache_dir.chmod(0o700)

    def prepare_device(self):
        import torch
        if self.device != 'cpu':
            if not torch.cuda.is_available():
                raise RuntimeError('Qwen3-TTS requiere GPU. Instala PyTorch ROCm/CUDA compatible en su venv; no se usará CPU automáticamente.')
            self.device = 'cuda'
        self.runtime = {'torch':torch.__version__, 'hip':torch.version.hip,
                        'gpu':torch.cuda.get_device_name(0) if self.device == 'cuda' else None}

    def cache_key(self, raw, reference_text):
        digest = hashlib.sha256()
        digest.update(self.model_id.encode())
        digest.update(b'\0')
        digest.update(raw)
        digest.update(b'\0')
        digest.update(reference_text.encode())
        return digest.hexdigest()

    @staticmethod
    def valid_prompt(prompt, torch):
        return (isinstance(prompt, dict) and set(prompt) == {'ref_code', 'ref_spk_embedding', 'x_vector_only_mode', 'icl_mode'}
                and all(isinstance(prompt[key], list) and len(prompt[key]) == 1 for key in prompt)
                and (prompt['ref_code'][0] is None or torch.is_tensor(prompt['ref_code'][0]))
                and torch.is_tensor(prompt['ref_spk_embedding'][0])
                and isinstance(prompt['x_vector_only_mode'][0], bool)
                and isinstance(prompt['icl_mode'][0], bool))

    def load_prompt(self, key, torch):
        prompt = self.prompts.get(key)
        if prompt is not None:
            self.stats['prompt_memory_hits'] += 1
            return prompt
        path = self.cache_dir / (key + '.pt') if self.cache_dir else None
        if not path or not path.is_file() or path.is_symlink():
            return None
        try:
            prompt = torch.load(path, map_location='cpu', weights_only=True)
            if not self.valid_prompt(prompt, torch):
                return None
        except Exception:
            return None
        self.prompts[key] = prompt
        self.stats['prompt_disk_hits'] += 1
        return prompt

    def save_prompt(self, key, prompt, torch):
        self.prompts[key] = prompt
        if not self.cache_dir:
            return
        descriptor, temporary = tempfile.mkstemp(prefix=key + '.', suffix='.tmp', dir=self.cache_dir)
        os.close(descriptor)
        try:
            torch.save(prompt, temporary)
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.cache_dir / (key + '.pt'))
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def voice_clone_prompt(self, raw, reference_text, torch):
        key = self.cache_key(raw, reference_text)
        prompt = self.load_prompt(key, torch)
        if prompt is not None:
            return prompt
        self.stats['prompt_misses'] += 1
        import numpy as np
        import soundfile as sf
        waveform, sample_rate = sf.read(io.BytesIO(raw), dtype='float32')
        # Qwen3-TTS 0.6B currently mutates a tuple while converting a
        # multichannel reference internally.  Supply an already-mono array
        # so stereo WAV profile samples use the same safe input path.
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1, dtype=np.float32)
        items = self.model.create_voice_clone_prompt(
            ref_audio=(waveform, sample_rate), ref_text=reference_text or None,
            x_vector_only_mode=not bool(reference_text))
        prompt = {
            'ref_code': [item.ref_code.detach().cpu() if item.ref_code is not None else None for item in items],
            'ref_spk_embedding': [item.ref_spk_embedding.detach().cpu() for item in items],
            'x_vector_only_mode': [bool(item.x_vector_only_mode) for item in items],
            'icl_mode': [bool(item.icl_mode) for item in items],
        }
        if not self.valid_prompt(prompt, torch):
            raise ValueError('Qwen3-TTS devolvió un prompt de clonación inválido.')
        self.save_prompt(key, prompt, torch)
        return prompt

    def speech(self, body):
        text, reference = body.get('input', ''), body.get('reference_audio', '')
        reference_text = body.get('reference_text', '')
        language = body.get('language', 'Spanish')
        if not isinstance(text, str) or not 0 < len(text.strip()) <= 5000:
            raise ValueError('Usa entre 1 y 5000 caracteres para la voz.')
        if not isinstance(language, str) or language not in self.LANGUAGES:
            raise ValueError('Idioma no admitido. Usa: ' + ', '.join(self.LANGUAGES) + '.')
        if not isinstance(reference_text, str) or len(reference_text) > 10000:
            raise ValueError('Transcripción de referencia inválida.')
        try:
            raw = base64.b64decode(reference, validate=True)
        except Exception as exc:
            raise ValueError('Referencia WAV inválida.') from exc
        if not 0 < len(raw) <= 20 * 1024**2:
            raise ValueError('Referencia WAV vacía o mayor de 20 MB.')
        with self.lock:
            started = time.perf_counter()
            # Finished WAVs stay in bounded RAM only, never in conversation logs.
            key = hashlib.sha256((self.cache_key(raw, reference_text) + '\0' + language + '\0' + text).encode()).hexdigest()
            if key in self.audio_cache:
                self.stats['audio_hits'] += 1
                self.audio_cache.move_to_end(key)
                self.timings = {'audio_cache_hit':True, 'total_seconds':round(time.perf_counter()-started,4)}
                return self.audio_cache[key]
            import torch
            if self.model is None:
                from qwen_tts import Qwen3TTSModel
                torch.set_num_threads(self.threads)
                self.prepare_device()
                dtype = torch.float32 if self.device == 'cpu' else torch.bfloat16
                self.model = Qwen3TTSModel.from_pretrained(
                    self.model_id, device_map=self.device, dtype=dtype, attn_implementation='sdpa')
                model_device = next(self.model.model.parameters()).device
                self.runtime['model_device'] = str(model_device)
                if self.device == 'cuda' and model_device.type != 'cuda':
                    self.model = None
                    raise RuntimeError('El modelo Qwen3-TTS no se ha cargado en GPU.')
            loaded = time.perf_counter()
            prompt = self.voice_clone_prompt(raw, reference_text, torch)
            prepared = time.perf_counter()
            # Dict prompts in qwen-tts 0.1.1 discard reference text. Reconstruct
            # the public item type so contextual cloning keeps its transcript.
            from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
            item = VoiceClonePromptItem(
                ref_code=prompt['ref_code'][0].to(self.device) if prompt['ref_code'][0] is not None else None,
                ref_spk_embedding=prompt['ref_spk_embedding'][0].to(self.device),
                x_vector_only_mode=prompt['x_vector_only_mode'][0],
                icl_mode=prompt['icl_mode'][0], ref_text=reference_text or None)
            with torch.inference_mode():
                audio, sample_rate = self.model.generate_voice_clone(
                    text=text, language=language, voice_clone_prompt=[item], non_streaming_mode=True)
            import soundfile as sf
            output = io.BytesIO()
            sf.write(output, audio[0], sample_rate, format='WAV')
            result = output.getvalue()
            if len(result) > 32 * 1024**2 or result[:4] != b'RIFF':
                raise ValueError('Qwen3-TTS no devolvió un WAV válido.')
            self.stats['syntheses'] += 1
            self.timings = {'audio_cache_hit':False, 'load_seconds':round(loaded-started,3),
                            'prompt_seconds':round(prepared-loaded,3),
                            'synthesis_seconds':round(time.perf_counter()-prepared,3),
                            'audio_seconds':round(len(audio[0])/sample_rate,3),
                            'total_seconds':round(time.perf_counter()-started,3)}
            self.audio_cache[key] = result
            self.audio_cache_bytes += len(result)
            while len(self.audio_cache)>16 or self.audio_cache_bytes>32*1024**2:
                _, old = self.audio_cache.popitem(last=False)
                self.audio_cache_bytes -= len(old)
            return result



class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

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
            return self.reply({'status':'ok', 'loaded':self.server.trial.model is not None,
                               'model':self.server.trial.model_id, 'device':self.server.trial.device,
                               'threads':self.server.trial.threads, 'cache':dict(self.server.trial.stats),
                               'runtime':dict(self.server.trial.runtime),
                               'timings':dict(self.server.trial.timings)})
        self.reply({'error':'Ruta desconocida'}, code=404)

    def do_POST(self):
        try:
            if self.path != '/tts':
                return self.reply({'error':'Ruta desconocida'}, code=404)
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 28 * 1024**2:
                raise ValueError('Petición de audio demasiado grande o vacía.')
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError('Petición inválida.')
            self.reply(self.server.trial.speech(body), 'audio/wav')
        except Exception as exc:
            self.reply({'error':str(exc)}, code=400)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8770)
    parser.add_argument('--model', default=os.environ.get('QWEN3_TTS_MODEL', 'Qwen/Qwen3-TTS-12Hz-0.6B-Base'))
    parser.add_argument('--device', default=os.environ.get('QWEN3_TTS_DEVICE', 'cuda'), choices=('auto', 'cpu', 'cuda'))
    parser.add_argument('--cache-dir', default=os.environ.get('QWEN3_TTS_CACHE_DIR'))
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.trial = Trial(args.model, args.device, args.cache_dir)
    server.trial.prepare_device()
    print(f'Qwen3-TTS trial: http://{args.host}:{args.port} · {args.model} · {args.device}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
