"""Audio validation and panel proxy tests; no models or microphone required."""
import io
import base64
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from urllib.request import Request, urlopen
from unittest.mock import patch
import zimage_audio_server as audio
import zimage_panel_grid as panel


class Tests(unittest.TestCase):
    def test_engine_rejects_bad_requests_before_loading(self):
        engine=audio.AudioEngine()
        for body in ({}, {'input':'x','voice':'../voice'}, {'input':'x','speed':99}):
            with self.assertRaises(ValueError):engine.speech(body)
        for body in ({}, {'audio':'bad!'}, {'audio':'YQ==','language':'invalid'}):
            with self.assertRaises(ValueError):engine.transcribe(body)
        self.assertIsNone(engine.tts)
        self.assertIsNone(engine.stt)

    def test_kokoclone_uses_selected_kokoro_voice_then_conversion(self):
        engine = audio.AudioEngine()
        reference = base64.b64encode(b'RIFF-reference').decode()
        with patch.object(engine, 'kokoro_speech', return_value=b'RIFF-dora') as kokoro, \
             patch.object(engine, 'kokoclone', return_value=b'RIFF-cloned') as clone:
            self.assertEqual(engine.speech({'engine':'kokoclone', 'input':'Hola', 'voice':'ef_dora',
                                            'speed':1.2, 'reference_audio':reference}), b'RIFF-cloned')
        kokoro.assert_called_once_with('Hola', 'ef_dora', 1.2)
        self.assertEqual(clone.call_args.args[0], '/convert')
        self.assertEqual(base64.b64decode(clone.call_args.args[1]['source']), b'RIFF-dora')
        self.assertEqual(clone.call_args.args[1]['reference'], reference)

    def test_kokoclone_direct_keeps_the_legacy_internal_base_voice(self):
        engine = audio.AudioEngine()
        reference = base64.b64encode(b'RIFF-reference').decode()
        with patch.object(engine, 'kokoro_speech') as kokoro, \
             patch.object(engine, 'kokoclone', return_value=b'RIFF-cloned') as clone:
            self.assertEqual(engine.speech({'engine':'kokoclone_direct', 'input':'Hola',
                                            'reference_audio':reference}), b'RIFF-cloned')
        kokoro.assert_not_called()
        self.assertEqual(clone.call_args.args[0], '/tts')
        self.assertEqual(clone.call_args.args[1], {'input':'Hola', 'lang':'es', 'reference':reference})

    def test_qwen_trial_receives_only_validated_clone_fields(self):
        engine = audio.AudioEngine()
        reference = base64.b64encode(b'RIFF-reference').decode()
        with patch.object(engine, 'qwen3tts', return_value=b'RIFF-qwen') as qwen:
            self.assertEqual(engine.speech({'engine':'qwen3tts', 'input':'Hola',
                                            'reference_audio':reference, 'reference_text':'Hola'}), b'RIFF-qwen')
        self.assertEqual(qwen.call_args.args[0], {'input':'Hola', 'reference_audio':reference, 'reference_text':'Hola'})

    def test_qwen_trial_falls_back_to_kokoclone_then_kokoro(self):
        engine = audio.AudioEngine()
        reference = base64.b64encode(b'RIFF-reference').decode()
        with patch.object(engine, 'qwen3tts', side_effect=ValueError('no disponible')), \
             patch.object(engine, 'kokoro_speech', return_value=b'RIFF-base') as kokoro, \
             patch.object(engine, 'kokoclone', return_value=b'RIFF-fallback') as clone:
            self.assertEqual(engine.speech({'engine':'qwen3tts', 'input':'Hola', 'reference_audio':reference}), b'RIFF-fallback')
        kokoro.assert_called_once_with('Hola', 'ef_dora', 1)
        self.assertEqual(clone.call_args.args[0], '/convert')


    def test_proxy_audio_and_settings(self):
        backend=ThreadingHTTPServer(('127.0.0.1',0),audio.Handler)
        backend.engine=audio.AudioEngine()
        worker=threading.Thread(target=backend.serve_forever,daemon=True);worker.start()
        with TemporaryDirectory() as temp:
            args=SimpleNamespace(lora_dir=temp,backend_host='127.0.0.1',backend_port=9877,
                                 mcp_url=None,chat_url='http://127.0.0.1:8086/v1',chat_model='',settings_file=str(Path(temp)/'settings.json'))
            app=panel.App(args)
            app.configure({**app.configuration(),'audio_url':f'http://127.0.0.1:{backend.server_port}'})
            self.assertEqual(panel.App(args).audio_url,app.audio_url)
            frozen=app.snapshot()
            server=ThreadingHTTPServer(('127.0.0.1',0),panel.Handler);server.app=app
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                with patch.object(backend.engine,'speech',return_value=b'RIFF-test-wave') as speech, patch.object(backend.engine,'transcribe',return_value={'text':'Hola'}) as stt:
                    for path,body in [('tts',{'input':'Hola','voice':'ef_dora'}),('stt',{'audio':'YQ==','language':'es'})]:
                        req=Request(f'http://127.0.0.1:{server.server_port}/api/audio/{path}',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                        with urlopen(req,timeout=5) as response:
                            result=response.read()
                            if path=='tts':self.assertEqual(response.headers.get_content_type(),'audio/wav');self.assertEqual(result,b'RIFF-test-wave')
                            else:self.assertEqual(json.loads(result),{'text':'Hola'})
                    speech.assert_called_once_with({'input':'Hola','voice':'ef_dora','engine':'kokoro'})
                    stt.assert_called_once_with({'audio':'YQ==','language':'es'})
                name='Character_(vrtlCharacter)'
                (Path(temp)/(name+'.safetensors')).touch()
                (Path(temp)/(name+'.wav')).write_bytes(b'RIFF-reference')
                with patch.object(backend.engine,'speech',return_value=b'RIFF-cloned') as speech:
                    req=Request(f'http://127.0.0.1:{server.server_port}/api/audio/tts',data=json.dumps({
                        'input':'Hola','engine':'kokoclone','lora_name':name,'reference_audio':'untrusted',
                        'reference_path':'/etc/passwd'}).encode(),headers={'Content-Type':'application/json'})
                    with urlopen(req,timeout=5) as response:self.assertEqual(response.read(),b'RIFF-cloned')
                    sent=speech.call_args.args[0]
                    self.assertEqual(base64.b64decode(sent['reference_audio']),b'RIFF-reference')
                    self.assertNotIn('reference_path',sent)
                    self.assertEqual(sent['engine'],'kokoclone')
                (Path(temp)/(name+'.txt')).write_text('Hola, esto es una referencia.', encoding='utf-8')
                with patch.object(backend.engine,'speech',return_value=b'RIFF-qwen') as speech:
                    req=Request(f'http://127.0.0.1:{server.server_port}/api/audio/tts',data=json.dumps({
                        'input':'Hola','engine':'qwen3tts','lora_name':name,
                        'reference_text':'no confiable'}).encode(),headers={'Content-Type':'application/json'})
                    with urlopen(req,timeout=5) as response:self.assertEqual(response.read(),b'RIFF-qwen')
                    sent=speech.call_args.args[0]
                    self.assertEqual(sent['reference_text'],'Hola, esto es una referencia.')
                    self.assertNotIn('lora_name',sent)
                    self.assertEqual(sent['engine'],'qwen3tts')
                (Path(temp)/(name+'.wav')).unlink()
                with self.assertRaises(Exception):urlopen(req,timeout=5)
                app.configure({**app.configuration(),'audio_url':''})
                self.assertNotEqual(frozen.audio_url,app.audio_url)
            finally:
                server.shutdown();server.server_close();thread.join()
                backend.shutdown();backend.server_close();worker.join()


if __name__=='__main__':unittest.main()
