"""Small CPU tests for cache correctness; no model download or synthesis."""
import base64
import importlib.util
import tempfile
import unittest
from unittest.mock import patch

from qwen3_tts_trial import Trial

class RequestValidationTests(unittest.TestCase):
    def test_language_is_validated_before_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            trial=Trial('model','cpu',directory)
            body={'input':'Hola','reference_audio':base64.b64encode(b'wave').decode(),'language':'Spanish (Spain)'}
            with self.assertRaisesRegex(ValueError,'Idioma no admitido'):
                trial.speech(body)

@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('qwen_tts'), 'Usa el venv de Qwen3-TTS')
class CacheTests(unittest.TestCase):
    def test_prompt_survives_restart_and_depends_on_reference_text(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            first=Trial('model','cpu',directory)
            prompt={'ref_code':[None],'ref_spk_embedding':[torch.ones(3)],'x_vector_only_mode':[True],'icl_mode':[False]}
            key=first.cache_key(b'wave','')
            first.save_prompt(key,prompt,torch)
            second=Trial('model','cpu',directory)
            self.assertIsNotNone(second.load_prompt(key,torch))
            self.assertEqual(second.stats['prompt_disk_hits'],1)
            self.assertIsNotNone(second.load_prompt(key,torch))
            self.assertEqual(second.stats['prompt_memory_hits'],1)
            self.assertIsNone(second.load_prompt(second.cache_key(b'wave','transcript'),torch))

    def test_finished_audio_cache_and_contextual_reference_text(self):
        import torch
        import numpy as np
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            trial=Trial('model','cpu',directory)
            calls=[]
            def generate(**kwargs):
                calls.append(kwargs)
                return [np.zeros(2400,dtype=np.float32)],24000
            trial.model=SimpleNamespace(generate_voice_clone=generate)
            prompt={'ref_code':[torch.ones(2,dtype=torch.long)],'ref_spk_embedding':[torch.ones(3)],'x_vector_only_mode':[False],'icl_mode':[True]}
            body={'input':'Hola','reference_audio':base64.b64encode(b'wave').decode(),'reference_text':'Referencia.'}
            with patch.object(trial,'voice_clone_prompt',return_value=prompt) as prepare:
                first=trial.speech(body)
                self.assertEqual(trial.speech(body),first)
                self.assertEqual(len(calls),1)
                self.assertEqual(prepare.call_count,1)
                self.assertEqual(calls[0]['voice_clone_prompt'][0].ref_text,'Referencia.')
                self.assertEqual(trial.stats['audio_hits'],1)
                trial.speech({**body,'language':'English'})
                self.assertEqual(calls[1]['language'],'English')
                self.assertEqual(calls[0]['language'],'Spanish')
                for index in range(17):trial.speech({**body,'input':str(index)})
                self.assertEqual(len(trial.audio_cache),16)
                trial.speech(body)
                self.assertEqual(len(calls),20)

    def test_gpu_required_never_silently_falls_back_to_cpu(self):
        with tempfile.TemporaryDirectory() as directory,patch('torch.cuda.is_available',return_value=False):
            for device in ('cuda','auto'):
                with self.assertRaisesRegex(RuntimeError,'requiere GPU'):
                    Trial('model',device,directory).prepare_device()

if __name__=='__main__':unittest.main()
