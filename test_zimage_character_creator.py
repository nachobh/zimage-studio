"""Offline tests: no model, web server, or GPU process is started."""
import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
import types
import wave
from unittest.mock import patch

from PIL import Image

from zimage_character_creator import CharacterCreator


class CharacterCreatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.creator = CharacterCreator(self.root / 'data', self.root / 'loras',
                                         model_dir=self.root / 'model')
        image = Image.new('RGB', (640, 640), '#2671a9')
        stream = io.BytesIO()
        image.save(stream, format='PNG')
        self.image = 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()
        small = Image.new('RGB', (32, 32), '#2671a9')
        stream = io.BytesIO()
        small.save(stream, format='PNG')
        self.small_image = 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()

    def make_model_ready(self):
        (self.root / 'model' / 'transformer').mkdir(parents=True, exist_ok=True)
        (self.root / 'model' / 'transformer' / 'part.safetensors').write_bytes(b'weight')
        (self.root / 'model' / 'vae').mkdir(exist_ok=True)
        (self.root / 'model' / 'vae' / 'model.safetensors').write_bytes(b'weight')
        (self.root / 'model' / 'text_encoder').mkdir(exist_ok=True)
        (self.root / 'model' / 'text_encoder' / 'model.safetensors').write_bytes(b'weight')
        (self.root / 'model' / 'model_index.json').write_text('{}')
        index = self.root / 'model' / 'transformer' / 'diffusion_pytorch_model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': {'a': 'part.safetensors'}}))
        self.assertTrue(self.creator.model_ready())

    def ready_character(self, name='Luna', image=None, photos=5):
        """Draft with captions, trigger and a complete-looking Z-Image checkpoint."""
        self.creator.create({'name': name, 'display_name': 'Luna'})
        for _ in range(photos):
            self.creator.add_photo(name, self.image if image is None else image)
        data = self.creator.get(name)
        captions = {photo: 'vrtlLuna portrait' for photo in data['photos']}
        self.creator.update(name, {'profile': data['profile'], 'captions': captions})
        self.make_model_ready()
        return self.creator.get(name)

    @staticmethod
    def make_wav(seconds=3):
        stream = io.BytesIO()
        with wave.open(stream, 'wb') as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(24000)
            audio.writeframes(b'\0\0' * (24000 * seconds))
        return stream.getvalue()

    def test_voice_wav_is_validated_replaceable_and_private_until_publish(self):
        name = 'Luna'
        self.creator.create({'name': name, 'display_name': 'Luna'})
        raw = self.make_wav()
        encoded = 'data:audio/wav;base64,' + base64.b64encode(raw).decode()
        data = self.creator.add_voice(name, encoded)
        self.assertEqual(data['voice']['duration'], 3.0)
        self.assertEqual(self.creator.voice(name), raw)
        with self.assertRaisesRegex(ValueError, 'WAV PCM'):
            self.creator.add_voice(name, 'data:audio/wav;base64,' + base64.b64encode(b'not a wav').decode())
        with self.assertRaisesRegex(ValueError, 'truncado'):
            self.creator.add_voice(name, 'data:audio/wav;base64,' + base64.b64encode(raw[:-20]).decode())
        self.creator.add_voice(name, 'data:audio/x-wav;base64,' + base64.b64encode(self.make_wav(4)).decode())
        self.assertEqual(self.creator.get(name)['voice']['duration'], 4.0)
        self.assertFalse((self.root / 'loras' / 'Luna.wav').exists())
        self.assertFalse(self.creator.remove_voice(name)['voice']['present'])

    def test_publish_copies_voice_as_lora_named_wav(self):
        name = 'Luna'
        self.creator.create({'name': name, 'display_name': 'Luna'})
        self.creator.add_photo(name, self.image)
        wav = self.make_wav()
        self.creator.add_voice(name, 'data:audio/wav;base64,' + base64.b64encode(wav).decode())
        (self.creator.directory(name) / 'adapter.safetensors').write_bytes(b'fake adapter')
        class FakeSafe:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def keys(self): return ['layer.lora_A.weight', 'layer.lora_B.weight']
        with patch.dict('sys.modules', {'safetensors': types.SimpleNamespace(safe_open=lambda *a, **k: FakeSafe())}):
            self.creator._publish(name)
        self.assertEqual((self.root / 'loras' / 'Luna.wav').read_bytes(), wav)
        self.assertEqual(self.creator.get(name)['state'], 'published')
        with self.assertRaisesRegex(ValueError, 'Ya hay archivos'):
            self.creator.create({'name': name, 'display_name': 'Luna'})

    def test_dataset_profile_and_photo_validation(self):
        for name in ('../outside', 'has/slash', '-bad', 'áé'):
            with self.assertRaises(ValueError):
                self.creator.create({'name': name, 'display_name': 'Persona'})
        data = self.creator.create({'name': 'Luna_vrtlLuna', 'display_name': 'Luna'})
        self.assertEqual(data['profile']['image_prompt'], 'vrtlLuna Luna mirando a cámara')
        self.assertFalse(data['photos'])
        with self.assertRaises(ValueError):
            self.creator.create({'name': data['name'], 'display_name': 'Otra'})
        data = self.creator.add_photo(data['name'], self.image)
        filename = data['photos'][0]
        self.assertTrue(self.creator.photo(data['name'], filename).startswith(b'\xff\xd8'))
        with self.assertRaises(ValueError):
            self.creator.photo(data['name'], '../character.json')
        with self.assertRaises(ValueError):
            self.creator.add_photo(data['name'], 'data:image/svg+xml;base64,YQ==')
        profile = {'system_prompt': 'Habla como Luna.', 'image_prompt': 'vrtlLuna retrato'}
        data = self.creator.update(data['name'], {'profile': profile, 'captions': {filename: 'vrtlLuna outdoors'}})
        self.assertEqual(data['captions'][filename], 'vrtlLuna outdoors')
        with self.assertRaises(ValueError):
            self.creator.update(data['name'], {'profile': profile, 'captions': {'../bad.jpg': 'oops'}})
        self.assertEqual(self.creator.remove_photo(data['name'], filename)['photos'], [])

    def test_training_preflight_requires_reviewed_captions_and_full_model(self):
        name = 'Luna'
        self.creator.create({'name': name, 'display_name': 'Luna'})
        for _ in range(5):
            self.creator.add_photo(name, self.image)
        with self.assertRaisesRegex(ValueError, 'caption'):
            self.creator.train(name)
        data = self.creator.get(name)
        profile = data['profile']
        captions = {photo: 'vrtlLuna portrait' for photo in data['photos']}
        self.creator.update(name, {'profile': profile, 'captions': captions})
        with self.assertRaisesRegex(ValueError, 'modelo Z-Image completo'):
            self.creator.train(name)
        (self.root / 'model' / 'transformer').mkdir(parents=True)
        (self.root / 'model' / 'transformer' / 'part.safetensors').write_bytes(b'weight')
        (self.root / 'model' / 'vae').mkdir()
        (self.root / 'model' / 'vae' / 'model.safetensors').write_bytes(b'weight')
        (self.root / 'model' / 'text_encoder').mkdir()
        (self.root / 'model' / 'text_encoder' / 'model.safetensors').write_bytes(b'weight')
        (self.root / 'model' / 'model_index.json').write_text('{}')
        index = self.root / 'model' / 'transformer' / 'diffusion_pytorch_model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': {'a': 'part.safetensors', 'b': 'missing.safetensors'}}))
        self.assertFalse(self.creator.model_ready())
        (self.root / 'model' / 'transformer' / 'missing.safetensors').write_bytes(b'weight')
        with patch.object(self.creator, '_start', return_value={'status': 'ok'}) as start:
            self.creator.train(name)
        self.assertEqual(start.call_args.args[1], 'training')
        self.assertEqual(json.loads((self.creator.directory(name) / 'captions.json').read_text()), captions)

    def test_photo_sizes_and_low_resolution_preflight(self):
        name = 'Luna'
        self.creator.create({'name': name, 'display_name': 'Luna'})
        for _ in range(4):
            self.creator.add_photo(name, self.image)
        self.creator.add_photo(name, self.small_image)
        data = self.creator.get(name)
        sizes = data['sizes']
        self.assertEqual(len(sizes), 5)
        self.assertIn([640, 640], sizes.values())
        self.assertIn([32, 32], sizes.values())
        captions = {photo: 'vrtlLuna portrait' for photo in data['photos']}
        self.creator.update(name, {'profile': data['profile'], 'captions': captions})
        with self.assertRaisesRegex(ValueError, 'llegan a 512 px'):
            self.creator.train(name, {'skip_small': True})
        self.assertIsNone(self.creator.get(name)['report'])

    def test_training_options_are_validated_and_forwarded(self):
        name = 'Luna'
        self.ready_character(name)
        options = {'epochs': 60, 'learning_rate': '0.0003', 'rank': 32, 'resolution': 768,
                   'centering_y': 0.25, 'min_size': 640}
        with patch.object(self.creator, '_start', return_value={'status': 'ok'}) as start:
            self.creator.train(name, options)
        command = start.call_args.args[2]
        for flag, value in (('--epochs', '60'), ('--rank', '32'), ('--resolution', '768'),
                            ('--learning-rate', '0.0003'), ('--min-size', '640'),
                            ('--centering-y', '0.25')):
            self.assertIn(flag, command)
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertNotIn('--keep-small', command)
        for bad, pattern in (({'epochs': 0}, 'epochs'), ({'rank': 999}, 'rank'),
                             ({'resolution': 500}, 'múltiplo'), ({'learning_rate': 5}, 'learning_rate'),
                             ({'centering_y': 2}, 'centering_y'), ({'min_size': 4096}, 'min_size')):
            with self.assertRaisesRegex(ValueError, pattern):
                self.creator.train(name, bad)
        with patch.object(self.creator, '_start', return_value={'status': 'ok'}) as start:
            self.creator.train(name, {'skip_small': False, 'epochs': 5})
        self.assertIn('--keep-small', start.call_args.args[2])

    def test_training_refuses_when_every_photo_is_below_the_minimum(self):
        name = 'Luna'
        self.creator.create({'name': name, 'display_name': 'Luna'})
        for _ in range(5):
            self.creator.add_photo(name, self.small_image)
        data = self.creator.get(name)
        captions = {photo: 'vrtlLuna portrait' for photo in data['photos']}
        self.creator.update(name, {'profile': data['profile'], 'captions': captions})
        with self.assertRaisesRegex(ValueError, 'px'):
            self.creator.train(name)
        self.make_model_ready()
        with patch.object(self.creator, '_start', return_value={'status': 'ok'}) as start:
            self.creator.train(name, {'min_size': 0})
        self.assertNotIn('--keep-small', start.call_args.args[2])
        self.assertIn('--min-size', start.call_args.args[2])

    def test_training_report_is_read_from_the_dataset(self):
        name = 'Luna'
        self.creator.create({'name': name, 'display_name': 'Luna'})
        self.assertIsNone(self.creator.get(name)['report'])
        (self.creator.directory(name) / 'training_report.json').write_text(
            json.dumps({'rel_median': 0.075, 'advice': ['Fuerza adecuada.']}), encoding='utf-8')
        report = self.creator.get(name)['report']
        self.assertEqual(report['rel_median'], 0.075)
        self.assertEqual(report['advice'], ['Fuerza adecuada.'])


    def test_upscale_small_photos_backs_up_keeps_names_and_restores(self):
        name = 'Luna'
        self.creator.create({'name': name, 'display_name': 'Luna'})
        self.creator.add_photo(name, self.small_image)   # 32×32
        self.creator.add_photo(name, self.image)         # 640×640
        data = self.creator.get(name)
        small = [p for p in data['photos'] if min(data['sizes'][p]) < 512]
        large = [p for p in data['photos'] if min(data['sizes'][p]) >= 512]
        self.assertEqual(len(small), 1)
        photos = self.creator.directory(name) / 'photos'
        original = (photos / small[0]).read_bytes()
        result = self.creator.upscale_photos(name, 512)
        self.assertEqual(result['upscaled'][0]['photo'], small[0])
        self.assertEqual(result['upscaled'][0]['before'], [32, 32])
        self.assertEqual(result['upscaled'][0]['after'], [512, 512])
        self.assertEqual(result['errors'], [])
        self.assertEqual([item['photo'] for item in result['skipped']], large)
        backup = self.creator.directory(name) / 'photos_original' / small[0]
        self.assertEqual(backup.read_bytes(), original)
        with Image.open(photos / small[0]) as image:
            self.assertEqual(min(image.size), 512)
        refreshed = self.creator.get(name)
        self.assertEqual(refreshed['photos'], data['photos'])
        self.assertEqual(refreshed['captions'], data['captions'])
        # Running it again must be a no-op, never a second copy of the original.
        second = self.creator.upscale_photos(name, 512)
        self.assertEqual(second['upscaled'], [])
        self.assertEqual(len(second['skipped']), 2)
        self.assertEqual(backup.read_bytes(), original)
        restored = self.creator.restore_photos(name)
        self.assertEqual(restored['restored'], small)
        self.assertEqual((photos / small[0]).read_bytes(), original)
        self.assertFalse((self.creator.directory(name) / 'photos_original').exists())

    def test_upscale_rejects_bad_targets_and_published_characters(self):
        name = 'Luna'
        self.creator.create({'name': name, 'display_name': 'Luna'})
        self.creator.add_photo(name, self.small_image)
        for target in (100, 4096, 'otra'):
            with self.assertRaises(ValueError):
                self.creator.upscale_photos(name, target)
        self.assertEqual(self.creator.upscale_photos(name, 512)['upscaled'][0]['after'], [512, 512])
        (self.creator.directory(name) / 'adapter.safetensors').write_bytes(b'fake adapter')

        class FakeSafe:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def keys(self): return ['layer.lora_A.weight', 'layer.lora_B.weight']

        with patch.dict('sys.modules', {'safetensors': types.SimpleNamespace(
                safe_open=lambda *a, **k: FakeSafe())}):
            self.creator._publish(name)
        self.assertEqual(self.creator.get(name)['state'], 'published')
        with self.assertRaisesRegex(ValueError, 'publicar'):
            self.creator.upscale_photos(name, 1024)
        with self.assertRaisesRegex(ValueError, 'publicar'):
            self.creator.restore_photos(name)


if __name__ == '__main__':
    unittest.main()
