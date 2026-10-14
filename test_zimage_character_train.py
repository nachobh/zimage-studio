"""Offline tests for the trainer helpers: no Torch, GPU or model is loaded."""
import io
import json
from pathlib import Path
import tempfile
import sys
import types
import unittest
from unittest.mock import patch

from PIL import Image

from zimage_character_train import REPORT_NAME, _strength, advice_for, build_parser, check_args, training_target_modules, usable_photos, write_report


class TrainerHelperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        (self.folder / 'photos').mkdir()

    def photo(self, name, size):
        stream = io.BytesIO()
        Image.new('RGB', size, '#2671a9').save(stream, format='JPEG')
        (self.folder / 'photos' / name).write_bytes(stream.getvalue())

    def test_usable_photos_skips_frames_below_the_minimum(self):
        self.photo('big.jpg', (640, 640))
        self.photo('small.jpg', (320, 240))
        kept, skipped = usable_photos(self.folder, 512)
        self.assertEqual([p.name for p in kept], ['big.jpg'])
        self.assertEqual([(p.name, side) for p, side in skipped], [('small.jpg', 240)])
        kept, skipped = usable_photos(self.folder, 512, keep_small=True)
        self.assertEqual(len(kept), 2)
        self.assertEqual(skipped, [])
        kept, skipped = usable_photos(self.folder, 0)
        self.assertEqual(len(kept), 2)
        self.assertEqual(skipped, [])

    def test_defaults_match_the_recommended_training_recipe(self):
        parser = build_parser()
        args = check_args(parser, parser.parse_args(['--dataset', 'd', '--model', 'm', '--output', 'o']))
        self.assertEqual((args.epochs, args.learning_rate, args.rank, args.resolution),
                         (40, 2e-4, 16, 512))
        self.assertEqual((args.min_size, args.centering_y, args.keep_small), (512, 0.3, False))
        for bad in (['--epochs', '0'], ['--rank', '999'], ['--resolution', '500'],
                    ['--learning-rate', '5'], ['--min-size', '4096'], ['--centering-y', '2']):
            with self.assertRaises(SystemExit):
                check_args(parser, parser.parse_args(['--dataset', 'd', '--model', 'm',
                                                      '--output', 'o'] + bad))

    def test_advice_reports_the_strength_band(self):
        weak = advice_for({'rel_median': 0.0003})
        self.assertIn('débil', weak[0])
        good = advice_for({'rel_median': 0.075})
        self.assertIn('adecuada', good[0])
        strong = advice_for({'rel_median': 0.25})
        self.assertIn('intenso', strong[0])
        unknown = advice_for({})
        self.assertIn('No se pudo medir', unknown[0])
        for lines in (weak, good, strong, unknown):
            self.assertTrue(any('CFG 0' in line for line in lines))
            self.assertTrue(any('0.6, 0.8, 1.0 y 1.2' in line for line in lines))

    def test_strength_uses_peft_base_layer_and_b_times_a(self):
        class Matrix:
            def __init__(self, rows): self.rows = rows
            def detach(self): return self
            def to(self, *_): return self
            def __matmul__(self, other):
                return Matrix([[sum(a * b for a, b in zip(row, col))
                                for col in zip(*other.rows)] for row in self.rows])
            def norm(self):
                return types.SimpleNamespace(item=lambda: sum(x*x for row in self.rows for x in row)**0.5)
        weights = {
            'block.to_q.base_layer.weight': Matrix([[1,1,1],[1,1,1],[1,1,1]]),
            'block.to_q.lora_A.default.weight': Matrix([[1,0,0],[0,1,0]]),
            'block.to_q.lora_B.default.weight': Matrix([[1,0],[0,1],[1,1]]),
        }
        model = types.SimpleNamespace(named_parameters=lambda: weights.items())
        with patch.dict(sys.modules, {'torch': types.SimpleNamespace(float32='float32')}):
            result = _strength(model)
        self.assertEqual(result['modules'], 1)
        self.assertAlmostEqual(result['median'], 2/3)

    def test_training_targets_only_main_layers(self):
        names = [f'layers.0.attention.{part}' for part in ('to_q','to_k','to_v','to_out.0')]
        names += [f'layers.0.feed_forward.{part}' for part in ('w1','w2','w3')]
        names += ['context_refiner.0.attention.to_q', 'noise_refiner.0.feed_forward.w1']
        model = types.SimpleNamespace(named_modules=lambda: [(name, object()) for name in names])
        self.assertEqual(training_target_modules(model), names[:7])
        with self.assertRaises(RuntimeError):
            training_target_modules(types.SimpleNamespace(named_modules=lambda: [('layers.0.attention.to_q', object())]))

    def test_write_report_replaces_the_previous_one(self):
        write_report(self.folder, {'rel_median': 0.05})
        path = self.folder / REPORT_NAME
        self.assertEqual(json.loads(path.read_text())['rel_median'], 0.05)
        write_report(self.folder, {'rel_median': 0.08})
        self.assertEqual(json.loads(path.read_text())['rel_median'], 0.08)
        self.assertEqual(list(self.folder.glob('.training-*')), [])


if __name__ == '__main__':
    unittest.main()
