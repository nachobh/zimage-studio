import contextlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import bootstrap_models as models


class BootstrapTests(unittest.TestCase):
    def test_panel_serves_configured_output_and_rejects_other_paths(self):
        import zimage_panel_grid as panel
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / 'outputs'
            output.mkdir()
            image = output / 'test.png'
            image.write_bytes(b'png fixture')
            handler = object.__new__(panel.Handler)
            handler.server = types.SimpleNamespace(app=types.SimpleNamespace(snapshot=lambda: None))
            with patch.dict(os.environ, {'ZIMAGE_OUTPUT_DIR':str(output)}), patch.object(handler, 'reply') as reply:
                handler.path = '/api/image?path=' + str(image)
                handler.do_GET()
                self.assertEqual(reply.call_args.args[0], b'png fixture')
                handler.path = '/api/image?path=' + str(root / 'outside.png')
                handler.do_GET()
                self.assertEqual(reply.call_args.args[1], 400)

    def test_dry_run_needs_no_python_models_or_network(self):
        with tempfile.TemporaryDirectory(prefix='studio space ') as folder:
            root = Path(folder)
            for name in ('start.sh', 'bootstrap_host.sh'):
                shutil.copyfile(Path(__file__).parent / name, root / name)
            before = set(root.iterdir())
            result = subprocess.run(['bash', str(root / 'start.sh'), '--dry-run'],
                                    cwd='/', env={**os.environ, 'PYTHON_BOOTSTRAP':'missing-python'},
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(before, set(root.iterdir()))

    def test_partial_base_is_resumed_and_custom_paths_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = root / 'external-base'
            base.mkdir()
            (base / 'model_index.json').write_text('{}')
            transformer = root / 'custom.gguf'
            transformer.write_bytes(b'existing')
            qwen = root / 'custom-chat.gguf'
            qwen.write_bytes(b'existing')
            projection = root / 'mmproj-BF16.gguf'
            projection.write_bytes(b'existing')
            snapshots, downloads = [], []
            def snapshot(*args, **kwargs):
                snapshots.append(kwargs['local_dir'])
            def download(repo, name, local_dir):
                downloads.append(name)
                p = Path(local_dir) / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b'fixture')
            hub = types.SimpleNamespace(snapshot_download=snapshot, hf_hub_download=download)
            env = {'ZIMAGE_BASE_PIPELINE':str(base), 'ZIMAGE_TRANSFORMER_GGUF':str(transformer),
                   'MODEL_FILE':str(qwen), 'LLAMA_MMPROJ':str(projection),
                   'ZIMAGE_TEXT_ENCODER_DIR':str(root/'external-text')}
            with patch.dict(os.environ, env), patch.dict(sys.modules, {'huggingface_hub':hub}), patch.object(sys, 'argv', ['bootstrap', '--data', str(root/'data')]), contextlib.redirect_stdout(io.StringIO()):
                models.main()
                models.main()
            self.assertEqual(snapshots, [str(base)])
            self.assertNotIn('custom.gguf', downloads)
            self.assertNotIn('custom-chat.gguf', downloads)
            self.assertEqual(len(downloads), 5)


if __name__ == '__main__':
    unittest.main()
