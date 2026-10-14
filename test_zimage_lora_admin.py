"""Unit tests for the live LoRA rescan path, with no model or GPU."""
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from zimage_lora_admin import reload_lora_adapters


class FakeTransformer:
    def __init__(self, fail=None):
        self.deleted = []
        self.fail = fail

    def delete_adapters(self, name):
        if name == self.fail:
            raise RuntimeError('test failure')
        self.deleted.append(name)


class LoraAdminTests(unittest.TestCase):
    def test_reload_replaces_adapters_and_skips_unsafe_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'New_Char.safetensors').touch()
            (root / 'bad;name.safetensors').touch()
            adapters = {'Old': '/old.safetensors'}
            transformer = FakeTransformer()

            def loader(name, path):
                adapters[name] = path
                return True

            result = reload_lora_adapters(SimpleNamespace(transformer=transformer), adapters, root, loader)
            self.assertEqual(transformer.deleted, ['Old'])
            self.assertEqual(result, {'status': 'ok', 'loaded': ['New_Char'],
                                      'failed': ['bad;name'], 'lora_count': 1})
            self.assertEqual(set(adapters), {'New_Char'})

    def test_unload_failure_aborts_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            adapters = {'Old': '/old.safetensors'}
            loader = Mock(return_value=True)
            result = reload_lora_adapters(SimpleNamespace(transformer=FakeTransformer('Old')),
                                          adapters, temporary, loader)
            self.assertEqual(result['status'], 'error')
            self.assertEqual(set(adapters), {'Old'})
            loader.assert_not_called()

    def test_missing_backend_or_directory_returns_error(self):
        self.assertEqual(reload_lora_adapters(None, {}, '/missing', lambda *_: True)['status'], 'error')


if __name__ == '__main__':
    unittest.main()
