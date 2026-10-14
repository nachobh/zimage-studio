import io
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gen_minimax_h3 as client
from PIL import Image


class MiniMaxClientTests(unittest.TestCase):
    def test_internal_comfy_status_is_atomic_minimal_and_path_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            job_dir=Path(directory)/('.job-'+'a'*32);job_dir.mkdir()
            target=job_dir/'comfyui.json'
            with patch.dict('os.environ',{'MINIMAX_H3_STATUS_FILE':str(target)}):
                client.write_internal_job_status('b'*32,'minimax_h3_'+'c'*32)
            self.assertEqual(__import__('json').loads(target.read_text()),{'prompt_id':'b'*32,'prefix':'minimax_h3_'+'c'*32})
            self.assertEqual(target.stat().st_mode&0o777,0o600)
            with patch.dict('os.environ',{'MINIMAX_H3_STATUS_FILE':str(Path(directory)/'outside.json')}):
                with self.assertRaises(ValueError):client.write_internal_job_status('b'*32,'minimax_h3_'+'c'*32)

    def test_graph_saves_one_diagnostic_frame_and_no_unused_audio_branch(self):
        args = SimpleNamespace(prompt="camina", width=960, height=544, seconds=5, steps=8, with_audio=False)
        graph, length = client.build_prompt("input.png", args, 42, "minimax_h3_test")
        self.assertEqual(length, 124)
        self.assertNotIn("14", graph)
        self.assertNotIn("4", graph)
        self.assertEqual(graph["18"]["inputs"]["length"], 1)
        self.assertEqual(graph["17"]["inputs"]["images"], ["18", 0])
        self.assertEqual(graph["16"]["inputs"]["filename_prefix"], "minimax_h3_test")

    def test_audio_option_connects_both_decoders_to_video(self):
        args = SimpleNamespace(prompt="camina", width=960, height=544, seconds=5, steps=8, with_audio=True)
        graph, _ = client.build_prompt("input.png", args, 42, "minimax_h3_test")
        self.assertEqual(graph["14"]["inputs"]["samples"], ["12", 0])
        self.assertEqual(graph["15"]["inputs"]["audio"], ["14", 0])

    def test_optional_start_and_end_frames(self):
        args = SimpleNamespace(prompt="una escena", width=320, height=480,
                               seconds=5, steps=8, with_audio=False)
        graph, _ = client.build_prompt(None, args, 42, "minimax_h3_test", "end.png")
        self.assertNotIn("5", graph)
        self.assertNotIn("first_frame", graph["6"]["inputs"])
        self.assertEqual(graph["6"]["inputs"]["last_frame"], ["19", 0])
        self.assertEqual(graph["19"]["inputs"]["image"], "end.png")
        graph, _ = client.build_prompt(None, args, 42, "minimax_h3_test")
        self.assertNotIn("last_frame", graph["6"]["inputs"])

    def test_guided_prompt_routes_voice_and_music_to_audio(self):
        answers = iter(["", "Una persona saluda", "320x480", "5", "s",
                        "Hola, ¿qué tal?", "s", "Piano suave", ""])
        with patch("builtins.input", side_effect=lambda _: next(answers)):
            args = client.guided_values()
        self.assertIsNone(args.image)
        self.assertIsNone(args.last_image)
        self.assertTrue(args.with_audio)
        self.assertIn("<d>[Spanish] Hola, ¿qué tal?</d>", args.prompt)
        self.assertIn("non_diegetic_music: Piano suave", args.prompt)

    def test_execution_error_is_reduced_to_one_useful_line(self):
        status = {"messages": [
            ["execution_start", {"prompt_id": "job"}],
            ["execution_error", {"node_id": "6", "node_type": "SamplerCustomAdvanced",
                                 "exception_type": "torch.OutOfMemoryError",
                                 "exception_message": "CUDA out of memory.\nTried to allocate 2.00 GiB\n",
                                 "current_inputs": {"samples": "tensor(device='cuda:0')"},
                                 "current_outputs": ["tensor(device='cuda:0')"]}],
        ]}
        message = client.describe_failure("job", status)
        self.assertIn("Memoria GPU insuficiente", message)
        self.assertIn("nodo 6", message)
        self.assertNotIn("tensor(", message)
        self.assertNotIn("current_inputs", message)
        generic = client.describe_failure("job", {"messages": [["execution_error", {
            "exception_type": "ValueError", "exception_message": "entrada no válida\nresto"}]]})
        self.assertEqual(generic, "ComfyUI falló: ValueError · entrada no válida")
        self.assertEqual(client.describe_failure("job", {"messages": []}), "ComfyUI falló en el trabajo job")

    def test_completed_job_never_selects_an_old_video(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "minimax_h3_old_00001_.mp4").write_bytes(b"old")
            source = output / "minimax_h3_new_00001_.mp4"
            source.write_bytes(b"new")
            with patch.object(client, "OUTPUT_DIR", output), patch.object(
                client, "get_json", return_value={"job": {"status": {"completed": True}}}
            ):
                self.assertEqual(client.wait_for_result("job", "minimax_h3_new", 1), source)
                with self.assertRaisesRegex(RuntimeError, "no produjo un único MP4"):
                    client.wait_for_result("job", "minimax_h3_missing", 1)

    def test_output_copy_does_not_replace_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            target = Path(directory) / "target.mp4"
            source.write_bytes(b"new")
            target.write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                client.copy_new(source, target)
            self.assertEqual(target.read_bytes(), b"keep")

    def test_upload_uses_unique_names_for_concurrent_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "input.png"
            Image.new("RGB", (8, 8), "blue").save(image)
            names = []

            def respond(req, timeout):
                names.append(re.search(rb'filename="([^"]+)"', req.data).group(1))
                return io.BytesIO(b'{"name":"uploaded.png"}')

            with patch.object(client.request, "urlopen", side_effect=respond):
                client.upload(image)
                client.upload(image)
            self.assertNotEqual(names[0], names[1])
            self.assertTrue(all(name.endswith(b".png") for name in names))


if __name__ == "__main__":
    unittest.main()
