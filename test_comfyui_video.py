import base64
from io import BytesIO
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import comfyui_video as video
import zimage_panel_grid as panel


def image_data_url():
    stream = BytesIO()
    Image.new("RGB", (2, 2), (12, 34, 56)).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


class _JsonResponse:
    """Minimal context manager compatible with json.load()."""

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def _fake_urlopen(stats):
    def open_url(target, *args, **kwargs):
        url = str(getattr(target, "full_url", target))
        if url.endswith("/system_stats"):
            return _JsonResponse(stats)
        return _JsonResponse({"queue_running": [], "queue_pending": []})
    return open_url


class VideoAdapterTests(unittest.TestCase):
    def test_media_audit_never_classifies_unknown_as_deletable(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(video,'OUTPUT_DIR',Path(directory)):
            (Path(directory)/('a'*32+'.mp4')).write_bytes(b'clip')
            (Path(directory)/('b'*32+'.mp4')).write_bytes(b'private-or-unknown')
            report=video.audit_media({'a'*32})
            self.assertEqual([item['classification'] for item in report['files']],['referenced_public','reference_unknown'])
            self.assertIn('no se eliminan',report['note'])

    def test_durable_job_record_omits_prompts_and_restores_completed_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'videos';state=root/'jobs';output.mkdir()
            job_id='a'*32;request_id='b'*32;(output/(job_id+'.mp4')).write_bytes(b'video')
            record={'id':job_id,'request_id':request_id,'status':'executing','created':1.0,
                    'prompt':'private words','video_generation':{'prompt':'private words'},'chat_id':'private'}
            with patch.object(video,'JOB_STATE_DIR',state),patch.object(video,'OUTPUT_DIR',output),\
                 patch.object(video,'_jobs',{}),patch.object(video,'_request_jobs',{}):
                video._persist_job(record)
                (state/(('c'*32)+'.json')).write_text('[]')
                raw=(state/(job_id+'.json')).read_text()
                self.assertNotIn('private words',raw);self.assertNotIn('chat_id',raw)
                video._jobs.clear();video._request_jobs.clear();video._restore_jobs()
                self.assertEqual(video.job(request_id=request_id)['status'],'done')
                self.assertEqual(set(video._jobs),{job_id})

    def test_import_video_publishes_new_opaque_done_job_without_overwrite(self):
        content=b'\x00\x00\x00\x18ftypisom'+b'x'*64
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(video,'OUTPUT_DIR',root/'videos'),patch.object(video,'JOB_STATE_DIR',root/'jobs'),patch.object(video,'_jobs',{}):
                job_id=video.import_video(content)
                self.assertRegex(job_id,r'^[0-9a-f]{32}$')
                self.assertEqual(video.video_path(job_id).read_bytes(),content)
                self.assertEqual(video.job(job_id)['status'],'done')
                self.assertTrue((root/'jobs'/(job_id+'.json')).is_file())

    def test_restart_reconciles_exact_comfy_prompt_and_recovers_mp4(self):
        job_id='a'*32;prompt_id='b'*32;prefix='minimax_h3_'+'c'*32
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'videos';state=root/'state';comfy=root/'comfy-output'
            output.mkdir();comfy.mkdir();(comfy/(prefix+'_00001_.mp4')).write_bytes(b'\x00\x00\x00\x18ftypisomclip')
            record={'id':job_id,'prompt_id':prompt_id,'prefix':prefix,'status':'interrupted','recovery_required':True,'created':1,'updated':1}
            with patch.object(video,'OUTPUT_DIR',output),patch.object(video,'JOB_STATE_DIR',state),patch.object(video,'COMFYUI_OUTPUT_DIR',comfy),patch.object(video,'_jobs',{job_id:record}),patch.object(video,'_request_jobs',{}),patch.object(video,'urlopen',return_value=_JsonResponse({prompt_id:{'status':{'completed':True}}})):
                restored=video.job(job_id)
                self.assertEqual(restored['status'],'done',restored)
                self.assertEqual((output/(job_id+'.mp4')).read_bytes(),(comfy/(prefix+'_00001_.mp4')).read_bytes())
                self.assertFalse(restored['recovery_required'])

    def test_restart_keeps_exact_comfy_prompt_executing_without_duplicate(self):
        job_id='a'*32;prompt_id='b'*32
        with tempfile.TemporaryDirectory() as directory:
            state=Path(directory)/'state'
            record={'id':job_id,'prompt_id':prompt_id,'prefix':'minimax_h3_'+'c'*32,'status':'interrupted','recovery_required':True,'created':1,'updated':1}
            with patch.object(video,'JOB_STATE_DIR',state),patch.object(video,'_jobs',{job_id:record}),patch.object(video,'_request_jobs',{}),patch.object(video,'urlopen',side_effect=[_JsonResponse({}),_JsonResponse({'queue_running':[[0,prompt_id]],'queue_pending':[]})]):
                self.assertEqual(video.job(job_id)['status'],'executing')
                self.assertTrue(video._jobs[job_id]['recovery_required'])

    def test_worker_records_comfy_prompt_id_before_process_finishes(self):
        job_id='a'*32;prompt_id='b'*32;prefix='minimax_h3_'+'c'*32
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'videos';job_dir=output/('.job-'+job_id);job_dir.mkdir(parents=True)
            image=job_dir/'reference.png';image.write_bytes(b'png');prompt=job_dir/'prompt.txt';prompt.write_text('private prompt')
            record={'id':job_id,'request_id':'d'*32,'status':'queued','created':1,'updated':1}
            class FakeProcess:
                returncode=0
                checks=0
                def __init__(self,command,**kwargs):self.command=command;self.env=kwargs['env']
                def poll(self):
                    self.checks+=1
                    if self.checks==1:
                        Path(self.env['MINIMAX_H3_STATUS_FILE']).write_text(json.dumps({'prompt_id':prompt_id,'prefix':prefix}));return None
                    Path(self.command[self.command.index('--output')+1]).write_bytes(b'\x00\x00\x00\x18ftypisomclip');return 0
            with patch.object(video,'OUTPUT_DIR',output),patch.object(video,'JOB_STATE_DIR',output/'.jobs'),patch.object(video,'_jobs',{job_id:record}),patch.object(video,'_waiting',[job_id]),patch.object(video,'_active',None),patch.object(video,'_preflight'),patch.object(video.subprocess,'Popen',FakeProcess),patch.object(video.time,'sleep'):
                video._run(job_id,image,prompt,output/(job_id+'.mp4'),False)
                self.assertEqual(video._jobs[job_id]['status'],'done',video._jobs[job_id].get('error'))
                self.assertEqual(video._jobs[job_id]['prompt_id'],prompt_id)
                self.assertEqual(video._jobs[job_id]['prefix'],prefix)
                saved=json.loads((output/'.jobs'/(job_id+'.json')).read_text())
                self.assertEqual(saved['prompt_id'],prompt_id)
                self.assertNotIn('private prompt',(output/'.jobs'/(job_id+'.json')).read_text())

    def test_panel_video_job_api_contract(self):
        def request(raw):
            server_side, client_side = socket.socketpair()
            try:
                client_side.sendall(raw)
                client_side.shutdown(socket.SHUT_WR)
                app = SimpleNamespace(snapshot=lambda: None)
                panel.Handler(server_side, ("local", 0), SimpleNamespace(app=app))
                server_side.shutdown(socket.SHUT_WR)
                chunks = []
                while data := client_side.recv(4096):
                    chunks.append(data)
                return b"".join(chunks).split(b"\r\n\r\n", 1)
            finally:
                server_side.close()
                client_side.close()

        body = b'{"prompt":"Hola","image_data":"test","with_audio":true}'
        raw = (b"POST /api/video/generate HTTP/1.1\r\nHost: localhost\r\nContent-Length: "
               + str(len(body)).encode() + b"\r\n\r\n" + body)
        with patch.object(video, "submit", return_value={"status": "queued", "job_id": "a" * 32}) as submit:
            headers, response = request(raw)
        self.assertIn(b"202 Accepted", headers)
        self.assertIn(b'"job_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"', response)
        submit.assert_called_once_with({"prompt": "Hola", "image_data": "test", "with_audio": True})
        with patch.object(video, "job", return_value={"status": "done", "video": "/api/video/file?id=" + "a" * 32}):
            headers, response = request(("GET /api/video/job?id=" + "a" * 32 + " HTTP/1.1\r\nHost: localhost\r\n\r\n").encode())
        self.assertIn(b"200 OK", headers)
        self.assertIn(b'"status": "done"', response)
        with patch.object(video,'submit',side_effect=video.VideoQueueFull('cola llena')):
            headers,response=request(raw)
        self.assertIn(b'409 Conflict',headers)
        self.assertIn('cola llena'.encode(),response)
        delete=("DELETE /api/video/job/"+'a'*32+" HTTP/1.1\r\nHost: localhost\r\n\r\n").encode()
        with patch.object(video,'cancel_job',return_value={'id':'a'*32,'status':'cancelled'}):headers,response=request(delete)
        self.assertIn(b'200 OK',headers);self.assertIn(b'"status": "cancelled"',response)
        with patch.object(video,'cancel_job',side_effect=RuntimeError('No se envió interrupt global')):headers,response=request(delete)
        self.assertIn(b'409 Conflict',headers)

    def test_panel_serves_video_ranges_and_rejects_invalid_ranges(self):
        def request(header):
            server_side, client_side = socket.socketpair()
            try:
                client_side.sendall(("GET /api/video/file?id=" + "a" * 32 + " HTTP/1.1\r\nHost: localhost\r\n" + header + "\r\n").encode())
                client_side.shutdown(socket.SHUT_WR)
                app = SimpleNamespace(snapshot=lambda: None)
                with patch.object(video, "video_path", return_value=video_file):
                    panel.Handler(server_side, ("local", 0), SimpleNamespace(app=app))
                server_side.shutdown(socket.SHUT_WR)
                chunks = []
                while data := client_side.recv(4096):
                    chunks.append(data)
                return b"".join(chunks).split(b"\r\n\r\n", 1)
            finally:
                server_side.close()
                client_side.close()

        with tempfile.TemporaryDirectory() as directory:
            video_file = Path(directory) / ("a" * 32 + ".mp4")
            video_file.write_bytes(b"0123456789")
            headers, body = request("Range: bytes=2-5\r\n")
            self.assertIn(b"206 Partial Content", headers)
            self.assertIn(b"Content-Range: bytes 2-5/10", headers)
            self.assertEqual(body, b"2345")
            headers, body = request("Range: bytes=-3\r\n")
            self.assertIn(b"206 Partial Content", headers)
            self.assertEqual(body, b"789")
            headers, body = request("Range: bytes=99-\r\n")
            self.assertIn(b"416 Range Not Satisfiable", headers)
            self.assertIn(b"Content-Range: bytes */10", headers)
            self.assertEqual(body, b"")
            headers, body = request("")
            self.assertIn(b"200 OK", headers)
            self.assertIn(b"Accept-Ranges: bytes", headers)
            self.assertEqual(body, b"0123456789")

    def test_rejects_external_mime_and_malformed_images(self):
        for value in ("https://example.invalid/image.png", "data:image/png;base64,not-base64"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                video._image_data(value)

    def test_failure_text_drops_tensor_dumps(self):
        raw = "\n".join([
            'Traceback (most recent call last):',
            '  File "/opt/ComfyUI/execution.py", line 424, in format_value',
            "RuntimeError: tensor(device='cuda:0', shape=(1, 64))",
            "  File \"/opt/ComfyUI/execution.py\", line 999, in execute",
            "RuntimeError: Memoria GPU insuficiente para MiniMax-H3 (nodo 6): sin memoria.",
        ])
        text = video._failure_text(raw)
        self.assertEqual(text, "RuntimeError: Memoria GPU insuficiente para MiniMax-H3 (nodo 6): sin memoria.")
        self.assertNotIn("tensor(", text)
        self.assertEqual(video._failure_text("\n".join(["tensor(device='cuda:0')"] * 50)), "")

    def test_submit_with_comfyui_down_fails_with_a_clean_message(self):
        with patch.object(video, "_alive", return_value=False), patch.object(video, "_start", return_value=False), \
             patch.object(video, "CLIENT", Path(__file__)), patch.object(video, "STARTER", Path(__file__)), \
             patch.object(video, "PYTHON", Path(__file__)):
            with self.assertRaisesRegex(RuntimeError, "MiniMax-H3 no está activo") as raised:
                video._preflight()
        self.assertNotIn("Traceback", str(raised.exception))

    def test_preflight_refuses_when_vram_is_low(self):
        stats = {"devices": [{"name": "cuda:0", "torch_vram_free": 1024**3}]}
        with patch.object(video, "urlopen", new=_fake_urlopen(stats)), \
             patch.object(video, "MIN_FREE_BYTES", 12 * 1024**3):
            with self.assertRaisesRegex(RuntimeError, "Memoria GPU libre insuficiente"):
                video._preflight()
        healthy = {"devices": [{"name": "cuda:0", "torch_vram_free": 40 * 1024**3}]}
        with patch.object(video, "urlopen", new=_fake_urlopen(healthy)), \
             patch.object(video, "MIN_FREE_BYTES", 12 * 1024**3):
            video._preflight()

    def test_preflight_prefers_total_device_free_over_torch_allocator_cache(self):
        # ComfyUI's torch_vram_free is the reclaimable allocator cache, not all
        # available VRAM. This reproduces the ROCm system_stats payload.
        stats = {"devices": [{
            "name": "cuda:0 AMD Radeon 8060S Graphics : native",
            "vram_total": 58886279168,
            "vram_free": 32390643712,
            "torch_vram_total": 2097152,
            "torch_vram_free": 2097152,
        }]}
        self.assertEqual(video._free_bytes(stats), 32390643712)
        with patch.object(video, "urlopen", new=_fake_urlopen(stats)), \
             patch.object(video, "MIN_FREE_BYTES", 12 * 1024**3):
            video._preflight()

    def test_preflight_releases_oom_resident_models_before_memory_check(self):
        low = {"devices": [{"name": "cuda:0", "vram_free": 1024**3}]}
        recovered = {"devices": [{"name": "cuda:0", "vram_free": 20 * 1024**3}]}
        calls = []
        released = False

        def open_url(target, *args, **kwargs):
            nonlocal released
            url = str(getattr(target, "full_url", target))
            calls.append(url.rsplit("/", 1)[-1])
            if url.endswith("/free"):
                released = True
                return _JsonResponse({})
            if url.endswith("/system_stats"):
                return _JsonResponse(recovered if released else low)
            return _JsonResponse({"queue_running": [], "queue_pending": []})

        with patch.object(video, "_alive", return_value=True), \
             patch.object(video, "urlopen", new=open_url), \
             patch.object(video, "MIN_FREE_BYTES", 12 * 1024**3):
            video._preflight()

        self.assertLess(calls.index("free"), max(i for i, name in enumerate(calls) if name == "system_stats"))

    def test_resolution_defaults_and_validation(self):
        self.assertEqual(video._resolution({}), (320, 480))
        self.assertEqual(video._resolution({"width": 640}), (640, 480))
        self.assertEqual(video._resolution({"height": 384}), (320, 384))
        self.assertEqual(video._resolution({"width": 640, "height": 384}), (640, 384))
        for value in ({"width": 100}, {"width": 1921}, {"height": 0}, {"width": True},
                      {"width": "640"}, {"height": 96.0}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                video._resolution(value)

    def test_submit_passes_resolution_to_the_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "videos"

            def reset():
                with video._lock:
                    video._jobs.clear()
                    video._active = None
                    video._waiting.clear()

            with patch.object(video, "OUTPUT_DIR", output), patch.object(video, "JOB_STATE_DIR", output/'.jobs'), patch.object(video, "CLIENT", Path(__file__)), \
                 patch.object(video, "STARTER", Path(__file__)), patch.object(video, "_preflight"), \
                 patch.object(video.threading, "Thread") as thread:
                reset()
                try:
                    video.submit({"image_data": image_data_url(), "prompt": "Hola", "width": 640, "height": 384})
                    self.assertEqual(thread.call_args.kwargs["args"][-2:], (640, 384))
                    reset()
                    video.submit({"image_data": image_data_url(), "prompt": "Hola"})
                    self.assertEqual(thread.call_args.kwargs["args"][-2:], (320, 480))
                    reset()
                    with self.assertRaisesRegex(ValueError, "múltiplos de 32"):
                        video.submit({"image_data": image_data_url(), "prompt": "Hola", "width": 100})
                finally:
                    reset()

    def test_submit_creates_one_opaque_job_and_private_input_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "videos"
            with patch.object(video, "OUTPUT_DIR", output), patch.object(video, "JOB_STATE_DIR", output/'.jobs'), patch.object(video, "CLIENT", Path(__file__)), \
                 patch.object(video, "STARTER", Path(__file__)), patch.object(video, "_preflight"), \
                 patch.object(video.threading, "Thread") as thread:
                with video._lock:
                    video._jobs.clear()
                    video._active = None
                    video._waiting.clear()
                request_id='c'*32
                request={"image_data": image_data_url(), "prompt": "<d>[Spanish] Hola.</d>", "with_audio": True,'request_id':request_id}
                result = video.submit(request)
                self.assertEqual(result["status"], "queued")
                self.assertRegex(result["job_id"], r"^[0-9a-f]{32}$")
                thread.return_value.start.assert_called_once()
                duplicate=video.submit(request)
                self.assertTrue(duplicate['idempotent']);self.assertEqual(duplicate['job_id'],result['job_id'])
                self.assertEqual(thread.call_count,1)
                with self.assertRaisesRegex(ValueError,'contenido distinto'):
                    video.submit({**request,'prompt':'Otro contenido'})
                job_directory = next(path for path in output.glob(".*") if path.name != '.jobs')
                self.assertEqual(job_directory.stat().st_mode & 0o777, 0o700)
                self.assertTrue((job_directory / "reference.png").is_file())
                self.assertTrue((job_directory / "prompt.txt").is_file())
                queued=video.submit({"image_data": image_data_url(), "prompt": "Otro"})
                self.assertEqual(queued['status'],'queued')
                self.assertEqual(thread.call_count,2)
                with video._queue_condition:video._jobs.clear();video._active=None;video._waiting.clear()

    def test_video_queue_accepts_one_active_plus_three_waiting_and_cancels_only_waiting(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'videos'
            with patch.object(video,'OUTPUT_DIR',output),patch.object(video,'JOB_STATE_DIR',output/'.jobs'),patch.object(video,'CLIENT',Path(__file__)),patch.object(video,'STARTER',Path(__file__)),patch.object(video,'_preflight') as preflight,patch.object(video.threading,'Thread'):
                with video._queue_condition:video._jobs.clear();video._waiting.clear();video._active=None
                created=[video.submit({'image_data':image_data_url(),'prompt':f'Escena {index}'}) for index in range(4)]
                preflight.assert_not_called()
                self.assertEqual([item['queue_position'] for item in created],[1,2,3,4])
                with self.assertRaises(video.VideoQueueFull):video.submit({'image_data':image_data_url(),'prompt':'Exceso'})
                cancelled=video.cancel_job(created[2]['job_id'])
                self.assertEqual(cancelled['status'],'cancelled')
                self.assertFalse((output/('.job-'+created[2]['job_id'])).exists())
                active_id=created[0]['job_id']
                with video._queue_condition:
                    video._waiting.remove(active_id);video._active=active_id;video._jobs[active_id]['status']='executing'
                with self.assertRaisesRegex(RuntimeError,'interrupt global'):
                    video.cancel_job(active_id)
                with video._queue_condition:video._jobs.clear();video._waiting.clear();video._active=None

    def test_video_lookup_accepts_only_opaque_ids_and_regular_mp4(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(video, "OUTPUT_DIR", Path(directory)):
            video_file = Path(directory) / ("a" * 32 + ".mp4")
            video_file.write_bytes(b"mp4")
            self.assertEqual(video.video_path("a" * 32), video_file)
            self.assertIsNone(video.video_path("../secret"))
            self.assertIsNone(video.job("../secret"))


if __name__ == "__main__":
    unittest.main()
