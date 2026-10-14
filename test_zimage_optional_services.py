import tempfile
import unittest
import os
import subprocess
from pathlib import Path
from unittest.mock import patch
import zimage_optional_services as services
import check_zimage_copies as copies

class Tests(unittest.TestCase):
    def test_restart_qwen3_tts_stops_and_starts_only_qwen_without_install(self):
        with patch.object(services,'stop') as stop,patch.object(services,'ensure') as ensure:
            services.restart_qwen3_tts(False)
        stop.assert_called_once_with('qwen3_tts_trial')
        ensure.assert_called_once_with('qwen3_tts_trial',False)

    def test_restart_qwen3_tts_entrypoints_are_isolated(self):
        local=Path(__file__).resolve().parent
        project=local/'zimage_panel_project' if (local/'zimage_panel_project').is_dir() else local
        launchers=[project/'start.sh',project/'start_instalacion_existente.sh',
                   local/'start.sh',local/'start_full_multimedia.sh']
        for launcher in launchers:
            if not launcher.is_file():continue
            with self.subTest(launcher=str(launcher)),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);(root/'zimage_optional_services.py').write_text(
                    'import sys\nfrom pathlib import Path\n'
                    'Path(__file__).with_name("events").write_text(" ".join(sys.argv[1:]))\n')
                if launcher.name=='start_instalacion_existente.sh':
                    (root/launcher.name).write_bytes(launcher.read_bytes())
                    existing=root/'installed';existing.mkdir()
                    (existing/'start.sh').write_text(
                        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" > "$(dirname "$0")/delegated"\n')
                    env=dict(os.environ,ZIMAGE_EXISTING_START=str(existing/'start.sh'))
                    command=root/launcher.name
                elif launcher.name=='start_full_multimedia.sh':
                    (root/launcher.name).write_bytes(launcher.read_bytes())
                    (root/'start.sh').write_text(
                        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" > "$(dirname "$0")/delegated"\n')
                    env=os.environ.copy();command=root/launcher.name
                else:
                    (root/'start.sh').write_bytes(launcher.read_bytes())
                    env=os.environ.copy();command=root/'start.sh'
                subprocess.run(['bash',str(command),'--restart-qwen3-tts'],env=env,text=True,capture_output=True,check=True)
                if launcher.name in ('start_instalacion_existente.sh','start_full_multimedia.sh'):
                    delegated=(root/'installed'/'delegated') if launcher.name=='start_instalacion_existente.sh' else (root/'delegated')
                    self.assertEqual(delegated.read_text().strip(),'--restart-qwen3-tts')
                else:
                    self.assertEqual((root/'events').read_text().strip(),'--restart-qwen3-tts --no-install')

    def test_existing_service_does_not_install_or_start(self):
        with patch.object(services,'healthy',return_value=True),patch.object(services.subprocess,'run') as run,patch.object(services.subprocess,'Popen') as spawn:
            services.ensure('audio',True)
        run.assert_not_called();spawn.assert_not_called()

    def test_missing_environment_no_install_fails_without_mutation(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(services,'ROOT',Path(temp)),patch.object(services,'healthy',return_value=False),patch.object(services.socket,'socket') as sock,patch.object(services.subprocess,'run') as run:
            sock.return_value.__enter__.return_value.connect_ex.return_value=1
            with self.assertRaises(RuntimeError):services.ensure('audio',False)
            self.assertEqual(list(Path(temp).iterdir()),[])
            run.assert_not_called()

    def test_https_status_distinguishes_certificate_from_listener(self):
        with tempfile.TemporaryDirectory() as temp:
          with patch.object(services,'ROOT',Path(temp)),patch.object(services,'listening',return_value=False):
            self.assertIn('sin certificado local y sin listener',services.https_status())
            tls=Path(temp)/'.tls';tls.mkdir();(tls/'server.crt').touch()
            self.assertIn('certificado presente, pero el panel no está escuchando',services.https_status())
          empty=Path(temp)/'sin-certificado';empty.mkdir()
          with patch.object(services,'ROOT',empty),patch.object(services,'listening',return_value=True):
            self.assertIn('falta server.crt',services.https_status())

    def test_integrity_reports_modified_and_missing(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(copies,'COMMON',['sample']):
            a=Path(temp)/'a';b=Path(temp)/'b';a.mkdir();b.mkdir()
            self.assertTrue(copies.differences(a,b))
            (a/'sample').write_text('same');(b/'sample').write_text('same')
            self.assertEqual(copies.differences(a,b),[])
            (b/'sample').write_text('different')
            self.assertTrue(copies.differences(a,b))

    def test_stop_rejects_pid_belonging_to_another_process(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(services,'ROOT',Path(temp)):
            state=Path(temp)/'.service-state';state.mkdir()
            (state/'audio.pid').write_text(str(os.getpid())+'\n')
            with patch.object(services.os,'kill') as kill:
                with self.assertRaisesRegex(RuntimeError,'no ejecuta'):
                    services.stop('audio')
            kill.assert_not_called()

    def test_existing_start_restart_stops_then_starts_defaults(self):
        local=Path(__file__).resolve().parent
        if not (local/'zimage_panel_project').is_dir(): local=local.parent
        if not (local/'start.sh').is_file(): self.skipTest('No hay instalación existente vecina')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'start.sh').write_bytes((local/'start.sh').read_bytes())
            (root/'zimage_optional_services.py').write_text(
                'import sys\nfrom pathlib import Path\n'
                'with Path(__file__).with_name("events").open("a") as f: '
                'f.write("optional " + " ".join(sys.argv[1:]) + "\\n")\n')
            (root/'zimage_stack.sh').write_text(
                'printf "stack %s\\n" "$*" >> "$(dirname "$0")/events"\n')
            subprocess.run(['bash',str(root/'start.sh'),'--restart','--with-qwen3-tts',
                            '--https-ip','<PANEL_IP>'],check=True,capture_output=True)
            self.assertEqual((root/'events').read_text().splitlines(),[
                'optional --stop','stack --stop',
                'optional --audio --qwen3-tts-trial --https-ip <PANEL_IP>',
                'stack '])

    def test_existing_installation_wrapper_publishes_panel_by_default(self):
        wrapper=Path(__file__).resolve().parent/'zimage_panel_project/start_instalacion_existente.sh'
        if not wrapper.is_file(): wrapper=Path(__file__).with_name('start_instalacion_existente.sh')
        with tempfile.TemporaryDirectory() as temp:
            fake=Path(temp)/'start.sh'
            fake.write_text('printf "%s\\n" "$ZIMAGE_PANEL_HOST" "$*"\n')
            env=dict(os.environ,ZIMAGE_EXISTING_START=str(fake))
            env.pop('ZIMAGE_PANEL_HOST',None)
            def invoke(extra):
                result=subprocess.run(['bash',str(wrapper),'--restart','--dry-run'],
                                      env=env|extra,text=True,capture_output=True,check=True)
                return result.stdout.splitlines()
            self.assertEqual(invoke({})[0],'0.0.0.0')
            self.assertEqual(invoke({'ZIMAGE_PANEL_HOST':'127.0.0.1'})[0],'127.0.0.1')

    def test_stop_entrypoints_never_start_install_or_download(self):
        local=Path(__file__).resolve().parent
        project=local/'zimage_panel_project' if (local/'zimage_panel_project').is_dir() else local
        existing=project.parent/'start.sh'
        for launcher in [project/'start.sh', existing]:
            if not launcher.is_file(): continue
            with self.subTest(launcher=str(launcher)),tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                (root/'start.sh').write_bytes(launcher.read_bytes())
                (root/'zimage_optional_services.py').write_text(
                    'import sys\nfrom pathlib import Path\n'
                    'with Path(__file__).with_name("events").open("a") as f: '
                    'f.write("optional " + " ".join(sys.argv[1:]) + "\\n")\n')
                (root/'zimage_stack.sh').write_text('printf "stack %s\\n" "$*" >> "$(dirname "$0")/events"\n')
                subprocess.run(['bash',str(root/'start.sh'),'--stop','--with-qwen3-tts'],check=True,capture_output=True)
                self.assertEqual((root/'events').read_text().splitlines(),['optional --stop','stack --stop'])
                (root/'events').unlink()
                subprocess.run(['bash',str(root/'start.sh'),'--stop','--dry-run'],check=True,capture_output=True)
                self.assertFalse((root/'events').exists())

    def test_comfy_missing_launcher_warns_unless_required(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(services,'comfy_launcher',return_value=Path(temp)/'ausente.sh'),patch.object(services.subprocess,'run') as run,patch('builtins.print') as log:
            services.ensure_comfy(False)
            run.assert_not_called()
            self.assertIn('continúa sin ComfyUI',log.call_args[0][0])
            with self.assertRaises(RuntimeError):services.ensure_comfy(True)
        with patch.dict(os.environ,{'ZIMAGE_COMFYUI_REQUIRED':'1'}),patch.object(services,'comfy_launcher',return_value=Path(tempfile.gettempdir())/'ausente-zimage.sh'),patch.object(services.subprocess,'run') as run:
            with self.assertRaises(RuntimeError):services.ensure_comfy()
            run.assert_not_called()

    def test_comfy_launcher_failure_warns_unless_required(self):
        with tempfile.TemporaryDirectory() as temp:
            launcher=Path(temp)/'lanzador.sh';launcher.write_text('#!/usr/bin/env bash\nexit 3\n')
            with patch.object(services,'comfy_launcher',return_value=launcher),patch('builtins.print') as log:
                services.ensure_comfy(False)
                self.assertIn('terminó con 3',log.call_args[0][0])
                with self.assertRaises(subprocess.CalledProcessError):services.ensure_comfy(True)

    def test_comfyui_is_enabled_by_default_and_opt_out(self):
        local=Path(__file__).resolve().parent
        project=local/'zimage_panel_project' if (local/'zimage_panel_project').is_dir() else local
        existing=project.parent/'start.sh'
        for launcher,rel in [(project/'start.sh','levantar_minimax_h3.sh'),(existing,'zimage_panel_project/levantar_minimax_h3.sh')]:
            if not launcher.is_file():continue
            with self.subTest(launcher=str(launcher)),tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                (root/'start.sh').write_bytes(launcher.read_bytes())
                (root/'bootstrap_host.sh').write_text(':\n')
                def run(args,check=True):
                    return subprocess.run(['bash',str(root/'start.sh'),*args],text=True,capture_output=True,check=check)
                target=root/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text('#!/usr/bin/env bash\nexit 0\n')
                default=run(['--dry-run'])
                self.assertIn('MiniMax',default.stdout)
                self.assertIn('--comfyui',default.stdout)
                self.assertNotIn('MiniMax',run(['--dry-run','--no-comfyui']).stdout)
                target.unlink()
                without=run(['--dry-run'])
                self.assertNotIn('MiniMax',without.stdout)
                self.assertIn('sin lanzador MiniMax-H3',without.stderr)
                explicit=run(['--dry-run','--with-comfyui'],check=False)
                self.assertEqual(explicit.returncode,2)
                self.assertIn('requiere un lanzador existente',explicit.stderr)

if __name__=='__main__':unittest.main()
