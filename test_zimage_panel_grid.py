"""CPU-only regression tests; fixture mode also supports browser interaction tests."""
import argparse
import base64
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from http.server import ThreadingHTTPServer

import zimage_panel_grid as panel
import zimage_mcp_server as mcp

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a3ioAAAAASUVORK5CYII=')
NAME = 'Aimee_Lou_Wood_(vrtlAimeeLouWood)'


def args(directory):
    return SimpleNamespace(lora_dir=str(directory), backend_host='127.0.0.1', backend_port=9877,
                           mcp_url=None, chat_url='http://127.0.0.1:8086/v1', chat_model='')


class ArchiveTests(unittest.TestCase):
    def test_profile_personality_presets_are_bounded_and_unique(self):
        preset={'id':'p-'+'a'*32,'name':'Cercana personalizada','revision':1,'system_prompt':'Habla con calidez.'}
        profile={'system_prompt':'base','image_prompt':'vrtlAda','personality_presets':[preset]}
        self.assertIs(panel.checked_profile(profile),profile)
        named={**preset,'id':'cercana_2026'}
        self.assertEqual(panel.checked_profile({**profile,'personality_presets':[named]})['personality_presets'],[named])
        with self.assertRaises(ValueError):
            panel.checked_profile({**profile,'personality_presets':[preset,preset]})
        with self.assertRaises(ValueError):
            panel.checked_profile({**profile,'personality_presets':[dict(preset,id='../../perfil')]})
        with self.assertRaises(ValueError):
            panel.checked_profile({**profile,'personality_presets':[dict(preset,id='builtin-close')]})

    def test_archive_roundtrip_and_rejections(self):
        import io
        import zipfile
        for kind in ('zimage-conversation', 'zimage-group'):
            raw=json.dumps({'format':kind,'version':1,'text':'Español e imágenes','image':base64.b64encode(PNG).decode()}).encode()
            packed=panel.conversation_archive(raw)
            self.assertEqual(panel.conversation_archive(packed,unpack=True),raw)
            with zipfile.ZipFile(io.BytesIO(packed)) as archive:self.assertIn('manifest.json',archive.namelist())
            old=io.BytesIO()
            with zipfile.ZipFile(old,'w') as archive:archive.writestr('conversation.json',raw)
            self.assertEqual(panel.conversation_archive(old.getvalue(),unpack=True),raw)
            bad=io.BytesIO()
            with zipfile.ZipFile(bad,'w') as archive:
                archive.writestr('conversation.json',raw);archive.writestr('manifest.json','{}')
            with self.assertRaisesRegex(ValueError,'manifiesto'):panel.conversation_archive(bad.getvalue(),unpack=True)
            with patch.object(panel,'CONVERSATION_LIMIT',len(raw)-1):
                with self.assertRaises(ValueError): panel.conversation_archive(packed,unpack=True)
        with self.assertRaises(ValueError): panel.conversation_archive(b'broken',unpack=True)
        with self.assertRaises(ValueError): panel.conversation_archive(b'{}')
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,'w') as archive: archive.writestr('../conversation.json',raw)
        with self.assertRaises(ValueError): panel.conversation_archive(buf.getvalue(),unpack=True)

    def test_archive_v2_detaches_mp4_deduplicates_and_remaps_import_ids(self):
        import io
        import zipfile
        import hashlib
        content=b'\x00\x00\x00\x18ftypisom'+b'x'*128
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/('a'*32+'.mp4');source.write_bytes(content)
            payload={'format':'zimage-conversation','version':1,'view':[
                {'kind':'video','src':'/api/video/file?id='+'a'*32,'job_id':'a'*32,'view_id':'v1'},
                {'kind':'video','src':'/api/video/file?id='+'b'*32,'job_id':'b'*32,'view_id':'v2'}]}
            raw=json.dumps(payload).encode()
            with patch.object(panel,'_archive_video',side_effect=[('a'*32,content),('b'*32,content)]):
                packed=panel.conversation_archive(raw,video_path=lambda _job:source)
            with zipfile.ZipFile(io.BytesIO(packed)) as archive:
                self.assertEqual(set(archive.namelist()),{'conversation.json','manifest.json','media/'+hashlib.sha256(content).hexdigest()+'.mp4'})
                self.assertEqual(archive.getinfo('media/'+hashlib.sha256(content).hexdigest()+'.mp4').compress_type,zipfile.ZIP_STORED)
                manifest=json.loads(archive.read('manifest.json'))
                self.assertEqual(manifest['archive_version'],2);self.assertEqual(len(manifest['videos']),2)
            ids=iter(('c'*32,'d'*32))
            with patch.object(panel,'_validate_imported_mp4') as validate:
                restored=json.loads(panel.conversation_archive(packed,unpack=True,video_importer=lambda _bytes:next(ids)))
            self.assertEqual([item['job_id'] for item in restored['view']],['c'*32,'c'*32])
            self.assertEqual([item['src'] for item in restored['view']],['/api/video/file?id='+'c'*32,'/api/video/file?id='+'c'*32])
            self.assertEqual(validate.call_count,1)

    def test_archive_v2_rejects_missing_or_changed_mp4_before_publish(self):
        import io
        import zipfile
        import hashlib
        content=b'\x00\x00\x00\x18ftypisom'+b'x'*64
        digest=hashlib.sha256(content).hexdigest();job='a'*32
        conversation=json.dumps({'format':'zimage-conversation','version':1,'view':[{'kind':'video','src':'/api/video/file?id='+job,'job_id':job}]}).encode()
        manifest=json.dumps({'archive_version':2,'videos':[{'job_id':job,'entry':'media/'+digest+'.mp4','sha256':digest,'size':len(content)}]}).encode()
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,'w') as archive:archive.writestr('conversation.json',conversation);archive.writestr('manifest.json',manifest);archive.writestr('media/'+digest+'.mp4',content+b'changed')
        importer=Mock()
        with patch.object(panel,'_validate_imported_mp4'):
            with self.assertRaises(ValueError):panel.conversation_archive(buf.getvalue(),unpack=True,video_importer=importer)
        importer.assert_not_called()
        empty_manifest=io.BytesIO()
        no_media=json.dumps({'format':'zimage-conversation','version':1,'view':[{'kind':'video','src':'/api/video/file?id='+job,'job_id':job}]}).encode()
        with zipfile.ZipFile(empty_manifest,'w') as archive:
            archive.writestr('conversation.json',no_media);archive.writestr('manifest.json',json.dumps({'archive_version':2,'videos':[]}))
        with self.assertRaisesRegex(ValueError,'no coinciden'):panel.conversation_archive(empty_manifest.getvalue(),unpack=True)


class PanelTests(unittest.TestCase):
    def test_only_identities_in_actual_prompt_keep_their_weight(self):
        second = 'Second_(vrtlSecond)'
        style = 'Style'
        for name in (second, style): (self.root / (name+'.safetensors')).touch()
        loras = [{'name':NAME,'weight':.4},{'name':second,'weight':.6},{'name':style,'weight':.3}]
        def effective(prompt):
            return panel.image_settings({'loras':loras,'image_prompt':prompt},self.app.profiles)['lora']
        self.assertEqual(effective('vrtlSecond portrait'),f'{NAME}:0.0;{second}:0.6;{style}:0.3')
        self.assertEqual(effective('vrtlSecondExtra portrait'),f'{NAME}:0.0;{second}:0.0;{style}:0.3')
        self.assertEqual(effective('vrtlAimeeLouWood and VRTLSECOND'),f'{NAME}:0.4;{second}:0.6;{style}:0.3')
        self.assertEqual(loras[0]['weight'],.4)

    def test_group_camera_is_forwarded_as_vision_content(self):
        src='data:image/png;base64,'+base64.b64encode(PNG).decode()
        body={'participants':[{'name':NAME}], 'speaker':NAME, 'tools_enabled':False,
              'transcript':[{'speaker':'Humano','text':'Mira esto','attachments':[{'src':src}]}]}
        with patch.object(self.app, 'chat', return_value={'messages':[]}) as chat:
            self.app.group_turn(body)
        history=chat.call_args.args[0]['messages']
        self.assertNotIn('base64',history[0]['content'])
        self.assertEqual(history[1]['content'][1]['image_url']['url'],src)
        body['transcript'][0]['attachments'][0]['src']='http://external/image.png'
        with self.assertRaises(ValueError): self.app.group_turn(body)

    def test_group_turn_uses_own_persona_and_shared_transcript(self):
        second = 'Second_(vrtlSecond)'
        (self.root / (second+'.safetensors')).touch()
        body = {'participants':[{'name':NAME,'weight':.4,'system_prompt':'Primera voz'},
                                {'name':second,'weight':.6,'system_prompt':'Segunda voz'}],
                'speaker':second,'parent_prompt':'Conversad sobre cine.',
                'transcript':[{'speaker':'Humano','text':'Hola'}, {'speaker':NAME,'text':'Prefiero comedias'}],
                'tools_enabled':False}
        with patch.object(self.app,'completion', return_value={'choices':[{'message':{'role':'assistant','content':'Yo también'}}]}) as call:
            result = self.app.group_turn(body)
        messages = call.call_args.args[0]['messages']
        self.assertIn('Segunda voz',messages[0]['content'])
        self.assertNotIn('Primera voz',messages[0]['content'])
        self.assertIn('Prefiero comedias',messages[1]['content'])
        self.assertEqual(result['speaker'],second)
        settings = panel.image_settings({'loras':[{'name':NAME,'weight':.4},{'name':second,'weight':.6}]},self.app.profiles)
        self.assertEqual(settings['lora'],f'{NAME}:0.4;{second}:0.6')
        with self.assertRaises(ValueError):
            panel.image_settings({'loras':[{'name':'../escape','weight':1}]},self.app.profiles)

    def test_group_human_request_is_explicit_and_image_tool_executes(self):
        image = self.root / 'result.png'
        image.write_bytes(PNG)
        body = {'participants':[{'name':NAME,'weight':.4}], 'speaker':NAME,
                'transcript':[{'speaker':'Humano','text':'Genera una imagen mirando a cámara'}]}
        replies = [
            {'choices':[{'message':{'role':'assistant','content':None,'tool_calls':[
                {'id':'groupimage','type':'function','function':{'name':'generate_image','arguments':'{"prompt":"vrtlAimeeLouWood portrait"}'}}]}}]},
            {'choices':[{'message':{'role':'assistant','content':'Aquí está.'}}]}]
        with patch.object(self.app,'completion',side_effect=replies) as completion, patch.object(mcp,'backend_request',return_value={'status':'ok','path':str(image)}) as backend:
            result = self.app.group_turn(body)
        payload = completion.call_args_list[0].args[0]
        self.assertIn('Genera una imagen mirando a cámara',payload['messages'][2]['content'])
        self.assertNotIn('Personaje seleccionado: ninguno',payload['messages'][0]['content'])
        self.assertEqual(backend.call_args.args[0]['lora'],NAME+':0.4')
        self.assertEqual(len(result['images']),1)
        body['transcript'].append({'speaker':NAME,'text':'Aquí está.'})
        with patch.object(self.app,'completion',return_value={'choices':[{'message':{'content':'Seguimos'}}]}) as completion:
            self.app.group_turn(body)
        self.assertNotIn('Petición del humano para este turno',completion.call_args.args[0]['messages'][2]['content'])

    def test_group_prompt_separates_spoken_dialogue_from_visual_triggers(self):
        body={'participants':[{'name':NAME}], 'speaker':NAME, 'tools_enabled':False,
              'transcript':[{'speaker':'Humano','text':'Descríbeme una imagen y dime qué ves.'}]}
        with patch.object(self.app,'completion',return_value={'choices':[{'message':{'content':'Te veo en un estudio luminoso.\n\nvrtlAimeeLouWood retrato en un estudio luminoso'}}]}) as completion:
            self.app.group_turn(body)
        system=completion.call_args.args[0]['messages'][0]['content']
        self.assertIn('No empieces el diálogo hablado con un trigger vrtl',system)
        self.assertIn('párrafo visual independiente que empiece por el trigger vrtl exacto',system)
        self.assertIn('instrucción visual, no diálogo',system)

    def test_group_plain_text_tool_prompt_generates_image(self):
        image = self.root / 'result.png'
        image.write_bytes(PNG)
        body = {'participants':[{'name':NAME,'weight':.4}], 'speaker':NAME,
                'transcript':[{'speaker':'Humano','text':'Genera una imagen mirando a cámara'}]}
        replies = [
            {'choices':[{'message':{'role':'assistant','content':'<tool name="generate_image">\n\nvrtlAimeeLouWood portrait'}}]},
            {'choices':[{'message':{'role':'assistant','content':'Aquí está.'}}]}]
        with patch.object(self.app, 'completion', side_effect=replies), \
             patch.object(mcp, 'backend_request', return_value={'status':'ok','path':str(image)}) as backend:
            result = self.app.group_turn(body)
        self.assertEqual(backend.call_args.args[0]['prompt'], 'vrtlAimeeLouWood portrait')
        self.assertEqual(backend.call_args.args[0]['lora'], NAME + ':0.4')
        self.assertEqual(len(result['images']), 1)

    def test_embedded_group_image_prompt_is_narrow(self):
        self.assertEqual(panel.embedded_group_image_prompt('<tool anything>\n\nvrtlAimeeLouWood portrait'),
                         'vrtlAimeeLouWood portrait')
        self.assertIsNone(panel.embedded_group_image_prompt('vrtlAimeeLouWood portrait'))
        self.assertIsNone(panel.embedded_group_image_prompt('<tool anything>\n\nportrait'))

    def test_progress_registry_and_mcp_forwarding(self):
        from zimage_progress import Progress
        p = Progress()
        p.update('one','sampling',3,9)
        self.assertEqual(p.get('one')['step'],3)
        self.assertEqual(p.get('other')['phase'],'waiting')
        p.update('one','done')
        self.assertEqual(p.get('one')['phase'],'done')
        with patch.object(mcp,'backend_request', return_value={'status':'ok','phase':'sampling','step':3,'total':9}) as backend:
            result=mcp.ZImageMCP('localhost',9877).call('generation_progress',{'request_id':'one'})
        self.assertEqual(backend.call_args.args[0],{'cmd':'progress','request_id':'one'})
        self.assertEqual(json.loads(panel.result_text(result))['step'],3)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        env=patch.dict('os.environ',{'ZIMAGE_DATA_DIR':str(self.root/'data')});env.start();self.addCleanup(env.stop)
        (self.root / (NAME + '.safetensors')).touch()
        self.app = panel.App(args(self.root))
        self.body = {'lora_name': NAME, 'system_prompt': 'Responde como Alba.', 'image_prompt': 'vrtlAimeeLouWood portrait',
                     'messages': [{'role':'user','content':'Genera una imagen'}], 'image_settings': {'seed': 42}, 'lora_weight': .7}

    def test_settings_persist_and_pin_routes_for_running_requests(self):
        self.app.settings_path = self.root / 'panel-settings.json'
        old = self.app.snapshot()
        config = self.app.configuration()
        config.update(chat_url='http://localhost:18086/v1', chat_model='other-model',
                      zimage_url='tcp://localhost:19877', mcp_url='')
        config['defaults']['width'] = 640
        self.app.configure(config)
        self.assertEqual(old.args.chat_url, 'http://127.0.0.1:8086/v1')
        self.assertEqual(self.app.snapshot().args.backend_port,19877)
        with patch.object(panel, 'http_json', return_value=({},{})) as http:
            self.app.snapshot().completion({'messages':[]})
        self.assertEqual(http.call_args.args[0], 'http://localhost:18086/v1/chat/completions')
        self.assertEqual(http.call_args.args[1]['model'], 'other-model')
        restarted_args = args(self.root)
        restarted_args.settings_file = str(self.app.settings_path)
        self.assertEqual(panel.App(restarted_args).configuration(), self.app.configuration())
        saved = self.app.settings_path.read_bytes()
        for changes in ({'zimage_url':'http://localhost:9877'}, {'chat_url':'file:///tmp/model'},
                        {'defaults':{**config['defaults'],'width':513}}):
            with self.assertRaises(ValueError): self.app.configure({**config,**changes})
            self.assertEqual(self.app.settings_path.read_bytes(),saved)
        with patch.object(panel, 'MCPBridge') as bridge:
            self.app.snapshot().bridge()
        bridge.assert_called_once_with('localhost',19877,None)

    def test_profiles_persist_metadata_and_never_overwrite_on_init(self):
        store = self.app.profiles
        self.assertEqual(store.initialize(), 1)
        original = store.load(NAME)
        self.assertIn('vrtlAimeeLouWood Aimee Lou Wood', original['image_prompt'])
        original.update(system_prompt='Personalidad de prueba', image_prompt='Imagen de prueba', custom={'keep': True})
        store.save(NAME, original)
        self.assertEqual(store.initialize(), 0)
        self.assertEqual(panel.Profiles(self.root).load(NAME), original)
        self.assertFalse(store.catalog()[0]['thumbnail'])
        (self.root / (NAME + '.png')).write_bytes(PNG)
        self.assertTrue(store.catalog()[0]['thumbnail'])
        for name in ('../outside', '/tmp/outside', 'missing'):
            with self.assertRaises(ValueError): store.save(name, original)
        with self.assertRaises(ValueError): store.save(NAME, {'system_prompt': 'incomplete'})

    def test_profile_history_restore_keeps_conversation_snapshots_out_of_scope(self):
        store=self.app.profiles;store.initialize();original=store.load(NAME)
        store.save(NAME,{**original,'system_prompt':'Nueva personalidad'})
        history=store.history(NAME);self.assertEqual(len(history),1)
        store.restore(NAME,history[0]['version_id'])
        self.assertEqual(store.load(NAME)['system_prompt'],original['system_prompt'])
        self.assertEqual(len(store.history(NAME)),2)

    def test_voice_reference_uses_profile_basename_without_path_escape(self):
        store = self.app.profiles
        self.assertEqual(store.voice_path(NAME), self.root / (NAME + '.wav'))
        other = self.root / 'alternativa.wav'
        other.write_bytes(b'RIFF')
        store.save(NAME, {**store.default(NAME), 'voice_wav': other.name})
        self.assertEqual(store.voice_path(NAME), other)
        self.assertEqual(store.voice_path(NAME,''),self.root/(NAME+'.wav'))
        self.assertEqual(store.voice_path(NAME,other.name),other)
        with self.assertRaises(ValueError):store.voice_path(NAME,'../outside.wav')
        with self.assertRaises(ValueError):store.voice_path(NAME,None)
        for invalid in ('../outside.wav', '/tmp/outside.wav', '.hidden.wav', 'other.mp3'):
            with self.assertRaises(ValueError):
                store.save(NAME, {**store.default(NAME), 'voice_wav': invalid})
        other.unlink()
        other.symlink_to(self.root / 'outside.wav')
        with self.assertRaises(ValueError): store.voice_path(NAME)

    def test_model_tool_mcp_image_and_history_roundtrip(self):
        image = self.root / 'result.png'
        image.write_bytes(PNG)
        requests, backend_calls = [], []
        def complete(payload):
            requests.append(copy.deepcopy(payload))
            if len(requests) == 1:
                return {'choices':[{'message':{'role':'assistant','content':None,'tool_calls':[
                    {'id':'image1','type':'function','function':{'name':'generate_image','arguments':json.dumps({
                        'prompt':'portrait', 'lora':'wrong:3', 'cfg':15, 'filename':'ignored', 'output_dir':'/tmp/ignored'})}}]}}]}
            return {'choices':[{'message':{'role':'assistant','content':'Aquí tienes la imagen.'}}]}
        def backend(payload, host, port):
            backend_calls.append(payload)
            return {'status':'ok','path':str(image)}
        with patch.object(self.app,'completion',side_effect=complete), patch.object(mcp,'backend_request',side_effect=backend):
            result = self.app.chat(self.body)
        self.assertEqual(backend_calls[0]['lora'], NAME + ':0.7')
        self.assertEqual(backend_calls[0]['cfg'], 0)
        self.assertEqual(backend_calls[0]['seed'], 42)
        self.assertNotIn('output_dir', backend_calls[0])
        self.assertEqual(result['images'][0]['src'], 'data:image/png;base64,' + base64.b64encode(PNG).decode())
        self.assertEqual(result['images'][0]['generation']['lora_name'], NAME)
        self.assertEqual(result['images'][0]['generation']['image_settings']['seed'],42)
        self.assertTrue(requests[0]['messages'][0]['content'].startswith('Responde como Alba.'))
        self.assertEqual(requests[1]['messages'][-1]['tool_call_id'], 'image1')
        self.assertNotIn(base64.b64encode(PNG).decode(), json.dumps(requests))
        self.assertEqual([m['role'] for m in result['messages']], ['assistant','tool','assistant'])

    def test_no_tools_and_no_lora_dont_inherit_identity(self):
        body = {**self.body, 'tools_enabled': False, 'lora_name': '', 'system_prompt':'Sistema sin personaje'}
        with patch.object(self.app,'completion', return_value={'choices':[{'message':{'role':'assistant','content':'Hola'}}]}) as call:
            self.app.chat(body)
        payload = call.call_args.args[0]
        self.assertNotIn('tools', payload)
        self.assertEqual(payload['messages'][0]['content'], 'Sistema sin personaje')

    def test_multimodal_history_reaches_chat(self):
        content = [{'type':'text','text':'Describe'}, {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(PNG).decode()}}]
        body = {**self.body, 'tools_enabled':False, 'messages':[{'role':'user','content':content}]}
        with patch.object(self.app, 'completion', return_value={'choices':[{'message':{'role':'assistant','content':'Imagen'}}]}) as call:
            self.app.chat(body)
        self.assertEqual(call.call_args.args[0]['messages'][1]['content'],content)

    def test_failed_tool_is_reported_and_returned_to_model(self):
        first = {'choices':[{'message':{'role':'assistant','content':None,'tool_calls':[
            {'id':'bad1','type':'function','function':{'name':'generate_image','arguments':'invalid json'}}]}}]}
        second = {'choices':[{'message':{'role':'assistant','content':'No se pudo generar.'}}]}
        with patch.object(self.app,'completion',side_effect=[first,second]), patch.object(mcp,'backend_request') as backend:
            result=self.app.chat(self.body)
        backend.assert_not_called()
        self.assertTrue(result['events'][0]['error'])
        self.assertEqual(result['images'], [])
        self.assertIn('error', result['messages'][1]['content'])

    def test_http_mcp_session_protocol(self):
        from email.message import Message
        headers = Message(); headers['Mcp-Session-Id'] = 'session123'
        init = {'result':{'protocolVersion':'2024-11-05'}}
        listing = {'result':{'tools':mcp.tool_definitions()}}
        with patch.object(panel,'http_json', side_effect=[(init,headers),(None,Message()),(listing,Message())]) as http:
            bridge=panel.MCPBridge('localhost',9877,'http://localhost:8765/mcp')
            bridge.initialize(); bridge.rpc('tools/list')
        self.assertEqual(http.call_args_list[1].args[1]['method'],'notifications/initialized')
        self.assertEqual(bridge.headers['Mcp-Session-Id'],'session123')
        self.assertEqual(http.call_args_list[2].args[1]['method'],'tools/list')


    def test_upscale_route_enlarges_with_lanczos_and_validates(self):
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen
        from PIL import Image
        buffer = io.BytesIO(); Image.new('RGB', (64, 48), (12, 180, 90)).save(buffer, 'PNG')
        src = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
        server = ThreadingHTTPServer(('127.0.0.1', 0), panel.Handler); server.app = self.app
        threading.Thread(target=server.serve_forever, daemon=True).start()

        def post(body):
            request = Request('http://127.0.0.1:%d/api/upscale' % server.server_address[1],
                              data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            try:
                with urlopen(request, timeout=10) as response:
                    return response.status, json.loads(response.read())
            except HTTPError as error:
                return error.code, json.loads(error.read())

        try:
            status, value = post({'src': src, 'scale': 2})
            self.assertEqual(status, 200)
            self.assertEqual((value['width'], value['height']), (128, 96))
            self.assertTrue(value['src'].startswith('data:image/png;base64,'))
            self.assertEqual(value['bytes'], len(base64.b64decode(value['src'].split(',', 1)[1])))
            status, value = post({'src': src, 'scale': 9})
            self.assertEqual(status, 400)
            self.assertIn('Factor', value['error'])
            status, value = post({'src': '/api/image?path=/etc/passwd', 'scale': 2})
            self.assertEqual(status, 400)
            self.assertIn('no permitida', value['error'])
            status, value = post({'src': 'nada', 'scale': 2})
            self.assertEqual(status, 400)
        finally:
            server.shutdown(); server.server_close()

    def test_upscale_rejects_sources_beyond_the_pixel_cap(self):
        from PIL import Image
        buffer = io.BytesIO(); Image.new('RGB', (4097, 8)).save(buffer, 'PNG')
        src = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
        with self.assertRaises(ValueError):
            panel.upscale_image(src, 2)
        with self.assertRaises(ValueError):
            panel.upscale_image('/api/image?path=' + str(self.root / 'missing.png'), 2)


def fixture(port):
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        os.environ['ZIMAGE_DATA_DIR']=str(root/'data')
        for name in (NAME, 'Magali_Vazquez_(vrtlMaguiAnsuz)'):
            (root/(name+'.safetensors')).touch()
        (root/(NAME+'.png')).write_bytes(PNG)
        app=panel.App(args(root)); app.profiles.initialize()
        app.completion=lambda payload: {'choices':[{'message':{'role':'assistant','content':
            'Respuesta de prueba para: '+payload['messages'][0]['content'].splitlines()[0]}}]}
        def backend(payload,host,port):
            if payload.get('cmd') == 'reload_loras':
                return {'status':'ok','loaded':['Aimee_Lou_Wood_(vrtlAimeeLouWood)'],'failed':[],'lora_count':1}
            return {'status':'ok','model_loaded':True,'lora_count':2}
        server=ThreadingHTTPServer(('127.0.0.1',port),panel.Handler); server.app=app
        store=panel.Store(root/'conversations')
        panel.conversation_store=lambda:store
        print(f'Fixture: http://127.0.0.1:{port}',flush=True)
        with patch.object(panel,'backend_request',side_effect=backend): server.serve_forever()


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--fixture',type=int); options,remaining=parser.parse_known_args()
    if options.fixture: fixture(options.fixture)
    else: unittest.main(argv=['test_zimage_panel_grid.py',*remaining])
