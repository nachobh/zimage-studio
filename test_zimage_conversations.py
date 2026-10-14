import base64
from io import BytesIO
from pathlib import Path
import socket
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from zimage_conversations import Store, Conflict, checked_members, checked_notes, checked_image_fields, checked_personality
import zimage_panel_auth
import zimage_panel_grid as panel


class Profiles:
    def path(self, name, suffix):
        if name != 'Ada_(vrtlAda)' or suffix != '.json':
            raise ValueError('LoRA no encontrado')
        return Path('/tmp/Ada.json')


def image_url():
    output = BytesIO()
    Image.new('RGB', (2, 2), 'red').save(output, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode()


class StoreTests(unittest.TestCase):
    def test_conversation_notes_contract(self):
        note={'id':'a'*32,'text':'Prefiere hablar de música.'}
        self.assertEqual(checked_notes([note]),[note])
        for value in ([dict(note,id='../x')],[dict(note,text='x'*501)],[note,note],
                      [{'id':'b'*32,'text':'x'*2001},{'id':'c'*32,'text':'y'*2001}],
                      [{'id':str(i).zfill(32),'text':'nota'} for i in range(11)]):
            with self.subTest(value=len(value)),self.assertRaises(ValueError):checked_notes(value)

    def test_personality_contract_rejects_unknown_fields_and_accepts_named_presets(self):
        valid={'version':1,'base_system_prompt':'base','preset_id':'cercana_2026','preset_revision':1,'builder_fields':{'tone':'cálido','response_length':'normal'}}
        self.assertEqual(checked_personality(valid),valid)
        for invalid in (
            {**valid,'version':True},
            {**valid,'unexpected':'value'},
            {**valid,'preset_id':'builtin-other'},
            {**valid,'builder_fields':{'tone':'ok','inject':'bad'}},
        ):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):checked_personality(invalid)

    def test_structured_image_fields_are_bounded(self):
        valid={'version':1,'subject':'vrtlAda','background':'beach'}
        self.assertEqual(checked_image_fields(valid),valid)
        for value in ({**valid,'subject':'x'*5001},{**valid,'output_dir':'/tmp'},
                      {**valid,'version':2},{**valid,'negative_prompt':1}):
            with self.assertRaises(ValueError):checked_image_fields(value)

    def test_http_auth_gate_login_and_csrf(self):
        def exchange(raw, auth):
            server_side, client_side = socket.socketpair()
            try:
                client_side.sendall(raw)
                client_side.shutdown(socket.SHUT_WR)
                panel.Handler(server_side, ('192.0.2.1', 1234), SimpleNamespace(auth=auth, app=SimpleNamespace(snapshot=lambda: None)))
                server_side.shutdown(socket.SHUT_WR)
                chunks=[]
                while data:=client_side.recv(4096): chunks.append(data)
                return b''.join(chunks)
            finally:
                server_side.close();client_side.close()
        with tempfile.TemporaryDirectory() as directory, patch.object(zimage_panel_auth, 'STORE_DIR', Path(directory)), \
             patch.dict('os.environ', {'ZIMAGE_PANEL_PASSWORD': 'contraseña-de-prueba-larga'}):
            auth=zimage_panel_auth.PanelAuth(True)
            denied=exchange(b'GET /api/conversations HTTP/1.1\r\nHost: localhost\r\n\r\n',auth)
            self.assertIn(b'401 Unauthorized',denied)
            body='{"password":"contraseña-de-prueba-larga"}'.encode()
            granted=exchange(b'POST /api/auth/login HTTP/1.1\r\nHost: localhost\r\nContent-Length: '+str(len(body)).encode()+b'\r\n\r\n'+body,auth)
            self.assertIn(b'200 OK',granted)
            self.assertIn(b'HttpOnly; SameSite=Strict',granted)
            cookie=granted.split(b'Set-Cookie: ',1)[1].split(b';',1)[0]
            session=exchange(b'GET /api/auth/session HTTP/1.1\r\nHost: localhost\r\nCookie: '+cookie+b'\r\n\r\n',auth)
            self.assertIn(b'200 OK',session)
            rejected=exchange(b'PATCH /api/conversations/'+b'a'*32+b' HTTP/1.1\r\nHost: localhost\r\nCookie: '+cookie+b'\r\nContent-Length: 2\r\n\r\n{}',auth)
            self.assertIn(b'403 Forbidden',rejected)

    def test_lan_auth_requires_password_and_session(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(zimage_panel_auth, 'STORE_DIR', Path(directory)), \
             patch.dict('os.environ', {'ZIMAGE_PANEL_PASSWORD': 'contraseña-de-prueba-larga'}):
            auth = zimage_panel_auth.PanelAuth(True)
            self.assertIsNone(auth.login('incorrecta', '192.0.2.1'))
            token, csrf = auth.login('contraseña-de-prueba-larga', '192.0.2.1')
            self.assertEqual(auth.session('zimage_session=' + token), (token, csrf))
            self.assertIsNone(auth.session('zimage_session=otro'))

    def test_members_validate_names_and_weights(self):
        profiles = Profiles()
        self.assertEqual(checked_members('individual', [{'name': 'Ada_(vrtlAda)'}], profiles)[0]['name'], 'Ada_(vrtlAda)')
        named={'id':'cercana_2026','name':'Cercana','revision':1,'system_prompt':'Habla cálido.'}
        members=checked_members('individual',[{'name':'Ada_(vrtlAda)','profile':{'system_prompt':'base','personality_presets':[named]}}],profiles)
        self.assertEqual(members[0]['profile']['personality_presets'],[named])
        for members in ([{'name': '../secret'}], [{'name': 'Ada_(vrtlAda)', 'weight': 4}]):
            with self.assertRaises(ValueError):
                checked_members('individual', members, profiles)

    def test_create_update_page_media_conflict_and_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / 'private')
            fake_mp4=b'\x00\x00\x00\x18ftypmp42' + b'0'*8
            video_url='data:video/mp4;base64,'+base64.b64encode(fake_mp4).decode()
            snapshot = {'version': 1, 'view': [{'role': 'user', 'text': 'Hola'},
                         {'role':'assistant','kind':'video','src':video_url},
                         {'role': 'assistant', 'text': 'Mira', 'src': image_url()}], 'messages': []}
            members = [{'name': 'Ada_(vrtlAda)', 'weight': .8}]
            item = store.create('individual', 'Ada', members, snapshot, migration_id='old:ada')
            self.assertEqual(store.create('individual', 'Ada', members, snapshot, migration_id='old:ada')['id'], item['id'])
            expected_media_bytes=len(fake_mp4)+len(base64.b64decode(image_url().split(',',1)[1]))
            self.assertEqual(item['storage_bytes'],expected_media_bytes)
            self.assertEqual(item['storage_scope'],'servidor · medios registrados')
            self.assertEqual(list((store.media_dir/'.pending').glob('*.json')),[])
            self.assertEqual(len(store.list()['items']), 1)
            self.assertEqual(store.list()['items'][0]['storage_bytes'],expected_media_bytes)
            self.assertEqual(item['preview'], 'Mira')
            self.assertEqual([m['text'] for m in store.messages(item['id'], limit=1)['items']], ['Mira'])
            self.assertEqual([m['text'] for m in store.messages(item['id'], before=2)['items']], ['Hola'])
            video_src=item['snapshot']['view'][1]['src']
            src = item['snapshot']['view'][2]['src']
            self.assertTrue(src.startswith('/api/conversations/media/'))
            self.assertTrue(store.media(src.rsplit('/', 1)[-1])[0].is_file())
            self.assertEqual(store.expand_media(src), image_url())
            self.assertEqual(store.media(video_src.rsplit('/',1)[-1])[1],'video/mp4')
            self.assertEqual(store.expand_media(video_src),video_url)
            with self.assertRaises(Conflict):
                store.update(item['id'], 0, title='Otro')
            updated = store.update(item['id'], item['revision'], snapshot={'version': 1, 'view': []})
            self.assertEqual(updated['revision'], 2)
            self.assertIsNone(store.media(src.rsplit('/', 1)[-1]))
            self.assertTrue(store.delete(item['id'], 2))
            self.assertEqual(store.list()['items'], [])

    def test_archive_and_cursor(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory))
            first = store.create('individual', 'Primero', [{'name': ''}])
            second = store.create('individual', 'Segundo', [{'name': ''}])
            page = store.list(limit=1)
            self.assertEqual(len(page['items']), 1)
            self.assertIsNotNone(page['next_cursor'])
            self.assertEqual(len(store.list(limit=1, cursor=page['next_cursor'])['items']), 1)
            store.update(first['id'], first['revision'], archived=True)
            self.assertEqual(len(store.list()['items']), 1)
            self.assertEqual(store.list(archived=True)['items'][0]['id'], first['id'])
            self.assertEqual(store.get(second['id'])['title'], 'Segundo')

    def test_visibility_migrates_old_rows_and_accepts_client_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = sqlite3.connect(root / 'conversations.sqlite3')
            legacy.executescript('''
                CREATE TABLE conversations (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, title TEXT NOT NULL,
                    members TEXT NOT NULL, snapshot TEXT NOT NULL, revision INTEGER NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    archived_at TEXT, pinned INTEGER NOT NULL DEFAULT 0,
                    preview TEXT NOT NULL DEFAULT '', migration_id TEXT UNIQUE);
            ''')
            legacy.execute('INSERT INTO conversations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                           ('b' * 32, 'individual', 'Antigua', '[]', '{"version":1,"view":[]}', 3,
                            '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', None, 0, '', None))
            legacy.execute('PRAGMA user_version=1')
            legacy.commit(); legacy.close()
            store = Store(root)
            with sqlite3.connect(root / 'conversations.sqlite3') as db:
                self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0], 2)
            migrated = store.get('b' * 32)
            self.assertEqual(migrated['visibility'], 'public')
            self.assertEqual(migrated['revision'], 3)
            self.assertEqual(store.list()['items'][0]['visibility'], 'public')
            client_id = 'c' * 32
            created = store.create('individual', 'Nueva', [{'name': ''}], conversation_id=client_id)
            self.assertEqual(created['id'], client_id)
            self.assertEqual(created['visibility'], 'public')
            with self.assertRaises(Conflict):
                store.create('individual', 'Otra', [{'name': ''}], conversation_id=client_id)
            with self.assertRaises(ValueError):
                store.create('individual', 'Otra', [{'name': ''}], conversation_id='corto')
            with self.assertRaises(ValueError):
                store.create('individual', 'Otra', [{'name': ''}], visibility='secreta')
            store.update(client_id, created['revision'], visibility='private')
            self.assertEqual(store.get(client_id)['visibility'], 'private')


if __name__ == '__main__':
    unittest.main()


def http_exchange(handler, raw, server):
    server_side, client_side = socket.socketpair()
    try:
        client_side.sendall(raw)
        client_side.shutdown(socket.SHUT_WR)
        handler(server_side, ('192.0.2.1', 1234), server)
        server_side.shutdown(socket.SHUT_WR)
        chunks = []
        while data := client_side.recv(4096): chunks.append(data)
        return b''.join(chunks)
    finally:
        server_side.close(); client_side.close()


class SessionTests(unittest.TestCase):
    PASSWORD = 'contraseña-de-prueba-larga'

    def make(self, directory):
        with patch.object(zimage_panel_auth, 'STORE_DIR', Path(directory)), \
             patch.dict('os.environ', {'ZIMAGE_PANEL_PASSWORD': self.PASSWORD}):
            return zimage_panel_auth.PanelAuth(True)

    def test_sessions_persist_hashed_and_purge_after_three_months(self):
        with tempfile.TemporaryDirectory() as directory:
            auth = self.make(directory)
            token, csrf = auth.login(self.PASSWORD, '192.0.2.1')
            stored = Path(directory, 'panel-sessions.json')
            self.assertNotIn(token, stored.read_text())
            self.assertEqual(stored.stat().st_mode & 0o777, 0o600)
            # A restart keeps recent sessions.
            self.assertEqual(self.make(directory).session('zimage_session=' + token), (token, csrf))
            # Age every entry beyond 90 days: the next start drops it.
            import json, time
            data = json.loads(stored.read_text())
            for item in data.values():
                item['created'] = time.time() - 91 * 86400
                item['expires'] = time.time() + 3600
            stored.write_text(json.dumps(data))
            self.assertIsNone(self.make(directory).session('zimage_session=' + token))
            self.assertEqual(json.loads(stored.read_text()), {})

    def test_corrupt_session_file_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'panel-sessions.json').write_text('{no json')
            auth = self.make(directory)
            self.assertIsNotNone(auth.login(self.PASSWORD, '192.0.2.1'))

    def test_logout_invalidates_session_and_requires_csrf(self):
        with tempfile.TemporaryDirectory() as directory:
            auth = self.make(directory)
            token, csrf = auth.login(self.PASSWORD, '192.0.2.1')
            server = SimpleNamespace(auth=auth, app=SimpleNamespace(snapshot=lambda: None))
            cookie = b'Cookie: zimage_session=' + token.encode() + b'\r\n'
            denied = http_exchange(panel.Handler, b'POST /api/auth/logout HTTP/1.1\r\nHost: localhost\r\n' + cookie + b'\r\n', server)
            self.assertIn(b'403 Forbidden', denied)
            self.assertIsNotNone(auth.session('zimage_session=' + token))
            done = http_exchange(panel.Handler, b'POST /api/auth/logout HTTP/1.1\r\nHost: localhost\r\n' + cookie
                                 + b'X-Zimage-CSRF: ' + csrf.encode() + b'\r\n\r\n', server)
            self.assertIn(b'200 OK', done)
            self.assertIn(b'Max-Age=0', done)
            self.assertIn(b'X-Content-Type-Options: nosniff', done)
            self.assertIsNone(auth.session('zimage_session=' + token))
            self.assertFalse(self.make(directory).session('zimage_session=' + token))

    def test_decompression_bomb_is_a_value_error(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            members = [{'name': 'Ada_(vrtlAda)', 'weight': .8, 'image_enabled': True, 'profile': {}}]
            snapshot = {'version': 1, 'view': [{'kind': 'image', 'src': image_url()}]}
            with patch.object(Image, 'MAX_IMAGE_PIXELS', 1):
                with self.assertRaisesRegex(ValueError, 'bomba'):
                    store.create('individual', 'Prueba', members, snapshot)


class MCPServerTests(unittest.TestCase):
    def test_rpc_rejects_malformed_messages(self):
        import zimage_mcp_server as mcp
        server = mcp.ZImageMCP('127.0.0.1', 1)
        self.assertEqual(server.rpc([])['error']['code'], -32600)
        self.assertEqual(server.rpc({'id': 7})['error']['code'], -32600)
        self.assertEqual(server.rpc({'id': 7, 'method': 5})['error']['code'], -32600)
        self.assertIsNone(server.rpc({'id': 7, 'result': {}}))
        self.assertIsNone(server.rpc({'method': 'notifications/initialized'}))
        self.assertEqual(server.rpc({'id': 1, 'method': 'ping'})['result'], {})

    def test_defaults_and_post_limit(self):
        import zimage_mcp_server as mcp
        from pathlib import Path as P
        self.assertEqual(mcp.DEFAULT_BACKEND_PORT, 9877)
        self.assertEqual(mcp.IO_TIMEOUT, 1800)
        schema = mcp.tool_definitions()[0]['inputSchema']['properties']['output_dir']['default']
        self.assertTrue(schema.endswith('zimage_outputs') and P(schema).is_absolute())
        server = SimpleNamespace(zimage=mcp.ZImageMCP('127.0.0.1', 1))
        too_big = b'POST /mcp HTTP/1.1\r\nHost: x\r\nContent-Length: ' + str(1024 ** 3 + 1).encode() + b'\r\n\r\n'
        self.assertIn(b'413', http_exchange(mcp.HTTPHandler, too_big, server).split(b'\r\n', 1)[0])
        ok = b'{"jsonrpc":"2.0","id":1,"method":"ping"}'
        reply = http_exchange(mcp.HTTPHandler, b'POST /mcp HTTP/1.1\r\nHost: x\r\nContent-Length: ' + str(len(ok)).encode() + b'\r\n\r\n' + ok, server)
        self.assertIn(b'200 OK', reply)
        self.assertIn(b'Access-Control-Allow-Origin: *', reply)
