import json
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from urllib.parse import quote

import zimage_media_audit as audit
from zimage_conversations import Store


class MediaAuditTests(unittest.TestCase):
    def test_read_only_report_separates_public_private_unknown_and_active_temps(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);conversations=root/'conversations';media=conversations/'media';videos=root/'videos';state=videos/'.jobs';images=root/'zimage-output';comfy=root/'comfy-output'
            media.mkdir(parents=True);state.mkdir(parents=True);images.mkdir();comfy.mkdir()
            db=sqlite3.connect(conversations/'conversations.sqlite3')
            db.executescript('CREATE TABLE media(id TEXT,mime TEXT,size INTEGER,sha256 TEXT,refcount INTEGER); CREATE TABLE conversations(snapshot TEXT,visibility TEXT);')
            png=b'png';zero=b'keep'
            db.execute('INSERT INTO media VALUES(?,?,?,?,?)',('1'*32,'image/png',len(png),hashlib.sha256(png).hexdigest(),2))
            db.execute('INSERT INTO media VALUES(?,?,?,?,?)',('7'*32,'image/jpeg',10,'f'*64,1))
            db.execute('INSERT INTO media VALUES(?,?,?,?,?)',('8'*32,'image/png',len(zero),hashlib.sha256(zero).hexdigest(),0));db.commit();db.close()
            (media/('1'*32+'.png')).write_bytes(png);(media/('2'*32+'.mp4')).write_bytes(b'mp4');(media/('8'*32+'.png')).write_bytes(zero)
            (videos/('3'*32+'.mp4')).write_bytes(b'mp4')
            (conversations/'conversations.sqlite3').chmod(0o600)
            (images/'linked.png').write_bytes(b'image');(images/'unknown.png').write_bytes(b'unknown')
            db=sqlite3.connect(conversations/'conversations.sqlite3');db.execute('INSERT INTO conversations VALUES(?,?)',('/api/video/file?id='+'3'*32+' /api/image?path='+quote(str(images/'linked.png')),'public'));db.commit();db.close()
            prefix='minimax_h3_'+'5'*32
            (state/(('4'*32)+'.json')).write_text(json.dumps({'id':'4'*32,'status':'executing','prompt_id':'5'*32,'prefix':prefix}))
            temp=videos/('.job-'+'4'*32);temp.mkdir();(temp/'reference.png').write_bytes(b'private')
            (comfy/(prefix+'_00001_.mp4')).write_bytes(b'generated')
            report=audit.audit(conversations,videos,images,comfy)
            self.assertEqual([item['classification'] for item in report['conversation_media']],['sqlite_referenced','reference_unknown','registry_zero_refcount'])
            self.assertEqual(report['media_registry'][1]['classification'],'referenced_file_missing')
            self.assertEqual(report['videos'][0]['classification'],'referenced_public')
            self.assertEqual(report['jobs'][0]['status'],'executing')
            self.assertEqual(report['temporary'][0]['classification'],'job_active')
            external=report['outside_managed_paths']['files']
            self.assertEqual([item['classification'] for item in external],['referenced_public','reference_unknown','known_comfy_job_output'])

    def test_delete_confirmed_never_deletes_reference_unknown_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);store=Store(root/'conversations');conversations=store.directory;videos=root/'videos';videos.mkdir()
            unknown=store.media_dir/(('a'*32)+'.png');unknown.write_bytes(b'keep')
            self.assertEqual(audit.main(['--conversations-dir',str(conversations),'--videos-dir',str(videos),'--image-output-dir',str(root/'none-img'),'--comfy-output-dir',str(root/'none-comfy'),'--delete-confirmed']),0)
            self.assertTrue(unknown.is_file())

    def test_cleanup_requires_durable_intent_and_shared_writer_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            store=Store(Path(directory)/'conversations');media_id='b'*32;data=b'unpublished';digest=hashlib.sha256(data).hexdigest()
            marker_dir=store.media_dir/'.pending';target=store.media_dir/(media_id+'.png')
            target.write_bytes(data)
            store._write_media_intent(media_id,'image/png','.png',data,digest)
            self.assertEqual(audit.cleanup_confirmed(store.directory),{'status':'ok','deleted':[media_id],'cleared_intents':[]})
            self.assertFalse(target.exists())
            self.assertFalse((marker_dir/(media_id+'.json')).exists())

            media_id='c'*32;target=store.media_dir/(media_id+'.png');target.write_bytes(data)
            store._write_media_intent(media_id,'image/png','.png',data,digest)
            with store.media_operation():
                self.assertEqual(audit.cleanup_confirmed(store.directory),{'status':'writer_active','deleted':[]})
            self.assertTrue(target.is_file())
            self.assertEqual(audit.cleanup_confirmed(store.directory),{'status':'ok','deleted':['c'*32],'cleared_intents':[]})

            media_id='d'*32;target=store.media_dir/(media_id+'.png');target.write_bytes(data)
            store._write_media_intent(media_id,'image/png','.png',data,digest)
            snapshot=json.dumps({'version':1,'view':[{'src':'/api/conversations/media/'+media_id}]})
            with store.connection() as db:
                db.execute('INSERT INTO conversations (id,kind,title,members,snapshot,revision,created_at,updated_at,visibility) VALUES (?,?,?,?,?,1,?,?,?)',( 'e'*32,'individual','Privada','[]',snapshot,'2026-10-01','2026-10-01','private'))
            self.assertEqual(audit.cleanup_confirmed(store.directory),{'status':'ok','deleted':[],'cleared_intents':[]})
            self.assertTrue(target.is_file())


if __name__=='__main__':unittest.main()
