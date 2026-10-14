#!/usr/bin/env python3
"""Read-only storage audit for panel videos and conversation media."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time

DATA_DIR=Path(os.environ.get('ZIMAGE_DATA_DIR',Path(__file__).resolve().parent/'data'))
CONVERSATIONS_DIR=Path(os.environ.get('ZIMAGE_CONVERSATIONS_DIR',DATA_DIR/'conversations'))
VIDEOS_DIR=Path(os.environ.get('MINIMAX_H3_OUTPUT_DIR',DATA_DIR/'videos'))
IMAGE_OUTPUT_DIR=Path(os.environ.get('ZIMAGE_OUTPUT_DIR',DATA_DIR/'outputs'))
COMFY_OUTPUT_DIR=Path(os.environ.get('MINIMAX_H3_COMFYUI_DIR','~/ComfyUI-minimax-h3'))/'output'
MEDIA_EXTENSIONS={'.png','.jpg','.webp','.mp4'}
EXTENSION_BY_MIME={'image/png':'.png','image/jpeg':'.jpg','image/webp':'.webp','video/mp4':'.mp4'}
MEDIA_ID=re.compile(r'[0-9a-f]{32}\Z')


def file_sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def _pending_intents(media_dir):
    pending=media_dir/'.pending';result=[]
    if not pending.is_dir() or pending.is_symlink():return result
    for marker in sorted(pending.glob('*.json')):
        if marker.is_symlink() or not MEDIA_ID.fullmatch(marker.stem):continue
        try:value=json.loads(marker.read_text(encoding='utf-8'))
        except (OSError,ValueError):
            result.append({'id':marker.stem,'classification':'invalid_intent'});continue
        expected={'version','id','mime','extension','size','sha256'}
        mime=value.get('mime') if isinstance(value,dict) else None
        extension=value.get('extension') if isinstance(value,dict) else None
        valid=(isinstance(value,dict) and set(value)==expected and type(value.get('version')) is int and value.get('version')==1 and
               value.get('id')==marker.stem and isinstance(mime,str) and isinstance(extension,str) and EXTENSION_BY_MIME.get(mime)==extension and
               type(value.get('size')) is int and value['size']>0 and
               isinstance(value.get('sha256'),str) and re.fullmatch(r'[0-9a-f]{64}',value['sha256']))
        result.append(value if valid else {'id':marker.stem,'classification':'invalid_intent'})
    return result


def cleanup_confirmed(conversations_dir):
    """Delete only marker-backed, unregistered files while publication is locked."""
    conversations_dir=Path(conversations_dir);media_dir=conversations_dir/'media'
    lock_path=media_dir/'.media-operation.lock';db_path=conversations_dir/'conversations.sqlite3'
    pending_dir=media_dir/'.pending'
    if media_dir.is_symlink() or not media_dir.is_dir() or pending_dir.is_symlink():return {'status':'media_directory_unavailable','deleted':[]}
    if not lock_path.is_file() or lock_path.is_symlink() or lock_path.resolve().parent!=media_dir.resolve():return {'status':'lock_unavailable','deleted':[]}
    with lock_path.open('rb') as lock_file:
        try:fcntl.flock(lock_file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {'status':'writer_active','deleted':[]}
        if not db_path.is_file() or db_path.is_symlink():return {'status':'database_unavailable','deleted':[]}
        uri=db_path.resolve().as_uri()+'?mode=ro'
        try:
            with sqlite3.connect(uri,uri=True,timeout=5) as db:
                rows={row[0] for row in db.execute('SELECT id FROM media')}
                snapshots=[row[0] for row in db.execute('SELECT snapshot FROM conversations')]
        except sqlite3.Error:return {'status':'database_unavailable','deleted':[]}
        referenced=set()
        for snapshot in snapshots:referenced.update(re.findall(r'/api/conversations/media/([0-9a-f]{32})',snapshot))
        deleted=[];cleared_intents=[]
        if pending_dir.resolve().parent!=media_dir.resolve():return {'status':'media_directory_unavailable','deleted':[]}
        for intent in _pending_intents(media_dir):
            if intent.get('classification')=='invalid_intent' or intent.get('id') in rows or intent.get('id') in referenced:continue
            media_id=intent['id'];marker=media_dir/'.pending'/(media_id+'.json')
            target=media_dir/(media_id+intent['extension']);temporary=media_dir/('.'+media_id)
            candidate=target if target.is_file() and not target.is_symlink() else temporary if temporary.is_file() and not temporary.is_symlink() else None
            if candidate is None:
                marker.unlink(missing_ok=True);cleared_intents.append(media_id);continue
            if candidate.resolve().parent!=media_dir.resolve():continue
            try:
                if candidate.stat().st_size!=intent['size'] or file_sha256(candidate)!=intent['sha256']:continue
                candidate.unlink()
                if candidate==target and temporary.is_file() and not temporary.is_symlink():temporary.unlink()
                marker.unlink(missing_ok=True)
                deleted.append(media_id)
            except OSError:continue
        return {'status':'ok','deleted':deleted,'cleared_intents':cleared_intents}


def audit(conversations_dir=CONVERSATIONS_DIR,videos_dir=VIDEOS_DIR,image_output_dir=IMAGE_OUTPUT_DIR,comfy_output_dir=COMFY_OUTPUT_DIR):
    conversations_dir=Path(conversations_dir);videos_dir=Path(videos_dir)
    image_output_dir=Path(image_output_dir);comfy_output_dir=Path(comfy_output_dir)
    media_dir=conversations_dir/'media';db_path=conversations_dir/'conversations.sqlite3'
    rows={};public_video_ids=set();public_image_paths=set();snapshot_media_ids=set();database_error=None
    if db_path.is_file() and not db_path.is_symlink():
        uri=db_path.resolve().as_uri()+'?mode=ro'
        try:
            with sqlite3.connect(uri,uri=True,timeout=5) as db:
                db.row_factory=sqlite3.Row
                for row in db.execute('SELECT id,mime,size,sha256,refcount FROM media'):
                    if re.fullmatch(r'[0-9a-f]{32}',row['id'] or ''):
                        rows[row['id']]={'mime':row['mime'],'size':row['size'],'sha256':row['sha256'],'refcount':row['refcount']}
                for row in db.execute('SELECT snapshot,visibility FROM conversations'):
                    snapshot_media_ids.update(re.findall(r'/api/conversations/media/([0-9a-f]{32})',row['snapshot']))
                    if row['visibility']!='public':continue
                    public_video_ids.update(re.findall(r'/api/video/file\?id=([0-9a-f]{32})',row['snapshot']))
                    for encoded in re.findall(r'/api/image\?path=([^"\s]+)',row['snapshot']):
                        try:
                            from urllib.parse import unquote
                            public_image_paths.add(Path(unquote(encoded)).resolve())
                        except (OSError,RuntimeError):pass
        except sqlite3.Error as error:
            database_error=f'{type(error).__name__}: {error}'
    media=[]
    if media_dir.is_dir() and not media_dir.is_symlink():
        for path in sorted(media_dir.iterdir()):
            if path.is_symlink() or not path.is_file() or path.suffix not in MEDIA_EXTENSIONS:continue
            item=rows.get(path.stem)
            classification='reference_unknown';integrity=None
            if item:
                try:
                    valid_ext=EXTENSION_BY_MIME.get(item['mime'])==path.suffix
                    integrity=valid_ext and path.stat().st_size==item['size'] and file_sha256(path)==item['sha256']
                except OSError:integrity=False
                classification=('sqlite_referenced' if item['refcount']>0 else 'registry_zero_refcount') if integrity else 'registry_file_mismatch'
            media.append({'id':path.stem,'size':path.stat().st_size,
                          'classification':classification,'integrity':integrity,
                          'record_size':item['size'] if item else None,'refcount':item['refcount'] if item else None})
    media_records=[]
    for media_id,item in rows.items():
        extension=EXTENSION_BY_MIME.get(item['mime']);path=media_dir/(media_id+extension) if extension else None
        exists=bool(path and not path.is_symlink() and path.is_file())
        if exists:
            try:valid=path.stat().st_size==item['size'] and file_sha256(path)==item['sha256']
            except OSError:valid=False
        else:valid=False
        classification=('referenced_file_missing' if item['refcount']>0 else 'registry_zero_refcount') if not exists else ('sqlite_referenced' if item['refcount']>0 and valid else 'registry_file_mismatch' if not valid else 'registry_zero_refcount')
        media_records.append({'id':media_id,'refcount':item['refcount'],'size':item['size'],'file_present':exists,'integrity':valid if exists else None,'classification':classification})
    lock_path=media_dir/'.media-operation.lock';writer_status='unavailable'
    if lock_path.is_file() and not lock_path.is_symlink():
        try:
            with lock_path.open('rb') as lock_file:
                try:
                    fcntl.flock(lock_file.fileno(),fcntl.LOCK_SH|fcntl.LOCK_NB);writer_status='idle'
                    fcntl.flock(lock_file.fileno(),fcntl.LOCK_UN)
                except BlockingIOError:writer_status='active'
        except OSError:writer_status='unavailable'
    publication_intents=[]
    for intent in _pending_intents(media_dir):
        media_id=intent.get('id')
        if intent.get('classification')=='invalid_intent':
            publication_intents.append(intent);continue
        if writer_status=='active':classification='writer_active'
        elif database_error or not db_path.is_file():classification='database_unknown'
        elif media_id in rows:classification='registered_or_already_published'
        elif media_id in snapshot_media_ids:classification='referenced_without_registry'
        else:
            path=media_dir/(media_id+intent['extension'])
            try:integrity=path.is_file() and not path.is_symlink() and path.stat().st_size==intent['size'] and file_sha256(path)==intent['sha256']
            except OSError:integrity=False
            classification='pending_unpublished_candidate' if integrity else 'pending_file_mismatch'
        publication_intents.append({'id':media_id,'size':intent['size'],'classification':classification})
    jobs=[];temporary=[];videos=[];job_by_id={}
    state_dir=videos_dir/'.jobs'
    if state_dir.is_dir() and not state_dir.is_symlink():
        for path in sorted(state_dir.glob('*.json')):
            if path.is_symlink() or not re.fullmatch(r'[0-9a-f]{32}\.json',path.name):continue
            try:value=json.loads(path.read_text(encoding='utf-8'))
            except (OSError,ValueError):continue
            if isinstance(value,dict):
                safe={key:value[key] for key in ('id','status','created','updated','size','prompt_id','prefix') if key in value}
                jobs.append(safe)
                if re.fullmatch(r'[0-9a-f]{32}',value.get('id','')):job_by_id[value['id']]=value.get('status','unknown')
    if videos_dir.is_dir() and not videos_dir.is_symlink():
        for path in sorted(videos_dir.iterdir()):
            if path.is_symlink():continue
            if path.is_dir() and re.fullmatch(r'\.job-[0-9a-f]{32}',path.name):
                job_id=path.name[5:];status=job_by_id.get(job_id)
                classification='job_active' if status in ('queued','executing') else 'job_terminal_cleanup_or_unknown' if status else 'owner_unknown'
                temporary.append({'name':path.name,'classification':classification,'job_status':status,'bytes':sum(child.stat().st_size for child in path.iterdir() if child.is_file() and not child.is_symlink())})
            elif path.is_file() and path.suffix=='.mp4' and re.fullmatch(r'[0-9a-f]{32}\.mp4',path.name):
                videos.append({'id':path.stem,'size':path.stat().st_size,'classification':'referenced_public' if path.stem in public_video_ids else 'reference_unknown'})
            elif path.is_file() and path.name.startswith('.import-'):
                temporary.append({'name':path.name,'classification':'import_owner_unknown','bytes':path.stat().st_size})
    external=[]
    for kind,root in (('zimage_output',image_output_dir),('comfyui_output',comfy_output_dir)):
        if not root.is_dir() or root.is_symlink():continue
        for path in sorted(root.iterdir()):
            if path.is_symlink() or not path.is_file():continue
            if kind=='zimage_output' and path.suffix.lower() not in {'.png','.jpg','.jpeg','.webp'}:continue
            if kind=='comfyui_output' and not re.fullmatch(r'minimax_h3_[0-9a-f]{32}_\d+_\.mp4',path.name):continue
            if kind=='zimage_output':
                classification='referenced_public' if path.resolve() in public_image_paths else 'reference_unknown'
            else:
                prefix=path.name.rsplit('_',2)[0]
                classification='known_comfy_job_output' if any(record.get('prefix')==prefix for record in jobs) else 'comfy_reference_unknown'
            external.append({'root':kind,'name':path.name,'size':path.stat().st_size,'classification':classification})
    return {'version':1,'read_only':True,'generated_at':int(time.time()),
            'database_present':db_path.is_file() and not db_path.is_symlink(),'database_error':database_error,
            'media_writer_status':writer_status,
            'conversation_media':media,'media_registry':media_records,'publication_intents':publication_intents,
            'videos':videos,'jobs':jobs,'temporary':temporary,
            'outside_managed_paths':{'scanned_roots':[str(path) for path in (image_output_dir,comfy_output_dir) if path.is_dir() and not path.is_symlink()],'files':external,'other_paths':'not_scanned'},
            'deletion_policy':'Sólo se permite borrar un medio de conversación si hay intención durable creada antes de publicarlo, no existe registro ni referencia en snapshot, y el hash/tamaño coinciden bajo el lock compartido. Medios desconocidos, outputs externos y vídeos de ComfyUI nunca se borran.'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conversations-dir',type=Path,default=CONVERSATIONS_DIR)
    parser.add_argument('--videos-dir',type=Path,default=VIDEOS_DIR)
    parser.add_argument('--image-output-dir',type=Path,default=IMAGE_OUTPUT_DIR)
    parser.add_argument('--comfy-output-dir',type=Path,default=COMFY_OUTPUT_DIR)
    parser.add_argument('--dry-run',action='store_true',default=True,help='Sólo informa; es el comportamiento predeterminado.')
    parser.add_argument('--delete-confirmed',action='store_true',help='Borra sólo archivos con intención durable, sin registro ni referencias y con hash/tamaño coincidentes; lo demás se conserva.')
    args=parser.parse_args(argv)
    result=audit(args.conversations_dir,args.videos_dir,args.image_output_dir,args.comfy_output_dir)
    if args.delete_confirmed:
        result['read_only']=False
        result['delete_requested']=cleanup_confirmed(args.conversations_dir)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
