#!/usr/bin/env python3
"""Isolated KokoClone voice conversion; CPU, persistent model, no arbitrary paths."""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from zimage_audio_server import Handler as BaseHandler


class Converter:
    def __init__(self, repo):
        self.repo=Path(repo).resolve()
        self.model=None
        self.lock=threading.Lock()

    def convert(self, body):
        import soundfile as sf
        files={}
        for name,seconds in [('reference',30),('source',180)]:
            data=base64.b64decode(body.get(name,''),validate=True)
            if not 0<len(data)<=20*1024**2:raise ValueError('Audio vacío o mayor de 20 MB.')
            info=sf.info(io.BytesIO(data))
            if not 0<info.duration<=seconds:raise ValueError(f'{name}: máximo {seconds} segundos.')
            files[name]=data
        with self.lock:
            if self.model is None:
                import torch
                torch.set_num_threads(4)
                sys.path.insert(0,str(self.repo))
                from core.cloner import KokoClone
                self.model=KokoClone()
            with tempfile.TemporaryDirectory(prefix='zimage-clone-') as temp:
                folder=Path(temp)
                for name,data in files.items():(folder/(name+'.wav')).write_bytes(data)
                self.model.convert(str(folder/'source.wav'),str(folder/'reference.wav'),str(folder/'output.wav'))
                result=(folder/'output.wav').read_bytes()
                if len(result)>32*1024**2:raise ValueError('Audio convertido demasiado grande.')
                return result

    def synthesize(self, body):
        import soundfile as sf
        text=body.get('input','')
        lang=body.get('lang','es')
        if not isinstance(text,str) or not 0<len(text.strip())<=5000:raise ValueError('Texto vacío o mayor de 5000 caracteres.')
        if lang not in ('es','en','fr','it','pt','hi','ja','zh'):raise ValueError('Idioma no compatible.')
        data=base64.b64decode(body.get('reference',''),validate=True)
        if not 0<len(data)<=20*1024**2:raise ValueError('Referencia WAV vacía o mayor de 20 MB.')
        info=sf.info(io.BytesIO(data))
        if not 0<info.duration<=30:raise ValueError('Referencia WAV: máximo 30 segundos.')
        with self.lock:
            if self.model is None:
                import torch
                torch.set_num_threads(4)
                sys.path.insert(0,str(self.repo))
                from core.cloner import KokoClone
                self.model=KokoClone()
            with tempfile.TemporaryDirectory(prefix='zimage-clone-') as temp:
                folder=Path(temp);(folder/'reference.wav').write_bytes(data)
                self.model.generate(text=text,lang=lang,reference_audio=str(folder/'reference.wav'),output_path=str(folder/'output.wav'))
                result=(folder/'output.wav').read_bytes()
                if len(result)>32*1024**2 or result[:4]!=b'RIFF':raise ValueError('KokoClone no devolvió un WAV válido.')
                return result


class Handler(BaseHandler):
    def do_GET(self):
        if self.path=='/health':return self.reply({'status':'ok','loaded':self.server.converter.model is not None})
        self.reply({'error':'Ruta desconocida'},code=404)

    def do_POST(self):
        try:
            if self.path not in ('/convert','/tts'):return self.reply({'error':'Ruta desconocida'},code=404)
            length=int(self.headers.get('Content-Length',0))
            if not 0<length<=56*1024**2:raise ValueError('Petición demasiado grande o vacía.')
            body=json.loads(self.rfile.read(length))
            if not isinstance(body,dict):raise ValueError('Petición inválida.')
            result=self.server.converter.synthesize(body) if self.path=='/tts' else self.server.converter.convert(body)
            return self.reply(result,'audio/wav')
        except Exception as exc:self.reply({'error':str(exc)},code=400)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8769)
    parser.add_argument('--repo',default=os.environ.get('ZIMAGE_KOKOCLONE_REPO',str(Path(__file__).resolve().parent/'vendor/kokoclone')))
    args=parser.parse_args()
    server=ThreadingHTTPServer((args.host,args.port),Handler)
    server.converter=Converter(args.repo)
    print(f'KokoClone: http://{args.host}:{args.port}',flush=True)
    server.serve_forever()
