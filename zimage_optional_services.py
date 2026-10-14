#!/usr/bin/env python3
"""Start optional voice services without touching model/image processes."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parent
SERVICES={
    'audio':('run_zimage_audio.sh','zimage_audio_server.py','ZIMAGE_AUDIO_VENV','.venv-audio-rocm10',8767),
    'kokoclone':('run_zimage_kokoclone.sh','zimage_kokoclone_server.py','ZIMAGE_KOKOCLONE_VENV','.venv-audio-rocm10',8769),
    'qwen3_tts_trial':('run_qwen3_tts_trial.sh','qwen3_tts_trial.py','QWEN3_TTS_VENV','.venv-qwen3-tts',8770),
}

def comfy_launcher():
    configured=os.environ.get('COMFYUI_START_SCRIPT')
    candidates=[Path(configured)] if configured else [ROOT/'levantar_minimax_h3.sh',ROOT/'zimage_panel_project'/'levantar_minimax_h3.sh']
    return next((p.resolve() for p in candidates if p.is_file()),candidates[0].resolve())

def ensure_comfy(required=None):
    """Start MiniMax-H3/ComfyUI through its launcher.

    A missing or failing launcher only aborts the run when ComfyUI was asked
    for explicitly (`--with-comfyui` exports ``ZIMAGE_COMFYUI_REQUIRED=1``);
    otherwise the stack keeps starting without video generation.
    """
    if required is None:required=os.environ.get('ZIMAGE_COMFYUI_REQUIRED')=='1'
    launcher=comfy_launcher()
    if not launcher.is_file():
        message=f'No existe COMFYUI_START_SCRIPT: {launcher}'
        if required:raise RuntimeError(message)
        print(f'Aviso: {message}; continúa sin ComfyUI.')
        return
    try:
        subprocess.run(['bash',str(launcher)],check=True,cwd=launcher.parent)
    except subprocess.CalledProcessError as error:
        if required:raise
        print(f'Aviso: el lanzador de ComfyUI terminó con {error.returncode}; continúa sin ComfyUI.')

def stop_comfy():
    launcher=comfy_launcher()
    if launcher.is_file():subprocess.run(['bash',str(launcher),'--stop'],check=True,cwd=launcher.parent)
    else:print(f'ComfyUI: no hay lanzador configurado ({launcher}); no se detiene ningún proceso.')

def healthy(port):
    try:
        with urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as response:
            return json.load(response).get('status')=='ok'
    except Exception:return False

def listening(port):
    """Return whether a local TCP listener accepts connections on *port*."""
    try:
        with socket.create_connection(('127.0.0.1',port),timeout=2):return True
    except OSError:return False

def https_status():
    certificate=(ROOT/'.tls/server.crt').is_file()
    listener=listening(8768)
    if listener and certificate:
        return 'HTTPS local :8768: escuchando; certificado presente.'
    if listener:
        return 'HTTPS local :8768: escuchando; falta server.crt en esta copia (puede pertenecer a otra copia).'
    if certificate:
        return 'HTTPS local :8768: certificado presente, pero el panel no está escuchando.'
    return 'HTTPS local :8768: sin certificado local y sin listener.'

def ensure(name,install):
    launcher,script,env,default,port=SERVICES[name]
    if healthy(port):
        if name == 'qwen3_tts_trial' and os.environ.get('QWEN3_TTS_DEVICE', 'cuda') != 'cpu':
            with urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as response:
                status=json.load(response)
            if status.get('device') != 'cuda' or not status.get('runtime',{}).get('gpu'):
                raise RuntimeError('Qwen3-TTS existente no confirma GPU. Usa --restart para sustituir el servicio gestionado.')
        print(f'{name}: disponible en loopback :{port}; se reutiliza el servicio existente.')
        return
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',port))==0:
            raise RuntimeError(f'Puerto {port} ocupado sin health válido; no se arranca un duplicado.')
    if install:subprocess.run(['bash',str(ROOT/launcher),'--install-only'],check=True)
    venv=Path(os.environ.get(env,str(ROOT/default)))
    if not (venv/'bin/python').is_file() and (ROOT.parent/default/'bin/python').is_file():
        venv=ROOT.parent/default
    python=venv/'bin/python'
    if not python.is_file():raise RuntimeError(f'Falta {python}. Ejecuta bash {launcher} --install-only.')
    state=ROOT/'.service-state';state.mkdir(exist_ok=True)
    args=[str(python),str(ROOT/script)]
    with (state/(name+'.log')).open('ab') as log:
        process=subprocess.Popen(args,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    (state/(name+'.pid')).write_text(str(process.pid)+'\n')
    for _ in range(40):
        if process.poll() is not None:raise RuntimeError(f'{name} terminó; consulta {state/(name+".log")}')
        if healthy(port):
            print(f'{name}: arrancado en loopback :{port}, PID {process.pid}. Modelos bajo demanda.')
            return
        time.sleep(.25)
    raise RuntimeError(f'{name} aún no está sano; consulta su log antes de volver a arrancarlo.')

def stop(name):
    """Stop only an optional process launched from this installation."""
    _,script,_,_,port=SERVICES[name]
    pidfile=ROOT/'.service-state'/(name+'.pid')
    if not pidfile.is_file():
        if listening(port):raise RuntimeError(f'{name} :{port} está ocupado sin PID gestionado; no se detiene otro proceso.')
        print(f'{name}: ya parado.')
        return
    value=pidfile.read_text().strip()
    if not value.isdecimal() or int(value)<2:
        raise RuntimeError(f'{name}: PID inválido en {pidfile}.')
    pid=int(value)
    cmdline=Path(f'/proc/{pid}/cmdline')
    try:argv=cmdline.read_bytes().split(b'\0')
    except FileNotFoundError:argv=[]
    if not argv:
        if listening(port):raise RuntimeError(f'{name} :{port} está ocupado; PID registrado {pid} ya no existe.')
        pidfile.unlink()
        print(f'{name}: PID obsoleto retirado; ya parado.')
        return
    if os.fsencode(str(ROOT/script)) not in argv:
        raise RuntimeError(f'{name}: PID {pid} no ejecuta {ROOT/script}; no se detiene.')
    os.kill(pid,signal.SIGTERM)
    for _ in range(50):
        try:argv=cmdline.read_bytes().split(b'\0')
        except FileNotFoundError:argv=[]
        if not argv and not listening(port):
            pidfile.unlink()
            print(f'{name}: detenido.')
            return
        time.sleep(.1)
    raise RuntimeError(f'{name}: PID {pid} sigue activo; no se lanza un duplicado.')

def restart_qwen3_tts(install):
    """Restart only the Qwen3-TTS service managed by this installation."""
    name='qwen3_tts_trial'
    stop(name)
    ensure(name,install)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--stop',action='store_true',help='Detiene servicios opcionales gestionados por esta instalación.')
    parser.add_argument('--audio',action='store_true')
    parser.add_argument('--kokoclone',action='store_true')
    parser.add_argument('--qwen3-tts-trial',action='store_true')
    parser.add_argument('--restart-qwen3-tts',action='store_true',help='Reinicia solo Qwen3-TTS gestionado por esta instalación.')
    parser.add_argument('--comfyui',action='store_true',help='Inicia el MiniMax-H3 local a través del lanzador configurado.')
    parser.add_argument('--no-install',action='store_true')
    parser.add_argument('--https-ip')
    args=parser.parse_args()
    if args.restart_qwen3_tts:
        if args.status or args.stop or args.audio or args.kokoclone or args.qwen3_tts_trial or args.comfyui or args.https_ip:
            parser.error('--restart-qwen3-tts se usa por separado (puede combinarse con --no-install).')
        restart_qwen3_tts(not args.no_install)
    elif args.status:
        for name,(*_,port) in SERVICES.items():print(f'{name} :{port}: '+('disponible (puede ser otra copia)' if healthy(port) else 'no disponible'))
        launcher=comfy_launcher()
        if launcher.is_file():subprocess.run(['bash',str(launcher),'--status'],check=True,cwd=launcher.parent)
        else:print(f'comfyui: sin lanzador configurado ({launcher})')
        print(https_status())
    elif args.stop:
        failures=[]
        actions=[stop_comfy, *(lambda name=name: stop(name) for name in reversed(tuple(SERVICES)))]
        for action in actions:
            try:action()
            except Exception as exc:
                print(f'No se pudo detener un servicio opcional: {exc}',file=sys.stderr)
                failures.append(exc)
        if failures:raise SystemExit(1)
    else:
        if args.https_ip:
            from setup_zimage_https import create
            import ipaddress
            address=str(ipaddress.ip_address(args.https_ip))
            cert=ROOT/'.tls/server.crt'
            if cert.exists():
                subprocess.run(['openssl','verify','-CAfile',str(ROOT/'.tls/ca.crt'),'-verify_ip',address,str(cert)],check=True)
                if not (ROOT/'.tls/server.key').is_file():raise RuntimeError('Falta server.key')
            else:create(ROOT/'.tls',[address,'127.0.0.1','::1'])
        if args.audio or args.kokoclone:ensure('audio',not args.no_install)
        if args.kokoclone:ensure('kokoclone',not args.no_install)
        if args.qwen3_tts_trial:ensure('qwen3_tts_trial',not args.no_install)
        if args.comfyui:ensure_comfy()
