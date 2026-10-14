#!/usr/bin/env python3
"""Create a private LAN CA and panel certificate without changing OS trust."""
import argparse
import ipaddress
import os
from pathlib import Path
import subprocess
import tempfile


def create(directory, ips):
    directory=Path(directory)
    addresses=[str(ipaddress.ip_address(value)) for value in ips]
    directory.mkdir(mode=0o700,parents=True,exist_ok=True)
    if any((directory/name).exists() for name in ('ca.key','ca.crt','server.key','server.crt')):
        raise ValueError('Ya existen certificados. Se conservan; usa otro directorio para una CA nueva.')
    with tempfile.TemporaryDirectory(dir=directory) as temp:
        tmp=Path(temp)
        def run(*args):subprocess.run(['openssl',*args],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        run('req','-x509','-newkey','rsa:3072','-nodes','-sha256','-days','3650','-subj','/CN=ZImage Local CA',
            '-addext','basicConstraints=critical,CA:TRUE,pathlen:0','-addext','keyUsage=critical,keyCertSign,cRLSign',
            '-keyout',str(tmp/'ca.key'),'-out',str(tmp/'ca.crt'))
        run('req','-new','-newkey','rsa:2048','-nodes','-subj','/CN=ZImage Panel',
            '-keyout',str(tmp/'server.key'),'-out',str(tmp/'server.csr'))
        (tmp/'extensions').write_text('basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\nsubjectAltName=DNS:localhost,'+','.join('IP:'+ip for ip in addresses)+'\n')
        run('x509','-req','-in',str(tmp/'server.csr'),'-CA',str(tmp/'ca.crt'),'-CAkey',str(tmp/'ca.key'),'-CAcreateserial',
            '-out',str(tmp/'server.crt'),'-days','365','-sha256','-extfile',str(tmp/'extensions'))
        run('verify','-CAfile',str(tmp/'ca.crt'),str(tmp/'server.crt'))
        for name in ('ca.key','ca.crt','server.key','server.crt'):
            os.chmod(tmp/name,0o600 if name.endswith('.key') else 0o644)
            os.replace(tmp/name,directory/name)
    print('Certificados creados en',directory)
    print('Instala SOLO ca.crt en el móvil como CA de confianza. No compartas ningún .key.')
    print('Reinicia solo el panel y abre https://'+addresses[0]+':8768')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',default=str(Path(__file__).resolve().parent/'.tls'))
    parser.add_argument('--ip',action='append',required=True)
    args=parser.parse_args()
    create(args.directory,[*args.ip,'127.0.0.1','::1'])
