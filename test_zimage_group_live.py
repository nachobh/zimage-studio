"""Live group/progress smoke test. --generate deliberately creates one 256px image."""
import argparse
import concurrent.futures
import json
import time
import uuid
from urllib.request import Request, urlopen
from urllib.parse import quote
from urllib.error import HTTPError


def api(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    try:
        with urlopen(Request('http://127.0.0.1:8766'+path, data=data,
                             headers={'Content-Type':'application/json'}), timeout=900) as response:
            return json.load(response)
    except HTTPError as exc:
        raise RuntimeError(exc.read().decode()) from exc


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--generate',action='store_true')
    args=parser.parse_args()
    names=[x['name'] for x in api('/api/loras')['lora_list']][:2]
    assert len(names)==2
    members=[{'name':n,'weight':.35,'system_prompt':'Eres un personaje ficticio. Responde con una sola frase breve.'} for n in names]
    transcript=[{'speaker':'Humano','text':'Presentaos brevemente; el segundo, saluda al primero.'}]
    for name in names:
        r=api('/api/group/turn',{'participants':members,'speaker':name,'transcript':transcript,'tools_enabled':False})
        assert r['speaker']==name and r['text']
        transcript.append({'speaker':name,'text':r['text']})
    print('PASS: dos turnos reales con historial compartido',flush=True)
    if not args.generate:
        return
    for _ in range(180):
        try:
            if api('/api/status').get('model_loaded'):
                break
        except (OSError, RuntimeError):
            pass
        time.sleep(1)
    else:
        raise RuntimeError('Z-Image no estuvo listo tras 180 segundos')
    key=str(uuid.uuid4())
    prompt=' '.join(api('/api/profile?name='+quote(n))['profile']['image_prompt'] for n in names)
    body={'request_id':key,'lora_name':'','loras':[{'name':n,'weight':.35} for n in names],
          'image_prompt':prompt+', two people in a studio portrait, neutral background',
          'image_settings':{'width':256,'height':256,'steps':9,'cfg':0,'seed':92418}}
    seen=[]
    with concurrent.futures.ThreadPoolExecutor() as pool:
        job=pool.submit(api,'/api/retry',body)
        while not job.done():
            p=api('/api/progress?request_id='+key)
            state=(p.get('phase'),p.get('step'),p.get('total'))
            if not seen or seen[-1]!=state:
                print('PROGRESS',state,flush=True);seen.append(state)
            time.sleep(.3)
        r=job.result()
    assert r.get('src','').startswith('data:image/')
    assert r['generation']['loras']==body['loras']
    assert any(s[0]=='sampling' and s[1]>0 for s in seen),seen
    print('PASS: imagen real con dos LoRAs y progreso de pasos por MCP',flush=True)


if __name__=='__main__':
    main()
