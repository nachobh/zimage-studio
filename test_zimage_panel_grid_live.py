"""Smoke test against local llama and MCP. --generate creates one small image."""
import argparse
import json
from types import SimpleNamespace
from zimage_panel_grid import App

parser=argparse.ArgumentParser()
parser.add_argument('--generate',action='store_true')
args=parser.parse_args()
app=App(SimpleNamespace(lora_dir='~/lora_models_peft',backend_host='127.0.0.1',backend_port=9877,
    mcp_url='http://127.0.0.1:8765/mcp',chat_url='http://127.0.0.1:8086/v1',chat_model=''))
body={'lora_name':'Aimee_Lou_Wood_(vrtlAimeeLouWood)','lora_weight':.8,
      'image_prompt':'vrtlAimeeLouWood Aimee Lou Wood, fully clothed headshot portrait, neutral background',
      'image_settings':{'width':256,'height':256,'steps':9,'cfg':0,'seed':4242},
      'system_prompt':'Eres un asistente de prueba. Sigue exactamente las instrucciones del usuario.'}
result=app.chat({**body,'tools_enabled':False,'messages':[{'role':'user','content':'Responde solo con PERFIL_OK.'}]})
print('CHAT:',result['messages'][-1]['content'],flush=True)
assert 'PERFIL_OK' in result['messages'][-1]['content']
result=app.chat({**body,'messages':[{'role':'user','content':'Llama una vez a zimage_status para comprobar el servidor de imágenes y dime si está cargado. No generes imágenes.'}]})
print('MCP STATUS:',[(e['name'],e['error']) for e in result['events']],flush=True)
assert any(e['name']=='zimage_status' and not e['error'] for e in result['events']), result
if args.generate:
    result=app.chat({**body,'messages':[{'role':'user','content':'Usa generate_image una sola vez para generar el retrato descrito en el prompt de imagen de referencia. No llames otras herramientas. Después confirma brevemente.'}]})
    print('MCP GENERATE:',[(e['name'],e['error']) for e in result['events']], 'images:',len(result['images']),flush=True)
    for event in result['events']:
        if event['name']=='generate_image':
            metadata=json.loads(event['text']); print('RESULT:',{k:metadata.get(k) for k in ('status','path','lora','duration')},flush=True)
    assert result['images'], result['events']
print('PASS: local integration',flush=True)
