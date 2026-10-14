#!/usr/bin/env python3
"""Read-only integrity check for files shared by the local and portable panels."""
import argparse
import hashlib
from pathlib import Path

COMMON='''zimage_panel_grid.py zimage_panel_grid.html zimage_panel_grid.js
zimage_character_creator.py zimage_character_creator.html zimage_character_creator.js
zimage_character_train.py
zimage_lora_admin.py
zimage_panel_group.html zimage_panel_group.js zimage_panel_progress.js
zimage_panel_chat_state.js zimage_panel_prompts.js zimage_panel_chat_tools.js zimage_panel_lora_eval.js
zimage_panel_messenger.html zimage_panel_messenger.js zimage_panel_messenger.css
zimage_conversations.py zimage_panel_auth.py zimage_panel_auth.js zimage_panel_login.html
zimage_panel_audio.js zimage_panel_video.js comfyui_video.py
zimage_audio_server.py zimage_kokoclone_server.py
run_zimage_audio.sh run_zimage_kokoclone.sh setup_zimage_https.py
zimage_optional_services.py check_zimage_copies.py zimage_media_audit.py
ZIMAGE_AUDIO.md ZIMAGE_HTTPS.md ZIMAGE_CAMBIOS_MULTIMEDIA.md
test_zimage_audio.py test_zimage_panel_grid.py test_zimage_panel_grid_browser.cjs test_zimage_panel_prompts.cjs test_zimage_panel_archive.cjs test_zimage_panel_lora_eval.cjs test_zimage_conversations.py
test_zimage_optional_services.py test_zimage_character_creator.py test_zimage_character_train.py test_zimage_lora_admin.py qwen3_tts_trial.py run_qwen3_tts_trial.sh
test_qwen3_tts_trial.py ZIMAGE_FUENTES_VOZ.md test_comfyui_video.py test_zimage_media_audit.py
GUIA_PLAN_EMULADOR_MENSAJERIA.md GUIA_PLAN_VIDEO_COMFYUI.md PLAN_MEJORAS_CHAT_PROMPTS_Y_MULTIMEDIA.md
GUIA_COMFYUI_VIDEO_ROCM.md PLAN_CORRECCIONES_PANEL.md
PLAN_REESCALADO.md GUIA_UPSCALER_ZIMAGE.md
gen_minimax_h3.py test_minimax_h3.py generar_audio_qwen3_tts.sh'''.split()
INTENTIONAL=['start.sh','zimage_stack.sh','zimage_stack_doc.sh','README.md','AGENTS.md',
             'GUIA_INSTALACION_CERO_ZIMAGE.md','GUIA_ARRANQUE_REINICIO_ZIMAGE.md','ZIMAGE_PANEL_PERSONAJES.md']

def differences(source,target):
    errors=[]
    for name in COMMON:
        a,b=source/name,target/name
        if not a.is_file() or not b.is_file():errors.append(name+': falta una copia')
        elif hashlib.sha256(a.read_bytes()).digest()!=hashlib.sha256(b.read_bytes()).digest():errors.append(name+': contenido diferente')
    return errors

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local',type=Path,required=True)
    parser.add_argument('--project',type=Path,required=True)
    args=parser.parse_args()
    errors=differences(args.local,args.project)
    for error in errors:print(error)
    print(f'{len(COMMON)} archivos comunes: '+('DIFERENCIAS' if errors else 'idénticos'))
    print('Adaptaciones de instalación (revisión manual): '+', '.join(INTENTIONAL))
    print('Se excluyen modelos, perfiles, ajustes, certificados, claves y entornos.')
    raise SystemExit(bool(errors))
