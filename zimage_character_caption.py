#!/usr/bin/env python3
"""Caption one private character dataset with the locally cached BLIP-2 model."""
import json
import os
from pathlib import Path
import sys


def main(directory):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from PIL import Image
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    folder = Path(directory)
    photos = sorted((folder / 'photos').glob('*.jpg'))
    if not photos:
        raise ValueError('No hay fotos')
    model_id = os.environ.get('ZIMAGE_BLIP_MODEL', 'Salesforce/blip2-opt-2.7b')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32
    print(f'Cargando BLIP-2 local en {device}', flush=True)
    processor = Blip2Processor.from_pretrained(model_id, local_files_only=True)
    model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype,
                                                          local_files_only=True).to(device)
    metadata = json.loads((folder / 'character.json').read_text(encoding='utf-8'))
    existing = metadata.get('captions', {})
    captions = {}
    for index, photo in enumerate(photos, 1):
        if existing.get(photo.name, '').strip():
            captions[photo.name] = existing[photo.name]
        else:
            with Image.open(photo) as image:
                inputs = processor(images=image.convert('RGB'), return_tensors='pt').to(device, dtype)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=80)
            caption = processor.decode(output[0], skip_special_tokens=True).strip()
            captions[photo.name] = metadata['trigger'] + ' ' + caption
        print(f'Caption {index}/{len(photos)}: {photo.name}', flush=True)
    from zimage_character_creator import atomic_json
    atomic_json(folder / 'captions.json', captions)


if __name__ == '__main__':
    main(sys.argv[1])
