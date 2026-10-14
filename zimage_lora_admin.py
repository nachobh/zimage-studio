"""Safe, serialized adapter rescan logic shared by Z-Image backends."""
import os
import re


def reload_lora_adapters(pipeline, adapters, directory, loader):
    if pipeline is None:
        return {'status': 'error', 'error': 'Modelo Z-Image no cargado'}
    if not os.path.isdir(directory):
        return {'status': 'error', 'error': f'No existe el directorio LoRA: {directory}'}

    unload_errors = []
    for name in list(adapters):
        try:
            pipeline.transformer.delete_adapters(name)
            adapters.pop(name, None)
        except Exception as exc:
            unload_errors.append(f'{name}: {str(exc)[:180]}')
    if unload_errors:
        return {'status': 'error', 'error': 'No se pudieron descargar todos los LoRAs; no se inició la recarga.',
                'unload_errors': unload_errors, 'lora_count': len(adapters)}

    loaded, failed = [], []
    directory = os.path.realpath(directory)
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith('.safetensors'):
            continue
        name = filename[:-len('.safetensors')]
        path = os.path.join(directory, filename)
        valid_file = (re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name) and
                      not os.path.islink(path) and os.path.isfile(path) and
                      os.path.dirname(os.path.realpath(path)) == directory)
        if not valid_file:
            failed.append(name)
            continue
        try:
            success = loader(name, path)
        except Exception:
            success = False
        (loaded if success else failed).append(name)
    return {'status': 'ok', 'loaded': loaded, 'failed': failed,
            'lora_count': len(adapters)}
