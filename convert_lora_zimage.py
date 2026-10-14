#!/usr/bin/env python3
"""
Convierte LoRAs de diferentes formatos al formato PEFT/Diffusers para Z-Image.
Z-Image usa estos módulos exactos:
  - layers.N.attention.to_q, .to_k, .to_v, .to_out.0, .to_out.1
  - layers.N.attention_out
  - layers.N.feed_forward.w1, .w2, .w3
  - layers.N.adaLN_modulation.0
  - noise_refiner.N.*
  - context_refiner.N.*
"""
import torch
import safetensors.torch as st
import os
import sys
import re

def detect_format(data):
    keys = list(data.keys())
    if any(k.startswith('__dit__[') for k in keys):
        return 'zimage_dit_compact'
    if any('diffusion_model.layers' in k for k in keys):
        return 'ostris'
    elif any('transformer.layers' in k for k in keys):
        return 'diffusers_transformer'
    elif any('lora_unet' in k for k in keys):
        return 'diffusers_unet'
    elif any('layers.' in k for k in keys):
        sample = keys[0]
        if '.lora_' in sample or '.lora-' in sample or '.lora_A' in sample or '.lora_B' in sample:
            return 'ostris'
        elif '.lora_down.' in sample or '.lora_up.' in sample:
            return 'diffusers_peft'
        elif '.alpha.alpha' in sample or '.weight.weight' in sample:
            return 'weird_duplicate'
        else:
            return 'unknown'
    else:
        return 'unknown'

def map_ostris_module_to_zimage(module_name, sub_module):
    """Mapea nombres de módulos Ostris a nombres Z-Image."""
    # Mapeo de nombres de módulos
    module_map = {
        'attention': 'attention',
        'feed_forward': 'feed_forward',
        'adaLN_modulation': 'adaLN_modulation',
    }
    if module_name not in module_map:
        return None, None
    mapped = module_map[module_name]
    # Mapeo de sub-módulos
    sub_map = {
        'to_out0': 'to_out.0',
        'to_out1': 'to_out.1',
        'to_out': 'to_out.0',
        'attention_qkv': 'to_q',  # qkv se divide en to_q, to_k, to_v
        'attention_kv': 'to_k',
    }
    if sub_module in sub_map:
        return mapped, sub_map[sub_module]
    return mapped, sub_module

def map_diffusers_unet_module(key):
    """Mapea nombres de módulos Diffusers UNet a nombres Z-Image.
    Formatos posibles:
    - lora_unet_layers_0_attention_to_q_lora_down.weight
    - lora_unet_context_refiner_0_attention_out.alpha
    - lora_unet_noise_refiner_0_attention_qkv.lora_up.weight
    """
    # Detectar si es context_refiner o noise_refiner
    prefix = None
    rest = key
    if 'lora_unet_context_refiner_' in key:
        prefix = 'context_refiner'
        rest = key.replace('lora_unet_context_refiner_', '')
    elif 'lora_unet_noise_refiner_' in key:
        prefix = 'noise_refiner'
        rest = key.replace('lora_unet_noise_refiner_', '')
    elif 'lora_unet_layers_' in key:
        prefix = 'layers'
        rest = key.replace('lora_unet_layers_', '')
    else:
        return None

    # rest ahora es algo como:
    # "0_attention_to_q_lora_down.weight" (con guiones bajos)
    # O "0_attention_out.lora_down.weight" (con puntos)
    # Separar el índice de capa
    parts = rest.split('_', 1)
    if len(parts) < 2:
        return None
    layer_idx = parts[0]
    remainder = parts[1]  # "attention_to_q_lora_down.weight" o "attention_out.lora_down.weight"

    # Separar el lora_type del final
    lora_part = None
    module_str = remainder
    # Primero verificar lora_down/lora_up (puede usar _ o .)
    for lp in ['lora_down', 'lora_up']:
        # Buscar _lp.weight o .lp.weight
        if '_' + lp + '.weight' in remainder:
            idx = remainder.index('_' + lp + '.weight')
            lora_part = lp
            module_str = remainder[:idx]
            break
        elif '.' + lp + '.weight' in remainder:
            idx = remainder.index('.' + lp + '.weight')
            lora_part = lp
            module_str = remainder[:idx]
            break
        elif '_' + lp in remainder:
            idx = remainder.index('_' + lp)
            lora_part = lp
            module_str = remainder[:idx]
            break
        elif '.' + lp in remainder:
            idx = remainder.index('.' + lp)
            lora_part = lp
            module_str = remainder[:idx]
            break
    # Luego verificar alpha (puede ser _alpha o .alpha)
    if not lora_part:
        if '.alpha' in remainder:
            idx = remainder.index('.alpha')
            lora_part = 'alpha'
            module_str = remainder[:idx]
        elif '_alpha' in remainder:
            idx = remainder.index('_alpha')
            lora_part = 'alpha'
            module_str = remainder[:idx]

    if not lora_part:
        return None

    # Convertir module_str de snake_case a punto
    # "attention_to_q" -> "attention.to_q"
    # "attention_out" -> "attention_out"
    # "feed_forward_w1" -> "feed_forward.w1"
    module_path = []
    current = module_str
    # Mapeo de nombres compuestos
    compound_map = {
        'attention_to_q': 'attention.to_q',
        'attention_to_k': 'attention.to_k',
        'attention_to_v': 'attention.to_v',
        'attention_to_out_0': 'attention.to_out.0',
        'attention_to_out_1': 'attention.to_out.1',
        'attention_qkv': 'attention.to_qkv',
        'attention_kv': 'attention.to_kv',
        'attention_out': 'attention_out',
        'feed_forward_w1': 'feed_forward.w1',
        'feed_forward_w2': 'feed_forward.w2',
        'feed_forward_w3': 'feed_forward.w3',
        'adaLN_modulation_0': 'adaLN_modulation.0',
    }
    if current in compound_map:
        module_path.append(compound_map[current])
    else:
        # Fallback: separar por _
        module_path.append(current.replace('_', '.'))

    if prefix == 'layers':
        return f"transformer.layers.{layer_idx}.{'.'.join(module_path)}.{lora_part}"
    else:
        return f"transformer.{prefix}.{layer_idx}.{'.'.join(module_path)}.{lora_part}"

def convert_ostris_to_zimage(data):
    """Convierte formato Ostris/AI-Toolkit al formato Z-Image PEFT."""
    converted = {}
    for key, value in data.items():
        if not key.startswith('diffusion_model.layers.'):
            continue
        # Formato: diffusion_model.layers.N.module.submodule...lora_A/B.weight
        # Pero submodule puede ser "to_out" y el siguiente es "0" (ej: to_out.0)
        parts = key.split('.')
        if len(parts) < 6:
            continue
        layer_idx = parts[2]
        module_name = parts[3]
        
        # Determinar el sub-módulo y el tipo de lora
        # Posibles formatos:
        # - layers.0.attention.to_k.lora_A.weight -> parts[4]=to_k, parts[5]=lora_A, parts[6]=weight
        # - layers.0.attention.to_out.0.lora_A.weight -> parts[4]=to_out, parts[5]=0, parts[6]=lora_A, parts[7]=weight
        # - layers.0.feed_forward.w1.lora_A.weight -> parts[4]=w1, parts[5]=lora_A, parts[6]=weight
        # - layers.0.adaLN_modulation.0.lora_A.weight -> parts[4]=adaLN_modulation, parts[5]=0, parts[6]=lora_A, parts[7]=weight
        
        if module_name in ('attention', 'feed_forward', 'adaLN_modulation'):
            # Verificar si es un caso de to_out.0 o adaLN_modulation.0
            if parts[4] in ('to_out', 'adaLN_modulation'):
                # Es un sub-módulo con número: to_out.0 o adaLN_modulation.0
                sub_module = parts[4] + '.' + parts[5]
                lora_type = parts[6] if len(parts) > 6 else ''
                if lora_type.endswith('.weight'):
                    lora_type = lora_type[:-7]
            else:
                sub_module = parts[4]
                lora_type = parts[5] if len(parts) > 5 else ''
                if lora_type.endswith('.weight'):
                    lora_type = lora_type[:-7]
        else:
            # Otros módulos (attention_qkv, etc.)
            sub_module = parts[4]
            lora_type = parts[5] if len(parts) > 5 else ''
            if lora_type.endswith('.weight'):
                lora_type = lora_type[:-7]
        
        mapped_module, mapped_sub = map_ostris_module_to_zimage(module_name, sub_module)
        if mapped_module is None:
            continue
        
        # Mapear lora_A -> lora_down, lora_B -> lora_up
        if lora_type == 'lora_A':
            lora_type = 'lora_down'
        elif lora_type == 'lora_B':
            lora_type = 'lora_up'
        elif lora_type == 'alpha':
            pass  # alpha se mantiene
        else:
            # Si no es un tipo conocido, usar como está
            pass
        
        new_key = f"layers.{layer_idx}.{mapped_module}.{mapped_sub}.{lora_type}"
        converted[new_key] = value
    return converted

def convert_diffusers_unet_to_zimage(data):
    """Convierte formato Diffusers UNet al formato Z-Image PEFT."""
    converted = {}
    for key, value in data.items():
        if not key.startswith('lora_unet_'):
            continue
        new_key = map_diffusers_unet_module(key)
        if new_key:
            converted[new_key] = value
    return converted

def convert_diffusers_transformer_to_zimage(data):
    """Convierte formato Diffusers transformer.layers al formato Z-Image."""
    converted = {}
    for key, value in data.items():
        if key.startswith('transformer.layers.'):
            parts = key.split('.')
            new_key = '.'.join(parts[1:])
            converted[new_key] = value
    return converted

def convert_zimage_dit_compact_to_peft(data):
    """Convierte el formato compacto __dit__[...] usado por algunos LoRA de Z-Image al formato PEFT/Diffusers."""
    converted = {}

    module_map = {
        'q': 'layers.{idx}.attention.to_q',
        'k': 'layers.{idx}.attention.to_k',
        'v': 'layers.{idx}.attention.to_v',
        'o': 'layers.{idx}.attention.to_out.0',
        'noise_refiner_q': 'noise_refiner.{idx}.attention.to_q',
        'noise_refiner_k': 'noise_refiner.{idx}.attention.to_k',
        'noise_refiner_v': 'noise_refiner.{idx}.attention.to_v',
        'noise_refiner_o': 'noise_refiner.{idx}.attention.to_out.0',
        'x_embedder': 'all_x_embedder.2-1',
        'linear_final': 'all_final_layer.2-1.linear',
    }

    pattern = re.compile(r'^__dit__\[(?P<stem>.+)\]__(?P<direction>down|up)__$')
    stem_pattern = re.compile(r'^t-(?P<module>.+)-(?P<idx>\d+)-\d+$')

    for key, value in data.items():
        match = pattern.match(key)
        if not match:
            continue

        inner = match.group('stem')
        direction = match.group('direction')
        stem_match = stem_pattern.match(inner)
        if not stem_match:
            continue

        module = stem_match.group('module')
        idx = stem_match.group('idx')
        target_template = module_map.get(module)
        if target_template is None:
            continue

        target = target_template.format(idx=idx)
        lora_part = 'lora_A.weight' if direction == 'down' else 'lora_B.weight'
        converted[f'{target}.{lora_part}'] = value

    return converted

def convert_weird_duplicate_to_zimage(data):
    """Convierte el formato raro con .alpha.alpha y .weight.weight."""
    converted = {}
    for key, value in data.items():
        # Formato: layers.N.attention.to.to_out0.alpha.alpha
        # -> layers.N.attention.to_out.0.alpha
        if '.alpha.alpha' in key:
            # Remover el .alpha extra
            clean = key.replace('.alpha.alpha', '.alpha')
            # Corregir to.to_out0 -> to_out.0
            clean = re.sub(r'\.to\.to_out(\d)', r'.to_out.\1', clean)
            # Corregir to.qkv -> to_qkv o dividir
            clean = re.sub(r'\.to\.to_qkv', '.to_qkv', clean)
            clean = re.sub(r'\.to\.to_q', '.to_q', clean)
            clean = re.sub(r'\.to\.to_k', '.to_k', clean)
            clean = re.sub(r'\.to\.to_v', '.to_v', clean)
            clean = re.sub(r'\.to\.to_out', '.to_out', clean)
            clean = re.sub(r'\.to\.attention_qkv', '.attention_qkv', clean)
            clean = re.sub(r'\.to\.attention_kv', '.attention_kv', clean)
            clean = re.sub(r'\.to\.attention_out', '.attention_out', clean)
            clean = re.sub(r'\.to\.', '.', clean)
            converted[clean] = value
        elif '.weight.weight' in key:
            clean = key.replace('.weight.weight', '.weight')
            clean = re.sub(r'\.to\.to_out(\d)', r'.to_out.\1', clean)
            clean = re.sub(r'\.to\.to_qkv', '.to_qkv', clean)
            clean = re.sub(r'\.to\.to_q', '.to_q', clean)
            clean = re.sub(r'\.to\.to_k', '.to_k', clean)
            clean = re.sub(r'\.to\.to_v', '.to_v', clean)
            clean = re.sub(r'\.to\.to_out', '.to_out', clean)
            clean = re.sub(r'\.to\.attention_qkv', '.attention_qkv', clean)
            clean = re.sub(r'\.to\.attention_kv', '.attention_kv', clean)
            clean = re.sub(r'\.to\.attention_out', '.attention_out', clean)
            clean = re.sub(r'\.to\.', '.', clean)
            converted[clean] = value
    return converted

def convert_lora(input_path, output_path, force_format=None):
    print(f"Convirtiendo: {os.path.basename(input_path)}")
    data = st.load_file(input_path)
    fmt = force_format if force_format else detect_format(data)
    print(f"  Formato detectado: {fmt}")
    print(f"  Keys totales: {len(data)}")
    if fmt == 'ostris':
        converted = convert_ostris_to_zimage(data)
    elif fmt == 'diffusers_transformer':
        converted = convert_diffusers_transformer_to_zimage(data)
    elif fmt == 'zimage_dit_compact':
        converted = convert_zimage_dit_compact_to_peft(data)
    elif fmt == 'diffusers_unet':
        converted = convert_diffusers_unet_to_zimage(data)
    elif fmt == 'weird_duplicate':
        converted = convert_weird_duplicate_to_zimage(data)
    else:
        print(f"  ERROR: Formato desconocido")
        return
    print(f"  Keys convertidas: {len(converted)}")
    if converted:
        print(f"  Ejemplos: {list(converted.keys())[:5]}")
    st.save_file(converted, output_path)
    print(f"  Guardado: {os.path.basename(output_path)}")
    print(f"  Tamaño: {os.path.getsize(output_path)/1024/1024:.0f} MB")
    print()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python3 convert_lora_zimage.py <input.safetensors> <output.safetensors> [formato]")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    fmt = sys.argv[3] if len(sys.argv) > 3 else None
    if not os.path.exists(input_path):
        print(f"Error: {input_path} no existe")
        sys.exit(1)
    convert_lora(input_path, output_path, fmt)
