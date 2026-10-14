"""Check that every serialized PEFT tensor actually reached its named adapter."""
import torch
from safetensors import safe_open


def verify_loaded_lora(model, path, adapter_name):
    expected_names = set()
    count = 0
    nonzero_b = 0
    with safe_open(str(path), framework="pt", device="cpu") as source:
        for key in source.keys():
            if key.endswith('.lora_A.weight'):
                module_name, direction = key[:-len('.lora_A.weight')], 'lora_A'
            elif key.endswith('.lora_B.weight'):
                module_name, direction = key[:-len('.lora_B.weight')], 'lora_B'
            else:
                raise ValueError(f'Clave no normalizada: {key}')
            module = model.get_submodule(module_name)
            actual = getattr(module, direction)[adapter_name].weight.detach()
            if actual.is_meta:
                raise ValueError(f'Peso sin materializar: {key}')
            expected = source.get_tensor(key).to(dtype=actual.dtype)
            if not torch.equal(actual.cpu(), expected):
                raise ValueError(f'El peso cargado no coincide: {key}')
            if direction == 'lora_B' and torch.count_nonzero(expected).item():
                nonzero_b += 1
            expected_names.add(f'{module_name}.{direction}.{adapter_name}.weight')
            count += 1
    actual_names = {
        name for name, _ in model.named_parameters()
        if any(f'.{d}.{adapter_name}.' in name for d in ('lora_A', 'lora_B'))
    }
    if actual_names != expected_names:
        raise ValueError(f'Cobertura distinta: extra={actual_names - expected_names}; '
                         f'ausentes={expected_names - actual_names}')
    if not count or not nonzero_b:
        raise ValueError('LoRA vacío o todas las matrices B son cero')
    return {'tensors_verified': count, 'nonzero_B_matrices': nonzero_b}
