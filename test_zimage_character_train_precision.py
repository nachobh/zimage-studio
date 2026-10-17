"""CPU numeric regressions; skipped in environments without Torch/Diffusers/PEFT."""
import unittest
try:
    import torch
except ImportError:
    torch = None

from zimage_character_train import prepare_trainable_parameters


@unittest.skipIf(torch is None, 'Requires the training Python with Torch')
class TrainingPrecisionTests(unittest.TestCase):
    def test_small_adam_update_survives_and_frozen_base_stays_bfloat16(self):
        model = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
        model.requires_grad_(False)
        frozen = model.weight.detach().clone()
        model.register_parameter('lora', torch.nn.Parameter(torch.tensor([0.03125], dtype=torch.bfloat16)))
        before = model.lora.item()
        control = torch.nn.Parameter(model.lora.detach().clone())
        parameters = prepare_trainable_parameters(model)
        optimizer = torch.optim.AdamW(parameters, lr=5e-5, weight_decay=0)
        control_optimizer = torch.optim.AdamW([control], lr=5e-5, weight_decay=0)
        model.lora.grad = torch.ones_like(model.lora)
        control.grad = torch.ones_like(control)
        optimizer.step()
        control_optimizer.step()
        self.assertEqual(control.item(), before, 'BF16 loses this update by rounding')
        self.assertAlmostEqual(before - model.lora.item(), 5e-5, places=8)
        self.assertEqual(model.lora.dtype, torch.float32)
        self.assertEqual(model.weight.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(model.weight, frozen))
        for key in ('exp_avg', 'exp_avg_sq'):
            self.assertEqual(optimizer.state[model.lora][key].dtype, torch.float32)

    def test_real_diffusers_injection_then_fp32_optimization(self):
        try:
            from diffusers.loaders.peft import PeftAdapterMixin
            from peft import LoraConfig
        except ImportError:
            self.skipTest('Requires Diffusers and PEFT')
        class Tiny(torch.nn.Module, PeftAdapterMixin):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)
        model = Tiny()
        model.requires_grad_(False)
        model.add_adapter(LoraConfig(r=4, lora_alpha=4, target_modules=['proj']))
        parameters = prepare_trainable_parameters(model)
        self.assertEqual(len(parameters), 2)
        self.assertTrue(all(p.dtype == torch.float32 for p in parameters))
        self.assertEqual(model.proj.base_layer.weight.dtype, torch.bfloat16)
        optimizer = torch.optim.AdamW(parameters, lr=5e-5)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            loss = model.proj(torch.ones(1, 16, dtype=torch.bfloat16)).float().square().mean()
        loss.backward()
        optimizer.step()
        self.assertTrue(all(optimizer.state[p]['exp_avg'].dtype == torch.float32 for p in parameters))

    def test_empty_adapter_is_rejected(self):
        model = torch.nn.Linear(1, 1)
        model.requires_grad_(False)
        with self.assertRaisesRegex(RuntimeError, 'entrenables'):
            prepare_trainable_parameters(model)


if __name__ == '__main__':
    unittest.main()
