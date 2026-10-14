"""Small GPU computation before downloading/loading models; never writes data."""
import torch

if not torch.cuda.is_available():
    raise SystemExit(f'GPU no disponible: torch={torch.__version__}, HIP={torch.version.hip}, CUDA={torch.version.cuda}')
with torch.inference_mode():
    x = torch.ones((32, 32), device='cuda', dtype=torch.bfloat16)
    y = x @ x
    torch.cuda.synchronize()
    assert float(y[0, 0]) == 32
print(f'[start] GPU comprobada: {torch.cuda.get_device_name(0)}; torch={torch.__version__}; HIP={torch.version.hip}; CUDA={torch.version.cuda}')
