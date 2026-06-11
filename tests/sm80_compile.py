import sys, torch
sys.path.insert(0,'.')
sys.path.insert(0,'third_party/flash-linear-attention')
from cula.kda.ampere_fused_fwd import cula_kda_prefill_ampere
B,T,H,D = 1,64,1,128
q = torch.randn(B,T,H,D, dtype=torch.bfloat16, device='cuda') * 0.1
k = torch.randn(B,T,H,D, dtype=torch.bfloat16, device='cuda') * 0.1
v = torch.randn(B,T,H,D, dtype=torch.bfloat16, device='cuda') * 0.1
g = torch.zeros(B,T,H,D, dtype=torch.float32, device='cuda')
b = torch.randn(B,T,H, dtype=torch.float32, device='cuda') * 0.1
print('Calling CuTe DSL backend...')
o, fs = cula_kda_prefill_ampere(q=q, k=k, v=v, g=g, beta=b, scale=D**-0.5)
print(f'Output: {o.shape}, mean={o.float().mean():.4f}')
print('DONE')