"""Validate SM80 CuTe DSL kernels against FLA Triton reference."""
import sys, torch
sys.path.insert(0,'.')
sys.path.insert(0,'third_party/flash-linear-attention')
import os
os.environ['CULA_USE_SM80_CUTEDSL'] = '1'

torch.manual_seed(42)
B, T, H, D = 1, 512, 1, 128  # 8 chunks of 64
C = 64

q = torch.randn(B, T, H, D, dtype=torch.bfloat16, device='cuda') * 0.1
k = torch.randn(B, T, H, D, dtype=torch.bfloat16, device='cuda') * 0.1
v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device='cuda') * 0.1
g = torch.randn(B, T, H, D, dtype=torch.float32, device='cuda') * 0.1
beta = torch.randn(B, T, H, dtype=torch.float32, device='cuda') * 0.1
initial_state = torch.randn(B, H, D, D, dtype=torch.float32, device='cuda') * 0.01
scale = D ** -0.5

# ── Reference: FLA Triton ──
from fla.ops.kda import chunk_kda
o_ref, state_ref = chunk_kda(
    q=q, k=k, v=v, g=g, beta=beta,
    scale=scale, initial_state=initial_state, output_final_state=True,
)
print(f'FLA Triton output mean: {o_ref.float().mean():.6f}')
print(f'FLA Triton state mean: {state_ref.float().mean():.6f}')

# ── CuTe DSL ──
from cula.kda.ampere_fused_fwd import cula_kda_prefill_ampere

o_cutedsl, state_cutedsl = cula_kda_prefill_ampere(
    q=q, k=k, v=v, g=g, beta=beta,
    scale=scale, initial_state=initial_state, output_final_state=True,
)

print(f'CuTeDSL output mean: {o_cutedsl.float().mean():.6f}')
print(f'CuTeDSL state mean: {state_cutedsl.float().mean():.6f}')

print(f'\nOutput max diff: {(o_ref.float() - o_cutedsl.float()).abs().max():.6f}')
print(f'State max diff:  {(state_ref.float() - state_cutedsl.float()).abs().max():.6f}')
print('DONE')
