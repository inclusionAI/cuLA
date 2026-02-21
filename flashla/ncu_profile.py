#!/usr/bin/env python3
"""NCU profiling script: compile first, then profile execution only."""
import torch
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from chunk_delta_h import ChunkDeltaRuleFwdH
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h as fla_fwd_h

B, T, H, K, V = 4, 4096, 64, 128, 128
BT = 64
NT = T // BT
device = "cuda"

torch.manual_seed(42)
k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16) * 0.1
w = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16) * 0.1
u = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16) * 0.1
g = torch.zeros(B, T, H, device=device, dtype=torch.float32)
gk = torch.zeros(B, T, H, K, device=device, dtype=torch.float32)
h0 = torch.zeros(B, H, K, V, device=device, dtype=torch.float32)
h_out = torch.zeros(B, NT, H, K, V, device=device, dtype=torch.bfloat16)
v_new = torch.zeros(B, T, H, V, device=device, dtype=torch.bfloat16)
ht = torch.zeros(B, H, K, V, device=device, dtype=torch.bfloat16)

stream = cutlass_torch.default_stream()

# ===== Compile our kernel =====
print("Compiling our kernel...")
kernel = ChunkDeltaRuleFwdH(chunk_size=BT, head_dim_k=K, head_dim_v=V)
kc, wc, uc = from_dlpack(k), from_dlpack(w), from_dlpack(u)
gc, gkc = from_dlpack(g), from_dlpack(gk)
h0c = from_dlpack(h0)
hc, vnc, htc = from_dlpack(h_out), from_dlpack(v_new), from_dlpack(ht)
args = (
    kc.iterator, wc.iterator, uc.iterator,
    gc.iterator, gkc.iterator,
    hc.iterator, vnc.iterator, h0c.iterator, htc.iterator,
    (B, T, H, K, V), 0, 0, 0, 0, 1, stream,
)
compiled = cute.compile(kernel, *args)

# Warmup both kernels
print("Warming up...")
compiled(*args)
fla_fwd_h(k=k, w=w, u=u, g=None, gk=None, initial_state=None,
          output_final_state=False, chunk_size=BT, save_new_value=True)
torch.cuda.synchronize()

# ===== Profile execution =====
print("Profiling...")
# Run our kernel
compiled(*args)
torch.cuda.synchronize()

# Run FLA kernel
fla_fwd_h(k=k, w=w, u=u, g=None, gk=None, initial_state=None,
          output_final_state=False, chunk_size=BT, save_new_value=True)
torch.cuda.synchronize()

print("Done.")
