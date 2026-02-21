"""
NCU profiling script for slow configs (g gating).
Profiles our kernel and FLA kernel under g-gating scenario.
"""
import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from chunk_delta_h import ChunkDeltaRuleFwdH
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h as fla_fwd_h

B, T, H, K, V, BT = 4, 4096, 64, 128, 128, 64
NT = T // BT
device = "cuda"
dtype = torch.bfloat16

torch.manual_seed(42)
k = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
w = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
u = torch.randn(B, T, H, V, device=device, dtype=dtype) * 0.1

# g gating (the slow case)
g = torch.randn(B, T, H, device=device, dtype=torch.float32) * 0.1
g = -torch.abs(g).cumsum(dim=1)

gk = torch.zeros(B, T, H, K, device=device, dtype=torch.float32)
h0 = torch.zeros(B, H, K, V, device=device, dtype=torch.float32)

# ===== Our kernel (with g gating) =====
h_out = torch.zeros(B, NT, H, K, V, device=device, dtype=dtype)
v_new_out = torch.zeros(B, T, H, V, device=device, dtype=dtype)
ht_out = torch.zeros(B, H, K, V, device=device, dtype=dtype)
kernel = ChunkDeltaRuleFwdH(chunk_size=BT, head_dim_k=K, head_dim_v=V)
stream = cutlass_torch.default_stream()
kc, wc, uc = from_dlpack(k), from_dlpack(w), from_dlpack(u)
gc, gkc = from_dlpack(g), from_dlpack(gk)
h0c, hc, vnc, htc = from_dlpack(h0), from_dlpack(h_out), from_dlpack(v_new_out), from_dlpack(ht_out)
args_g = (kc.iterator, wc.iterator, uc.iterator, gc.iterator, gkc.iterator,
          hc.iterator, vnc.iterator, h0c.iterator, htc.iterator,
          (B, T, H, K, V), 1, 0, 0, 0, 1, stream)  # use_g=1
compiled_g = cute.compile(kernel, *args_g)

# ===== Our kernel (without gating, for comparison) =====
kernel2 = ChunkDeltaRuleFwdH(chunk_size=BT, head_dim_k=K, head_dim_v=V)
h_out2 = torch.zeros(B, NT, H, K, V, device=device, dtype=dtype)
v_new2 = torch.zeros(B, T, H, V, device=device, dtype=dtype)
ht2 = torch.zeros(B, H, K, V, device=device, dtype=dtype)
hc2, vnc2, htc2 = from_dlpack(h_out2), from_dlpack(v_new2), from_dlpack(ht2)
g0 = torch.zeros(B, T, H, device=device, dtype=torch.float32)
gc0 = from_dlpack(g0)
args_no_g = (kc.iterator, wc.iterator, uc.iterator, gc0.iterator, gkc.iterator,
             hc2.iterator, vnc2.iterator, h0c.iterator, htc2.iterator,
             (B, T, H, K, V), 0, 0, 0, 0, 1, stream)  # use_g=0
compiled_no_g = cute.compile(kernel2, *args_no_g)

# Warmup all
for _ in range(3):
    compiled_g(*args_g)
    compiled_no_g(*args_no_g)
    fla_fwd_h(k=k, w=w, u=u, g=g, gk=None, initial_state=None,
              output_final_state=False, chunk_size=BT, save_new_value=True)
    fla_fwd_h(k=k, w=w, u=u, g=None, gk=None, initial_state=None,
              output_final_state=False, chunk_size=BT, save_new_value=True)
torch.cuda.synchronize()

# === Profiled launches ===
# Launch 1: Our kernel WITH g gating (the slow one)
compiled_g(*args_g)
torch.cuda.synchronize()

# Launch 2: Our kernel WITHOUT g gating (fast baseline)
compiled_no_g(*args_no_g)
torch.cuda.synchronize()

# Launch 3: FLA kernel WITH g gating
fla_fwd_h(k=k, w=w, u=u, g=g, gk=None, initial_state=None,
          output_final_state=False, chunk_size=BT, save_new_value=True)
torch.cuda.synchronize()

# Launch 4: FLA kernel WITHOUT g gating
fla_fwd_h(k=k, w=w, u=u, g=None, gk=None, initial_state=None,
          output_final_state=False, chunk_size=BT, save_new_value=True)
torch.cuda.synchronize()

print("Done. 4 profiled launches: ours_g, ours_no_g, fla_g, fla_no_g")
