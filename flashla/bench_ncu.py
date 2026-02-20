"""NCU profiling: our CuTe DSL kernel + FLA Triton kernel."""
import torch
from chunk_delta_h import ChunkDeltaRuleFwdH
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h as fla_fwd_h
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

B, T, H, K, V, BT = 4, 4096, 64, 128, 128, 64
NT = T // BT
device = "cuda"
dtype = torch.bfloat16

torch.manual_seed(42)
k = torch.randn(B,T,H,K, device=device, dtype=dtype) * 0.1
w = torch.randn(B,T,H,K, device=device, dtype=dtype) * 0.1
u = torch.randn(B,T,H,V, device=device, dtype=dtype) * 0.1
g = torch.zeros(B,T,H, device=device, dtype=torch.float32)
gk = torch.zeros(B,T,H,K, device=device, dtype=torch.float32)
h0 = torch.zeros(B,H,K,V, device=device, dtype=torch.float32)

# Our kernel
h_out = torch.zeros(B,NT,H,K,V, device=device, dtype=dtype)
v_new_out = torch.zeros(B,T,H,V, device=device, dtype=dtype)
ht_out = torch.zeros(B,H,K,V, device=device, dtype=dtype)
kernel = ChunkDeltaRuleFwdH(chunk_size=BT, head_dim_k=K, head_dim_v=V)
stream = cutlass_torch.default_stream()
kc, wc, uc = from_dlpack(k), from_dlpack(w), from_dlpack(u)
gc, gkc = from_dlpack(g), from_dlpack(gk)
h0c, hc, vnc, htc = from_dlpack(h0), from_dlpack(h_out), from_dlpack(v_new_out), from_dlpack(ht_out)
args = (kc.iterator, wc.iterator, uc.iterator, gc.iterator, gkc.iterator,
        hc.iterator, vnc.iterator, h0c.iterator, htc.iterator,
        (B,T,H,K,V), 0, 0, 0, 0, 1, stream)
compiled = cute.compile(kernel, *args)

# Warmup both outside profiler
for _ in range(3):
    compiled(*args)
torch.cuda.synchronize()
for _ in range(3):
    fla_fwd_h(k=k, w=w, u=u, g=None, gk=None, initial_state=None,
              output_final_state=False, chunk_size=BT, save_new_value=True)
torch.cuda.synchronize()

# Profiled launches (ncu --kernel-id ::regex:kernel_cutlass:1 or --launch-skip N)
compiled(*args)
torch.cuda.synchronize()

fla_fwd_h(k=k, w=w, u=u, g=None, gk=None, initial_state=None,
          output_final_state=False, chunk_size=BT, save_new_value=True)
torch.cuda.synchronize()
print("Done.")
