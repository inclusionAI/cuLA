"""Debug persistent kernel hang with printf — H=1, num_iters=2."""
import os, sys
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

import torch
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, '/ossfs/workspace/fwd_o/flashla/flashla')
from fwd_o import ChunkGlaFwdO, build_chunk_indices

device = 'cuda'
dtype = torch.bfloat16
BT, K, V = 64, 128, 128
scale = K ** -0.5
stream = cutlass_torch.default_stream()

sm_count = torch.cuda.get_device_properties(0).multi_processor_count
print(f"SM count: {sm_count}")

# H=1. Need total_nt > sm_count to get num_iters=2.
# 160 seqs of 64 tokens → 160 chunks → 160 WUs.
# 160 > 152 → max_iters = ceil(160/152) = 2 for first 8 CTAs.
H = 1
num_seqs = 160
seq_len_each = 64
seq_lens = [seq_len_each] * num_seqs
T_total = sum(seq_lens)
total_nt = sum((s + BT - 1) // BT for s in seq_lens)  # 160
num_wu = total_nt * H  # 160
max_iters = (num_wu + sm_count - 1) // sm_count

print(f"H={H}, num_seqs={num_seqs}, total_nt={total_nt}, WUs={num_wu}, max_iters={max_iters}")
sys.stdout.flush()

cu_list = [0]
for s in seq_lens:
    cu_list.append(cu_list[-1] + s)
ci = build_chunk_indices(seq_lens, BT=BT, device=device)
cu = torch.tensor(cu_list, dtype=torch.int32, device=device)

torch.manual_seed(42)
q = torch.randn(T_total, H, K, dtype=dtype, device=device)
v = torch.randn(T_total, H, V, dtype=dtype, device=device)
g = torch.randn(T_total, H, K, dtype=torch.float32, device=device) * 0.1
h = torch.randn(total_nt, H, K, V, dtype=dtype, device=device) * 0.01
A = torch.randn(T_total, H, BT, dtype=dtype, device=device) * 0.1
o = torch.zeros(T_total, H, V, dtype=dtype, device=device)

ps = (num_seqs, T_total, H, K, V)
kernel = ChunkGlaFwdO(chunk_size=BT, head_dim_k=K, head_dim_v=V, scale=scale,
                      is_varlen=True, persistent=True)
print(f"persistent={kernel.persistent}, occ={kernel.min_occupancy}")
print("Compiling...")
sys.stdout.flush()

compiled = cute.compile(kernel,
    from_dlpack(q.detach()).iterator, from_dlpack(v.detach()).iterator,
    from_dlpack(g.detach()).iterator, from_dlpack(h.detach()).iterator,
    from_dlpack(o.detach()).iterator, from_dlpack(A.detach()).iterator,
    from_dlpack(cu.detach()).iterator, from_dlpack(ci.detach()).iterator,
    ps, total_nt, stream)
print("Compiled. Running (expect printf output)...")
sys.stdout.flush()

compiled(
    from_dlpack(q.detach()).iterator, from_dlpack(v.detach()).iterator,
    from_dlpack(g.detach()).iterator, from_dlpack(h.detach()).iterator,
    from_dlpack(o.detach()).iterator, from_dlpack(A.detach()).iterator,
    from_dlpack(cu.detach()).iterator, from_dlpack(ci.detach()).iterator,
    ps, total_nt, stream)
torch.cuda.synchronize()
print("DONE - no hang!")
