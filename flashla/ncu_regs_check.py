#!/usr/bin/env python3
"""Quick NCU check: does changing num_regs_cuda actually affect launch register count?"""

import numpy as np
import torch
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from chunk_delta_h import ChunkDeltaRuleFwdH

num_seqs, total_T, H, K, V, BT = 20, 8192, 64, 128, 128, 64
device, dtype = "cuda", torch.bfloat16
BV = 64

seq_lens = [total_T // num_seqs] * num_seqs
seq_lens[-1] += total_T - sum(seq_lens)
cu_seqlens = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
for i, l in enumerate(seq_lens):
    cu_seqlens[i + 1] = cu_seqlens[i] + l
NTs = [(l + BT - 1) // BT for l in seq_lens]
chunk_offsets = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
for i, nt in enumerate(NTs):
    chunk_offsets[i + 1] = chunk_offsets[i] + nt
total_NT = sum(NTs)

torch.manual_seed(42)
k = torch.randn(1, total_T, H, K, device=device, dtype=dtype) * 0.1
w = torch.randn(1, total_T, H, K, device=device, dtype=dtype) * 0.1
u = torch.randn(1, total_T, H, V, device=device, dtype=dtype) * 0.1
gk = torch.randn(1, total_T, H, K, device=device, dtype=torch.float32) * -0.01
gk = gk.cumsum(dim=1)
h0 = torch.randn(num_seqs, H, K, V, device=device, dtype=torch.float32) * 0.01
g_tensor = torch.zeros(1, total_T, H, device=device, dtype=torch.float32)
h_out = torch.zeros(1, total_NT, H, K, V, device=device, dtype=dtype)
v_new_out = torch.zeros(1, total_T, H, V, device=device, dtype=dtype)
ht_out = torch.zeros(num_seqs, H, K, V, device=device, dtype=torch.float32)
workspace = torch.zeros(num_seqs * 128, dtype=torch.uint8, device=device)

import sys
num_regs = int(sys.argv[1]) if len(sys.argv) > 1 else 240
print(f"Using num_regs_cuda={num_regs}")

kernel = ChunkDeltaRuleFwdH(
    chunk_size=BT, head_dim_k=K, head_dim_v=V,
    is_varlen=True, BV=BV, num_stages=2, min_occupancy=1,
    persistent=True, num_regs_cuda=num_regs,
)
stream = cutlass_torch.default_stream()
kc, wc, uc = from_dlpack(k), from_dlpack(w), from_dlpack(u)
gc, gkc = from_dlpack(g_tensor), from_dlpack(gk)
h0c = from_dlpack(h0)
hc, vnc, htc = from_dlpack(h_out), from_dlpack(v_new_out), from_dlpack(ht_out)
csc, coc, wsc = from_dlpack(cu_seqlens), from_dlpack(chunk_offsets), from_dlpack(workspace)
args = (
    kc.iterator, wc.iterator, uc.iterator,
    gc.iterator, gkc.iterator,
    hc.iterator, vnc.iterator, h0c.iterator, htc.iterator,
    csc.iterator, coc.iterator, wsc.iterator,
    (int(num_seqs), int(total_T), H, K, V), int(total_NT),
    0, 1, 1, 1, 1, stream,
)
compiled = cute.compile(kernel, *args)
for _ in range(3):
    compiled(*args)
torch.cuda.synchronize()
print("Profiling...")
compiled(*args)
torch.cuda.synchronize()
print("Done.")
