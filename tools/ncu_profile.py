#!/usr/bin/env python3
"""NCU profiling script: compile first, then profile execution only.

Usage:
  # Profile our kernel only:
  ncu -o kda_report --set full python ncu_profile.py

  # Profile both ours + FLA:
  ncu -o kda_report --set full --launch-count 2 python ncu_profile.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from flashla.chunk_delta_h import ChunkDeltaRuleFwdH

# ---------- Config ----------
# Varlen scenario: 20 seqs, total 8192 tokens
num_seqs = 20
total_T = 8192
H, K, V, BT = 20, 128, 128, 64
BV = 64
num_stages = 2
min_occupancy = 1
seed = 42
ratio = 3.0
device = "cuda"
dtype = torch.bfloat16

# ---------- Generate varlen data ----------
rng = np.random.RandomState(seed)
log_weights = rng.uniform(0, np.log(ratio), num_seqs)
weights = np.exp(log_weights)
raw_lens = weights / weights.sum() * total_T
seq_lens = np.maximum(np.round(raw_lens).astype(int), 1)
diff = total_T - seq_lens.sum()
if diff > 0:
    indices = np.argsort(seq_lens)
    for i in range(abs(diff)):
        seq_lens[indices[i % num_seqs]] += 1
elif diff < 0:
    indices = np.argsort(-seq_lens)
    for i in range(abs(diff)):
        seq_lens[indices[i % num_seqs]] -= 1

cu_seqlens = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
for i, l in enumerate(seq_lens):
    cu_seqlens[i + 1] = cu_seqlens[i] + l

NTs = [(int(l) + BT - 1) // BT for l in seq_lens]
chunk_offsets = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
for i, nt in enumerate(NTs):
    chunk_offsets[i + 1] = chunk_offsets[i] + nt
total_NT = int(sum(NTs))

torch.manual_seed(seed)
k = torch.randn(1, total_T, H, K, device=device, dtype=dtype) * 0.1
w = torch.randn(1, total_T, H, K, device=device, dtype=dtype) * 0.1
u = torch.randn(1, total_T, H, V, device=device, dtype=dtype) * 0.1
gk_raw = torch.randn(1, total_T, H, K, device=device, dtype=torch.float32) * 0.1
gk = torch.zeros_like(gk_raw)
for i in range(num_seqs):
    bos = cu_seqlens[i].item()
    eos = cu_seqlens[i + 1].item()
    gk[:, bos:eos] = -torch.abs(gk_raw[:, bos:eos]).cumsum(dim=1)
h0 = torch.randn(num_seqs, H, K, V, device=device, dtype=torch.float32) * 0.01

g_tensor = torch.zeros(1, total_T, H, device=device, dtype=torch.float32)
h_out = torch.zeros(1, total_NT, H, K, V, device=device, dtype=dtype)
v_new_out = torch.zeros(1, total_T, H, V, device=device, dtype=dtype)
ht_out = torch.zeros(num_seqs, H, K, V, device=device, dtype=dtype)
workspace = torch.zeros(num_seqs * 128, dtype=torch.uint8, device=device)

# ---------- Compile ----------
print("Compiling kernel...")
kernel = ChunkDeltaRuleFwdH(
    chunk_size=BT, head_dim_k=K, head_dim_v=V,
    is_varlen=True, BV=BV, num_stages=num_stages, min_occupancy=min_occupancy
)
stream = cutlass_torch.default_stream()

kc, wc, uc = from_dlpack(k), from_dlpack(w), from_dlpack(u)
gc, gkc = from_dlpack(g_tensor), from_dlpack(gk)
h0c = from_dlpack(h0)
hc, vnc, htc = from_dlpack(h_out), from_dlpack(v_new_out), from_dlpack(ht_out)
csc = from_dlpack(cu_seqlens)
coc = from_dlpack(chunk_offsets)
wsc = from_dlpack(workspace)

args = (
    kc.iterator, wc.iterator, uc.iterator,
    gc.iterator, gkc.iterator,
    hc.iterator, vnc.iterator, h0c.iterator, htc.iterator,
    csc.iterator, coc.iterator, wsc.iterator,
    (num_seqs, total_T, H, K, V), total_NT,
    0, 1, 1, 1, 1,
    stream,
)
compiled = cute.compile(kernel, *args)
print("Compiled.")

# ---------- Warmup ----------
print("Warming up...")
for _ in range(3):
    compiled(*args)
torch.cuda.synchronize()

# ---------- Profile ----------
print("Profiling...")
compiled(*args)
torch.cuda.synchronize()
print("Done.")
