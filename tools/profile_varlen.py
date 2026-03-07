#!/usr/bin/env python3
"""Profile varlen kernel with ncu: H=64, ratio=2-3x."""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from flashla.chunk_delta_h import ChunkDeltaRuleFwdH


def generate_seq_lens(num_seqs, total_T, ratio, seed=42):
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
    return list(seq_lens)


def make_cu_seqlens(seq_lens, device="cuda"):
    cu = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=device)
    for i, l in enumerate(seq_lens):
        cu[i + 1] = cu[i] + l
    return cu


def make_chunk_offsets(seq_lens, BT, device="cuda"):
    NTs = [(int(l) + BT - 1) // BT for l in seq_lens]
    co = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=device)
    for i, nt in enumerate(NTs):
        co[i + 1] = co[i] + nt
    return co, int(sum(NTs))


def run_profile(num_seqs=20, total_T=8192, H=64, K=128, V=128, BT=64, ratio=2.0):
    device = "cuda"
    dtype = torch.bfloat16

    seq_lens = generate_seq_lens(num_seqs, total_T, ratio)
    cu_seqlens = make_cu_seqlens(seq_lens, device)
    chunk_offsets, total_NT = make_chunk_offsets(seq_lens, BT, device)

    print(f"Config: {num_seqs} seqs, H={H}, ratio={ratio}x, total_T={total_T}")
    print(f"  seq_lens: min={min(seq_lens)}, max={max(seq_lens)}, actual_ratio={max(seq_lens)/min(seq_lens):.1f}x")
    print(f"  total_NT={total_NT} chunks")

    torch.manual_seed(42)
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

    kernel = ChunkDeltaRuleFwdH(chunk_size=BT, head_dim_k=K, head_dim_v=V, is_varlen=True)
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
        (int(num_seqs), int(total_T), H, K, V), int(total_NT),
        0, 1, 1, 1, 1,
        stream,
    )

    compiled = cute.compile(kernel, *args)

    # Warmup
    for _ in range(3):
        compiled(*args)
    torch.cuda.synchronize()

    # --- FLA kernel ---
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h as fla_fwd_h
    cu_seqlens_long = cu_seqlens.long()

    for _ in range(3):
        fla_fwd_h(k=k, w=w, u=u, g=None, gk=gk, initial_state=h0,
                  output_final_state=True, chunk_size=BT, save_new_value=True,
                  cu_seqlens=cu_seqlens_long)
    torch.cuda.synchronize()

    # Profiled runs (ncu will capture these)
    print("\n=== Profiled run: Our kernel ===")
    compiled(*args)
    torch.cuda.synchronize()

    print("\n=== Profiled run: FLA kernel ===")
    fla_fwd_h(k=k, w=w, u=u, g=None, gk=gk, initial_state=h0,
              output_final_state=True, chunk_size=BT, save_new_value=True,
              cu_seqlens=cu_seqlens_long)
    torch.cuda.synchronize()

    print("\nDone.")


if __name__ == "__main__":
    run_profile(num_seqs=20, H=64, ratio=2.0)
