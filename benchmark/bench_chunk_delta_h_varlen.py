#!/usr/bin/env python3
"""Benchmark varlen chunk_delta_h: SM100 CuTe DSL vs FLA Triton.

Target scenario: total_T=8192, 20-25 variable-length sequences,
shortest:longest ratio ~2-4x.
"""

import argparse
import math
import time
import numpy as np
import torch

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

from flashla.chunk_delta_h import ChunkDeltaRuleFwdH


def generate_seq_lens(num_seqs, total_T, ratio, seed=42):
    """Generate variable-length sequences with given total and ratio.
    
    ratio: max_len / min_len
    """
    rng = np.random.RandomState(seed)
    # Generate random weights with controlled ratio
    # Use log-uniform to get good spread
    log_weights = rng.uniform(0, np.log(ratio), num_seqs)
    weights = np.exp(log_weights)
    # Scale to match total_T
    raw_lens = weights / weights.sum() * total_T
    # Round to integers, ensure minimum length of 1
    seq_lens = np.maximum(np.round(raw_lens).astype(int), 1)
    # Adjust to match total_T exactly
    diff = total_T - seq_lens.sum()
    if diff > 0:
        # Add to shortest sequences
        indices = np.argsort(seq_lens)
        for i in range(abs(diff)):
            seq_lens[indices[i % num_seqs]] += 1
    elif diff < 0:
        # Remove from longest sequences
        indices = np.argsort(-seq_lens)
        for i in range(abs(diff)):
            seq_lens[indices[i % num_seqs]] -= 1
    
    assert seq_lens.sum() == total_T
    assert all(s > 0 for s in seq_lens)
    return list(seq_lens)


def make_cu_seqlens(seq_lens, device="cuda"):
    """Create cu_seqlens from sequence lengths."""
    cu = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=device)
    for i, l in enumerate(seq_lens):
        cu[i + 1] = cu[i] + l
    return cu


def make_chunk_offsets(seq_lens, BT, device="cuda"):
    """Create chunk_offsets from sequence lengths."""
    NTs = [(int(l) + BT - 1) // BT for l in seq_lens]
    co = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=device)
    for i, nt in enumerate(NTs):
        co[i + 1] = co[i] + nt
    return co, int(sum(NTs))


def bench_fn(fn, warmup=5, n_iter=20):
    """Benchmark a function, return median time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def run_varlen_benchmark(num_seqs, total_T, H, K, V, BT, ratio,
                         use_gk=True, use_h0=True, store_ht=True, save_vnew=True,
                         seed=42, persistent=True):
    """Run one varlen benchmark config: our kernel vs FLA."""
    device = "cuda"
    dtype = torch.bfloat16
    
    seq_lens = generate_seq_lens(num_seqs, total_T, ratio, seed=seed)
    cu_seqlens = make_cu_seqlens(seq_lens, device)
    chunk_offsets, total_NT = make_chunk_offsets(seq_lens, BT, device)
    
    min_len, max_len = min(seq_lens), max(seq_lens)
    actual_ratio = max_len / min_len
    
    torch.manual_seed(seed)
    k = torch.randn(1, total_T, H, K, device=device, dtype=dtype) * 0.1
    w = torch.randn(1, total_T, H, K, device=device, dtype=dtype) * 0.1
    u = torch.randn(1, total_T, H, V, device=device, dtype=dtype) * 0.1
    
    gk = None
    h0 = None
    if use_gk:
        gk_raw = torch.randn(1, total_T, H, K, device=device, dtype=torch.float32) * 0.1
        # Per-sequence cumsum
        gk = torch.zeros_like(gk_raw)
        for i in range(num_seqs):
            bos = cu_seqlens[i].item()
            eos = cu_seqlens[i + 1].item()
            gk[:, bos:eos] = -torch.abs(gk_raw[:, bos:eos]).cumsum(dim=1)
    if use_h0:
        h0 = torch.randn(num_seqs, H, K, V, device=device, dtype=torch.float32) * 0.01

    # ========== FLA kernel ==========
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h as fla_fwd_h
    
    cu_seqlens_long = cu_seqlens.long()
    
    def fla_fn():
        fla_fwd_h(
            k=k, w=w, u=u,
            g=None, gk=gk,
            initial_state=h0,
            output_final_state=store_ht,
            chunk_size=BT,
            save_new_value=save_vnew,
            cu_seqlens=cu_seqlens_long,
        )
    
    fla_ms = bench_fn(fla_fn)

    # ========== Our SM100 kernel ==========
    g_tensor = torch.zeros(1, total_T, H, device=device, dtype=torch.float32)
    gk_tensor = gk if gk is not None else torch.zeros(1, total_T, H, K, device=device, dtype=torch.float32)
    h0_tensor = h0 if h0 is not None else torch.zeros(num_seqs, H, K, V, device=device, dtype=torch.float32)
    
    h_out = torch.zeros(1, total_NT, H, K, V, device=device, dtype=dtype)
    v_new_out = torch.zeros(1, total_T, H, V, device=device, dtype=dtype)
    ht_out = torch.zeros(num_seqs, H, K, V, device=device, dtype=torch.float32)
    workspace = torch.zeros(num_seqs * 128, dtype=torch.uint8, device=device)
    
    kernel = ChunkDeltaRuleFwdH(chunk_size=BT, head_dim_k=K, head_dim_v=V, is_varlen=True, persistent=persistent)
    stream = cutlass_torch.default_stream()
    
    kc, wc, uc = from_dlpack(k), from_dlpack(w), from_dlpack(u)
    gc, gkc = from_dlpack(g_tensor), from_dlpack(gk_tensor)
    h0c = from_dlpack(h0_tensor)
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
        0, int(use_gk), int(use_h0), int(store_ht), int(save_vnew),
        stream,
    )
    
    compiled = cute.compile(kernel, *args)
    
    def our_fn():
        compiled(*args)
    
    our_ms = bench_fn(our_fn)
    
    return our_ms, fla_ms, seq_lens, actual_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_T", type=int, default=8192)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--head_dim_k", type=int, default=128)
    parser.add_argument("--head_dim_v", type=int, default=128)
    parser.add_argument("--persistent", type=int, nargs="+", default=None,
                        help="Persistent kernel mode (default: [1]). "
                             "0=non-persistent (free HW scheduling), 1=persistent.")
    args = parser.parse_args()
    
    K, V, BT = args.head_dim_k, args.head_dim_v, args.chunk_size
    total_T = args.total_T
    persist_list = args.persistent if args.persistent else [1]
    
    configs = [
        # (num_seqs, H, ratio, use_gk, use_h0, store_ht, save_vnew, description)
        # Core target scenario: 20-25 seqs, ratio 2-4x, all features
        (20, 16,  2.0, True, True, True, True, "20 seqs, ratio=2x, H=16"),
        (20, 32,  2.0, True, True, True, True, "20 seqs, ratio=2x, H=32"),
        (20, 64,  2.0, True, True, True, True, "20 seqs, ratio=2x, H=64"),
        (25, 16,  3.0, True, True, True, True, "25 seqs, ratio=3x, H=16"),
        (25, 32,  3.0, True, True, True, True, "25 seqs, ratio=3x, H=32"),
        (25, 64,  3.0, True, True, True, True, "25 seqs, ratio=3x, H=64"),
        (20, 64,  4.0, True, True, True, True, "20 seqs, ratio=4x, H=64"),
        (25, 64,  4.0, True, True, True, True, "25 seqs, ratio=4x, H=64"),
        # Fewer features
        (25, 64,  3.0, False, False, False, True, "25 seqs, ratio=3x, H=64 (minimal)"),
        # More/fewer sequences
        (10, 64,  3.0, True, True, True, True, "10 seqs, ratio=3x, H=64"),
        (30, 64,  3.0, True, True, True, True, "30 seqs, ratio=3x, H=64"),
        (40, 64,  3.0, True, True, True, True, "40 seqs, ratio=3x, H=64"),
    ]
    
    # Build variant keys: persistent mode combinations
    variants = persist_list
    
    print(f"Varlen Benchmark: total_T={total_T}, K={K}, V={V}, BT={BT}, persist={persist_list}")
    
    # Build header
    def var_label(p):
        return 'Persistent' if p else 'Free'
    var_cols = "".join(f" {var_label(v):>12}" for v in variants)
    print(f"{'Config':<45}{var_cols} {'FLA':>10} {'Best':>7} {'MinL':>5} {'MaxL':>5} {'Ratio':>6}")
    print("-" * (45 + 12 * len(variants) + 10 + 7 + 5 + 5 + 6 + 6))
    
    all_speedups = {v: [] for v in variants}
    for (num_seqs, H, ratio, use_gk, use_h0, store_ht, save_vnew, desc) in configs:
        fla_ms = None
        results = {}
        for p in variants:
            our_ms, fla_ms_cur, seq_lens, actual_ratio = run_varlen_benchmark(
                num_seqs, total_T, H, K, V, BT, ratio,
                use_gk, use_h0, store_ht, save_vnew,
                persistent=bool(p),
            )
            results[p] = our_ms
            fla_ms = fla_ms_cur  # same for all variants
        
        best_var = min(results, key=results.get)
        best_sp = fla_ms / results[best_var]
        min_l, max_l = min(seq_lens), max(seq_lens)
        
        var_str = "".join(f" {results[v]:>11.3f}ms" for v in variants)
        print(f"{desc:<45}{var_str} {fla_ms:>9.3f}ms {best_sp:>6.2f}x {min_l:>5} {max_l:>5} {actual_ratio:>5.1f}x")
        
        for v in variants:
            all_speedups[v].append(fla_ms / results[v])
    
    print("-" * (45 + 12 * len(variants) + 10 + 7 + 5 + 5 + 6 + 6))
    for v in variants:
        geo = math.exp(sum(math.log(s) for s in all_speedups[v]) / len(all_speedups[v]))
        print(f"{'Geomean '+var_label(v):<45} {geo:>6.2f}x")


if __name__ == "__main__":
    main()
