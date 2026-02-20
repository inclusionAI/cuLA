#!/usr/bin/env python3
"""
Comprehensive benchmark: disambiguate all prior claims.

Prior version benchmarked our h-kernel vs FLA's FULL forward pass
(chunk_gated_delta_rule which includes q*h attention + h-state + output).
That's an apples-to-oranges comparison.

This script benchmarks ALL three:
  A) Our h-kernel  vs  FLA's h-kernel  (apples-to-apples)
  B) Our h-kernel  vs  FLA full forward (the old unfair comparison)
  C) FLA h-kernel  vs  FLA full forward (to show the gap)

Plus: tests with/without v_new save, with/without gating, with/without h0/ht.
"""

import argparse
import math
import time
import torch

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

from chunk_delta_h import ChunkDeltaRuleFwdH
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h as fla_fwd_h
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_full_fwd


def bench_fn(fn, warmup=5, rep=20):
    """CUDA event timing."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


def make_our_kernel_fn(B, T, H, K, V, BT, k, w, u, g, gk, h0,
                       use_g, use_gk, use_h0, store_ht, save_vnew):
    NT = T // BT
    device = "cuda"
    dtype = torch.bfloat16

    g_tensor = g if g is not None else torch.zeros(B, T, H, device=device, dtype=torch.float32)
    gk_tensor = gk if gk is not None else torch.zeros(B, T, H, K, device=device, dtype=torch.float32)
    h0_tensor = h0 if h0 is not None else torch.zeros(B, H, K, V, device=device, dtype=torch.float32)

    h_out = torch.zeros(B, NT, H, K, V, device=device, dtype=dtype)
    v_new_out = torch.zeros(B, T, H, V, device=device, dtype=dtype)
    ht_out = torch.zeros(B, H, K, V, device=device, dtype=dtype)

    kernel = ChunkDeltaRuleFwdH(chunk_size=BT, head_dim_k=K, head_dim_v=V)
    stream = cutlass_torch.default_stream()

    kc, wc, uc = from_dlpack(k), from_dlpack(w), from_dlpack(u)
    gc, gkc = from_dlpack(g_tensor), from_dlpack(gk_tensor)
    h0c = from_dlpack(h0_tensor)
    hc, vnc, htc = from_dlpack(h_out), from_dlpack(v_new_out), from_dlpack(ht_out)

    args = (
        kc.iterator, wc.iterator, uc.iterator,
        gc.iterator, gkc.iterator,
        hc.iterator, vnc.iterator, h0c.iterator, htc.iterator,
        (B, T, H, K, V),
        int(use_g), int(use_gk), int(use_h0), int(store_ht), int(save_vnew),
        stream,
    )
    compiled = cute.compile(kernel, *args)

    def fn():
        compiled(*args)
    return fn


def make_fla_h_fn(k, w, u, g, gk, h0, store_ht, save_vnew, BT):
    def fn():
        fla_fwd_h(
            k=k, w=w, u=u,
            g=g, gk=gk,
            initial_state=h0,
            output_final_state=store_ht,
            chunk_size=BT,
            save_new_value=save_vnew,
        )
    return fn


def make_fla_full_fn(B, T, H, K, V, k, w, u, gk_full, save_vnew):
    """FLA full forward: chunk_gated_delta_rule(q, k, v, gk, beta, ...)
    Note: this does the ENTIRE forward pass including q*h attention.
    The 'q' and 'beta' inputs are additional vs the h-kernel.
    """
    q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16) * 0.1
    v = u.clone()
    beta = torch.ones(B, T, H, device="cuda", dtype=torch.float32)
    if gk_full is None:
        gk_full = torch.zeros(B, T, H, K, device="cuda", dtype=torch.float32)
    scale = K ** -0.5
    _fla_full = fla_full_fwd  # capture in closure

    def fn():
        _fla_full(
            q, k, v, gk_full, beta,
            scale=scale,
            initial_state=None,
            output_final_state=False,
        )
    return fn


def run_all(B, T, H, K, V, BT, use_g, use_gk, use_h0, store_ht, save_vnew):
    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(42)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
    w = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
    u = torch.randn(B, T, H, V, device=device, dtype=dtype) * 0.1
    g = None
    gk = None
    h0 = None
    if use_g:
        g = torch.randn(B, T, H, device=device, dtype=torch.float32) * 0.1
        g = -torch.abs(g).cumsum(dim=1)
    if use_gk:
        gk = torch.randn(B, T, H, K, device=device, dtype=torch.float32) * 0.1
        gk = -torch.abs(gk).cumsum(dim=1)
    if use_h0:
        h0 = torch.randn(B, H, K, V, device=device, dtype=torch.float32) * 0.01

    # A) Our h-kernel
    our_fn = make_our_kernel_fn(B, T, H, K, V, BT, k, w, u, g, gk, h0,
                                 use_g, use_gk, use_h0, store_ht, save_vnew)
    our_ms = bench_fn(our_fn)

    # B) FLA h-kernel
    fla_h_fn = make_fla_h_fn(k, w, u, g, gk, h0, store_ht, save_vnew, BT)
    fla_h_ms = bench_fn(fla_h_fn)

    return our_ms, fla_h_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--head_dim_k", type=int, default=128)
    parser.add_argument("--head_dim_v", type=int, default=128)
    args = parser.parse_args()

    K, V, BT = args.head_dim_k, args.head_dim_v, args.chunk_size

    # ======================================================================
    # Part 1: Fair comparison (h-kernel vs h-kernel) with different configs
    # ======================================================================
    print("=" * 90)
    print("Part 1: OUR h-kernel vs FLA h-kernel (apples-to-apples, same function)")
    print("=" * 90)

    configs_h = [
        # (B, T, H, use_g, use_gk, use_h0, store_ht, save_vnew, description)
        # --- Baseline: no gating, no v_new save ---
        (4, 4096, 64, False, False, False, False, False, "baseline (no vn save)"),
        # --- With v_new save ---
        (4, 4096, 64, False, False, False, False, True, "with v_new save"),
        # --- With g gating ---
        (4, 4096, 64, True, False, False, False, False, "g gate, no vn"),
        (4, 4096, 64, True, False, False, False, True, "g gate + vn"),
        # --- With h0/ht ---
        (4, 4096, 64, False, False, True, True, True, "h0+ht+vn"),
        # --- All features ---
        (4, 4096, 64, True, False, True, True, True, "g+h0+ht+vn"),
        # --- Different sizes ---
        (1, 2048, 64, False, False, False, False, True, "small: B1 T2k"),
        (2, 2048, 64, False, False, False, False, True, "medium: B2 T2k"),
        (8, 2048, 64, False, False, False, False, True, "large: B8 T2k"),
        (4, 2048, 64, False, False, False, False, True, "B4 T2k"),
        (4, 8192, 64, False, False, False, False, True, "B4 T8k"),
    ]

    print(f"\n{'Description':<30} {'Ours (ms)':>10} {'FLA-h (ms)':>10} {'Ratio':>10}")
    print("-" * 64)

    speedups_h = []
    for cfg in configs_h:
        B, T, H, ug, ugk, uh0, uht, uvn, desc = cfg
        our_ms, fla_h_ms = run_all(B, T, H, K, V, BT, ug, ugk, uh0, uht, uvn)
        ratio = fla_h_ms / our_ms
        speedups_h.append(ratio)
        print(f"{desc:<30} {our_ms:>10.3f} {fla_h_ms:>10.3f} {ratio:>9.2f}x")

    geo_h = math.exp(sum(math.log(s) for s in speedups_h) / len(speedups_h))
    print("-" * 64)
    print(f"{'Geometric mean':<30} {'':>10} {'':>10} {geo_h:>9.2f}x")

    # ======================================================================
    # Part 2: Old unfair comparison (our h-kernel vs FLA full forward)
    # ======================================================================
    print("\n" + "=" * 90)
    print("Part 2: OUR h-kernel vs FLA FULL forward (the old unfair comparison)")
    print("  FLA full = chunk_gated_delta_rule() includes: WY preprocess + h-state + q*h attention + output")
    print("  Our kernel = h-state only")
    print("=" * 90)

    configs_full = [
        (4, 4096, 64),
        (8, 4096, 64),
        (4, 8192, 64),
    ]

    print(f"\n{'Config':<30} {'Ours-h (ms)':>12} {'FLA-full (ms)':>14} {'Ratio':>10}")
    print("-" * 70)

    for B, T, H in configs_full:
        device = "cuda"
        dtype = torch.bfloat16
        torch.manual_seed(42)
        k = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
        w = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
        u = torch.randn(B, T, H, V, device=device, dtype=dtype) * 0.1

        # Our h-kernel (no gating, no v_new save, matching old benchmark)
        our_fn = make_our_kernel_fn(B, T, H, K, V, BT, k, w, u, None, None, None,
                                     False, False, False, False, False)
        our_ms = bench_fn(our_fn)

        # FLA full forward
        gk_full = torch.zeros(B, T, H, K, device=device, dtype=torch.float32)
        fla_full_fn = make_fla_full_fn(B, T, H, K, V, k, w, u, gk_full, False)
        fla_full_ms = bench_fn(fla_full_fn)

        ratio = fla_full_ms / our_ms
        label = f"B={B} T={T} H={H}"
        print(f"{label:<30} {our_ms:>12.3f} {fla_full_ms:>14.3f} {ratio:>9.2f}x")

    # ======================================================================
    # Part 3: FLA h-kernel vs FLA full forward (shows what fraction h is of total)
    # ======================================================================
    print("\n" + "=" * 90)
    print("Part 3: FLA h-kernel vs FLA FULL forward (h-kernel is what fraction of total?)")
    print("=" * 90)

    print(f"\n{'Config':<30} {'FLA-h (ms)':>12} {'FLA-full (ms)':>14} {'h/full %':>10}")
    print("-" * 70)

    for B, T, H in configs_full:
        device = "cuda"
        dtype = torch.bfloat16
        torch.manual_seed(42)
        k = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
        w = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
        u = torch.randn(B, T, H, V, device=device, dtype=dtype) * 0.1

        fla_h_fn = make_fla_h_fn(k, w, u, None, None, None, False, True, BT)
        fla_h_ms = bench_fn(fla_h_fn)

        gk_full = torch.zeros(B, T, H, K, device=device, dtype=torch.float32)
        fla_full_fn = make_fla_full_fn(B, T, H, K, V, k, w, u, gk_full, False)
        fla_full_ms = bench_fn(fla_full_fn)

        pct = fla_h_ms / fla_full_ms * 100
        label = f"B={B} T={T} H={H}"
        print(f"{label:<30} {fla_h_ms:>12.3f} {fla_full_ms:>14.3f} {pct:>9.1f}%")

    # ======================================================================
    # Part 4: Impact of save_v_new on our kernel
    # ======================================================================
    print("\n" + "=" * 90)
    print("Part 4: Our kernel: impact of save_v_new (element-wise GMEM write)")
    print("=" * 90)

    B, T, H = 4, 4096, 64
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
    w = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
    u = torch.randn(B, T, H, V, device=device, dtype=dtype) * 0.1

    fn_no_vn = make_our_kernel_fn(B, T, H, K, V, BT, k, w, u, None, None, None,
                                   False, False, False, False, False)
    fn_with_vn = make_our_kernel_fn(B, T, H, K, V, BT, k, w, u, None, None, None,
                                     False, False, False, False, True)
    ms_no_vn = bench_fn(fn_no_vn)
    ms_with_vn = bench_fn(fn_with_vn)
    print(f"\n  Without v_new save: {ms_no_vn:.3f} ms")
    print(f"  With v_new save:    {ms_with_vn:.3f} ms")
    print(f"  Overhead:           {(ms_with_vn - ms_no_vn):.3f} ms ({(ms_with_vn/ms_no_vn - 1)*100:.1f}%)")

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"\n  Our h-kernel vs FLA h-kernel (fair):  geo mean = {geo_h:.2f}x")
    print(f"  (< 1.0 means we are SLOWER than FLA)")
    print()
    print("  The prior version's '2.55x speedup' claim compared our h-kernel (one sub-kernel)")
    print("  against FLA's FULL chunk_gated_delta_rule forward pass (which includes WY preprocess")
    print("  + h-state recurrence + q*h chunk attention + output projection).")
    print("  That was an apples-to-oranges comparison.")


if __name__ == "__main__":
    main()
