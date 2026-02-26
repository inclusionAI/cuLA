#!/usr/bin/env python3
"""
Fair benchmark: our SM100 CuTe DSL h-kernel vs FLA's Triton h-kernel.

Both kernels compute exactly the same thing:
  - Input:  k, w, u, g (pre-processed WY representation inputs)
  - Output: h_out (per-chunk state), v_new, ht (final state)
  - Math:   h_out[t] = h; v_new = u - w@h; v_new *= gate; h = decay*h + k^T @ v_new

We call each kernel's h-state function directly, bypassing any outer wrappers.
"""

import argparse
import time
import torch
import triton

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

from chunk_delta_h import ChunkDeltaRuleFwdH
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h as fla_fwd_h


def bench_fn(fn, warmup=5, rep=20):
    """Benchmark using CUDA events for precise GPU timing."""
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


def run_benchmark(B, T, H, K, V, BT, use_g, use_gk, use_h0, store_ht, save_vnew):
    NT = T // BT
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

    # ========== FLA kernel ==========
    def fla_fn():
        fla_fwd_h(
            k=k, w=w, u=u,
            g=g, gk=gk,
            initial_state=h0,
            output_final_state=store_ht,
            chunk_size=BT,
            save_new_value=save_vnew,
        )

    fla_ms = bench_fn(fla_fn)

    # ========== Our SM100 kernel ==========
    g_tensor = g if g is not None else torch.zeros(B, T, H, device=device, dtype=torch.float32)
    gk_tensor = gk if gk is not None else torch.zeros(B, T, H, K, device=device, dtype=torch.float32)
    h0_tensor = h0 if h0 is not None else torch.zeros(B, H, K, V, device=device, dtype=torch.float32)

    h_out = torch.zeros(B, NT, H, K, V, device=device, dtype=dtype)
    v_new_out = torch.zeros(B, T, H, V, device=device, dtype=dtype)
    ht_out = torch.zeros(B, H, K, V, device=device, dtype=torch.float32)

    kernel = ChunkDeltaRuleFwdH(chunk_size=BT, head_dim_k=K, head_dim_v=V)
    stream = cutlass_torch.default_stream()

    kc, wc, uc = from_dlpack(k), from_dlpack(w), from_dlpack(u)
    gc, gkc = from_dlpack(g_tensor), from_dlpack(gk_tensor)
    h0c = from_dlpack(h0_tensor)
    hc, vnc, htc = from_dlpack(h_out), from_dlpack(v_new_out), from_dlpack(ht_out)
    cu_seqlens_d = torch.zeros(2, dtype=torch.int32, device=device)
    chunk_offsets_d = torch.zeros(2, dtype=torch.int32, device=device)
    workspace_d = torch.zeros(128, dtype=torch.uint8, device=device)
    csd = from_dlpack(cu_seqlens_d)
    cod = from_dlpack(chunk_offsets_d)
    wsd = from_dlpack(workspace_d)

    args = (
        kc.iterator, wc.iterator, uc.iterator,
        gc.iterator, gkc.iterator,
        hc.iterator, vnc.iterator, h0c.iterator, htc.iterator,
        csd.iterator, cod.iterator, wsd.iterator,
        (B, T, H, K, V), NT,
        int(use_g), int(use_gk), int(use_h0), int(store_ht), int(save_vnew),
        stream,
    )
    compiled = cute.compile(kernel, *args)

    def our_fn():
        compiled(*args)

    our_ms = bench_fn(our_fn)

    return our_ms, fla_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--head_dim_k", type=int, default=128)
    parser.add_argument("--head_dim_v", type=int, default=128)
    args = parser.parse_args()

    K, V, BT = args.head_dim_k, args.head_dim_v, args.chunk_size

    configs = [
        # (B, T, H, use_g, use_gk, use_h0, store_ht, save_vnew)
        # --- T=8192, all features (gk, h0, ht, vn), sweep B × H ---
        (1,  8192, 16,  False, True, True, True, True),
        (1,  8192, 32,  False, True, True, True, True),
        (1,  8192, 64,  False, True, True, True, True),
        (1,  8192, 128, False, True, True, True, True),
        (2,  8192, 16,  False, True, True, True, True),
        (2,  8192, 32,  False, True, True, True, True),
        (2,  8192, 64,  False, True, True, True, True),
        (2,  8192, 128, False, True, True, True, True),
        (4,  8192, 16,  False, True, True, True, True),
        (4,  8192, 32,  False, True, True, True, True),
        (4,  8192, 64,  False, True, True, True, True),
        (4,  8192, 128, False, True, True, True, True),
        (8,  8192, 16,  False, True, True, True, True),
        (8,  8192, 32,  False, True, True, True, True),
        (8,  8192, 64,  False, True, True, True, True),
        (8,  8192, 128, False, True, True, True, True),
        (16, 8192, 16,  False, True, True, True, True),
        (16, 8192, 32,  False, True, True, True, True),
        (16, 8192, 64,  False, True, True, True, True),
        # --- T=8192, minimal features, sweep B × H ---
        (1,  8192, 64,  False, False, False, False, True),
        (4,  8192, 64,  False, False, False, False, True),
        (8,  8192, 64,  False, False, False, False, True),
        # --- Original configs for reference ---
        (4, 4096, 64, False, True,  True,  True,  True),
        (8, 4096, 64, False, True,  True,  True,  True),
    ]

    print(f"{'Config':<40} {'Ours (ms)':>10} {'FLA (ms)':>10} {'Speedup':>10}")
    print("-" * 74)

    speedups = []
    for (B, T, H, use_g, use_gk, use_h0, store_ht, save_vnew) in configs:
        label = f"B={B} T={T} H={H}"
        flags = []
        if use_g: flags.append("g")
        if use_gk: flags.append("gk")
        if use_h0: flags.append("h0")
        if store_ht: flags.append("ht")
        if save_vnew: flags.append("vn")
        if flags:
            label += f" [{','.join(flags)}]"

        our_ms, fla_ms = run_benchmark(B, T, H, K, V, BT, use_g, use_gk, use_h0, store_ht, save_vnew)
        sp = fla_ms / our_ms
        speedups.append(sp)
        print(f"{label:<40} {our_ms:>10.3f} {fla_ms:>10.3f} {sp:>9.2f}x")

    import math
    geo_mean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    print("-" * 74)
    print(f"{'Geometric mean speedup':<40} {'':>10} {'':>10} {geo_mean:>9.2f}x")


if __name__ == "__main__":
    main()
