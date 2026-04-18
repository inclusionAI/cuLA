#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Benchmark: la_decode (CuTe DSL) vs fla fused_recurrent_simple_gla
Lightning Attention single-token decode (T=1) performance comparison.

Both compute:
    h_new = exp(g_gamma) * h_old + k ⊗ v
    o = (q * scale) @ h_new
    (write back h_new)

where g_gamma is the per-head log decay from Lightning Attention.

Two comparison modes for fairness:
  1. "kernel-only": Direct kernel calls with pre-allocated buffers on both sides.
     fla side:  pre-allocate o/ht/o_sum, call fused_recurrent_fwd_kernel directly,
                then torch.sum(o, dim=0, out=o_sum). Includes reduction (intrinsic
                to fla's NK=2 K-block tiling), but no per-call allocation.
     cute side: pre-create compiled + stream_handle, call compiled() directly.

  2. "wrapper": Full fused_recurrent_fwd() vs linear_attention_decode() call paths.
     fla allocates o [NK,B,1,H,V] fp32 + ht [B,H,K,V] fp32 + o.sum(0) per call.
     la_decode does dict lookup + CUstream() per call.

Usage:
    python benchmarks/bench_la_decode_vs_fla.py
    python benchmarks/bench_la_decode_vs_fla.py --heads 64 --head-dim 128
    python benchmarks/bench_la_decode_vs_fla.py --batch-sizes 1 8 64 256
"""

import argparse
import os
import sys

os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))  # Enable fast ops in FLA for fair comparison

import cuda.bindings.driver as cuda_drv
import torch
import triton

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fla.ops.common.fused_recurrent import fused_recurrent_fwd, fused_recurrent_fwd_kernel

from cula.ops.la_decode import _get_compiled_kernel, linear_attention_decode
from cula.utils import USE_FAST_MATH


# ─────────────────────────────────────────────────────────────────────────────
# Timing utility
# ─────────────────────────────────────────────────────────────────────────────
def benchmark_fn(fn, warmup=30, rep=200):
    """Benchmark using CUDA events. Returns IQR-mean time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

    for i in range(rep):
        starts[i].record()
        fn()
        ends[i].record()

    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    n = len(times)
    iqr = times[n // 4 : 3 * n // 4]
    return sum(iqr) / len(iqr)


# ─────────────────────────────────────────────────────────────────────────────
# Core benchmark for one configuration
# ─────────────────────────────────────────────────────────────────────────────
def run_config(B, H, K, V, layer_idx, num_layers):
    device = "cuda"
    dtype = torch.bfloat16
    scale = K**-0.5

    # Per-head log decay (Lightning Attention formula)
    g_gamma = -(8 / H * (1 - layer_idx / num_layers)) * torch.arange(H, device=device, dtype=torch.float32)
    decay_scales = -g_gamma  # la_decode convention

    # ── Random inputs ──────────────────────────────────────────────────────
    torch.manual_seed(42)
    q_4d = torch.randn(B, 1, H, K, device=device, dtype=dtype)
    k_4d = torch.randn(B, 1, H, K, device=device, dtype=dtype)
    v_4d = torch.randn(B, 1, H, V, device=device, dtype=dtype)
    state_init = torch.randn(B, H, K, V, device=device, dtype=torch.float32) * 0.01

    # ── fla reference output ───────────────────────────────────────────────
    state_fla = state_init.clone()
    with torch.no_grad():
        o_fla_fp32, ht_fla = fused_recurrent_fwd(
            q_4d,
            k_4d,
            v_4d,
            g_gamma=g_gamma,
            scale=scale,
            initial_state=state_fla,
            output_final_state=True,
        )
    o_fla = o_fla_fp32.to(dtype)

    # ── la_decode output ───────────────────────────────────────────────────
    state_cute = state_init.clone().permute(0, 1, 3, 2).reshape(B * H, V, K).contiguous()
    q_3d = q_4d.squeeze(1)
    k_3d = k_4d.squeeze(1)
    v_3d = v_4d.squeeze(1)
    out_cute = torch.zeros(B, H, V, device=device, dtype=dtype)
    s_offsets = torch.arange(B, device=device, dtype=torch.int32)

    with torch.no_grad():
        linear_attention_decode(
            q_3d,
            k_3d,
            v_3d,
            state_cute,
            out_cute,
            softmax_scale=scale,
            stride_q=0,
            stride_k=0,
            stride_v=0,
            stride_s=0,
            stride_o=0,
            s_offsets=s_offsets,
            decay_scales=decay_scales,
            HEAD_DIM=K,
            K_SPLIT_DIM=K,
            V_SPLIT_DIM=V,
        )

    # ── Correctness ────────────────────────────────────────────────────────
    o_fla_cmp = o_fla.squeeze(1).float()
    o_cute_cmp = out_cute.float()
    rmse = torch.sqrt(torch.mean((o_cute_cmp - o_fla_cmp) ** 2)).item()
    max_ref = torch.abs(o_fla_cmp).max().item()
    rel_maxdiff = torch.abs(o_cute_cmp - o_fla_cmp).max().item() / (max_ref + 1e-8)

    state_cute_back = state_cute.reshape(B, H, V, K).permute(0, 1, 3, 2).contiguous()
    state_rmse = torch.sqrt(torch.mean((state_cute_back - ht_fla.float()) ** 2)).item()

    # ==================================================================
    # Mode 1: KERNEL-ONLY (pre-allocated everything, minimal host overhead)
    # ==================================================================

    # fla kernel: pre-allocate o, ht, o_sum buffers
    BK_fla = min(triton.next_power_of_2(K), 64)
    BV_fla = min(triton.next_power_of_2(V), 64)
    NK = triton.cdiv(K, BK_fla)
    NV = triton.cdiv(V, BV_fla)
    fla_o_buf = torch.empty(NK, B, 1, H, V, device=device, dtype=torch.float32)
    fla_ht_buf = torch.empty(B, H, K, V, device=device, dtype=torch.float32)
    fla_o_sum = torch.empty(B, 1, H, V, device=device, dtype=torch.float32)
    fla_state_k = state_init.clone()
    grid_fla = (NV, NK, B * H)

    def kernel_fla():
        fused_recurrent_fwd_kernel[grid_fla](
            q=q_4d,
            k=k_4d,
            v=v_4d,
            g=None,
            g_gamma=g_gamma,
            gk=None,
            gv=None,
            o=fla_o_buf,
            h0=fla_state_k,
            ht=fla_ht_buf,
            cu_seqlens=None,
            scale=scale,
            B=B,
            T=1,
            H=H,
            K=K,
            V=V,
            BK=BK_fla,
            BV=BV_fla,
            USE_G=False,
            USE_G_GAMMA=True,
            USE_GK=False,
            USE_GV=False,
            REVERSE=False,
        )
        torch.sum(fla_o_buf, dim=0, out=fla_o_sum)

    # cute kernel: pre-create compiled + stream handle
    cute_state_k = state_init.clone().permute(0, 1, 3, 2).reshape(B * H, V, K).contiguous()
    out_cute_k = torch.empty(B, H, V, device=device, dtype=dtype)
    cache = _get_compiled_kernel(B, 1, H, K, V, scale, USE_FAST_MATH)
    compiled_cute = cache["compiled"]
    stream_handle = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)

    def kernel_cute():
        compiled_cute(cute_state_k, decay_scales, q_3d, k_3d, v_3d, out_cute_k, s_offsets, stream_handle)

    with torch.no_grad():
        kernel_fla_ms = benchmark_fn(kernel_fla)
        kernel_cute_ms = benchmark_fn(kernel_cute)

    # ==================================================================
    # Mode 2: WRAPPER (full call path as used in production)
    # ==================================================================
    wrap_fla_state = state_init.clone()
    wrap_cute_state = state_init.clone().permute(0, 1, 3, 2).reshape(B * H, V, K).contiguous()
    wrap_cute_out = torch.empty(B, H, V, device=device, dtype=dtype)

    def wrapper_fla():
        fused_recurrent_fwd(
            q_4d,
            k_4d,
            v_4d,
            g_gamma=g_gamma,
            scale=scale,
            initial_state=wrap_fla_state,
            output_final_state=True,
        )

    def wrapper_cute():
        linear_attention_decode(
            q_3d,
            k_3d,
            v_3d,
            wrap_cute_state,
            wrap_cute_out,
            softmax_scale=scale,
            stride_q=0,
            stride_k=0,
            stride_v=0,
            stride_s=0,
            stride_o=0,
            s_offsets=s_offsets,
            decay_scales=decay_scales,
            HEAD_DIM=K,
            K_SPLIT_DIM=K,
            V_SPLIT_DIM=V,
        )

    with torch.no_grad():
        wrap_fla_ms = benchmark_fn(wrapper_fla)
        wrap_cute_ms = benchmark_fn(wrapper_cute)

    return {
        "B": B,
        "kernel_fla_ms": kernel_fla_ms,
        "kernel_cute_ms": kernel_cute_ms,
        "kernel_speedup": kernel_fla_ms / kernel_cute_ms,
        "wrap_fla_ms": wrap_fla_ms,
        "wrap_cute_ms": wrap_cute_ms,
        "wrap_speedup": wrap_fla_ms / wrap_cute_ms,
        "rmse": rmse,
        "rel_maxdiff": rel_maxdiff,
        "state_rmse": state_rmse,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Benchmark la_decode vs fla fused_recurrent for decode")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    )
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--layer-idx", type=int, default=12)
    parser.add_argument("--num-layers", type=int, default=24)
    args = parser.parse_args()

    H, K, V = args.heads, args.head_dim, args.head_dim

    print("Lightning Attention Decode Benchmark")
    print("  la_decode (CuTe DSL) vs fla fused_recurrent_fwd (Triton)")
    print(f"  H={H}, K={K}, V={V}, layer={args.layer_idx}/{args.num_layers}")
    print("  dtype=bf16, state=fp32, T=1")

    # ── Kernel-only comparison ──────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print("  Mode 1: KERNEL-ONLY (pre-allocated buffers, direct kernel dispatch)")
    print("  fla: kernel + sum(0) with pre-allocated out=; cute: compiled() with pre-created stream")
    print(f"{'=' * 100}")
    print(
        f"{'B':>5} | {'fla (ms)':>10} | {'cute (ms)':>10} | "
        f"{'speedup':>8} | {'RMSE':>10} | {'Rel MaxDiff':>12} | {'State RMSE':>12}"
    )
    print("─" * 90)

    results = []
    for B in args.batch_sizes:
        r = run_config(B, H, K, V, args.layer_idx, args.num_layers)
        results.append(r)
        print(
            f"{r['B']:>5} | {r['kernel_fla_ms']:>10.4f} | {r['kernel_cute_ms']:>10.4f} | "
            f"{r['kernel_speedup']:>7.2f}x | {r['rmse']:>10.6f} | "
            f"{r['rel_maxdiff']:>12.6f} | {r['state_rmse']:>12.8f}"
        )

    # ── Wrapper comparison ──────────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print("  Mode 2: WRAPPER (fused_recurrent_fwd vs linear_attention_decode, full call path)")
    print("  fla: alloc o[NK,B,1,H,V]+ht[B,H,K,V] + kernel + sum(0); cute: cache lookup + CUstream + kernel")
    print(f"{'=' * 100}")
    print(f"{'B':>5} | {'fla (ms)':>10} | {'cute (ms)':>10} | {'speedup':>8}")
    print("─" * 50)

    for r in results:
        print(f"{r['B']:>5} | {r['wrap_fla_ms']:>10.4f} | {r['wrap_cute_ms']:>10.4f} | {r['wrap_speedup']:>7.2f}x")

    print()
    print("Notes:")
    print("  Kernel-only: both sides use pre-allocated buffers, direct kernel dispatch.")
    print("               fla: fused_recurrent_fwd_kernel + torch.sum(o,0,out=) [NK=2 K-blocks].")
    print("               cute: compiled() with pre-created stream handle.")
    print("  Wrapper:     fla: fused_recurrent_fwd allocs o+ht per call + kernel + o.sum(0).")
    print("               cute: linear_attention_decode does cache lookup + CUstream() per call.")
    print("  Both modes:  same g_gamma decay, same softmax_scale, state write-back included.")

    return results


if __name__ == "__main__":
    main()
