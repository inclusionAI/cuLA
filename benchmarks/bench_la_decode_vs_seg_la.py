#!/usr/bin/env python3
"""
Benchmark: la_decode (CuTe DSL) vs seg_la_d_kernel (Triton)
Lightning Attention single-token decode (T=1) performance comparison.

Both compute the same recurrence:
    state_new = exp(-decay) * state_old + k ⊗ v
    o = (q * scale) @ state_new
    (write back state_new)

Two comparison modes for fairness:
  1. "kernel-only": Direct kernel calls with pre-allocated buffers on both sides.
     seg_la side: pre-allocate tmp, call seg_la_d_kernel directly (no torch.empty).
     cute side:   pre-create compiled + stream handle, call compiled() directly.
     Pure GPU kernel performance comparison with minimal host overhead.

  2. "wrapper": Full seg_la_fwd() vs linear_attention_decode() call paths.
     seg_la_fwd internally allocates torch.empty(tmp) per call + Python setup.
     linear_attention_decode does dict cache lookup + CUstream creation per call.
     Represents real-world end-to-end call overhead.

Usage:
    python benchmarks/bench_la_decode_vs_seg_la.py
    python benchmarks/bench_la_decode_vs_seg_la.py --heads 64
    python benchmarks/bench_la_decode_vs_seg_la.py --batch-sizes 1 8 64 256
"""

import os
import sys
import argparse

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

import torch
import cuda.bindings.driver as cuda_drv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cula.lightning.la_decode import linear_attention_decode, _get_compiled_kernel
from cula.seg_la import seg_la_fwd, seg_la_d_kernel, SegLaMeta


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
def run_config(B, H, D, layer_idx, num_layers):
    device = "cuda"
    dtype = torch.bfloat16
    scale = D ** -0.5

    # Per-head decay (Lightning Attention formula)
    decay_scales = (8 / H * (1 - layer_idx / num_layers)) * torch.arange(
        H, device=device, dtype=torch.float32
    )

    # ── Random inputs ──────────────────────────────────────────────────────
    torch.manual_seed(42)
    q_flat = torch.randn(B, H, D, device=device, dtype=dtype)
    k_flat = torch.randn(B, H, D, device=device, dtype=dtype)
    v_flat = torch.randn(B, H, D, device=device, dtype=dtype)
    state_init_4d = torch.randn(B, H, D, D, device=device, dtype=torch.float32) * 0.01

    # ── seg_la meta ────────────────────────────────────────────────────────
    s_offsets = torch.arange(B, device=device, dtype=torch.int32)
    q_offsets = torch.arange(B + 1, device=device, dtype=torch.int32)
    q_lengths = torch.ones(B, device=device, dtype=torch.int32)
    s_scales = torch.ones(B, device=device, dtype=torch.int32)
    meta = SegLaMeta(
        batch_size=B, max_q_length=1,
        q_offsets=q_offsets, s_offsets=s_offsets,
        q_lengths=q_lengths, s_scales=s_scales,
    )
    cute_s_offsets = torch.arange(B, device=device, dtype=torch.int32)

    # ── seg_la correctness reference ───────────────────────────────────────
    state_seg = state_init_4d.clone().reshape(B, H * D * D).contiguous()
    with torch.no_grad():
        o_seg = seg_la_fwd(q_flat, k_flat, v_flat, state_seg, decay_scales, meta, softmax_scale=scale)

    # ── la_decode correctness ──────────────────────────────────────────────
    state_cute = (
        state_init_4d.clone()
        .permute(0, 1, 3, 2)
        .reshape(B * H, D, D)
        .contiguous()
    )
    out_cute = torch.zeros(B, H, D, device=device, dtype=dtype)
    with torch.no_grad():
        linear_attention_decode(
            q_flat, k_flat, v_flat, state_cute, out_cute,
            softmax_scale=scale,
            stride_q=0, stride_k=0, stride_v=0, stride_s=0, stride_o=0,
            s_offsets=cute_s_offsets, decay_scales=decay_scales,
            HEAD_DIM=D, K_SPLIT_DIM=D, V_SPLIT_DIM=D,
        )

    # ── Correctness check ─────────────────────────────────────────────────
    o_seg_f, o_cute_f = o_seg.float(), out_cute.float()
    rmse = torch.sqrt(torch.mean((o_cute_f - o_seg_f) ** 2)).item()
    max_ref = torch.abs(o_seg_f).max().item()
    rel_maxdiff = torch.abs(o_cute_f - o_seg_f).max().item() / (max_ref + 1e-8)

    state_seg_after = state_seg.reshape(B, H, D, D)
    state_cute_after = state_cute.reshape(B, H, D, D).permute(0, 1, 3, 2).contiguous()
    state_rmse = torch.sqrt(
        torch.mean((state_cute_after.float() - state_seg_after.float()) ** 2)
    ).item()

    # ==================================================================
    # Mode 1: KERNEL-ONLY (pre-allocated everything, minimal host overhead)
    # ==================================================================

    # seg_la: pre-allocate tmp buffer (same as seg_la_fwd does internally)
    if B <= 128:
        K_SPLIT_DIM, V_SPLIT_DIM = 128, 32
        num_warps, num_stages = 2, 2
    else:
        K_SPLIT_DIM, V_SPLIT_DIM = 128, 64
        num_warps, num_stages = 2, 3
    k_dim_block = D // K_SPLIT_DIM  # 1 for D=128
    v_dim_block = D // V_SPLIT_DIM
    seg_tmp = torch.empty((k_dim_block, B, H, D), device=device, dtype=dtype)
    seg_state_k = state_init_4d.clone().reshape(B, H * D * D).contiguous()
    grid_seg = (B, H, k_dim_block * v_dim_block)

    def kernel_seg_la():
        seg_la_d_kernel[grid_seg](
            q_flat, k_flat, v_flat, seg_state_k, seg_tmp, scale,
            q_flat.stride(0), k_flat.stride(0), v_flat.stride(0),
            seg_state_k.stride(0), seg_tmp.stride(0),
            s_offsets, decay_scales,
            HEAD_DIM=D, K_SPLIT_DIM=K_SPLIT_DIM, V_SPLIT_DIM=V_SPLIT_DIM,
            num_warps=num_warps, num_stages=num_stages,
        )

    # cute: pre-create compiled kernel + stream handle
    cute_state_k = (
        state_init_4d.clone()
        .permute(0, 1, 3, 2)
        .reshape(B * H, D, D)
        .contiguous()
    )
    out_cute_k = torch.empty(B, H, D, device=device, dtype=dtype)
    # Trigger compilation (already done above), get cached compiled object
    cache = _get_compiled_kernel(B, 1, H, D, D, scale)
    compiled_cute = cache["compiled"]
    stream_handle = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)

    def kernel_cute():
        compiled_cute(cute_state_k, decay_scales, q_flat, k_flat, v_flat,
                      out_cute_k, cute_s_offsets, stream_handle)

    with torch.no_grad():
        kernel_seg_ms = benchmark_fn(kernel_seg_la)
        kernel_cute_ms = benchmark_fn(kernel_cute)

    # ==================================================================
    # Mode 2: WRAPPER (full call path as used in production)
    # ==================================================================
    wrap_seg_state = state_init_4d.clone().reshape(B, H * D * D).contiguous()
    wrap_cute_state = (
        state_init_4d.clone()
        .permute(0, 1, 3, 2)
        .reshape(B * H, D, D)
        .contiguous()
    )
    wrap_cute_out = torch.empty(B, H, D, device=device, dtype=dtype)

    def wrapper_seg_la():
        seg_la_fwd(q_flat, k_flat, v_flat, wrap_seg_state, decay_scales, meta, softmax_scale=scale)

    def wrapper_cute():
        linear_attention_decode(
            q_flat, k_flat, v_flat, wrap_cute_state, wrap_cute_out,
            softmax_scale=scale,
            stride_q=0, stride_k=0, stride_v=0, stride_s=0, stride_o=0,
            s_offsets=cute_s_offsets, decay_scales=decay_scales,
            HEAD_DIM=D, K_SPLIT_DIM=D, V_SPLIT_DIM=D,
        )

    with torch.no_grad():
        wrap_seg_ms = benchmark_fn(wrapper_seg_la)
        wrap_cute_ms = benchmark_fn(wrapper_cute)

    return {
        "B": B,
        "kernel_seg_ms": kernel_seg_ms,
        "kernel_cute_ms": kernel_cute_ms,
        "kernel_speedup": kernel_seg_ms / kernel_cute_ms,
        "wrap_seg_ms": wrap_seg_ms,
        "wrap_cute_ms": wrap_cute_ms,
        "wrap_speedup": wrap_seg_ms / wrap_cute_ms,
        "rmse": rmse,
        "rel_maxdiff": rel_maxdiff,
        "state_rmse": state_rmse,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark la_decode (CuTe) vs seg_la (Triton) for decode"
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    )
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--layer-idx", type=int, default=12)
    parser.add_argument("--num-layers", type=int, default=24)
    args = parser.parse_args()

    H, D = args.heads, args.head_dim

    print(f"Lightning Attention Decode Benchmark")
    print(f"  la_decode (CuTe DSL) vs seg_la_d_kernel (Triton)")
    print(f"  H={H}, D={D}, layer={args.layer_idx}/{args.num_layers}")
    print(f"  dtype=bf16, state=fp32, T=1")

    # ── Kernel-only comparison ──────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  Mode 1: KERNEL-ONLY (pre-allocated buffers, direct kernel dispatch)")
    print(f"{'='*100}")
    print(
        f"{'B':>5} | {'seg_la (ms)':>11} | {'cute (ms)':>10} | "
        f"{'speedup':>8} | {'RMSE':>10} | {'Rel MaxDiff':>12} | {'State RMSE':>12}"
    )
    print("─" * 92)

    results = []
    for B in args.batch_sizes:
        r = run_config(B, H, D, args.layer_idx, args.num_layers)
        results.append(r)
        print(
            f"{r['B']:>5} | {r['kernel_seg_ms']:>11.4f} | {r['kernel_cute_ms']:>10.4f} | "
            f"{r['kernel_speedup']:>7.2f}x | {r['rmse']:>10.6f} | "
            f"{r['rel_maxdiff']:>12.6f} | {r['state_rmse']:>12.8f}"
        )

    # ── Wrapper comparison ──────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  Mode 2: WRAPPER (seg_la_fwd vs linear_attention_decode, full call path)")
    print(f"{'='*100}")
    print(
        f"{'B':>5} | {'seg_la (ms)':>11} | {'cute (ms)':>10} | {'speedup':>8}"
    )
    print("─" * 50)

    for r in results:
        print(
            f"{r['B']:>5} | {r['wrap_seg_ms']:>11.4f} | {r['wrap_cute_ms']:>10.4f} | "
            f"{r['wrap_speedup']:>7.2f}x"
        )

    print()
    print("Notes:")
    print("  Kernel-only: both sides use pre-allocated output buffers, direct kernel dispatch.")
    print("               seg_la: call seg_la_d_kernel with pre-allocated tmp.")
    print("               cute:   call compiled() with pre-created stream handle.")
    print("  Wrapper:     seg_la_fwd does torch.empty(tmp) + Python setup per call.")
    print("               linear_attention_decode does dict lookup + CUstream() per call.")
    print("  Both modes:  state updated in-place, same decay_scales and softmax_scale.")


if __name__ == "__main__":
    main()
