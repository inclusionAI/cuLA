#!/usr/bin/env python3
# Copyright (c) 2025 ANTGROUP. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Test suite for ChunkGlaFwdO kernel.
Tests correctness against:
  1. Pure PyTorch reference
  2. Triton chunk_gla_fwd_o_gk (from flash-linear-attention)
"""

import os
import sys
import argparse
import pytest
import torch
import torch.nn.functional as F

# Add flash-linear-attention to path for Triton reference
sys.path.insert(0, "/ossfs/workspace/flash-linear-attention")

# Import directly from the module file to avoid flashla package __init__ (requires cudac)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "fwd_o", os.path.join(os.path.dirname(__file__), "..", "flashla", "fwd_o.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ChunkGlaFwdO = _mod.ChunkGlaFwdO
reference_chunk_gla_fwd_o = _mod.reference_chunk_gla_fwd_o
chunk_gla_fwd_o = _mod.chunk_gla_fwd_o
build_chunk_indices = _mod.build_chunk_indices
build_chunk_offsets = _mod.build_chunk_offsets


def triton_chunk_gla_fwd_o(q, v, g, h, A, scale, chunk_size=64):
    """Call the Triton reference kernel."""
    from fla.ops.gla.chunk import chunk_gla_fwd_o_gk
    return chunk_gla_fwd_o_gk(
        q=q, v=v, g=g, A=A, h=h,
        scale=scale, chunk_size=chunk_size,
        use_exp2=True,
    )


def assert_close(name, ref, out, atol=0.005):
    diff = (ref.float() - out.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    passed = max_diff < atol
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    if not passed:
        # Find worst location
        flat_idx = diff.view(-1).argmax().item()
        print(f"    Worst at flat_idx={flat_idx}")
        print(f"    ref={ref.view(-1)[flat_idx].item():.6f}, out={out.view(-1)[flat_idx].item():.6f}")
    return passed


# ===================== Reference (PyTorch) Tests =====================

@pytest.mark.parametrize("B,T,H,K,V", [
    (1, 64, 1, 128, 128),
    (2, 128, 2, 128, 128),
    (2, 256, 4, 128, 128),
    (4, 1024, 4, 128, 128),
    (1, 192, 2, 128, 128),   # Non-aligned T (not multiple of 64)
])
def test_reference_vs_triton(B, T, H, K, V):
    """Verify PyTorch reference matches Triton kernel."""
    BT = 64
    NT = (T + BT - 1) // BT
    scale = K ** -0.5
    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    g = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    h = torch.randn(B * NT, H, K, V, dtype=dtype, device=device) * 0.01
    A = torch.randn(B, T, H, BT, dtype=dtype, device=device) * 0.1

    o_ref = reference_chunk_gla_fwd_o(q, v, g, h, A, scale, BT)
    o_triton = triton_chunk_gla_fwd_o(q, v, g, h, A, scale, BT)

    assert assert_close(f"ref_vs_triton B={B} T={T} H={H}", o_ref, o_triton, atol=0.01)


# ===================== CuTe DSL vs Reference Tests =====================

@pytest.mark.parametrize("B,T,H,K,V", [
    (1, 64, 1, 128, 128),
    (2, 128, 2, 128, 128),
    (2, 256, 4, 128, 128),
    (4, 1024, 4, 128, 128),
    (1, 192, 2, 128, 128),
])
def test_cute_dsl_vs_reference(B, T, H, K, V):
    """Verify CuTe DSL kernel matches PyTorch reference (non-varlen)."""
    BT = 64
    NT = (T + BT - 1) // BT
    scale = K ** -0.5
    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    g = torch.randn(B, T, H, K, dtype=torch.float32, device=device) * 0.1
    h = torch.randn(B * NT, H, K, V, dtype=dtype, device=device) * 0.01
    A = torch.randn(B, T, H, BT, dtype=dtype, device=device) * 0.1

    o_ref = reference_chunk_gla_fwd_o(q, v, g, h, A, scale, BT)

    o_cute = torch.zeros_like(q[:, :, :, :V])
    chunk_gla_fwd_o(q, v, g, h, o_cute, A, scale, chunk_size=BT,
                    is_varlen=False, persistent=True)
    torch.cuda.synchronize()

    assert assert_close(f"CuTe_vs_ref B={B} T={T} H={H}", o_ref, o_cute, atol=0.02)


@pytest.mark.parametrize("seq_lens", [
    [64],
    [128, 64],
    [256, 128, 64],
    [192, 192],
])
def test_cute_dsl_varlen_vs_reference(seq_lens):
    """Verify CuTe DSL varlen kernel matches non-varlen reference per sequence."""
    H, K, V = 2, 128, 128
    BT = 64
    scale = K ** -0.5
    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(42)
    T_total = sum(seq_lens)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    cu_seqlens = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(seq_lens_t, 0)

    chunk_indices = build_chunk_indices(seq_lens_t, BT=BT, device=device)

    q = torch.randn(T_total, H, K, dtype=dtype, device=device)
    v = torch.randn(T_total, H, V, dtype=dtype, device=device)
    g = torch.randn(T_total, H, K, dtype=torch.float32, device=device) * 0.1

    total_nt = sum((s + BT - 1) // BT for s in seq_lens)
    h = torch.randn(total_nt, H, K, V, dtype=dtype, device=device) * 0.01
    A = torch.randn(T_total, H, BT, dtype=dtype, device=device) * 0.1

    o_cute = torch.zeros(T_total, H, V, dtype=dtype, device=device)
    chunk_gla_fwd_o(q, v, g, h, o_cute, A, scale, chunk_size=BT,
                    cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
                    is_varlen=True, persistent=True)
    torch.cuda.synchronize()

    # Compare per-sequence against non-varlen reference
    nt_offset = 0
    for i, slen in enumerate(seq_lens):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        nt_i = (slen + BT - 1) // BT
        qi = q[start:end].unsqueeze(0)
        vi = v[start:end].unsqueeze(0)
        gi = g[start:end].unsqueeze(0)
        hi = h[nt_offset:nt_offset + nt_i]
        Ai = A[start:end].unsqueeze(0)
        oi_ref = reference_chunk_gla_fwd_o(qi, vi, gi, hi, Ai, scale, BT)
        oi_cute = o_cute[start:end].unsqueeze(0)
        assert assert_close(f"varlen_seq{i} slen={slen}", oi_ref, oi_cute, atol=0.02)
        nt_offset += nt_i


# ===================== Manual Correctness Test =====================

def run_correctness_tests():
    """Run correctness tests manually (not pytest)."""
    device = "cuda"
    dtype = torch.bfloat16
    BT = 64

    configs = [
        # (B, T, H, K, V, description)
        (1, 64, 1, 128, 128, "minimal single chunk"),
        (2, 128, 2, 128, 128, "2 chunks"),
        (2, 256, 4, 128, 128, "4 heads, 4 chunks"),
        (4, 1024, 4, 128, 128, "standard KDA config"),
        (1, 192, 2, 128, 128, "non-aligned T"),
        (3, 512, 8, 128, 128, "larger H"),
    ]

    all_passed = True
    for B, T, H, K, V, desc in configs:
        print(f"\n--- Test: {desc} (B={B}, T={T}, H={H}, K={K}, V={V}) ---")
        NT = (T + BT - 1) // BT
        scale = K ** -0.5

        torch.manual_seed(42)
        q = torch.randn(B, T, H, K, dtype=dtype, device=device)
        v = torch.randn(B, T, H, V, dtype=dtype, device=device)
        g = torch.randn(B, T, H, K, dtype=torch.float32, device=device) * 0.1
        h = torch.randn(B * NT, H, K, V, dtype=dtype, device=device) * 0.01
        A = torch.randn(B, T, H, BT, dtype=dtype, device=device) * 0.1

        # PyTorch reference
        o_ref = reference_chunk_gla_fwd_o(q, v, g, h, A, scale, BT)

        # CuTe DSL kernel
        try:
            o_cute = torch.zeros_like(q[:, :, :, :V])
            chunk_gla_fwd_o(q, v, g, h, o_cute, A, scale, chunk_size=BT,
                            is_varlen=False, persistent=True)
            torch.cuda.synchronize()
            passed = assert_close("CuTe DSL vs Ref", o_ref, o_cute, atol=0.02)
            all_passed = all_passed and passed
        except Exception as e:
            print(f"  CuTe DSL failed: {e}")
            all_passed = False

        # Triton reference
        try:
            o_triton = triton_chunk_gla_fwd_o(q, v, g, h, A, scale, BT)
            passed = assert_close("Triton vs Ref", o_ref, o_triton, atol=0.02)
            all_passed = all_passed and passed
        except Exception as e:
            print(f"  Triton failed: {e}")
            all_passed = False

    print(f"\n{'='*50}")
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


def run_benchmark(B=4, T=4096, H=4, K=128, V=128, num_iters=100):
    """Benchmark CuTe DSL and Triton kernels."""
    BT = 64
    NT = (T + BT - 1) // BT
    scale = K ** -0.5
    device = "cuda"
    dtype = torch.bfloat16

    print(f"\n=== Benchmark: B={B}, T={T}, H={H}, K={K}, V={V} ===")

    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    g = torch.randn(B, T, H, K, dtype=torch.float32, device=device) * 0.1
    h = torch.randn(B * NT, H, K, V, dtype=dtype, device=device) * 0.01
    A = torch.randn(B, T, H, BT, dtype=dtype, device=device) * 0.1
    o = torch.zeros(B, T, H, V, dtype=dtype, device=device)

    # --- CuTe DSL ---
    # Warmup (first call triggers compilation)
    chunk_gla_fwd_o(q, v, g, h, o, A, scale, chunk_size=BT,
                    is_varlen=False, persistent=True)
    torch.cuda.synchronize()

    for _ in range(5):
        chunk_gla_fwd_o(q, v, g, h, o, A, scale, chunk_size=BT,
                        is_varlen=False, persistent=True)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        chunk_gla_fwd_o(q, v, g, h, o, A, scale, chunk_size=BT,
                        is_varlen=False, persistent=True)
    end.record()
    torch.cuda.synchronize()
    cute_ms = start.elapsed_time(end) / num_iters

    # --- Triton ---
    for _ in range(10):
        triton_chunk_gla_fwd_o(q, v, g, h, A, scale, BT)
    torch.cuda.synchronize()

    start.record()
    for _ in range(num_iters):
        triton_chunk_gla_fwd_o(q, v, g, h, A, scale, BT)
    end.record()
    torch.cuda.synchronize()
    triton_ms = start.elapsed_time(end) / num_iters

    total_bytes = (q.nelement() + v.nelement() + g.nelement() + h.nelement() +
                   A.nelement()) * 2  # bf16 = 2 bytes
    total_bytes += v.nelement() * 2  # output

    print(f"  CuTe DSL: {cute_ms:.3f} ms, {total_bytes / (cute_ms * 1e-3) / 1e9:.1f} GB/s")
    print(f"  Triton:   {triton_ms:.3f} ms, {total_bytes / (triton_ms * 1e-3) / 1e9:.1f} GB/s")
    print(f"  Speedup:  {triton_ms / cute_ms:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="correctness",
                        choices=["correctness", "benchmark", "both"])
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--T", type=int, default=4096)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--V", type=int, default=128)
    args = parser.parse_args()

    if args.test in ("correctness", "both"):
        run_correctness_tests()

    if args.test in ("benchmark", "both"):
        run_benchmark(B=args.B, T=args.T, H=args.H, K=args.K, V=args.V)
