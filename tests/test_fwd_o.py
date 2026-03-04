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
        g = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
        h = torch.randn(B * NT, H, K, V, dtype=dtype, device=device) * 0.01
        A = torch.randn(B, T, H, BT, dtype=dtype, device=device) * 0.1

        # PyTorch reference
        o_ref = reference_chunk_gla_fwd_o(q, v, g, h, A, scale, BT)

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
    """Benchmark Triton kernel."""
    BT = 64
    NT = (T + BT - 1) // BT
    scale = K ** -0.5
    device = "cuda"
    dtype = torch.bfloat16

    print(f"\n=== Benchmark: B={B}, T={T}, H={H}, K={K}, V={V} ===")

    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    g = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    h = torch.randn(B * NT, H, K, V, dtype=dtype, device=device) * 0.01
    A = torch.randn(B, T, H, BT, dtype=dtype, device=device) * 0.1

    # Warmup
    for _ in range(10):
        triton_chunk_gla_fwd_o(q, v, g, h, A, scale, BT)
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iters):
        triton_chunk_gla_fwd_o(q, v, g, h, A, scale, BT)
    end.record()
    torch.cuda.synchronize()

    triton_ms = start.elapsed_time(end) / num_iters
    total_bytes = (q.nelement() + v.nelement() + g.nelement() + h.nelement() +
                   A.nelement()) * 2  # bf16 = 2 bytes
    total_bytes += v.nelement() * 2  # output
    bw_gb = total_bytes / (triton_ms * 1e-3) / 1e9

    print(f"  Triton: {triton_ms:.3f} ms, {bw_gb:.1f} GB/s")


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
