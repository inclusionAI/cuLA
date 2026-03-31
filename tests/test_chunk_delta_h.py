#!/usr/bin/env python3
# Copyright (c) 2025 ANTGROUP. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test suite for ChunkDeltaRuleFwdH CuTe DSL kernel.
Tests correctness against FLA's Triton reference (chunk_gated_delta_rule_fwd_h).
"""

import argparse
import os
import sys

import pytest
import torch

# ─── FLA reference ───
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h as fla_fwd_h

# ─── CuTe DSL kernel (via importlib to avoid cula __init__ requiring cudac) ───
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "chunk_delta_h", os.path.join(os.path.dirname(__file__), "..", "cula", "ops", "chunk_delta_h.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
chunk_gated_delta_rule_fwd_h = _mod.chunk_gated_delta_rule_fwd_h


BT = 64
device = "cuda"


def run_fla_ref(k, w, u, g=None, gk=None, initial_state=None, output_final_state=False, save_new_value=True, cu_seqlens=None):
    """Call FLA's Triton kernel as reference."""
    return fla_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=BT,
        save_new_value=save_new_value,
        cu_seqlens=cu_seqlens,
    )


def run_cute_dsl(k, w, u, g=None, gk=None, initial_state=None, output_final_state=False, save_new_value=True, cu_seqlens=None):
    """Call CuTe DSL kernel wrapper (FLA-compatible API) and return (h_out, v_new, ht)."""
    return chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=BT,
        save_new_value=save_new_value,
        cu_seqlens=cu_seqlens,
    )


# ===================== Pytest parametrized tests =====================


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("H", [1, 4])
@pytest.mark.parametrize("T", [64, 128, 256])
@pytest.mark.parametrize("K", [128])
@pytest.mark.parametrize("V", [128])
@pytest.mark.parametrize("use_gk", [False, True])
@pytest.mark.parametrize("use_h0", [False, True])
def test_h_against_fla(B, H, T, K, V, use_gk, use_h0):
    """Test CuTe DSL h_out matches FLA's Triton kernel."""
    torch.manual_seed(42)

    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device) * 0.1
    w = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device) * 0.1
    u = torch.randn(B, T, H, V, dtype=torch.bfloat16, device=device) * 0.1

    h0 = None
    if use_h0:
        h0 = torch.randn(B, H, K, V, dtype=torch.float32, device=device) * 0.01

    gk_val = None
    if use_gk:
        gk_val = torch.randn(B, T, H, K, dtype=torch.float32, device=device) * 0.1
        gk_val = -torch.abs(gk_val).cumsum(dim=1)

    # FLA reference
    ref_h, ref_vnew, ref_ht = run_fla_ref(
        k,
        w,
        u,
        gk=gk_val,
        initial_state=h0,
        output_final_state=use_h0,
        save_new_value=True,
    )

    # CuTe DSL kernel
    our_h, our_vnew, our_ht = run_cute_dsl(
        k,
        w,
        u,
        gk=gk_val,
        initial_state=h0,
        output_final_state=use_h0,
        save_new_value=True,
    )

    # Compare h_out: FLA returns [B, NT, H, K, V], ours is [B, NT, H, K, V]
    torch.testing.assert_close(
        our_h.float(),
        ref_h.float(),
        atol=1e-2,
        rtol=1e-2,
        msg=f"h_out mismatch B={B} H={H} T={T} gk={use_gk} h0={use_h0}",
    )

    # Compare v_new
    if ref_vnew is not None and our_vnew is not None:
        torch.testing.assert_close(
            our_vnew.float(),
            ref_vnew.float(),
            atol=1e-2,
            rtol=1e-2,
            msg=f"v_new mismatch B={B} H={H} T={T} gk={use_gk} h0={use_h0}",
        )

    # Compare ht (final state)
    if use_h0 and ref_ht is not None and our_ht is not None:
        torch.testing.assert_close(
            our_ht.float(),
            ref_ht.float(),
            atol=1e-2,
            rtol=1e-2,
            msg=f"ht mismatch B={B} H={H} T={T} gk={use_gk} h0={use_h0}",
        )


@pytest.mark.parametrize(
    "B,T,H,K,V",
    [
        (1, 64, 1, 128, 128),
        (2, 128, 4, 128, 128),
        (4, 512, 4, 128, 128),
    ],
)
def test_vnew_no_gating(B, T, H, K, V):
    """Test v_new output without gating matches FLA."""
    torch.manual_seed(42)
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device) * 0.1
    w = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device) * 0.1
    u = torch.randn(B, T, H, V, dtype=torch.bfloat16, device=device) * 0.1

    ref_h, ref_vnew, _ = run_fla_ref(k, w, u, save_new_value=True)
    our_h, our_vnew, _ = run_cute_dsl(k, w, u, save_new_value=True)

    torch.testing.assert_close(
        our_vnew.float(),
        ref_vnew.float(),
        atol=1e-2,
        rtol=1e-2,
        msg=f"v_new no-gating mismatch B={B} T={T} H={H}",
    )


# ===================== Varlen pytest tests =====================


def _make_varlen_inputs(seq_lens, H, K, V, use_gk=False, use_h0=False, seed=42):
    """Create varlen-packed tensors in FLA convention: [1, T_total, H, D]."""
    T_total = sum(seq_lens)
    num_seqs = len(seq_lens)
    cu_seqlens_list = [0]
    for sl in seq_lens:
        cu_seqlens_list.append(cu_seqlens_list[-1] + sl)

    torch.manual_seed(seed)
    k = torch.randn(1, T_total, H, K, dtype=torch.bfloat16, device=device) * 0.1
    w = torch.randn(1, T_total, H, K, dtype=torch.bfloat16, device=device) * 0.1
    u = torch.randn(1, T_total, H, V, dtype=torch.bfloat16, device=device) * 0.1

    gk_val = None
    if use_gk:
        # Per-sequence cumsum (reset at sequence boundaries)
        gk_val = torch.zeros(1, T_total, H, K, dtype=torch.float32, device=device)
        for i in range(num_seqs):
            bos, eos = cu_seqlens_list[i], cu_seqlens_list[i + 1]
            seg = torch.randn(1, eos - bos, H, K, dtype=torch.float32, device=device) * 0.1
            gk_val[:, bos:eos] = -torch.abs(seg).cumsum(dim=1)

    h0 = None
    if use_h0:
        h0 = torch.randn(num_seqs, H, K, V, dtype=torch.float32, device=device) * 0.01

    cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=device)
    return k, w, u, gk_val, h0, cu_seqlens


@pytest.mark.parametrize(
    "seq_lens",
    [
        [128, 128],
        [50, 192, 100],
        [33, 128, 200, 95],
    ],
)
@pytest.mark.parametrize("H", [1, 4])
@pytest.mark.parametrize("use_gk", [False, True])
@pytest.mark.parametrize("use_h0", [False, True])
def test_varlen_against_fla(seq_lens, H, use_gk, use_h0):
    """Test varlen CuTe DSL h_out/v_new/ht matches FLA's Triton kernel."""
    K, V = 128, 128
    k, w, u, gk_val, h0, cu_seqlens = _make_varlen_inputs(
        seq_lens,
        H,
        K,
        V,
        use_gk=use_gk,
        use_h0=use_h0,
    )

    ref_h, ref_vnew, ref_ht = run_fla_ref(
        k,
        w,
        u,
        gk=gk_val,
        initial_state=h0,
        output_final_state=use_h0,
        save_new_value=True,
        cu_seqlens=cu_seqlens,
    )
    our_h, our_vnew, our_ht = run_cute_dsl(
        k,
        w,
        u,
        gk=gk_val,
        initial_state=h0,
        output_final_state=use_h0,
        save_new_value=True,
        cu_seqlens=cu_seqlens,
    )

    torch.testing.assert_close(
        our_h.float(),
        ref_h.float(),
        atol=1e-2,
        rtol=1e-2,
        msg=f"varlen h_out mismatch seqs={seq_lens} H={H} gk={use_gk} h0={use_h0}",
    )
    if ref_vnew is not None and our_vnew is not None:
        torch.testing.assert_close(
            our_vnew.float(),
            ref_vnew.float(),
            atol=1e-2,
            rtol=1e-2,
            msg=f"varlen v_new mismatch seqs={seq_lens} H={H} gk={use_gk} h0={use_h0}",
        )
    if use_h0 and ref_ht is not None and our_ht is not None:
        torch.testing.assert_close(
            our_ht.float(),
            ref_ht.float(),
            atol=1e-2,
            rtol=1e-2,
            msg=f"varlen ht mismatch seqs={seq_lens} H={H} gk={use_gk} h0={use_h0}",
        )


def test_varlen_vs_nonvarlen():
    """Test that varlen with a single sequence matches non-varlen output."""
    H, K, V = 2, 128, 128
    T = 256

    torch.manual_seed(42)
    k = torch.randn(1, T, H, K, dtype=torch.bfloat16, device=device) * 0.1
    w = torch.randn(1, T, H, K, dtype=torch.bfloat16, device=device) * 0.1
    u = torch.randn(1, T, H, V, dtype=torch.bfloat16, device=device) * 0.1

    # Non-varlen
    h_nv, vnew_nv, _ = run_cute_dsl(k, w, u, save_new_value=True)

    # Varlen with single sequence (should be identical)
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)
    h_vl, vnew_vl, _ = run_cute_dsl(k, w, u, save_new_value=True, cu_seqlens=cu_seqlens)

    torch.testing.assert_close(
        h_nv.float(),
        h_vl.float(),
        atol=1e-6,
        rtol=1e-6,
        msg="varlen vs non-varlen h_out mismatch for single sequence",
    )
    torch.testing.assert_close(
        vnew_nv.float(),
        vnew_vl.float(),
        atol=1e-6,
        rtol=1e-6,
        msg="varlen vs non-varlen v_new mismatch for single sequence",
    )


# ===================== Manual test runner =====================


def run_correctness_tests():
    """Run correctness tests manually (not pytest)."""
    configs = [
        (1, 64, 1, 128, 128, False, False, "minimal"),
        (1, 128, 1, 128, 128, True, False, "gk only"),
        (1, 128, 1, 128, 128, False, True, "h0 only"),
        (1, 128, 1, 128, 128, True, True, "gk + h0"),
        (2, 256, 4, 128, 128, True, True, "multi-batch multi-head"),
        (4, 512, 4, 128, 128, False, False, "larger no gating"),
        (4, 1024, 8, 128, 128, True, True, "large with gk + h0"),
    ]

    all_passed = True
    for B, T, H, K, V, use_gk, use_h0, desc in configs:
        print(f"\n--- {desc} (B={B} T={T} H={H} K={K} V={V} gk={use_gk} h0={use_h0}) ---")
        torch.manual_seed(42)

        k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device) * 0.1
        w = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device) * 0.1
        u = torch.randn(B, T, H, V, dtype=torch.bfloat16, device=device) * 0.1

        h0_val = torch.randn(B, H, K, V, dtype=torch.float32, device=device) * 0.01 if use_h0 else None
        gk_val = None
        if use_gk:
            gk_val = torch.randn(B, T, H, K, dtype=torch.float32, device=device) * 0.1
            gk_val = -torch.abs(gk_val).cumsum(dim=1)

        try:
            ref_h, ref_vnew, ref_ht = run_fla_ref(
                k,
                w,
                u,
                gk=gk_val,
                initial_state=h0_val,
                output_final_state=use_h0,
                save_new_value=True,
            )
            our_h, our_vnew, our_ht = run_cute_dsl(
                k,
                w,
                u,
                gk=gk_val,
                initial_state=h0_val,
                output_final_state=use_h0,
                save_new_value=True,
            )

            h_diff = (our_h.float() - ref_h.float()).abs().max().item()
            vnew_diff = (our_vnew.float() - ref_vnew.float()).abs().max().item() if ref_vnew is not None else 0
            ht_diff = 0
            if use_h0 and ref_ht is not None and our_ht is not None:
                ht_diff = (our_ht.float() - ref_ht.float()).abs().max().item()

            passed = h_diff < 0.02 and vnew_diff < 0.02 and ht_diff < 0.02
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] h_diff={h_diff:.6f} vnew_diff={vnew_diff:.6f} ht_diff={ht_diff:.6f}")
            all_passed = all_passed and passed
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback

            traceback.print_exc()
            all_passed = False

    print(f"\n{'=' * 50}")
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


def run_benchmark(B=4, T=4096, H=64, K=128, V=128, num_iters=20):
    """Benchmark CuTe DSL vs FLA Triton."""
    print(f"\n=== Benchmark: B={B}, T={T}, H={H}, K={K}, V={V} ===")

    torch.manual_seed(42)
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device) * 0.1
    w = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device) * 0.1
    u = torch.randn(B, T, H, V, dtype=torch.bfloat16, device=device) * 0.1
    gk = torch.randn(B, T, H, K, dtype=torch.float32, device=device) * 0.1
    gk = -torch.abs(gk).cumsum(dim=1)
    h0 = torch.randn(B, H, K, V, dtype=torch.float32, device=device) * 0.01

    # --- CuTe DSL ---
    # Warmup (triggers compilation)
    chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        gk=gk,
        initial_state=h0,
        output_final_state=True,
        chunk_size=BT,
        save_new_value=True,
    )
    torch.cuda.synchronize()

    for _ in range(5):
        chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            gk=gk,
            initial_state=h0,
            output_final_state=True,
            chunk_size=BT,
            save_new_value=True,
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            gk=gk,
            initial_state=h0,
            output_final_state=True,
            chunk_size=BT,
            save_new_value=True,
        )
    end.record()
    torch.cuda.synchronize()
    cute_ms = start.elapsed_time(end) / num_iters

    # --- FLA Triton ---
    for _ in range(5):
        fla_fwd_h(k=k, w=w, u=u, gk=gk, initial_state=h0, output_final_state=True, chunk_size=BT, save_new_value=True)
    torch.cuda.synchronize()

    start.record()
    for _ in range(num_iters):
        fla_fwd_h(k=k, w=w, u=u, gk=gk, initial_state=h0, output_final_state=True, chunk_size=BT, save_new_value=True)
    end.record()
    torch.cuda.synchronize()
    fla_ms = start.elapsed_time(end) / num_iters

    print(f"  CuTe DSL: {cute_ms:.3f} ms")
    print(f"  FLA:      {fla_ms:.3f} ms")
    print(f"  Speedup:  {fla_ms / cute_ms:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="correctness", choices=["correctness", "benchmark", "both"])
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--T", type=int, default=4096)
    parser.add_argument("--H", type=int, default=64)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--V", type=int, default=128)
    args = parser.parse_args()

    if args.test in ("correctness", "both"):
        run_correctness_tests()

    if args.test in ("benchmark", "both"):
        run_benchmark(B=args.B, T=args.T, H=args.H, K=args.K, V=args.V)
