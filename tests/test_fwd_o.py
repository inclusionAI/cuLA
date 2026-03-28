#!/usr/bin/env python3
# Copyright (c) 2025 ANTGROUP. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test suite for ChunkGlaFwdO kernel (non-varlen + varlen).

Tests correctness against:
  1. Pure PyTorch reference (per-sequence gold standard)
  2. Triton chunk_gla_fwd_o_gk (from flash-linear-attention)
  3. CuTe DSL non-varlen kernel (consistency: pack vs pad-batch)

Varlen test dimensions:
  - Various seq count / length combinations (1 seq to 40+)
  - Aligned (multiple of BT=64) and non-aligned seq lengths
  - Single-chunk and multi-chunk sequences
  - Mixed short + long sequences
  - Tail chunks (seq length not divisible by BT)
  - Edge: all-same lengths, single token per seq, length == BT
  - Multiple head counts (1, 4, 32, 64)
  - Zero / near-zero gates, large gates
  - Zero h (pure intra-chunk), zero A (pure inter-chunk)

K=128, V=128, BT=64, dtype=bf16, use_exp2=True throughout.
"""

import argparse
import os
import random
import sys

import pytest
import torch

# Import directly from the module file to avoid cula package __init__ (requires cudac)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "fwd_o", os.path.join(os.path.dirname(__file__), "..", "cula", "ops", "fwd_o.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ChunkGlaFwdO = _mod.ChunkGlaFwdO
reference_chunk_gla_fwd_o = _mod.reference_chunk_gla_fwd_o
chunk_gla_fwd_o = _mod.chunk_gla_fwd_o
build_chunk_indices = _mod.build_chunk_indices
build_chunk_offsets = _mod.build_chunk_offsets

from fla.ops.gla.chunk import chunk_gla_fwd_o_gk as triton_chunk_gla_fwd_o_gk  # noqa: E402

# ── Constants ──
K, V, BT = 128, 128, 64
DTYPE = torch.bfloat16
DEVICE = "cuda"


# ═══════════════════════════════════════════════════════════════════════
# Helpers — non-varlen
# ═══════════════════════════════════════════════════════════════════════


def triton_chunk_gla_fwd_o(q, v, g, h, A, scale, chunk_size=64):
    """Call the Triton reference kernel (non-varlen). FLA expects h as [B*NT, H, K, V]."""
    return triton_chunk_gla_fwd_o_gk(
        q=q,
        v=v,
        g=g,
        A=A,
        h=h.flatten(0, 1),
        scale=scale,
        chunk_size=chunk_size,
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
        flat_idx = diff.view(-1).argmax().item()
        print(f"    Worst at flat_idx={flat_idx}")
        print(f"    ref={ref.view(-1)[flat_idx].item():.6f}, out={out.view(-1)[flat_idx].item():.6f}")
    return passed


# ═══════════════════════════════════════════════════════════════════════
# Helpers — varlen
# ═══════════════════════════════════════════════════════════════════════


def make_cu_seqlens(seq_lens, device=DEVICE):
    """Build cu_seqlens [N+1] int32 from list of lengths."""
    cu = [0]
    for s in seq_lens:
        cu.append(cu[-1] + s)
    return torch.tensor(cu, dtype=torch.int32, device=device)


def gen_varlen_data(seq_lens, H, seed=42, g_scale=0.1, h_scale=0.01, zero_h=False, zero_A=False, zero_g=False):
    """Generate all tensors for a varlen fwd_o test case.

    Returns dict with q, v, g, h, A, o, cu_seqlens, chunk_indices, scale,
    seq_lens, T_total, total_nt, H.
    All token-indexed tensors are 4D: [1, T_total, H, *] (B=1 for varlen).
    """
    torch.manual_seed(seed)
    T_total = sum(seq_lens)
    total_nt = sum((s + BT - 1) // BT for s in seq_lens)
    scale = K**-0.5

    q = torch.randn(1, T_total, H, K, dtype=DTYPE, device=DEVICE)
    v = torch.randn(1, T_total, H, V, dtype=DTYPE, device=DEVICE)

    if zero_g:
        g = torch.zeros(1, T_total, H, K, dtype=torch.float32, device=DEVICE)
    else:
        g = torch.randn(1, T_total, H, K, dtype=torch.float32, device=DEVICE) * g_scale

    if zero_h:
        h = torch.zeros(1, total_nt, H, K, V, dtype=DTYPE, device=DEVICE)
    else:
        h = torch.randn(1, total_nt, H, K, V, dtype=DTYPE, device=DEVICE) * h_scale

    if zero_A:
        A = torch.zeros(1, T_total, H, BT, dtype=DTYPE, device=DEVICE)
    else:
        A = torch.randn(1, T_total, H, BT, dtype=DTYPE, device=DEVICE) * 0.1

    o = torch.zeros(1, T_total, H, V, dtype=DTYPE, device=DEVICE)
    cu_seqlens = make_cu_seqlens(seq_lens)
    chunk_indices = build_chunk_indices(seq_lens, BT=BT, device=DEVICE)

    return dict(
        q=q,
        v=v,
        g=g,
        h=h,
        A=A,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        seq_lens=seq_lens,
        T_total=T_total,
        total_nt=total_nt,
        H=H,
    )


def run_cute_varlen(d):
    """Run CuTe DSL varlen kernel, return output [T_total, H, V]."""
    o = d["o"].zero_()
    chunk_gla_fwd_o(
        q=d["q"],
        v=d["v"],
        g=d["g"],
        h=d["h"],
        o=o,
        A=d["A"],
        scale=d["scale"],
        chunk_size=BT,
        cu_seqlens=d["cu_seqlens"],
        chunk_indices=d["chunk_indices"],
        is_varlen=True,
        persistent=True,
    )
    torch.cuda.synchronize()
    return o.squeeze(0)  # [1, T_total, H, V] → [T_total, H, V]


def run_pytorch_ref_per_seq(d):
    """Run PyTorch reference per-sequence, return [T_total, H, V]."""
    seq_lens = d["seq_lens"]
    o_ref = torch.zeros(d["T_total"], d["H"], V, dtype=DTYPE, device=DEVICE)
    nt_offset = 0
    cu = d["cu_seqlens"]
    for i, slen in enumerate(seq_lens):
        start = cu[i].item()
        end = cu[i + 1].item()
        nt_i = (slen + BT - 1) // BT
        qi = d["q"][:, start:end]
        vi = d["v"][:, start:end]
        gi = d["g"][:, start:end]
        hi = d["h"][:, nt_offset : nt_offset + nt_i]
        Ai = d["A"][:, start:end]
        oi = reference_chunk_gla_fwd_o(qi, vi, gi, hi, Ai, d["scale"], BT)
        o_ref[start:end] = oi.squeeze(0)
        nt_offset += nt_i
    return o_ref


def run_fla_varlen(d):
    """Run FLA Triton varlen kernel, return [T_total, H, V]."""
    cu_fla = d["cu_seqlens"].to(torch.int64)
    h_flat = d["h"].flatten(0, 1)  # [1, NT, H, K, V] -> [NT, H, K, V]
    o_fla = triton_chunk_gla_fwd_o_gk(
        q=d["q"],
        v=d["v"],
        g=d["g"],
        A=d["A"],
        h=h_flat,
        scale=d["scale"],
        cu_seqlens=cu_fla,
        chunk_size=BT,
        use_exp2=True,
    )
    return o_fla.squeeze(0)


def run_cute_non_varlen_per_seq(d):
    """Run CuTe DSL non-varlen kernel per-sequence, return [T_total, H, V]."""
    seq_lens = d["seq_lens"]
    o_ref = torch.zeros(d["T_total"], d["H"], V, dtype=DTYPE, device=DEVICE)
    nt_offset = 0
    cu = d["cu_seqlens"]
    for i, slen in enumerate(seq_lens):
        start = cu[i].item()
        end = cu[i + 1].item()
        nt_i = (slen + BT - 1) // BT
        qi = d["q"][:, start:end]
        vi = d["v"][:, start:end]
        gi = d["g"][:, start:end]
        hi = d["h"][:, nt_offset : nt_offset + nt_i]
        Ai = d["A"][:, start:end]
        oi = torch.zeros(1, slen, d["H"], V, dtype=DTYPE, device=DEVICE)
        chunk_gla_fwd_o(
            q=qi,
            v=vi,
            g=gi,
            h=hi,
            o=oi,
            A=Ai,
            scale=d["scale"],
            chunk_size=BT,
            is_varlen=False,
            persistent=True,
        )
        torch.cuda.synchronize()
        o_ref[start:end] = oi.squeeze(0)
        nt_offset += nt_i
    return o_ref


def check_accuracy(name, ref, out, atol=0.02):
    """Check accuracy with detailed metrics. Asserts max_diff < atol."""
    ref_f = ref.float()
    out_f = out.float()
    diff = (ref_f - out_f).abs()
    max_diff = diff.max().item()
    denom = ref_f.abs().max().item()
    rel_max = max_diff / denom if denom > 0 else 0.0
    rmse = diff.flatten().square().mean().sqrt().item()
    rms_ref = ref_f.flatten().square().mean().sqrt().item()
    rmse_ratio = rmse / (rms_ref + 1e-8)

    passed = max_diff < atol
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: max_diff={max_diff:.6f} rel_max={rel_max:.6f} rmse_ratio={rmse_ratio:.6f}")
    if not passed:
        flat_idx = diff.view(-1).argmax().item()
        print(
            f"    Worst at flat_idx={flat_idx}: "
            f"ref={ref_f.view(-1)[flat_idx].item():.6f}, "
            f"out={out_f.view(-1)[flat_idx].item():.6f}"
        )
    assert passed, f"{name}: max_diff={max_diff:.6f} exceeds atol={atol}"
    return max_diff, rel_max, rmse_ratio


def check_per_seq(name, ref, out, d, atol=0.02):
    """Check accuracy per-sequence, to pinpoint which seq fails."""
    cu = d["cu_seqlens"]
    for i, slen in enumerate(d["seq_lens"]):
        start = cu[i].item()
        end = cu[i + 1].item()
        check_accuracy(f"{name} seq[{i}] len={slen}", ref[start:end], out[start:end], atol=atol)


# ═══════════════════════════════════════════════════════════════════════
# Non-varlen tests: Reference (PyTorch) vs Triton
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "B,T,H,K,V",
    [
        (1, 64, 1, 128, 128),
        (2, 128, 2, 128, 128),
        (2, 256, 4, 128, 128),
        (4, 1024, 4, 128, 128),
        (1, 192, 2, 128, 128),  # Non-aligned T (not multiple of 64)
    ],
)
def test_reference_vs_triton(B, T, H, K, V):
    """Verify PyTorch reference matches Triton kernel."""
    NT = (T + BT - 1) // BT
    scale = K**-0.5

    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, dtype=DTYPE, device=DEVICE)
    v = torch.randn(B, T, H, V, dtype=DTYPE, device=DEVICE)
    g = torch.randn(B, T, H, K, dtype=DTYPE, device=DEVICE) * 0.1
    h = torch.randn(B, NT, H, K, V, dtype=DTYPE, device=DEVICE) * 0.01
    A = torch.randn(B, T, H, BT, dtype=DTYPE, device=DEVICE) * 0.1

    o_ref = reference_chunk_gla_fwd_o(q, v, g, h, A, scale, BT)
    o_triton = triton_chunk_gla_fwd_o(q, v, g, h, A, scale, BT)

    assert assert_close(f"ref_vs_triton B={B} T={T} H={H}", o_ref, o_triton, atol=0.02)


# ═══════════════════════════════════════════════════════════════════════
# Non-varlen tests: CuTe DSL vs Reference
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "B,T,H,K,V",
    [
        (1, 64, 1, 128, 128),
        (2, 128, 2, 128, 128),
        (2, 256, 4, 128, 128),
        (4, 1024, 4, 128, 128),
        (1, 192, 2, 128, 128),
    ],
)
def test_cute_dsl_vs_reference(B, T, H, K, V):
    """Verify CuTe DSL kernel matches PyTorch reference (non-varlen)."""
    NT = (T + BT - 1) // BT
    scale = K**-0.5

    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, dtype=DTYPE, device=DEVICE)
    v = torch.randn(B, T, H, V, dtype=DTYPE, device=DEVICE)
    g = torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE) * 0.1
    h = torch.randn(B, NT, H, K, V, dtype=DTYPE, device=DEVICE) * 0.01
    A = torch.randn(B, T, H, BT, dtype=DTYPE, device=DEVICE) * 0.1

    o_ref = reference_chunk_gla_fwd_o(q, v, g, h, A, scale, BT)

    o_cute = torch.zeros_like(q[:, :, :, :V])
    chunk_gla_fwd_o(q, v, g, h, o_cute, A, scale, chunk_size=BT, is_varlen=False, persistent=True)
    torch.cuda.synchronize()

    assert assert_close(f"CuTe_vs_ref B={B} T={T} H={H}", o_ref, o_cute, atol=0.02)


# ═══════════════════════════════════════════════════════════════════════
# Varlen tests: CuTe varlen vs PyTorch reference (per-sequence gold standard)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "seq_lens,H",
    [
        # Single sequence
        ([64], 4),
        ([128], 4),
        ([192], 4),  # non-aligned (3 chunks, last is 64)
        ([100], 4),  # non-aligned (2 chunks, last is 36)
        ([1], 4),  # single token
        ([63], 4),  # < BT
        ([65], 4),  # BT + 1
        # Two sequences
        ([64, 64], 4),
        ([128, 64], 4),
        ([64, 128], 4),  # short before long
        ([100, 200], 4),  # both non-aligned
        ([256, 1], 4),  # long + single token
        # Multiple sequences
        ([64, 64, 64], 4),
        ([128, 64, 192], 4),
        ([256, 128, 64, 64], 4),
        ([100, 150, 200, 250, 300], 4),  # 5 seqs all non-aligned
        # Same-length sequences
        ([64, 64, 64, 64], 4),
        ([128, 128, 128], 4),
        ([100, 100, 100], 4),  # all non-aligned, same
        # Many short sequences
        ([64] * 10, 4),
        ([2] * 20, 4),  # very short seqs
        ([BT] * 8, 4),  # all exactly BT
        # Mixed extreme lengths
        ([1, 1024], 4),
        ([1024, 1], 4),
        ([1, 64, 1, 128, 1], 4),
        # Larger scale
        ([256, 512, 128, 64, 256], 64),
        ([1024, 2048], 64),
    ],
    ids=lambda x: str(x),
)
def test_cute_varlen_vs_torch_baseline(seq_lens, H):
    """CuTe DSL varlen must match per-sequence PyTorch baseline."""
    d = gen_varlen_data(seq_lens, H)
    o_ref = run_pytorch_ref_per_seq(d)
    o_cute = run_cute_varlen(d)
    check_accuracy(f"CuTe vs torch seqs={seq_lens} H={H}", o_ref, o_cute)
    check_per_seq("CuTe_per_seq", o_ref, o_cute, d)


# ═══════════════════════════════════════════════════════════════════════
# Varlen tests: FLA Triton varlen vs PyTorch baseline
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "seq_lens,H",
    [
        # Single sequence
        ([64], 4),
        ([128], 4),
        ([192], 4),
        ([100], 4),
        ([1], 4),
        ([63], 4),
        ([65], 4),
        # Two sequences
        ([64, 64], 4),
        ([128, 64], 4),
        ([100, 200], 4),
        ([256, 1], 4),
        # Multiple sequences
        ([64, 64, 64], 4),
        ([128, 64, 192], 4),
        ([256, 128, 64, 64], 4),
        ([100, 150, 200, 250, 300], 4),
        # Same-length / many short
        ([64, 64, 64, 64], 4),
        ([128, 128, 128], 4),
        ([2] * 20, 4),
        # Mixed extreme
        ([1, 1024], 4),
        ([1024, 1], 4),
        ([1, 64, 1, 128, 1], 4),
        # Larger scale
        ([256, 512, 128, 64, 256], 64),
        ([1024, 2048], 64),
    ],
    ids=lambda x: str(x),
)
def test_fla_varlen_vs_torch_baseline(seq_lens, H):
    """FLA Triton varlen must match per-sequence PyTorch baseline."""
    d = gen_varlen_data(seq_lens, H)
    o_ref = run_pytorch_ref_per_seq(d)
    o_fla = run_fla_varlen(d)
    check_accuracy(f"FLA vs torch seqs={seq_lens} H={H}", o_ref, o_fla)
    check_per_seq("FLA_per_seq", o_ref, o_fla, d)


# ═══════════════════════════════════════════════════════════════════════
# Varlen tests: CuTe varlen vs CuTe non-varlen (consistency)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "seq_lens,H",
    [
        ([64], 4),
        ([128, 64], 4),
        ([256, 128, 64], 4),
        ([100, 200], 4),
        ([64, 64, 64, 64], 4),
        ([512, 256], 32),
    ],
    ids=lambda x: str(x),
)
def test_cute_varlen_vs_non_varlen(seq_lens, H):
    """CuTe varlen output must match per-sequence non-varlen output."""
    d = gen_varlen_data(seq_lens, H)
    o_varlen = run_cute_varlen(d)
    o_non_varlen = run_cute_non_varlen_per_seq(d)
    check_accuracy(f"varlen vs non-varlen seqs={seq_lens} H={H}", o_non_varlen, o_varlen)
    check_per_seq("consistency", o_non_varlen, o_varlen, d)


# ═══════════════════════════════════════════════════════════════════════
# Varlen tests: Component isolation (pure inter-chunk, pure intra-chunk)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "seq_lens",
    [
        [64, 128],
        [100, 200, 64],
        [256, 512],
    ],
)
def test_pure_inter_chunk(seq_lens):
    """With A=0, output is purely from inter-chunk (q*g @ h). Must match torch baseline."""
    H = 4
    d = gen_varlen_data(seq_lens, H, zero_A=True)
    o_ref = run_pytorch_ref_per_seq(d)
    o_cute = run_cute_varlen(d)
    o_fla = run_fla_varlen(d)
    check_accuracy(f"pure_inter (A=0) FLA vs torch seqs={seq_lens}", o_ref, o_fla)
    check_accuracy(f"pure_inter (A=0) CuTe vs torch seqs={seq_lens}", o_ref, o_cute)


@pytest.mark.parametrize(
    "seq_lens",
    [
        [64, 128],
        [100, 200, 64],
        [256, 512],
    ],
)
def test_pure_intra_chunk(seq_lens):
    """With h=0, output is purely from intra-chunk (A @ v). Must match torch baseline."""
    H = 4
    d = gen_varlen_data(seq_lens, H, zero_h=True)
    o_ref = run_pytorch_ref_per_seq(d)
    o_cute = run_cute_varlen(d)
    o_fla = run_fla_varlen(d)
    check_accuracy(f"pure_intra (h=0) FLA vs torch seqs={seq_lens}", o_ref, o_fla)
    check_accuracy(f"pure_intra (h=0) CuTe vs torch seqs={seq_lens}", o_ref, o_cute)


@pytest.mark.parametrize(
    "seq_lens",
    [
        [64, 128],
        [100, 200, 64],
    ],
)
def test_zero_gate(seq_lens):
    """With g=0, exp2(g)=1 so qg=q. Must match torch baseline."""
    H = 4
    d = gen_varlen_data(seq_lens, H, zero_g=True)
    o_ref = run_pytorch_ref_per_seq(d)
    o_cute = run_cute_varlen(d)
    o_fla = run_fla_varlen(d)
    check_accuracy(f"zero_gate (g=0) FLA vs torch seqs={seq_lens}", o_ref, o_fla)
    check_accuracy(f"zero_gate (g=0) CuTe vs torch seqs={seq_lens}", o_ref, o_cute)


# ═══════════════════════════════════════════════════════════════════════
# Varlen tests: Various head counts
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("H", [1, 2, 4, 8, 16, 32, 64])
def test_head_counts(H):
    """Test varlen across different head counts, both FLA and CuTe vs torch baseline."""
    seq_lens = [128, 64, 192]
    d = gen_varlen_data(seq_lens, H)
    o_ref = run_pytorch_ref_per_seq(d)
    o_cute = run_cute_varlen(d)
    o_fla = run_fla_varlen(d)
    check_accuracy(f"H={H} FLA vs torch", o_ref, o_fla)
    check_accuracy(f"H={H} CuTe vs torch", o_ref, o_cute)


# ═══════════════════════════════════════════════════════════════════════
# Varlen tests: Determinism
# ═══════════════════════════════════════════════════════════════════════


def test_determinism():
    """Multiple calls with same input must produce identical output."""
    seq_lens = [256, 128, 100]
    H = 4
    d = gen_varlen_data(seq_lens, H)

    o1 = run_cute_varlen(d).clone()
    o2 = run_cute_varlen(d).clone()
    o3 = run_cute_varlen(d).clone()

    diff_12 = (o1.float() - o2.float()).abs().max().item()
    diff_13 = (o1.float() - o3.float()).abs().max().item()
    print(f"  Determinism: diff(run1,run2)={diff_12:.8f} diff(run1,run3)={diff_13:.8f}")
    assert diff_12 == 0.0, f"Non-deterministic: diff_12={diff_12}"
    assert diff_13 == 0.0, f"Non-deterministic: diff_13={diff_13}"


# ═══════════════════════════════════════════════════════════════════════
# Varlen tests: Random configs (stress test)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("seed", list(range(10)))
def test_random_varlen_configs(seed):
    """Random seq count (2-15), random lengths (1-500), random H in {4,16,32,64}.
    Both FLA and CuTe compared against torch baseline."""
    rng = random.Random(seed + 1000)
    n_seqs = rng.randint(2, 15)
    seq_lens = [rng.randint(1, 500) for _ in range(n_seqs)]
    H = rng.choice([4, 16, 32, 64])
    print(f"  seed={seed}: n_seqs={n_seqs} H={H} lens={seq_lens}")

    d = gen_varlen_data(seq_lens, H, seed=seed)
    o_ref = run_pytorch_ref_per_seq(d)
    o_cute = run_cute_varlen(d)
    o_fla = run_fla_varlen(d)
    check_accuracy(f"random seed={seed} FLA vs torch", o_ref, o_fla, atol=0.02)
    check_accuracy(f"random seed={seed} CuTe vs torch", o_ref, o_cute, atol=0.02)


# ═══════════════════════════════════════════════════════════════════════
# Varlen tests: Gate magnitude (numerical stress)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("g_scale", [0.01, 0.1, 0.5, 1.0])
def test_gate_magnitude(g_scale):
    """Various gate magnitudes to test numerical stability."""
    seq_lens = [128, 64, 100]
    H = 4
    d = gen_varlen_data(seq_lens, H, g_scale=g_scale)
    o_ref = run_pytorch_ref_per_seq(d)
    o_cute = run_cute_varlen(d)
    o_fla = run_fla_varlen(d)
    # Larger gates → larger values → relax atol proportionally
    atol = 0.02 * max(1.0, 2**g_scale)
    check_accuracy(f"g_scale={g_scale} FLA vs torch", o_ref, o_fla, atol=atol)
    check_accuracy(f"g_scale={g_scale} CuTe vs torch", o_ref, o_cute, atol=atol)


# ═══════════════════════════════════════════════════════════════════════
# Varlen tests: Persistent vs non-persistent consistency
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "seq_lens,H",
    [
        ([128, 64], 4),
        ([256, 128, 64], 4),
        ([100, 200, 300], 32),
    ],
)
def test_persistent_non_varlen_consistency(seq_lens, H):
    """persistent=True non-varlen per-seq must agree with varlen persistent."""
    d = gen_varlen_data(seq_lens, H)
    o_varlen = run_cute_varlen(d)
    o_non_varlen = run_cute_non_varlen_per_seq(d)
    check_accuracy(f"persistent consistency seqs={seq_lens} H={H}", o_non_varlen, o_varlen, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# Varlen tests: Sequence boundary isolation
# ═══════════════════════════════════════════════════════════════════════


def test_sequence_boundary_isolation():
    """Changing data in one sequence must not affect others."""
    seq_lens = [128, 192, 64]
    H = 4
    d = gen_varlen_data(seq_lens, H, seed=77)
    o1 = run_cute_varlen(d).clone()

    # Perturb seq[1] data
    cu = d["cu_seqlens"]
    s1_start = cu[1].item()
    s1_end = cu[2].item()
    d["q"][:, s1_start:s1_end] = torch.randn_like(d["q"][:, s1_start:s1_end]) * 10
    d["v"][:, s1_start:s1_end] = torch.randn_like(d["v"][:, s1_start:s1_end]) * 10

    o2 = run_cute_varlen(d).clone()

    # Check seq[0] unchanged
    s0_end = cu[1].item()
    diff_s0 = (o1[:s0_end].float() - o2[:s0_end].float()).abs().max().item()
    print(f"  seq[0] diff after perturbing seq[1]: {diff_s0:.8f}")
    assert diff_s0 == 0.0, f"Cross-contamination detected in seq[0]: diff={diff_s0}"

    # Check seq[2] unchanged
    s2_start = cu[2].item()
    diff_s2 = (o1[s2_start:].float() - o2[s2_start:].float()).abs().max().item()
    print(f"  seq[2] diff after perturbing seq[1]: {diff_s2:.8f}")
    assert diff_s2 == 0.0, f"Cross-contamination detected in seq[2]: diff={diff_s2}"


# ═══════════════════════════════════════════════════════════════════════
# Varlen tests: 3-way cross-validation
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "seq_lens,H",
    [
        ([64, 128], 4),
        ([100, 200, 64], 4),
        ([256, 512, 128], 32),
        ([64, 64, 64, 64, 64], 64),
    ],
)
def test_three_way_cross_validation(seq_lens, H):
    """All three implementations must agree with torch baseline."""
    d = gen_varlen_data(seq_lens, H)
    o_ref = run_pytorch_ref_per_seq(d)
    o_fla = run_fla_varlen(d)
    o_cute = run_cute_varlen(d)

    check_accuracy(f"torch vs FLA seqs={seq_lens}", o_ref, o_fla, atol=0.02)
    check_accuracy(f"torch vs CuTe seqs={seq_lens}", o_ref, o_cute, atol=0.02)
    check_accuracy(f"FLA vs CuTe seqs={seq_lens}", o_fla, o_cute, atol=0.02)


# ═══════════════════════════════════════════════════════════════════════
# Manual correctness test runner
# ═══════════════════════════════════════════════════════════════════════


def run_correctness_tests():
    """Run correctness tests manually (not pytest)."""
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
    for B, T, H, _K, _V, desc in configs:
        print(f"\n--- Test: {desc} (B={B}, T={T}, H={H}, K={K}, V={V}) ---")
        NT = (T + BT - 1) // BT
        scale = K**-0.5

        torch.manual_seed(42)
        q = torch.randn(B, T, H, K, dtype=DTYPE, device=DEVICE)
        v = torch.randn(B, T, H, V, dtype=DTYPE, device=DEVICE)
        g = torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE) * 0.1
        h = torch.randn(B, NT, H, K, V, dtype=DTYPE, device=DEVICE) * 0.01
        A = torch.randn(B, T, H, BT, dtype=DTYPE, device=DEVICE) * 0.1

        o_ref = reference_chunk_gla_fwd_o(q, v, g, h, A, scale, BT)

        try:
            o_cute = torch.zeros_like(q[:, :, :, :V])
            chunk_gla_fwd_o(q, v, g, h, o_cute, A, scale, chunk_size=BT, is_varlen=False, persistent=True)
            torch.cuda.synchronize()
            passed = assert_close("CuTe DSL vs Ref", o_ref, o_cute, atol=0.02)
            all_passed = all_passed and passed
        except Exception as e:
            print(f"  CuTe DSL failed: {e}")
            all_passed = False

        try:
            o_triton = triton_chunk_gla_fwd_o(q, v, g, h, A, scale, BT)
            passed = assert_close("Triton vs Ref", o_ref, o_triton, atol=0.02)
            all_passed = all_passed and passed
        except Exception as e:
            print(f"  Triton failed: {e}")
            all_passed = False

    print(f"\n{'=' * 50}")
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


def run_benchmark(B=4, T=4096, H=4, num_iters=100):
    """Benchmark CuTe DSL and Triton kernels."""
    NT = (T + BT - 1) // BT
    scale = K**-0.5

    print(f"\n=== Benchmark: B={B}, T={T}, H={H}, K={K}, V={V} ===")

    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, dtype=DTYPE, device=DEVICE)
    v = torch.randn(B, T, H, V, dtype=DTYPE, device=DEVICE)
    g = torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE) * 0.1
    h = torch.randn(B, NT, H, K, V, dtype=DTYPE, device=DEVICE) * 0.01
    A = torch.randn(B, T, H, BT, dtype=DTYPE, device=DEVICE) * 0.1
    o = torch.zeros(B, T, H, V, dtype=DTYPE, device=DEVICE)

    # --- CuTe DSL ---
    chunk_gla_fwd_o(q, v, g, h, o, A, scale, chunk_size=BT, is_varlen=False, persistent=True)
    torch.cuda.synchronize()

    for _ in range(5):
        chunk_gla_fwd_o(q, v, g, h, o, A, scale, chunk_size=BT, is_varlen=False, persistent=True)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        chunk_gla_fwd_o(q, v, g, h, o, A, scale, chunk_size=BT, is_varlen=False, persistent=True)
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

    total_bytes = (q.nelement() + v.nelement() + g.nelement() + h.nelement() + A.nelement()) * 2  # bf16 = 2 bytes
    total_bytes += v.nelement() * 2  # output

    print(f"  CuTe DSL: {cute_ms:.3f} ms, {total_bytes / (cute_ms * 1e-3) / 1e9:.1f} GB/s")
    print(f"  Triton:   {triton_ms:.3f} ms, {total_bytes / (triton_ms * 1e-3) / 1e9:.1f} GB/s")
    print(f"  Speedup:  {triton_ms / cute_ms:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="correctness", choices=["correctness", "benchmark", "both"])
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--T", type=int, default=4096)
    parser.add_argument("--H", type=int, default=4)
    args = parser.parse_args()

    if args.test in ("correctness", "both"):
        run_correctness_tests()

    if args.test in ("benchmark", "both"):
        run_benchmark(B=args.B, T=args.T, H=args.H)
