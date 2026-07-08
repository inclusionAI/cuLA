#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
# Licensed under the Apache License, Version 2.0.
"""Tests for SM90 intra-card CP: dispatch routing + numerical accuracy.

Mirrors tests/test_intracard_cp.py (SM100 version) but targets the Hopper
(SM90) `kda_prefill_hopper_opt` / `kda_prefill_hopper_auto` path.

Three reference levels:
  - cuLA basic (kda_prefill_hopper)         — same C++ kernel, no CP scheduling
                                              → verifies CP scheduling is value-preserving
  - cuLA opt with auto_cp=False             — opt Python wrapper but CP disabled
                                              → isolates the CP code paths
  - FLA chunk_kda (cross-impl reference)    → source of truth for end-to-end output

The CP path is exercised through:
  - kda_prefill_hopper_auto  (router picks opt when shape benefits from CP)
  - kda_prefill_hopper_opt(auto_cp=True)  (force CP entry; bypasses router)
"""

from __future__ import annotations

import math
import pathlib
import sys

import pytest
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fla.ops.kda import chunk_kda as fla_chunk_kda  # noqa: E402
from fla.utils import assert_close  # noqa: E402

from cula.kda import (  # noqa: E402
    kda_prefill_hopper,
    kda_prefill_hopper_auto,
    kda_prefill_hopper_opt,
)
from cula.kda.cp_context import _calc_cp_seqs  # noqa: E402
from cula.utils import get_device_sm_count  # noqa: E402

BT, D = 64, 128
DEVICE = "cuda"
DTYPE = torch.bfloat16
LOWER_BOUND = -5.0

# Tolerances — same convention as tests/test_intracard_cp.py:
#   * Same-kernel (CP scheduling only): torch.testing.assert_close
#                                       (CP-on vs CP-off both go through cuLA kernels)
#   * Cross-impl (vs FLA):              fla.utils.assert_close(ratio=...)
ATOL_SAME_KERNEL = 1e-2
RTOL_SAME_KERNEL = 1e-2
RATIO_VS_FLA = 0.015  # bf16 cross-impl noise band (matches SM100 test)
RATIO_STRESS = 1e-6  # deterministic re-run: drift implies race


pytestmark = [
    pytest.mark.sm90_only,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


# ============================== Helpers ==============================


def _cu_from_seq_lens(seq_lens, device=DEVICE):
    cu = [0]
    for s in seq_lens:
        cu.append(cu[-1] + s)
    return torch.tensor(cu, dtype=torch.int32, device=device)


def make_varlen_inputs(seq_lens, H, *, use_h0=False, seed=42):
    """Build varlen-packed B=1 inputs for kda_prefill_hopper_*."""
    total = sum(seq_lens)
    N = len(seq_lens)
    cu = _cu_from_seq_lens(seq_lens)
    torch.manual_seed(seed)
    q = torch.randn(1, total, H, D, dtype=DTYPE, device=DEVICE)
    k = torch.randn(1, total, H, D, dtype=DTYPE, device=DEVICE)
    v = torch.randn(1, total, H, D, dtype=DTYPE, device=DEVICE)
    g = -torch.rand(1, total, H, D, dtype=torch.float32, device=DEVICE).abs() * 0.5
    beta = torch.randn(1, total, H, dtype=torch.float32, device=DEVICE).sigmoid().to(DTYPE)
    A_log = torch.randn(H, dtype=torch.float32, device=DEVICE)
    dt_bias = torch.randn(H, D, dtype=torch.float32, device=DEVICE)
    h0 = torch.randn(N, H, D, D, dtype=torch.float32, device=DEVICE) * 0.1 if use_h0 else None
    return q, k, v, g, beta, h0, A_log, dt_bias, cu


# ---- entry points under test ----


def _common_cula_kw(q, k, v, g, beta, h0, A_log, dt_bias, cu):
    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=1.0 / math.sqrt(D),
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=LOWER_BOUND,
        cu_seqlens=cu,
    )


def run_cula_basic(q, k, v, g, beta, h0, A_log, dt_bias, cu):
    return kda_prefill_hopper(**_common_cula_kw(q, k, v, g, beta, h0, A_log, dt_bias, cu))


def run_cula_opt_no_cp(q, k, v, g, beta, h0, A_log, dt_bias, cu):
    return kda_prefill_hopper_opt(
        **_common_cula_kw(q, k, v, g, beta, h0, A_log, dt_bias, cu),
        auto_cp=False,
    )


def run_cula_opt_cp(q, k, v, g, beta, h0, A_log, dt_bias, cu):
    """Force CP entry through opt wrapper."""
    return kda_prefill_hopper_opt(
        **_common_cula_kw(q, k, v, g, beta, h0, A_log, dt_bias, cu),
        auto_cp=True,
    )


def run_cula_auto(q, k, v, g, beta, h0, A_log, dt_bias, cu):
    """Adaptive router — exercises the production entry point."""
    return kda_prefill_hopper_auto(
        **_common_cula_kw(q, k, v, g, beta, h0, A_log, dt_bias, cu),
    )


def run_fla(q, k, v, g, beta, h0, A_log, dt_bias, cu):
    """FLA reference. cuLA returns ht as [N, HV, V, K]; FLA's default layout
    is [N, HV, K, V]. We pass ``transpose_state_layout=True`` so its output
    matches cuLA's layout — no manual transpose needed before assert_close.
    """
    return fla_chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=1.0 / math.sqrt(D),
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=LOWER_BOUND,
        cu_seqlens=cu.long(),
        transpose_state_layout=True,
    )


# ---- assertions ----


def _assert_same_kernel(name, actual, ref):
    """torch.testing.assert_close with tight atol/rtol (CP-on vs CP-off use the
    same C++ kernel; the only delta is per-chunk recurrence reordering)."""
    if actual is None or ref is None:
        assert actual is ref, f"{name}: one is None and the other isn't"
        return
    torch.testing.assert_close(
        actual.float(),
        ref.float(),
        atol=ATOL_SAME_KERNEL,
        rtol=RTOL_SAME_KERNEL,
        msg=lambda m: f"{name}: {m}",
    )


def assert_cp_engages(cu, H):
    """Fail fast if _calc_cp_seqs won't engage CP for this shape — without
    that, the test silently checks CP-off vs CP-off.
    """
    num_sms = get_device_sm_count(torch.device(DEVICE))
    use_cp, cp_cu, *_ = _calc_cp_seqs(
        cu,
        BT,
        H,
        num_sms,
        raw_cu_seqlens_cpu=cu.cpu(),
    )
    assert use_cp and cp_cu is not None, f"_calc_cp_seqs returned use_cp=False for cu={cu.tolist()} H={H}"
    n_sub = int(cp_cu.numel() - 1)
    raw_batch = int(cu.numel() - 1)
    assert n_sub > raw_batch, f"CP didn't split: n_sub={n_sub} == raw_batch={raw_batch}"


# ====================== Dispatch path: CP vs no-CP ======================
# Verifies kda_prefill_hopper_opt(auto_cp=True) routes through CP and matches
# the same-kernel no-CP baseline (kda_prefill_hopper).

DISPATCH_CONFIGS = [
    # (seq_lens, H, use_h0)
    ([32768], 4, False),
    ([32768], 4, True),
    ([65536], 4, True),
    ([32768], 8, False),
    ([65536], 8, True),
    ([16384, 16384], 4, True),
    ([28672, 4096], 4, True),
    ([131072, 1024], 4, False),
]


@pytest.mark.parametrize("seq_lens,H,use_h0", DISPATCH_CONFIGS)
def test_cp_matches_basic_baseline(seq_lens, H, use_h0):
    """CP-on (opt+auto_cp) output equals basic baseline (no-CP)."""
    q, k, v, g, beta, h0, A_log, dt_bias, cu = make_varlen_inputs(
        seq_lens,
        H,
        use_h0=use_h0,
    )
    assert_cp_engages(cu, H)
    with torch.inference_mode():
        o_base, ht_base = run_cula_basic(q, k, v, g, beta, h0, A_log, dt_bias, cu)
        o_cp, ht_cp = run_cula_opt_cp(q, k, v, g, beta, h0, A_log, dt_bias, cu)
    _assert_same_kernel("o", o_cp, o_base)
    _assert_same_kernel("ht", ht_cp, ht_base)


@pytest.mark.parametrize("seq_lens,H,use_h0", DISPATCH_CONFIGS)
def test_auto_router_matches_basic_baseline(seq_lens, H, use_h0):
    """kda_prefill_hopper_auto output (whatever path it picks) equals basic baseline."""
    q, k, v, g, beta, h0, A_log, dt_bias, cu = make_varlen_inputs(
        seq_lens,
        H,
        use_h0=use_h0,
    )
    with torch.inference_mode():
        o_base, ht_base = run_cula_basic(q, k, v, g, beta, h0, A_log, dt_bias, cu)
        o_auto, ht_auto = run_cula_auto(q, k, v, g, beta, h0, A_log, dt_bias, cu)
    _assert_same_kernel("o", o_auto, o_base)
    _assert_same_kernel("ht", ht_auto, ht_base)


def test_cp_off_matches_basic_baseline():
    """opt with auto_cp=False must match basic (no CP, no fused-pre divergence)."""
    seq_lens, H, use_h0 = [32768], 4, True
    q, k, v, g, beta, h0, A_log, dt_bias, cu = make_varlen_inputs(seq_lens, H, use_h0=use_h0)
    with torch.inference_mode():
        o_base, ht_base = run_cula_basic(q, k, v, g, beta, h0, A_log, dt_bias, cu)
        o_off, ht_off = run_cula_opt_no_cp(q, k, v, g, beta, h0, A_log, dt_bias, cu)
    _assert_same_kernel("o", o_off, o_base)
    _assert_same_kernel("ht", ht_off, ht_base)


# ====================== Cross-impl: CP vs FLA ======================

VS_FLA_CONFIGS = [
    ([32768], 4),
    ([65536], 4),
    ([32768], 8),
    ([16384, 16384], 4),
    ([28672, 4096], 4),
    ([131072, 1024], 4),
]


# Irregular varlen lengths
IRREGULAR_VARLEN_CONFIGS = [
    ([1], 4),
    ([63], 4),
    ([64], 4),
    ([65], 4),
    ([129], 4),
    ([1, 63, 64, 65, 129], 4),
    ([1, 63, 64, 65, 129], 8),
    ([129, 65, 64, 63, 1], 4),
    ([1024, 1, 63, 65, 129], 4),
    ([4096, 1, 63, 64, 65, 129], 4),
    ([4096, 1, 63, 64, 65, 129], 8),
    ([8192, 1, 31, 63, 65, 127, 129, 255], 4),
    ([1] * 8 + [63] * 4 + [129] * 2, 4),
    ([255, 257, 511, 513], 4),
]


@pytest.mark.parametrize("seq_lens,H", IRREGULAR_VARLEN_CONFIGS)
def test_irregular_varlen_vs_fla(seq_lens, H):
    """Irregular varlen lengths"""
    q, k, v, g, beta, _, A_log, dt_bias, cu = make_varlen_inputs(seq_lens, H, use_h0=False)
    with torch.inference_mode():
        o_fla, ht_fla = run_fla(q, k, v, g, beta, None, A_log, dt_bias, cu)
        o_opt, ht_opt = run_cula_opt_no_cp(q, k, v, g, beta, None, A_log, dt_bias, cu)
    assert_close(f"o (cu={cu.tolist()},H={H})", o_fla, o_opt, ratio=RATIO_VS_FLA)
    assert_close(f"ht (cu={cu.tolist()},H={H})", ht_fla, ht_opt, ratio=RATIO_VS_FLA)


@pytest.mark.parametrize("seq_lens,H", IRREGULAR_VARLEN_CONFIGS)
def test_irregular_varlen_opt_matches_basic(seq_lens, H):
    """Irregular varlen: opt path (may take fused gate+l2norm) equals basic baseline."""
    q, k, v, g, beta, _, A_log, dt_bias, cu = make_varlen_inputs(seq_lens, H, use_h0=False)
    with torch.inference_mode():
        o_base, ht_base = run_cula_basic(q, k, v, g, beta, None, A_log, dt_bias, cu)
        o_opt, ht_opt = run_cula_opt_no_cp(q, k, v, g, beta, None, A_log, dt_bias, cu)
    _assert_same_kernel("o", o_opt, o_base)
    _assert_same_kernel("ht", ht_opt, ht_base)


@pytest.mark.parametrize("seq_lens,H", VS_FLA_CONFIGS)
def test_cp_vs_fla(seq_lens, H):
    """CP output matches FLA chunk_kda reference (cross-impl)."""
    q, k, v, g, beta, _, A_log, dt_bias, cu = make_varlen_inputs(seq_lens, H, use_h0=False)
    assert_cp_engages(cu, H)
    with torch.inference_mode():
        o_fla, ht_fla = run_fla(q, k, v, g, beta, None, A_log, dt_bias, cu)
        o_cp, ht_cp = run_cula_opt_cp(q, k, v, g, beta, None, A_log, dt_bias, cu)
    assert_close(f"o (cu={cu.tolist()},H={H})", o_fla, o_cp, ratio=RATIO_VS_FLA)
    assert_close(f"ht (cu={cu.tolist()},H={H})", ht_fla, ht_cp, ratio=RATIO_VS_FLA)


# ====================== Final state ht correctness ======================
# Per-sequence ht must be independently correct for prefill→decode handoff.

FINAL_STATE_CONFIGS = [
    ([65536], 4, False),
    ([65536], 4, True),
    ([65536, 16384], 4, True),
    ([28672, 4096], 4, False),
    ([131072, 1024], 4, True),
]


@pytest.mark.parametrize("seq_lens,H,use_h0", FINAL_STATE_CONFIGS)
def test_cp_final_state_per_seq(seq_lens, H, use_h0):
    """Each sequence's ht matches basic baseline independently (no cross-leakage)."""
    q, k, v, g, beta, h0, A_log, dt_bias, cu = make_varlen_inputs(seq_lens, H, use_h0=use_h0)
    assert_cp_engages(cu, H)
    with torch.inference_mode():
        _, ht_base = run_cula_basic(q, k, v, g, beta, h0, A_log, dt_bias, cu)
        _, ht_cp = run_cula_opt_cp(q, k, v, g, beta, h0, A_log, dt_bias, cu)
    assert ht_cp is not None and ht_cp.shape == ht_base.shape, (
        f"shape mismatch: cp={tuple(ht_cp.shape)} base={tuple(ht_base.shape)}"
    )
    for i in range(len(seq_lens)):
        _assert_same_kernel(f"ht[{i}] (len={seq_lens[i]})", ht_cp[i], ht_base[i])


# ====================== Stress: race / non-determinism ======================
# CP's per-chunk preprocess + main kernel — re-running same inputs must
# produce bit-identical outputs (no race, no order-dependence).

STRESS_ITERS = 50


@pytest.mark.parametrize(
    "seq_lens,H,use_h0",
    [
        pytest.param([65536], 4, True, id="single-64K-H4-h0"),
        pytest.param([65536, 4096], 4, True, id="multi-64K+4K-H4-h0"),
    ],
)
def test_cp_stress_repeat(seq_lens, H, use_h0):
    """Run CP N times; every iter must match the first (deterministic)."""
    q, k, v, g, beta, h0, A_log, dt_bias, cu = make_varlen_inputs(
        seq_lens,
        H,
        use_h0=use_h0,
        seed=20260516,
    )
    assert_cp_engages(cu, H)
    with torch.inference_mode():
        o_ref, ht_ref = run_cula_opt_cp(q, k, v, g, beta, h0, A_log, dt_bias, cu)
        torch.cuda.synchronize()
        for i in range(STRESS_ITERS):
            o_i, ht_i = run_cula_opt_cp(q, k, v, g, beta, h0, A_log, dt_bias, cu)
            torch.cuda.synchronize()
            assert_close(f"iter {i} o", o_ref, o_i, ratio=RATIO_STRESS)
            assert_close(f"iter {i} ht", ht_ref, ht_i, ratio=RATIO_STRESS)


# ====================== h0=None equivalence ======================
# We patched cp_context.py so raw_h0=None synthesizes a zero pool. Verify the
# kernel result is numerically equivalent to passing an explicit zero h0.


def test_cp_h0_none_equiv_h0_zeros():
    """h0=None must produce identical ht to h0=zeros (no implicit init drift)."""
    seq_lens, H = [65536, 4096], 4
    assert_cp_engages(_cu_from_seq_lens(seq_lens), H)
    q, k, v, g, beta, _, A_log, dt_bias, cu = make_varlen_inputs(
        seq_lens,
        H,
        use_h0=False,
        seed=20260501,
    )
    h0_zeros = torch.zeros(len(seq_lens), H, D, D, dtype=torch.float32, device=DEVICE)
    with torch.inference_mode():
        o_none, ht_none = run_cula_opt_cp(q, k, v, g, beta, None, A_log, dt_bias, cu)
        o_zeros, ht_zeros = run_cula_opt_cp(q, k, v, g, beta, h0_zeros, A_log, dt_bias, cu)
    torch.cuda.synchronize()
    o_diff = (o_none.float() - o_zeros.float()).abs().max().item()
    ht_diff = (ht_none.float() - ht_zeros.float()).abs().max().item()
    assert o_diff < 1e-3, f"o:  h0=None vs h0=zeros max abs diff {o_diff:.4e}"
    assert ht_diff < 1e-4, f"ht: h0=None vs h0=zeros max abs diff {ht_diff:.4e}"


# ====================== CP bypass: shapes where _calc_cp_seqs returns False ======================
# When CP heuristic says "don't split", auto_cp=True must produce bit-identical
# output to basic (because the kernel takes the same path).

BYPASS_CONFIGS = [
    ([2048], 8),  # H=8 single seq T<=2048 → no CP
    ([16384], 64),  # H=64 → CP never fires (per _calc_cp_seqs H>=64 branch)
    ([4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096], 8),  # native_grid 64 >> 16
    ([131072] + [1024] * 5, 8),  # raw_batch big enough that native_grid > 16
]


@pytest.mark.parametrize("seq_lens,H", BYPASS_CONFIGS)
def test_cp_bypass_matches_basic(seq_lens, H):
    """When CP heuristic skips, auto_cp=True must be a no-op (same output as basic)."""
    q, k, v, g, beta, _, A_log, dt_bias, cu = make_varlen_inputs(seq_lens, H, use_h0=False)
    num_sms = get_device_sm_count(torch.device(DEVICE))
    use_cp, cp_cu, *_ = _calc_cp_seqs(cu, BT, H, num_sms, raw_cu_seqlens_cpu=cu.cpu())
    n_sub = int(cp_cu.numel() - 1) if cp_cu is not None else 0
    assert not use_cp or n_sub == len(seq_lens), (
        f"expected bypass for cu={cu.tolist()} H={H}, got use_cp={use_cp} n_sub={n_sub}"
    )
    with torch.inference_mode():
        o_base, ht_base = run_cula_basic(q, k, v, g, beta, None, A_log, dt_bias, cu)
        o_cp, ht_cp = run_cula_opt_cp(q, k, v, g, beta, None, A_log, dt_bias, cu)
    _assert_same_kernel("o", o_cp, o_base)
    _assert_same_kernel("ht", ht_cp, ht_base)
