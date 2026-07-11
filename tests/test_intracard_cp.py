#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
# Licensed under the Apache License, Version 2.0.
"""Tests for intracard CP: dispatch routing + numerical accuracy.

Two reference levels are used:
  - cuLA no-CP baseline (same kernel, no CP scheduling) — verifies dispatch
    plumbing and that CP scheduling is value-preserving.
  - Pure-PyTorch fp32 reference — source of truth for kernel correctness;
    any deviation here is a real CP / kernel bug, not a cross-impl gap.

The CP path is exercised via two entry points:
  - ``chunk_gated_delta_rule_fwd_h`` with ``CULA_INTRACARD_CP=1`` + inference_mode
  - ``intracard_fwd_h`` (direct, bypasses the heuristic)
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import pytest
import torch

# Make cuLA importable when tests run from a fresh checkout (no `pip install -e`).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h as fla_fwd_h  # noqa: E402
from fla.utils import assert_close  # noqa: E402  (RMSE-relative + atol short-circuit + NaN check)

from cula.ops.kda.sm100.cp.chunk_delta_h import (  # noqa: E402
    compute_subseq_len,
    intracard_fwd_h,
    prepare_subseq_cu_seqlens,
    should_use_intracard_cp,
)
from cula.ops.kda.sm100.delta_h import chunk_gated_delta_rule_fwd_h  # noqa: E402
from cula.utils import get_device_sm_count  # noqa: E402

# Constants & tolerances — aligned with existing cuLA tests (see below).
BT, K, V = 64, 128, 128
DEVICE = "cuda"
# Tolerances aligned with existing cuLA tests:
#   * Same-kernel (CP scheduling only): torch.testing.assert_close(atol=1e-2, rtol=1e-2)
#                                       — matches tests/test_chunk_delta_h.py CP block
#   * Cross-impl / vs ref:              fla.utils.assert_close(ratio=...)
#                                       — matches tests/test_kda_compare_fla.py
ATOL_SAME_KERNEL = 1e-2
RTOL_SAME_KERNEL = 1e-2
RATIO_VS_REF = 0.005  # RMSE / RMS(ref) — matches FLA test_gated_delta.py fwd
RATIO_VS_FLA = 0.015  # cross-impl gap measured ~1.27% (TF32 MMA vs Triton fp32)
RATIO_STRESS = 1e-6  # deterministic re-run: drift would indicate race


pytestmark = [
    pytest.mark.sm100_only,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


# ============================== Helpers ==============================


def make_varlen_inputs(seq_lens, H, *, use_gk=False, use_h0=False, seed=42):
    """Build varlen-packed B=1 inputs for chunk_gated_delta_rule_fwd_h."""
    total = sum(seq_lens)
    N = len(seq_lens)
    cu = [0]
    for s in seq_lens:
        cu.append(cu[-1] + s)

    torch.manual_seed(seed)
    k = torch.randn(1, total, H, K, dtype=torch.bfloat16, device=DEVICE) * 0.02
    w = torch.randn(1, total, H, K, dtype=torch.bfloat16, device=DEVICE) * 0.02
    u = torch.randn(1, total, H, V, dtype=torch.bfloat16, device=DEVICE) * 0.02

    gk = None
    if use_gk:
        gk = torch.zeros(1, total, H, K, dtype=torch.float32, device=DEVICE)
        for i in range(N):
            bos, eos = cu[i], cu[i + 1]
            seg = torch.randn(1, eos - bos, H, K, dtype=torch.float32, device=DEVICE) * 0.01
            gk[:, bos:eos] = -torch.abs(seg).cumsum(dim=1)

    h0 = torch.randn(N, H, K, V, dtype=torch.float32, device=DEVICE) * 0.01 if use_h0 else None
    return k, w, u, gk, h0, torch.tensor(cu, dtype=torch.int32, device=DEVICE)


def run_cula_no_cp(k, w, u, gk, h0, cu, **kw):
    return chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        gk=gk,
        initial_state=h0,
        chunk_size=BT,
        cu_seqlens=cu,
        _no_cp=True,
        **kw,
    )


def run_cula_cp(k, w, u, gk, h0, cu, **kw):
    """Auto-dispatch via env + inference_mode."""
    old = os.environ.get("CULA_INTRACARD_CP")
    os.environ["CULA_INTRACARD_CP"] = "1"
    try:
        with torch.inference_mode():
            return chunk_gated_delta_rule_fwd_h(
                k=k,
                w=w,
                u=u,
                gk=gk,
                initial_state=h0,
                chunk_size=BT,
                cu_seqlens=cu,
                **kw,
            )
    finally:
        if old is None:
            os.environ.pop("CULA_INTRACARD_CP", None)
        else:
            os.environ["CULA_INTRACARD_CP"] = old


def run_intracard_direct(k, w, u, gk, h0, cu, *, output_final_state=True, save_new_value=True):
    """Direct CP call — skips the auto-dispatch heuristic.

    intracard_fwd_h is a pure executor that raises NotSplittableError when the
    post-split occupancy guard rejects; mirror the production caller's graceful
    fallback to the serial path so configs that don't engage CP still return.
    """
    from cula.ops.kda.policy import NotSplittableError

    try:
        return intracard_fwd_h(
            k=k,
            w=w,
            u=u,
            gk=gk,
            initial_state=h0,
            output_final_state=output_final_state,
            chunk_size=BT,
            save_new_value=save_new_value,
            cu_seqlens=cu,
            cu_seqlens_cpu=cu.cpu(),
        )
    except NotSplittableError:
        return chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            gk=gk,
            initial_state=h0,
            output_final_state=output_final_state,
            chunk_size=BT,
            save_new_value=save_new_value,
            cu_seqlens=cu,
            _no_cp=True,
        )


def run_fla(k, w, u, gk, h0, cu, **kw):
    return fla_fwd_h(
        k=k,
        w=w,
        u=u,
        gk=gk,
        initial_state=h0,
        chunk_size=BT,
        cu_seqlens=cu,
        **kw,
    )


def pytorch_ref(k, w, u, *, gk=None, initial_state=None, cu_seqlens, save_new_value=True):
    """Pure-PyTorch fp32 reference for varlen chunk_gated_delta_rule_fwd_h.

    Mirrors the per-chunk math FLA's Triton kernel implements:
        v_new = u - w @ h
        h    *= exp2(gk_last)      # (if gk)
        h    += k^T @ v_new
    """
    assert k.shape[0] == 1, "varlen reference expects packed B=1"
    _, total, H, head_k = k.shape
    head_v = u.shape[-1]
    cu = cu_seqlens.cpu().tolist()
    N = len(cu) - 1
    total_c = sum(math.ceil((cu[i + 1] - cu[i]) / BT) for i in range(N))

    h_out = torch.empty(1, total_c, H, head_k, head_v, dtype=torch.bfloat16, device=k.device)
    v_out = torch.empty_like(u) if save_new_value else None
    ht_out = torch.empty(N, H, head_k, head_v, dtype=torch.float32, device=k.device)

    ci = 0
    for s in range(N):
        bos, eos = cu[s], cu[s + 1]
        h = (
            initial_state[s].float().clone()
            if initial_state is not None
            else torch.zeros(H, head_k, head_v, dtype=torch.float32, device=k.device)
        )
        for cs in range(bos, eos, BT):
            ce = min(cs + BT, eos)
            h_out[0, ci] = h.to(torch.bfloat16)
            w_c = w[0, cs:ce].permute(1, 0, 2).float()
            k_c = k[0, cs:ce].permute(1, 0, 2).float()
            u_c = u[0, cs:ce].permute(1, 0, 2).float()
            v_new = u_c - torch.matmul(w_c, h)
            if v_out is not None:
                v_out[0, cs:ce] = v_new.permute(1, 0, 2).to(torch.bfloat16)
            if gk is not None:
                gk_last = gk[0, cs:ce].permute(1, 0, 2).float()[:, -1, :]
                h = h * torch.exp2(gk_last).unsqueeze(-1)
            h = h + torch.matmul(k_c.transpose(-2, -1), v_new)
            ci += 1
        ht_out[s] = h
    return h_out, v_out, ht_out


def _assert_same_kernel(name, actual, ref):
    """torch.testing.assert_close — matches tests/test_chunk_delta_h.py."""
    if actual is None or ref is None:
        assert actual is ref, f"{name}: one is None and other isn't"
        return
    torch.testing.assert_close(
        actual.float(),
        ref.float(),
        atol=ATOL_SAME_KERNEL,
        rtol=RTOL_SAME_KERNEL,
        msg=lambda m: f"{name}: {m}",
    )


def assert_cp_splits(cu, H, total_T):
    """Fail fast if the strategy doesn't even try to engage CP for this config.

    Note: we do NOT assert the post-split SM guard (total_subseqs * 2 * H <= num_sms).
    intracard_fwd_h falls back gracefully to the non-CP path when that guard rejects,
    so the test still exercises a valid code path even if CP scheduling itself doesn't
    engage.
    """
    cu_cpu = cu.cpu()
    num_sms = get_device_sm_count(torch.device(DEVICE))
    assert should_use_intracard_cp(cu_cpu, num_sms, H, BT), (
        "should_use_intracard_cp returned False — config does not trigger CP"
    )
    max_seq = int(torch.diff(cu_cpu).max().item())
    subseq_len = compute_subseq_len(max_seq, num_sms, H, BT, num_seqs=len(cu_cpu) - 1)
    _, split_info, _ = prepare_subseq_cu_seqlens(cu_cpu, subseq_len, BT)
    assert split_info, "config must exercise the split path"


def test_forced_cp_not_splittable_raises():
    """use_intracard_cp=True on an unsplittable shape must raise NotSplittableError."""
    from cula.ops.kda.policy import NotSplittableError

    # A single one-chunk sequence cannot be meaningfully split.
    cu = torch.tensor([0, BT], dtype=torch.int32, device=DEVICE)
    k = torch.randn(1, BT, 1, K, device=DEVICE, dtype=torch.bfloat16)
    w = torch.randn(1, BT, 1, K, device=DEVICE, dtype=torch.bfloat16)
    u = torch.randn(1, BT, 1, V, device=DEVICE, dtype=torch.bfloat16)
    with torch.inference_mode(), pytest.raises(NotSplittableError):
        chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, cu_seqlens=cu, use_intracard_cp=True)


# ====================== Dispatch path: CP vs no-CP ======================
# Verifies chunk_gated_delta_rule_fwd_h routes to CP under env+inference_mode,
# and matches the same-kernel no-CP baseline.

DISPATCH_CONFIGS = [
    ([32768], 4, False),
    ([32768], 4, True),
    ([65536], 4, True),
    ([32768], 8, True),
    ([32768, 256, 32768], 4, True),
    ([65536, 128], 4, False),
    ([32768, 32768, 32768], 4, True),
    ([65536, 256, 128, 64], 8, True),
]


@pytest.mark.parametrize("seq_lens,H,use_gk", DISPATCH_CONFIGS)
def test_cp_autodispatch_matches_baseline(seq_lens, H, use_gk):
    """CP auto-dispatch output equals no-CP baseline (same kernel).

    Tolerance: `torch.testing.assert_close(atol=1e-2, rtol=1e-2)` —
    matches the CP block in tests/test_chunk_delta_h.py.
    """
    k, w, u, gk, _, cu = make_varlen_inputs(seq_lens, H, use_gk=use_gk)
    h_base, v_base, _ = run_cula_no_cp(k, w, u, gk, None, cu)
    h_cp, v_cp, _ = run_cula_cp(k, w, u, gk, None, cu)
    _assert_same_kernel("h", h_cp, h_base)
    _assert_same_kernel("v_new", v_cp, v_base)


@pytest.mark.parametrize("seq_lens,H", [([32768], 4), ([32768, 256, 32768], 4)])
def test_cp_autodispatch_with_h0(seq_lens, H):
    """CP path preserves h0 input and ht output."""
    k, w, u, gk, h0, cu = make_varlen_inputs(seq_lens, H, use_gk=True, use_h0=True)
    h_base, v_base, ht_base = run_cula_no_cp(
        k,
        w,
        u,
        gk,
        h0,
        cu,
        output_final_state=True,
    )
    h_cp, v_cp, ht_cp = run_cula_cp(
        k,
        w,
        u,
        gk,
        h0,
        cu,
        output_final_state=True,
    )
    _assert_same_kernel("h", h_cp, h_base)
    _assert_same_kernel("v_new", v_cp, v_base)
    _assert_same_kernel("ht", ht_cp, ht_base)


@pytest.mark.parametrize("T,H", [(32768, 4), (65536, 4), (32768, 8)])
def test_cp_autodispatch_vs_fla(T, H):
    """CP output matches FLA Triton reference (cross-impl).

    Tolerance: FLA's `assert_close` ratio=0.005 (RMSE/RMS <= 0.5%) —
    same as FLA tests/ops/test_gated_delta.py for fwd outputs.
    """
    k, w, u, gk, _, cu = make_varlen_inputs([T], H, use_gk=True)
    h_fla, _, _ = run_fla(k, w, u, gk, None, cu)
    h_cp, _, _ = run_cula_cp(k, w, u, gk, None, cu)
    assert_close(f"h (T={T},H={H})", h_fla, h_cp, ratio=RATIO_VS_FLA)


# ====================== Accuracy: vs PyTorch fp32 reference ======================
# Direct entry intracard_fwd_h, ground truth = pure-PyTorch fp32.

ACCURACY_CONFIGS = [
    ([65536], 4, False, False),
    ([65536], 4, True, True),
    ([65536, 512], 4, True, True),
    ([65536, 256, 32768], 4, True, False),
    ([65536, 128], 4, False, True),
    ([131072], 4, True, True),
    ([65536, 512, 256, 128], 4, True, False),
    ([65536, 1024, 8192], 8, True, True),
]


@pytest.mark.parametrize("seq_lens,H,use_gk,use_h0", ACCURACY_CONFIGS)
def test_intracard_cp_vs_pytorch_ref(seq_lens, H, use_gk, use_h0):
    """CP output (h, v_new, ht) matches PyTorch fp32 reference.

    Tolerance: FLA's `assert_close` ratio=0.005 (RMSE/RMS <= 0.5%).
    """
    k, w, u, gk, h0, cu = make_varlen_inputs(
        seq_lens,
        H,
        use_gk=use_gk,
        use_h0=use_h0,
        seed=20260428,
    )
    assert_cp_splits(cu, H, k.shape[1])
    with torch.inference_mode():
        ref_h, ref_v, ref_ht = pytorch_ref(
            k,
            w,
            u,
            gk=gk,
            initial_state=h0,
            cu_seqlens=cu,
        )
        cp_h, cp_v, cp_ht = run_intracard_direct(k, w, u, gk, h0, cu)
    torch.cuda.synchronize()
    assert_close("h", ref_h, cp_h, ratio=RATIO_VS_REF)
    assert_close("v_new", ref_v, cp_v, ratio=RATIO_VS_REF)
    assert_close("ht", ref_ht, cp_ht, ratio=RATIO_VS_REF)


# ====================== Final state ht correctness ======================
# Per-sequence ht must be independently correct for prefill→decode handoff.

FINAL_STATE_CONFIGS = [
    ([65536], 4, False, False),
    ([65536], 4, True, True),
    ([65536], 8, True, True),
    ([65536, 16384], 4, True, True),
    ([65536, 512, 16384], 4, True, False),
]


@pytest.mark.parametrize("seq_lens,H,use_gk,use_h0", FINAL_STATE_CONFIGS)
def test_intracard_cp_final_state_per_seq(seq_lens, H, use_gk, use_h0):
    """Each sequence's ht matches PyTorch ref independently (no cross-leakage)."""
    k, w, u, gk, h0, cu = make_varlen_inputs(
        seq_lens,
        H,
        use_gk=use_gk,
        use_h0=use_h0,
        seed=20260430,
    )
    assert_cp_splits(cu, H, k.shape[1])
    with torch.inference_mode():
        _, _, ref_ht = pytorch_ref(
            k,
            w,
            u,
            gk=gk,
            initial_state=h0,
            cu_seqlens=cu,
            save_new_value=False,
        )
        _, _, cp_ht = run_intracard_direct(
            k,
            w,
            u,
            gk,
            h0,
            cu,
            save_new_value=False,
        )
    torch.cuda.synchronize()
    assert cp_ht is not None and cp_ht.shape == ref_ht.shape
    for i in range(len(seq_lens)):
        assert_close(f"ht[{i}] (len={seq_lens[i]})", ref_ht[i], cp_ht[i], ratio=RATIO_VS_REF)


# ====================== Stress: race / non-determinism ======================
# CP uses dynamic atomicAdd scheduling + multi-sub-seq merge — re-running the
# same inputs must produce the same outputs (no race, no order-dependence).

STRESS_ITERS = 100


@pytest.mark.parametrize(
    "seq_lens,H,use_gk,use_h0",
    [
        pytest.param([65536], 4, True, True, id="single-64K-H4-gk-h0"),
        pytest.param([65536, 4096], 4, True, True, id="multi-64K+4K-H4-gk-h0"),
    ],
)
def test_intracard_cp_stress_repeat(seq_lens, H, use_gk, use_h0):
    """Run CP N times; every iter must match the first (race detection).

    Tolerance: ratio=1e-6 — deterministic CP should not drift across runs.
    Uses `assert_close`'s atol short-circuit (abs <= 1e-6 → auto-pass).
    """
    k, w, u, gk, h0, cu = make_varlen_inputs(
        seq_lens,
        H,
        use_gk=use_gk,
        use_h0=use_h0,
        seed=20260516,
    )
    assert_cp_splits(cu, H, k.shape[1])
    with torch.inference_mode():
        ref_h, ref_v, ref_ht = run_intracard_direct(k, w, u, gk, h0, cu)
        torch.cuda.synchronize()
        for i in range(STRESS_ITERS):
            cp_h, cp_v, cp_ht = run_intracard_direct(k, w, u, gk, h0, cu)
            torch.cuda.synchronize()
            assert_close(f"iter {i} h", ref_h, cp_h, ratio=RATIO_STRESS)
            assert_close(f"iter {i} v", ref_v, cp_v, ratio=RATIO_STRESS)
            assert_close(f"iter {i} ht", ref_ht, cp_ht, ratio=RATIO_STRESS)


def test_intracard_cp_h0_none_equiv_h0_zeros():
    """h0=None must produce identical ht to h0=zeros (no implicit init)."""
    seq_lens, H = [65536, 4096], 4
    k, w, u, gk, _, cu = make_varlen_inputs(seq_lens, H, use_gk=True, seed=20260501)
    assert_cp_splits(cu, H, k.shape[1])
    h0_zeros = torch.zeros(len(seq_lens), H, K, V, dtype=torch.float32, device=DEVICE)
    with torch.inference_mode():
        _, _, ht_none = run_intracard_direct(
            k,
            w,
            u,
            gk,
            None,
            cu,
            save_new_value=False,
        )
        _, _, ht_zeros = run_intracard_direct(
            k,
            w,
            u,
            gk,
            h0_zeros,
            cu,
            save_new_value=False,
        )
    torch.cuda.synchronize()
    diff = (ht_none.float() - ht_zeros.float()).abs().max().item()
    assert diff < 1e-4, f"h0=None vs h0=zeros diff {diff:.4e}"
