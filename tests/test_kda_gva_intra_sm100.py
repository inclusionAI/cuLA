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

"""Unit tests for SM100 KDA GVA (HV > HQK) support in chunk_kda_fwd_intra.

The SM100 kernels (kda_fwd_intra / kda_fwd_recomp_w_u) now accept:
  * q, k with head-dim ``HQK``
  * v, g, beta with head-dim ``HV`` where ``HV = group_size * HQK`` (group_size >= 1)

This file verifies that the cuLA GVA path produces numerically matching results
compared to the FLA Triton reference, where the FLA reference does not natively
support GVA and therefore receives ``k`` replicated along the head axis to
``HV`` heads. Both uniform-length and varlen layouts are covered, and an
additional degeneracy test asserts that ``HV == HQK`` (group_size == 1) keeps
the non-GVA behaviour untouched.
"""

from __future__ import annotations

import pytest
import torch
from einops import rearrange
from fla.modules.l2norm import l2norm_fwd
from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra as fla_chunk_kda_fwd_intra
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.index import prepare_chunk_indices
from fla.utils import assert_close, device

from cula.kda.chunk_intra import chunk_kda_fwd_intra as cula_chunk_kda_fwd_intra
from cula.utils import prepare_uniform_cu_seqlens

pytestmark = pytest.mark.sm100_only


# =========================================================================
# Helpers
# =========================================================================

def _repeat_head(x: torch.Tensor, group_size: int, head_dim: int = 2) -> torch.Tensor:
    """Replicate ``x`` along the head axis by ``group_size``.

    Mirrors GVA's broadcasting semantics: each QK head is paired with
    ``group_size`` consecutive V heads, so ``k[..., h_qk, :]`` is used by
    ``v[..., h_qk * group_size : (h_qk + 1) * group_size, :]``.
    """
    return x.repeat_interleave(group_size, dim=head_dim).contiguous()


def _make_gva_inputs(
    B: int,
    T: int,
    HQK: int,
    HV: int,
    D: int,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None = None,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
):
    """Construct inputs for chunk_kda_fwd_intra in GVA layout.

    Returns:
        q, k         : (B, T, HQK, D)         dtype
        v            : (B, T, HV,  D)         dtype
        g            : (B, T, HV,  D)         float32, after kda_gate_chunk_cumsum
        beta         : (B, T, HV)             float32 in (0, 1)
        scale        : float
        cu_seqlens   : (N+1,) int32 or None
        chunk_indices: (NT, 2) int32 or None
    """
    assert HV % HQK == 0 and HV >= HQK, f"invalid HV/HQK: {HV}/{HQK}"

    torch.manual_seed(seed)
    scale = D ** (-0.5)

    # QK are in HQK head space; V / gates / beta live in HV space.
    q = torch.randn(B, T, HQK, D, dtype=dtype, device=device)
    k = torch.randn(B, T, HQK, D, dtype=dtype, device=device)
    v = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    g_raw = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    beta = torch.randn(B, T, HV, dtype=torch.float, device=device).sigmoid()

    # l2-normalise q/k so that scale/gate ranges match production use.
    q, _ = l2norm_fwd(q)
    k, _ = l2norm_fwd(k)

    # FLA gate cumsum only supports packed batch (B=1) when cu_seqlens is set.
    if B != 1:
        q, k, v, g_raw, beta = map(
            lambda x: rearrange(x, "b t ... -> 1 (b t) ..."),
            (q, k, v, g_raw, beta),
        )

    # Per-HV gate preprocessing (cumsum inside chunks).
    A_log = torch.randn(HV, dtype=torch.float, device=device)
    dt_bias = torch.randn(HV * D, dtype=torch.float, device=device)

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    )
    g = kda_gate_chunk_cumsum(
        g=g_raw,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=RCP_LN2,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        lower_bound=-5.0,
    )
    return q, k, v, g, beta, scale, cu_seqlens, chunk_indices


def _run_fla_ref(q, k_hqk, v, g, beta, scale, cu_seqlens, chunk_indices, chunk_size, group_size, disable_recompute):
    """Reference: replicate k along head axis to HV, then call FLA intra.

    FLA's chunk_kda_fwd_intra assumes H == HQK == HV (no GVA), so we construct
    the HV-head view of k and q before invoking it.
    """
    k_hv = _repeat_head(k_hqk, group_size)
    q_hv = _repeat_head(q, group_size)
    return fla_chunk_kda_fwd_intra(
        q=q_hv,
        k=k_hv,
        v=v,
        gk=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=True,
        disable_recompute=disable_recompute,
    )


def _run_cula_gva(q, k, v, g, beta, scale, cu_seqlens, chunk_indices, chunk_size, disable_recompute):
    return cula_chunk_kda_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=True,
        disable_recompute=disable_recompute,
    )


def _assert_intra_outputs_match(ref, tri, disable_recompute: bool) -> None:
    """Compare cuLA vs FLA on user-visible intra outputs.

    We intentionally skip ``Aqk``: the cuLA SM100 fused kernel does not
    materialise every off-diagonal slot that FLA's multi-kernel path writes,
    and the FLA reference can contain NaNs in unused ``Aqk`` entries.  The
    downstream tensors ``w`` / ``u`` / ``kg`` (and ``Akk``) are the meaningful
    correctness signals and match the benchmark's comparison strategy.
    """
    w_r, u_r, qg_r, kg_r, _Aqk_r, Akk_r = ref
    w_c, u_c, qg_c, kg_c, _Aqk_c, Akk_c = tri

    assert Akk_c.shape == Akk_r.shape, (Akk_c.shape, Akk_r.shape)
    assert w_c.shape == w_r.shape, (w_c.shape, w_r.shape)
    assert u_c.shape == u_r.shape, (u_c.shape, u_r.shape)
    assert kg_c.shape == kg_r.shape, (kg_c.shape, kg_r.shape)

    assert_close("Akk", Akk_r, Akk_c, 0.008)
    assert_close("w", w_r, w_c, 0.008)
    assert_close("u", u_r, u_c, 0.008)
    assert_close("kg", kg_r, kg_c, 0.005)

    if disable_recompute:
        assert qg_c is not None and qg_r is not None
        assert qg_c.shape == qg_r.shape, (qg_c.shape, qg_r.shape)
        assert_close("qg", qg_r, qg_c, 0.005)
    else:
        assert qg_c is None, "cuLA must not materialise qg when disable_recompute=False"


# =========================================================================
# Uniform-length tests
# =========================================================================

@pytest.mark.parametrize("disable_recompute", [False, True], ids=["recomp", "no_recomp"])
@pytest.mark.parametrize(
    ("B", "T", "HQK", "group_size", "D"),
    [
        pytest.param(*cfg, id="B{}-T{}-HQK{}-gs{}-D{}".format(*cfg))
        for cfg in [
            # group_size == 2: classic GVA 2:1
            (1, 256, 2, 2, 128),
            (2, 512, 4, 2, 128),
            # group_size == 4: wider grouping
            (1, 1024, 2, 4, 128),
            (2, 1024, 4, 4, 128),
            # Non-multiple-of-BT sequence length to stress boundary handling.
            (1, 500, 2, 2, 128),
            (1, 1000, 4, 2, 128),
        ]
    ],
)
def test_gva_intra_uniform(B, T, HQK, group_size, D, disable_recompute):
    """cuLA GVA path must match FLA(k-replicated-to-HV) for uniform seqlens."""
    HV = HQK * group_size
    chunk_size = 64

    cu_seqlens = prepare_uniform_cu_seqlens(B, T, torch.device(device), torch.int32)
    q, k, v, g, beta, scale, cu_seqlens, chunk_indices = _make_gva_inputs(
        B=B, T=T, HQK=HQK, HV=HV, D=D, chunk_size=chunk_size, cu_seqlens=cu_seqlens,
    )

    # cuLA GVA path (k in HQK head space).
    w_c, u_c, qg_c, kg_c, Aqk_c, Akk_c = _run_cula_gva(
        q, k, v, g, beta, scale, cu_seqlens, chunk_indices, chunk_size, disable_recompute,
    )

    # FLA reference (k replicated to HV).
    w_r, u_r, qg_r, kg_r, Aqk_r, Akk_r = _run_fla_ref(
        q, k, v, g, beta, scale, cu_seqlens, chunk_indices, chunk_size, group_size, disable_recompute,
    )

    _assert_intra_outputs_match(
        (w_r, u_r, qg_r, kg_r, Aqk_r, Akk_r),
        (w_c, u_c, qg_c, kg_c, Aqk_c, Akk_c),
        disable_recompute,
    )


# =========================================================================
# Varlen tests
# =========================================================================

@pytest.mark.parametrize("disable_recompute", [False, True], ids=["recomp", "no_recomp"])
@pytest.mark.parametrize(
    ("HQK", "group_size", "D", "cu_seqlens"),
    [
        pytest.param(*cfg, id="HQK{}-gs{}-D{}-ns{}".format(cfg[0], cfg[1], cfg[2], len(cfg[3]) - 1))
        for cfg in [
            (2, 2, 128, [0, 256, 500, 1000]),
            (4, 2, 128, [0, 100, 300, 1200, 2000]),
            (2, 4, 128, [0, 15, 100, 300, 1200, 2048]),
            # Simulated realistic trace.
            (
                4, 2, 128,
                [0, 247, 699, 982, 1688, 1985, 2383, 3081, 3526, 3973, 4096],
            ),
        ]
    ],
)
def test_gva_intra_varlen(HQK, group_size, D, cu_seqlens, disable_recompute):
    """GVA correctness under variable-length (packed) inputs."""
    HV = HQK * group_size
    chunk_size = 64

    cu_seqlens_t = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    T = int(cu_seqlens_t[-1].item())
    # Packed layout uses B=1 and a flat time axis.
    q, k, v, g, beta, scale, cu_seqlens_t, chunk_indices = _make_gva_inputs(
        B=1, T=T, HQK=HQK, HV=HV, D=D, chunk_size=chunk_size, cu_seqlens=cu_seqlens_t,
    )

    w_c, u_c, qg_c, kg_c, Aqk_c, Akk_c = _run_cula_gva(
        q, k, v, g, beta, scale, cu_seqlens_t, chunk_indices, chunk_size, disable_recompute,
    )
    w_r, u_r, qg_r, kg_r, Aqk_r, Akk_r = _run_fla_ref(
        q, k, v, g, beta, scale, cu_seqlens_t, chunk_indices, chunk_size, group_size, disable_recompute,
    )

    _assert_intra_outputs_match(
        (w_r, u_r, qg_r, kg_r, Aqk_r, Akk_r),
        (w_c, u_c, qg_c, kg_c, Aqk_c, Akk_c),
        disable_recompute,
    )


# =========================================================================
# Degeneracy: HV == HQK must match the non-GVA (same-shape) reference
# =========================================================================

@pytest.mark.parametrize("disable_recompute", [False, True], ids=["recomp", "no_recomp"])
@pytest.mark.parametrize(
    ("B", "T", "H", "D"),
    [
        pytest.param(*cfg, id="B{}-T{}-H{}-D{}".format(*cfg))
        for cfg in [
            (1, 512, 4, 128),
            (2, 1024, 4, 128),
        ]
    ],
)
def test_gva_intra_degenerate_equals_non_gva(B, T, H, D, disable_recompute):
    """When HV == HQK, the GVA code path must be byte-for-byte equivalent
    to the non-GVA path that existed before this change.

    We do not have a separate "non-GVA" entrypoint, but we can assert the
    cuLA path matches FLA with *no* head replication (group_size=1), which
    exercises the ``HV == HQK`` fast-path inside the new kernels.
    """
    chunk_size = 64
    cu_seqlens = prepare_uniform_cu_seqlens(B, T, torch.device(device), torch.int32)
    q, k, v, g, beta, scale, cu_seqlens, chunk_indices = _make_gva_inputs(
        B=B, T=T, HQK=H, HV=H, D=D, chunk_size=chunk_size, cu_seqlens=cu_seqlens,
    )

    w_c, u_c, qg_c, kg_c, Aqk_c, Akk_c = _run_cula_gva(
        q, k, v, g, beta, scale, cu_seqlens, chunk_indices, chunk_size, disable_recompute,
    )
    # group_size=1 → no replication; identical input shape to cuLA.
    w_r, u_r, qg_r, kg_r, Aqk_r, Akk_r = fla_chunk_kda_fwd_intra(
        q=q, k=k, v=v, gk=g, beta=beta, scale=scale,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size, chunk_indices=chunk_indices,
        safe_gate=True, disable_recompute=disable_recompute,
    )

    _assert_intra_outputs_match(
        (w_r, u_r, qg_r, kg_r, Aqk_r, Akk_r),
        (w_c, u_c, qg_c, kg_c, Aqk_c, Akk_c),
        disable_recompute,
    )


# =========================================================================
# Shape / contract sanity checks (run even without a reference)
# =========================================================================

@pytest.mark.parametrize("group_size", [1, 2, 4])
def test_gva_intra_output_shapes(group_size):
    """All outputs of chunk_kda_fwd_intra must live in HV-head space."""
    B, T, HQK, D = 1, 256, 2, 128
    HV = HQK * group_size
    chunk_size = 64
    cu_seqlens = prepare_uniform_cu_seqlens(B, T, torch.device(device), torch.int32)
    q, k, v, g, beta, scale, cu_seqlens, chunk_indices = _make_gva_inputs(
        B=B, T=T, HQK=HQK, HV=HV, D=D, chunk_size=chunk_size, cu_seqlens=cu_seqlens,
    )
    w, u, qg, kg, Aqk, Akk = _run_cula_gva(
        q, k, v, g, beta, scale, cu_seqlens, chunk_indices, chunk_size, disable_recompute=True,
    )

    assert Aqk.shape == (B, T, HV, chunk_size), Aqk.shape
    assert Akk.shape == (B, T, HV, chunk_size), Akk.shape
    assert w.shape == (B, T, HV, D), w.shape
    assert u.shape == (B, T, HV, D), u.shape
    assert kg.shape == (B, T, HV, D), kg.shape
    assert qg is not None and qg.shape == (B, T, HV, D), (None if qg is None else qg.shape)


# =========================================================================
# Negative / assertion tests
# =========================================================================

def test_gva_intra_rejects_non_multiple_ratio():
    """HV must be a positive integer multiple of HQK."""
    B, T, HQK, HV, D = 1, 128, 3, 5, 128  # 5 % 3 != 0
    chunk_size = 64
    cu_seqlens = prepare_uniform_cu_seqlens(B, T, torch.device(device), torch.int32)
    # We intentionally do not use _make_gva_inputs because the assert fires
    # before kernel launch on the python side.
    dtype = torch.bfloat16
    q = torch.randn(B, T, HQK, D, dtype=dtype, device=device)
    k = torch.randn(B, T, HQK, D, dtype=dtype, device=device)
    v = torch.randn(B, T, HV, D, dtype=dtype, device=device)
    g = torch.randn(B, T, HV, D, dtype=torch.float, device=device)
    beta = torch.randn(B, T, HV, dtype=torch.float, device=device).sigmoid()

    with pytest.raises((AssertionError, RuntimeError), match=r"multiple|h_v"):
        cula_chunk_kda_fwd_intra(
            q=q, k=k, v=v, gk=g, beta=beta, scale=D ** -0.5,
            cu_seqlens=cu_seqlens, chunk_size=chunk_size,
            safe_gate=True, disable_recompute=False,
        )
