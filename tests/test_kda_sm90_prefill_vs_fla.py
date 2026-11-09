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

# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""Precision tests for SM90 CuTeDSL prefill vs FLA baseline.

Forward-only: the SM90 path has no backward, so only ``o`` and ``ht`` are compared.

NOTE: the SM90 kernel keeps its recurrence state in **VK-transposed** layout
(``[B, H, V, K]``) while FLA uses KV layout (``[B, H, K, V]``).
``initial_state`` is transposed when passed to cuLA, and cuLA's ``ht`` is
transposed back to KV layout before comparison.
"""

import pytest
import torch
from fla.ops.kda.chunk import chunk_kda as fla_chunk_kda
from fla.utils import assert_close, device, get_err_ratio

from cula.kda import flashkda_prefill as cula_kda_prefill

pytestmark = [
    pytest.mark.sm90_only,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]

D = 128

# (B, T, H, with_state) — non-aligned T exercises tail-chunk handling (CHUNK=16).
_DENSE = [
    pytest.param(1, 63, 1, False, marks=pytest.mark.kda_fast, id="B1-T63-H1-tail"),
    pytest.param(1, 256, 2, False, marks=pytest.mark.kda_fast, id="B1-T256-H2-aligned"),
    pytest.param(1, 500, 2, False, marks=pytest.mark.kda_fast, id="B1-T500-H2-tail"),
    pytest.param(1, 512, 2, True, marks=pytest.mark.kda_fast, id="B1-T512-H2-init_state"),
    pytest.param(2, 512, 2, False, marks=pytest.mark.kda_fast, id="B2-T512-H2-stride"),
    pytest.param(1, 1024, 4, True, marks=pytest.mark.kda_fast, id="B1-T1024-H4-init_state"),
    pytest.param(2, 1024, 4, False, marks=pytest.mark.kda_slow, id="B2-T1024-H4"),
]

_VARLEN = [
    pytest.param([0, 15], 2, False, marks=pytest.mark.kda_fast, id="cu[0,15]-H2"),
    pytest.param([0, 256, 500, 1000], 4, False, marks=pytest.mark.kda_slow, id="cu[0,256,500,1000]-H4"),
    pytest.param([0, 15, 100, 300, 1200], 4, True, marks=pytest.mark.kda_slow, id="cu[0,15,100,300,1200]-H4-init"),
]


def _make_inputs(B, T, H, *, with_state, n_state=None, seed=42):
    torch.manual_seed(seed)
    q = torch.rand(B, T, H, D, dtype=torch.bfloat16, device=device)
    k = torch.rand(B, T, H, D, dtype=torch.bfloat16, device=device)
    v = torch.rand(B, T, H, D, dtype=torch.bfloat16, device=device)
    g = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(H, dtype=torch.float32, device=device)
    dt_bias = torch.randn(H * D, dtype=torch.float32, device=device)
    beta = torch.randn(B, T, H, dtype=torch.float32, device=device).sigmoid().to(torch.bfloat16)
    n_state = B if n_state is None else n_state
    h0 = torch.randn(n_state, H, D, D, dtype=torch.float32, device=device) if with_state else None
    return q, k, v, g, beta, A_log, dt_bias, h0


@pytest.mark.parametrize(("B", "T", "H", "with_state"), _DENSE)
def test_prefill_dense_matches_fla(B, T, H, with_state):
    q, k, v, g, beta, A_log, dt_bias, h0 = _make_inputs(B, T, H, with_state=with_state)

    with torch.no_grad():
        ref_o, ref_ht = fla_chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=h0,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=-5.0,
        )

        h0_vk = h0.transpose(-2, -1).contiguous() if h0 is not None else None
        tri_o, tri_ht_vk = cula_kda_prefill(
            q,
            k,
            v,
            g,
            beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=h0_vk,
            output_final_state=True,
            safe_gate=True,
            lower_bound=-5.0,
        )
    tri_ht = tri_ht_vk.transpose(-2, -1)

    assert_close("o", ref_o, tri_o, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@pytest.mark.parametrize(("cu_seqlens", "H", "with_state"), _VARLEN)
def test_prefill_varlen_matches_fla(cu_seqlens, H, with_state):
    total_t = cu_seqlens[-1]
    q, k, v, g, beta, A_log, dt_bias, h0 = _make_inputs(1, total_t, H, with_state=with_state, n_state=len(cu_seqlens) - 1)
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    cu_cpu = cu.cpu()

    with torch.no_grad():
        ref_o, ref_ht = fla_chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=h0,
            cu_seqlens=cu,
            cu_seqlens_cpu=cu_cpu,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=-5.0,
        )

        h0_vk = h0.transpose(-2, -1).contiguous() if h0 is not None else None
        tri_o, tri_ht_vk = cula_kda_prefill(
            q,
            k,
            v,
            g,
            beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=h0_vk,
            cu_seqlens=cu,
            cu_seqlens_cpu=cu_cpu,
            output_final_state=True,
            safe_gate=True,
            lower_bound=-5.0,
        )
    tri_ht = tri_ht_vk.transpose(-2, -1)

    assert_close("o", ref_o, tri_o, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@pytest.mark.kda_fast
def test_prefill_workspace_reuse_across_shapes():
    """Back-to-back calls with different shapes share the grow-only workspace
    arena; earlier shapes must keep matching FLA after the arena has been
    re-carved for larger and smaller shapes (including a tail-chunk one)."""
    shapes = [(1, 512, 2), (1, 1024, 2), (2, 512, 2), (1, 500, 2), (1, 512, 2)]
    for B, T, H in shapes:
        q, k, v, g, beta, A_log, dt_bias, _ = _make_inputs(B, T, H, with_state=False)
        with torch.no_grad():
            ref_o, _ = fla_chunk_kda(
                q,
                k,
                v,
                g,
                beta,
                A_log=A_log,
                dt_bias=dt_bias,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                safe_gate=True,
                lower_bound=-5.0,
            )
            tri_o, _ = cula_kda_prefill(
                q,
                k,
                v,
                g,
                beta,
                A_log=A_log,
                dt_bias=dt_bias,
                output_final_state=True,
                safe_gate=True,
                lower_bound=-5.0,
            )
        assert_close("o", ref_o, tri_o, 0.005)


@pytest.mark.kda_fast
def test_prefill_repeated_tokens_inverse_precision():
    """Near-collinear keys stress the old fp16 Neumann inverse."""
    B, T, H = 1, 16, 1
    torch.manual_seed(321)
    token = torch.linspace(-1.0, 1.0, D, dtype=torch.float32, device=device)
    token = torch.nn.functional.normalize(token, dim=0).to(torch.bfloat16)
    q = token.view(1, 1, 1, D).expand(B, T, H, D).contiguous()
    k = q.clone()
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device)
    # sigmoid(-80) makes the intra-chunk decay nearly one, so consecutive
    # normalized keys produce a strictly-lower L with values close to one.
    g = torch.full_like(q, -80.0)
    beta = torch.full((B, T, H), 0.999, dtype=torch.bfloat16, device=device)
    A_log = torch.zeros(H, dtype=torch.float32, device=device)
    dt_bias = torch.zeros(H, D, dtype=torch.float32, device=device)

    with torch.no_grad():
        ref_o, ref_ht = fla_chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            A_log=A_log,
            dt_bias=dt_bias,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=-5.0,
        )
        tri_o, tri_ht_vk = cula_kda_prefill(
            q,
            k,
            v,
            g,
            beta,
            A_log=A_log,
            dt_bias=dt_bias,
            output_final_state=True,
            safe_gate=True,
            lower_bound=-5.0,
            use_intracard_cp=False,
        )
    tri_ht = tri_ht_vk.transpose(-2, -1)

    o_err_ratio = get_err_ratio(ref_o, tri_o)
    ht_err_ratio = get_err_ratio(ref_ht, tri_ht)
    assert o_err_ratio < 0.005, f"repeated-token output precision regressed: {o_err_ratio=:g}"
    assert ht_err_ratio < 0.01, f"repeated-token final-state precision regressed: {ht_err_ratio=:g}"
