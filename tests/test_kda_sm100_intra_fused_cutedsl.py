# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils.constant import RCP_LN2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cula.kda.chunk_intra import chunk_kda_fwd_intra as csrc_chunk_kda_fwd_intra
from cula.ops.kda.sm100.intra_fused import chunk_kda_fwd_intra_sm100_equal
from cula.ops.kda.sm100.recompute_wu import recompute_w_u_from_preprocessed


def _requires_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 CUDA device required")


def test_fused_intra_and_preprocessed_wu_match_csrc():
    _requires_sm100()
    torch.manual_seed(2)
    device = torch.device("cuda")
    batch, seqlen, heads, dim, chunk_size = 1, 256, 4, 128, 64
    scale = dim**-0.5
    lower_bound = -5.0

    q = F.normalize(torch.randn(batch, seqlen, heads, dim, device=device).float(), dim=-1).bfloat16()
    k = F.normalize(torch.randn(batch, seqlen, heads, dim, device=device).float(), dim=-1).bfloat16()
    g = torch.randn(batch, seqlen, heads, dim, device=device, dtype=torch.bfloat16)
    beta = torch.randn(batch, seqlen, heads, device=device).sigmoid().bfloat16()
    A_log = torch.randn(heads, device=device)
    dt_bias = torch.randn(heads * dim, device=device)

    gk = kda_gate_chunk_cumsum(
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=RCP_LN2,
        chunk_size=chunk_size,
        lower_bound=lower_bound,
    )
    w_ref, u_ref, _, kg_ref, Aqk_ref, Akk_ref = csrc_chunk_kda_fwd_intra(
        q=q,
        k=k,
        v=k,
        gk=gk,
        beta=beta,
        scale=scale,
        chunk_size=chunk_size,
        safe_gate=True,
    )

    k_scaled, kg, q_scaled, _, Aqk, Akk = chunk_kda_fwd_intra_sm100_equal(
        q=q,
        k=k,
        g=g,
        beta=beta,
        A_log=A_log,
        scale=scale,
        dt_bias=dt_bias,
        safe_gate=True,
        lower_bound=lower_bound,
        fp32_akk_inv=True,
    )
    w, u = recompute_w_u_from_preprocessed(k_scaled, k, beta, Akk)

    exp_gk = torch.exp2(gk)
    torch.testing.assert_close(k_scaled, (k.float() * exp_gk).bfloat16(), rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(q_scaled, (q.float() * exp_gk).bfloat16(), rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(kg, kg_ref, rtol=1e-2, atol=2e-3)

    row = torch.arange(seqlen, device=device) % chunk_size
    col = torch.arange(chunk_size, device=device)
    lower = (col[None, :] <= row[:, None]).view(1, seqlen, 1, chunk_size).expand_as(Akk)
    torch.testing.assert_close(Aqk[lower], Aqk_ref[lower], rtol=1e-2, atol=2e-3)
    torch.testing.assert_close(Akk[lower], Akk_ref[lower], rtol=1e-2, atol=2e-3)
    torch.testing.assert_close(Aqk[~lower], torch.zeros_like(Aqk[~lower]), rtol=0, atol=0)
    torch.testing.assert_close(Akk[~lower], torch.zeros_like(Akk[~lower]), rtol=0, atol=0)
    torch.testing.assert_close(w, w_ref, rtol=1e-2, atol=2e-3)
    torch.testing.assert_close(u, u_ref, rtol=1e-2, atol=2e-3)
