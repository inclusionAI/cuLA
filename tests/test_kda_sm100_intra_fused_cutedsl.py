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
from cula.ops.kda.sm100.intra_fused import (
    chunk_kda_fwd_intra_sm100_equal,
    chunk_kda_fwd_intra_sm100_from_gk,
    chunk_kda_fwd_intra_sm100_varlen,
)
from cula.ops.kda.sm100.recompute_wu import recompute_w_u_fwd


def _requires_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 CUDA device required")


def _chunk_fp64_oracle(q, k, v, gk, beta, scale, chunk_size):
    """Small-shape oracle for ranking csrc and CuTeDSL rounding error."""
    batch, seqlen, heads, dim = q.shape
    chunks = seqlen // chunk_size

    def chunked(x):
        return x.double().view(batch, chunks, chunk_size, heads, -1).permute(0, 1, 3, 2, 4)

    q_c = chunked(q)
    k_c = chunked(k)
    v_c = chunked(v)
    g_c = chunked(gk)
    beta_c = beta.double().view(batch, chunks, chunk_size, heads).permute(0, 1, 3, 2)
    exp_g = torch.exp2(g_c)
    q_g = q_c * exp_g
    k_g = k_c * exp_g
    k_inv_g = k_c / exp_g
    aqk = torch.tril(torch.matmul(q_g, k_inv_g.transpose(-1, -2)) * scale)
    eye = torch.eye(chunk_size, dtype=torch.float64, device=q.device)
    mat = eye + torch.tril(torch.matmul(k_g, k_inv_g.transpose(-1, -2)), diagonal=-1) * beta_c.unsqueeze(-1)
    akk = torch.linalg.inv(mat)
    w = torch.matmul(akk, (k_c * beta_c.unsqueeze(-1)) * exp_g)
    u = torch.matmul(akk, v_c * beta_c.unsqueeze(-1))

    def unchunk(x):
        return x.permute(0, 1, 3, 2, 4).reshape(batch, seqlen, heads, x.shape[-1])

    return unchunk(aqk), unchunk(akk), unchunk(w), unchunk(u)


def _rel_rmse_fp64(actual, oracle):
    diff = actual.double() - oracle
    return torch.sqrt(torch.mean(diff.square()) / torch.mean(oracle.square())).item()


def test_csrc_boundary_port_intra_wu_strict_fp64_precision():
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

    Aqk, Akk = chunk_kda_fwd_intra_sm100_from_gk(
        q=q,
        k=k,
        gk=gk,
        beta=beta,
        scale=scale,
    )
    w, u, _, kg = recompute_w_u_fwd(k, k, beta, Akk, gk)
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
    assert torch.equal(Aqk, Aqk_ref)

    aqk_oracle, akk_oracle, w_oracle, u_oracle = _chunk_fp64_oracle(q, k, k, gk, beta, scale, chunk_size)
    precision_rows = []
    for name, candidate, baseline, oracle in (
        ("Aqk", Aqk, Aqk_ref, aqk_oracle),
        ("Akk", Akk, Akk_ref, akk_oracle),
        ("w", w, w_ref, w_oracle),
        ("u", u, u_ref, u_oracle),
    ):
        candidate_error = _rel_rmse_fp64(candidate, oracle)
        baseline_error = _rel_rmse_fp64(baseline, oracle)
        precision_rows.append((name, candidate_error, baseline_error))
    regressions = [row for row in precision_rows if row[1] > row[2] * (1.0 + 1e-6)]
    assert not regressions, "; ".join(
        f"{name}: CuTeDSL={candidate_error:.6e}, csrc={baseline_error:.6e}"
        for name, candidate_error, baseline_error in precision_rows
    )


def test_uniform_varlen_uses_equal_fp16_path_bitwise():
    _requires_sm100()
    torch.manual_seed(7)
    device = torch.device("cuda")
    batch, seqlen, heads, dim = 2, 256, 4, 128
    q = F.normalize(torch.randn(batch, seqlen, heads, dim, device=device).float(), dim=-1).bfloat16()
    k = F.normalize(torch.randn(batch, seqlen, heads, dim, device=device).float(), dim=-1).bfloat16()
    g = torch.randn(batch, seqlen, heads, dim, device=device, dtype=torch.bfloat16)
    beta = torch.randn(batch, seqlen, heads, device=device).sigmoid().bfloat16()
    a_log = torch.full((heads,), -4.0, device=device)
    dt_bias = torch.zeros(heads * dim, device=device)
    kwargs = dict(
        A_log=a_log,
        dt_bias=dt_bias,
        safe_gate=True,
        lower_bound=-5.0,
        fp32_akk_inv=True,
        kscaled_fp16=True,
    )
    equal = chunk_kda_fwd_intra_sm100_equal(q=q, k=k, g=g, beta=beta, **kwargs)
    packed = chunk_kda_fwd_intra_sm100_varlen(
        q=q.flatten(0, 1).unsqueeze(0),
        k=k.flatten(0, 1).unsqueeze(0),
        g=g.flatten(0, 1).unsqueeze(0),
        beta=beta.flatten(0, 1).unsqueeze(0),
        cu_seqlens=torch.tensor([0, seqlen, 2 * seqlen], dtype=torch.int32, device=device),
        seq_lens=[seqlen, seqlen],
        **kwargs,
    )
    for equal_tensor, packed_tensor in zip(equal, packed, strict=True):
        expected = equal_tensor.flatten(0, 1).unsqueeze(0)
        torch.testing.assert_close(packed_tensor, expected, rtol=0, atol=0, equal_nan=True)
