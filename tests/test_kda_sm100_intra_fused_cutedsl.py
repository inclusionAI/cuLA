# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.constant import RCP_LN2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cula.kda.chunk_intra import chunk_kda_fwd_intra as csrc_chunk_kda_fwd_intra
from cula.ops.kda.sm100.intra_fused import (
    chunk_kda_fwd_intra_sm100_equal,
    chunk_kda_fwd_intra_sm100_from_gk,
    chunk_kda_fwd_intra_sm100_varlen,
)
from cula.ops.kda.sm100.recompute_wu import recompute_w_u_fwd

pytestmark = pytest.mark.cula_slow


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


@pytest.mark.parametrize("beta_dtype", [torch.bfloat16, torch.float32])
def test_csrc_boundary_port_intra_wu_strict_fp64_precision(beta_dtype):
    _requires_sm100()
    torch.manual_seed(2)
    device = torch.device("cuda")
    batch, seqlen, heads, dim, chunk_size = 1, 256, 4, 128, 64
    scale = dim**-0.5
    lower_bound = -5.0

    q = F.normalize(torch.randn(batch, seqlen, heads, dim, device=device).float(), dim=-1).bfloat16()
    k = F.normalize(torch.randn(batch, seqlen, heads, dim, device=device).float(), dim=-1).bfloat16()
    g = torch.randn(batch, seqlen, heads, dim, device=device, dtype=torch.bfloat16)
    beta = torch.randn(batch, seqlen, heads, device=device).sigmoid().to(beta_dtype)
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
        fp32_akk_inv=True,
    )
    w, u, _, kg = recompute_w_u_fwd(k, k, beta, Akk, gk)
    assert torch.equal(kg, kg_ref), f"kg differs bitwise: max_abs={(kg.float() - kg_ref.float()).abs().max().item()}"

    row = torch.arange(seqlen, device=device) % chunk_size
    col = torch.arange(chunk_size, device=device)
    lower = (col[None, :] <= row[:, None]).view(1, seqlen, 1, chunk_size).expand_as(Akk)
    assert torch.equal(Aqk, Aqk_ref), f"Aqk differs bitwise: max_abs={(Aqk.float() - Aqk_ref.float()).abs().max().item()}"
    assert torch.equal(Akk, Akk_ref), f"Akk differs bitwise: max_abs={(Akk.float() - Akk_ref.float()).abs().max().item()}"
    torch.testing.assert_close(Aqk[~lower], torch.zeros_like(Aqk[~lower]), rtol=0, atol=0)
    torch.testing.assert_close(Akk[~lower], torch.zeros_like(Akk[~lower]), rtol=0, atol=0)
    assert torch.equal(w, w_ref), f"w differs bitwise: max_abs={(w.float() - w_ref.float()).abs().max().item()}"
    assert torch.equal(u, u_ref), f"u differs bitwise: max_abs={(u.float() - u_ref.float()).abs().max().item()}"

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


@pytest.mark.parametrize("beta_dtype", [torch.bfloat16, torch.float32])
def test_csrc_boundary_port_intra_wu_larger_shape_bitwise(beta_dtype):
    _requires_sm100()
    torch.manual_seed(11)
    device = torch.device("cuda")
    batch, seqlen, heads, dim, chunk_size = 2, 512, 8, 128, 64
    scale = dim**-0.5

    q = F.normalize(torch.randn(batch, seqlen, heads, dim, device=device).float(), dim=-1).bfloat16()
    k = F.normalize(torch.randn(batch, seqlen, heads, dim, device=device).float(), dim=-1).bfloat16()
    gk = torch.randn(batch, seqlen, heads, dim, device=device, dtype=torch.float32) * 0.02
    beta = torch.randn(batch, seqlen, heads, device=device).sigmoid().to(beta_dtype)

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
        fp32_akk_inv=True,
    )
    w, u, _, kg = recompute_w_u_fwd(k, k, beta, Akk, gk)

    for name, actual, expected in (
        ("Aqk", Aqk, Aqk_ref),
        ("Akk", Akk, Akk_ref),
        ("kg", kg, kg_ref),
        ("w", w, w_ref),
        ("u", u, u_ref),
    ):
        assert torch.equal(actual, expected), (
            f"{name} differs bitwise: max_abs={(actual.float() - expected.float()).abs().max().item()}"
        )

    # The csrc-boundary kernel must overwrite the complete Aqk/Akk output,
    # including the causal upper triangle; correctness cannot depend on the
    # initial contents of the cached output buffers.
    Aqk.fill_(float("nan"))
    Akk.fill_(float("nan"))
    Aqk_repeat, Akk_repeat = chunk_kda_fwd_intra_sm100_from_gk(
        q=q,
        k=k,
        gk=gk,
        beta=beta,
        scale=scale,
        fp32_akk_inv=True,
    )
    assert Aqk_repeat.data_ptr() != Aqk.data_ptr()
    assert Akk_repeat.data_ptr() != Akk.data_ptr()
    assert torch.equal(Aqk_repeat, Aqk_ref)
    assert torch.equal(Akk_repeat, Akk_ref)


@pytest.mark.parametrize("beta_dtype", [torch.bfloat16, torch.float32])
def test_csrc_boundary_port_intra_wu_gva_bitwise(beta_dtype):
    _requires_sm100()
    torch.manual_seed(17)
    device = torch.device("cuda")
    batch, seqlen, h_qk, h_v, dim, chunk_size = 1, 256, 2, 4, 128, 64
    scale = dim**-0.5

    q = F.normalize(torch.randn(batch, seqlen, h_qk, dim, device=device).float(), dim=-1).bfloat16()
    k = F.normalize(torch.randn(batch, seqlen, h_qk, dim, device=device).float(), dim=-1).bfloat16()
    v = torch.randn(batch, seqlen, h_v, dim, device=device, dtype=torch.bfloat16)
    gk = torch.randn(batch, seqlen, h_v, dim, device=device, dtype=torch.float32) * 0.02
    beta = torch.randn(batch, seqlen, h_v, device=device).sigmoid().to(beta_dtype)

    w_ref, u_ref, qg_ref, kg_ref, Aqk_ref, Akk_ref = csrc_chunk_kda_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=gk,
        beta=beta,
        scale=scale,
        chunk_size=chunk_size,
        safe_gate=True,
        disable_recompute=True,
    )
    Aqk, Akk = chunk_kda_fwd_intra_sm100_from_gk(
        q=q,
        k=k,
        gk=gk,
        beta=beta,
        scale=scale,
        fp32_akk_inv=True,
    )
    w, u, qg, kg = recompute_w_u_fwd(k, v, beta, Akk, gk, q=q)

    for name, actual, expected in (
        ("Aqk", Aqk, Aqk_ref),
        ("Akk", Akk, Akk_ref),
        ("kg", kg, kg_ref),
        ("qg", qg, qg_ref),
        ("w", w, w_ref),
        ("u", u, u_ref),
    ):
        assert torch.equal(actual, expected), (
            f"{name} differs bitwise: max_abs={(actual.float() - expected.float()).abs().max().item()}"
        )


@pytest.mark.parametrize("beta_dtype", [torch.bfloat16, torch.float32])
def test_csrc_boundary_port_intra_wu_gva_partial_varlen_bitwise(beta_dtype):
    _requires_sm100()
    torch.manual_seed(23)
    device = torch.device("cuda")
    cu_offsets = [0, 63, 193]
    total_tokens, h_qk, h_v, dim, chunk_size = cu_offsets[-1], 2, 4, 128, 64
    scale = dim**-0.5

    q = F.normalize(torch.randn(1, total_tokens, h_qk, dim, device=device).float(), dim=-1).bfloat16()
    k = F.normalize(torch.randn(1, total_tokens, h_qk, dim, device=device).float(), dim=-1).bfloat16()
    v = torch.randn(1, total_tokens, h_v, dim, device=device, dtype=torch.bfloat16)
    gk = torch.randn(1, total_tokens, h_v, dim, device=device, dtype=torch.float32) * 0.02
    beta = torch.randn(1, total_tokens, h_v, device=device).sigmoid().to(beta_dtype)
    cu_seqlens = torch.tensor(cu_offsets, dtype=torch.int32, device=device)
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    w_ref, u_ref, qg_ref, kg_ref, Aqk_ref, Akk_ref = csrc_chunk_kda_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=gk,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=True,
        disable_recompute=True,
    )
    Aqk, Akk = chunk_kda_fwd_intra_sm100_from_gk(
        q=q,
        k=k,
        gk=gk,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        fp32_akk_inv=True,
    )
    w, u, qg, kg = recompute_w_u_fwd(
        k,
        v,
        beta,
        Akk,
        gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        q=q,
    )

    for name, actual, expected in (
        ("Aqk", Aqk, Aqk_ref),
        ("Akk", Akk, Akk_ref),
        ("kg", kg, kg_ref),
        ("qg", qg, qg_ref),
        ("w", w, w_ref),
        ("u", u, u_ref),
    ):
        assert torch.equal(actual, expected), (
            f"{name} differs bitwise: max_abs={(actual.float() - expected.float()).abs().max().item()}"
        )


def test_csrc_boundary_port_intra_multistream_isolation():
    _requires_sm100()
    device = torch.device("cuda")
    batch, seqlen, heads, dim = 1, 256, 4, 128
    inputs = []
    refs = []
    for seed in (31, 37):
        torch.manual_seed(seed)
        q = F.normalize(torch.randn(batch, seqlen, heads, dim, device=device).float(), dim=-1).bfloat16()
        k = F.normalize(torch.randn(batch, seqlen, heads, dim, device=device).float(), dim=-1).bfloat16()
        gk = torch.randn(batch, seqlen, heads, dim, device=device, dtype=torch.float32) * 0.02
        beta = torch.rand(batch, seqlen, heads, device=device, dtype=torch.float32)
        inputs.append((q, k, gk, beta))
        refs.append(
            csrc_chunk_kda_fwd_intra(
                q=q,
                k=k,
                v=k,
                gk=gk,
                beta=beta,
                scale=dim**-0.5,
                safe_gate=True,
            )[-2:]
        )

    # Compile before the concurrent launch so this test exercises only runtime
    # stream isolation, not compiler initialization.
    chunk_kda_fwd_intra_sm100_from_gk(*inputs[0], scale=dim**-0.5)
    torch.cuda.synchronize()
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    outputs = []
    for stream, args in zip(streams, inputs, strict=True):
        with torch.cuda.stream(stream):
            outputs.append(chunk_kda_fwd_intra_sm100_from_gk(*args, scale=dim**-0.5))
    torch.cuda.synchronize()

    assert outputs[0][0].data_ptr() != outputs[1][0].data_ptr()
    assert outputs[0][1].data_ptr() != outputs[1][1].data_ptr()
    for (Aqk, Akk), (Aqk_ref, Akk_ref) in zip(outputs, refs, strict=True):
        assert torch.equal(Aqk, Aqk_ref)
        assert torch.equal(Akk, Akk_ref)


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
