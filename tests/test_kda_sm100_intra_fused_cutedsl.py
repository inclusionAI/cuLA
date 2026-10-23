# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

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
    batch, seqlen, heads, _ = q.shape
    chunks = seqlen // chunk_size

    def chunked(x):
        return x.double().view(batch, chunks, chunk_size, heads, -1).permute(0, 1, 3, 2, 4)

    q_c, k_c, v_c, g_c = (chunked(x) for x in (q, k, v, gk))
    beta_c = beta.double().view(batch, chunks, chunk_size, heads).permute(0, 1, 3, 2)
    exp_g = torch.exp2(g_c)
    aqk = torch.tril(torch.matmul(q_c * exp_g, (k_c / exp_g).transpose(-1, -2)) * scale)
    eye = torch.eye(chunk_size, dtype=torch.float64, device=q.device)
    mat = eye + torch.tril(torch.matmul(k_c * exp_g, (k_c / exp_g).transpose(-1, -2)), diagonal=-1) * beta_c.unsqueeze(-1)
    akk = torch.linalg.inv(mat)
    w = torch.matmul(akk, k_c * beta_c.unsqueeze(-1) * exp_g)
    u = torch.matmul(akk, v_c * beta_c.unsqueeze(-1))

    def unchunk(x):
        return x.permute(0, 1, 3, 2, 4).reshape(batch, seqlen, heads, x.shape[-1])

    return tuple(unchunk(x) for x in (aqk, akk, w, u))


def _rel_rmse(actual, reference):
    diff = actual.double() - reference
    return torch.sqrt(torch.mean(diff.square()) / torch.mean(reference.square())).item()


@pytest.mark.parametrize("beta_dtype", [torch.bfloat16, torch.float32])
def test_exact_port_preserves_csrc_precision_contract(beta_dtype):
    _requires_sm100()
    torch.manual_seed(2)
    batch, seqlen, heads, dim, chunk_size = 1, 256, 4, 128, 64
    scale = dim**-0.5
    q = F.normalize(torch.randn(batch, seqlen, heads, dim, device="cuda").float(), dim=-1).bfloat16()
    k = F.normalize(torch.randn(batch, seqlen, heads, dim, device="cuda").float(), dim=-1).bfloat16()
    v = torch.randn(batch, seqlen, heads, dim, device="cuda", dtype=torch.bfloat16)
    gk = torch.randn(batch, seqlen, heads, dim, device="cuda", dtype=torch.float32) * 0.02
    beta = torch.rand(batch, seqlen, heads, device="cuda", dtype=beta_dtype)

    Aqk, Akk = chunk_kda_fwd_intra_sm100_from_gk(q, k, gk, beta, scale=scale)
    w, u, _, _ = recompute_w_u_fwd(k, v, beta, Akk, gk)
    aqk_ref, akk_ref, w_ref, u_ref = _chunk_fp64_oracle(q, k, v, gk, beta, scale, chunk_size)
    # Aqk is emitted as bf16 after the native-compatible MMA reduction. The
    # bound is fixed just above the removed implementation's 1.818e-3 error
    # for this deterministic vector.
    assert _rel_rmse(Aqk, aqk_ref) < 2e-3
    # The removed TF32 inverse baseline measured 5.55e-4 on the same vector.
    assert _rel_rmse(Akk, akk_ref) < 6e-4
    assert _rel_rmse(w, w_ref) < 2e-3
    assert _rel_rmse(u, u_ref) < 2e-3


def test_exact_port_isolates_concurrent_streams():
    _requires_sm100()
    inputs = []
    references = []
    for seed in (31, 37):
        torch.manual_seed(seed)
        q = torch.randn(1, 256, 4, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        gk = torch.randn(1, 256, 4, 128, device="cuda", dtype=torch.float32) * 0.02
        beta = torch.rand(1, 256, 4, device="cuda", dtype=torch.float32)
        args = (q, k, gk, beta)
        inputs.append(args)
        references.append(chunk_kda_fwd_intra_sm100_from_gk(*args))

    torch.cuda.synchronize()
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    outputs = []
    for stream, args in zip(streams, inputs, strict=True):
        with torch.cuda.stream(stream):
            outputs.append(chunk_kda_fwd_intra_sm100_from_gk(*args))
    for stream in streams:
        stream.synchronize()

    assert outputs[0][0].data_ptr() != outputs[1][0].data_ptr()
    assert outputs[0][1].data_ptr() != outputs[1][1].data_ptr()
    for actual, expected in zip(outputs, references, strict=True):
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])


def test_uniform_varlen_uses_equal_fp16_path_bitwise():
    _requires_sm100()
    torch.manual_seed(7)
    batch, seqlen, heads, dim = 2, 256, 4, 128
    q = F.normalize(torch.randn(batch, seqlen, heads, dim, device="cuda").float(), dim=-1).bfloat16()
    k = F.normalize(torch.randn(batch, seqlen, heads, dim, device="cuda").float(), dim=-1).bfloat16()
    g = torch.randn(batch, seqlen, heads, dim, device="cuda", dtype=torch.bfloat16)
    beta = torch.rand(batch, seqlen, heads, device="cuda", dtype=torch.bfloat16)
    kwargs = dict(
        A_log=torch.full((heads,), -4.0, device="cuda"),
        dt_bias=torch.zeros(heads * dim, device="cuda"),
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
        cu_seqlens=torch.tensor([0, seqlen, 2 * seqlen], dtype=torch.int32, device="cuda"),
        seq_lens=[seqlen, seqlen],
        **kwargs,
    )
    for equal_tensor, packed_tensor in zip(equal, packed, strict=True):
        expected = equal_tensor.flatten(0, 1).unsqueeze(0)
        torch.testing.assert_close(packed_tensor, expected, rtol=0, atol=0, equal_nan=True)
