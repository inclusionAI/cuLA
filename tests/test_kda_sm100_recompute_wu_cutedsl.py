# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from cula.ops.kda.sm100.recompute_wu import recompute_w_u_from_preprocessed, recompute_w_u_fwd

pytestmark = pytest.mark.cula_slow


def _requires_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 CUDA device required")


@pytest.mark.parametrize("beta_dtype", [torch.float32, torch.bfloat16])
def test_recompute_wu_repeated_calls_are_bitwise_deterministic(beta_dtype):
    _requires_sm100()
    torch.manual_seed(5)
    batch, seqlen, h_qk, h_v, dim, chunk_size = 2, 63, 2, 4, 128, 64
    k = torch.randn(batch, seqlen, h_qk, dim, device="cuda", dtype=torch.bfloat16) * 0.1
    q = torch.randn_like(k) * 0.1
    v = torch.randn(batch, seqlen, h_v, dim, device="cuda", dtype=torch.bfloat16) * 0.1
    beta = torch.rand(batch, seqlen, h_v, device="cuda", dtype=beta_dtype)
    gk = torch.randn(batch, seqlen, h_v, dim, device="cuda", dtype=torch.float32) * 0.02
    A = torch.randn(batch, seqlen, h_v, chunk_size, device="cuda", dtype=torch.bfloat16) * 0.02

    first = recompute_w_u_fwd(k, v, beta, A, gk, q=q)
    second = recompute_w_u_fwd(k, v, beta, A, gk, q=q)
    for expected, actual in zip(first, second, strict=True):
        assert torch.equal(expected, actual)


@pytest.mark.cula_fast
def test_preprocessed_recompute_wu_matches_torch():
    _requires_sm100()
    torch.manual_seed(1)
    batch, seqlen, heads, dim, chunk_size = 1, 256, 4, 128, 64

    k_scaled = torch.randn(batch, seqlen, heads, dim, device="cuda", dtype=torch.bfloat16) * 0.1
    v = torch.randn_like(k_scaled) * 0.1
    beta = torch.rand(batch, seqlen, heads, device="cuda", dtype=torch.bfloat16)
    A = torch.randn(batch, seqlen, heads, chunk_size, device="cuda", dtype=torch.bfloat16) * 0.02
    row = torch.arange(seqlen, device="cuda") % chunk_size
    col = torch.arange(chunk_size, device="cuda")
    A.masked_fill_((col[None, :] > row[:, None]).view(1, seqlen, 1, chunk_size), 0)

    w, u = recompute_w_u_from_preprocessed(k_scaled, v, beta, A)
    w_ref = torch.empty_like(w)
    u_ref = torch.empty_like(u)
    k_beta = (k_scaled.float() * beta.float().unsqueeze(-1)).bfloat16()
    v_beta = (v.float() * beta.float().unsqueeze(-1)).bfloat16()
    for start in range(0, seqlen, chunk_size):
        end = start + chunk_size
        A_tile = A[:, start:end].float()
        w_ref[:, start:end] = torch.einsum("bmhk,bkhd->bmhd", A_tile, k_beta[:, start:end].float())
        u_ref[:, start:end] = torch.einsum("bmhk,bkhd->bmhd", A_tile, v_beta[:, start:end].float())

    torch.testing.assert_close(w, w_ref, rtol=1e-2, atol=2e-3)
    torch.testing.assert_close(u, u_ref, rtol=1e-2, atol=2e-3)
