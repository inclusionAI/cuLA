# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from fla.ops.utils import prepare_chunk_indices

import cula.cudac as cula_cuda
from cula.ops.kda.sm100.recompute_wu import recompute_w_u_fwd


def _requires_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 CUDA device required")


@pytest.mark.parametrize("beta_dtype", [torch.float32, torch.bfloat16])
def test_recompute_wu_matches_csrc(beta_dtype):
    _requires_sm100()
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, seqlen, heads, dim, chunk_size = 2, 256, 16, 128, 64

    k = torch.randn(1, batch * seqlen, heads, dim, device=device, dtype=torch.bfloat16) * 0.1
    v = torch.randn_like(k) * 0.1
    beta = torch.rand(1, batch * seqlen, heads, device=device, dtype=beta_dtype)
    gk = torch.randn(1, batch * seqlen, heads, dim, device=device, dtype=torch.float32) * 0.02
    A = torch.randn(1, batch * seqlen, heads, chunk_size, device=device, dtype=torch.bfloat16) * 0.02
    cu_seqlens = torch.tensor([0, seqlen, 2 * seqlen], dtype=torch.int32, device=device)
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    w_ref = torch.empty_like(k)
    u_ref = torch.empty_like(v)
    kg_ref = torch.empty_like(k)
    cula_cuda.recompute_w_u_cuda(
        k,
        v,
        beta,
        A,
        gk,
        cu_seqlens,
        chunk_indices,
        w_ref,
        u_ref,
        kg_ref,
        chunk_size,
        None,
        None,
    )
    w, u, _, kg = recompute_w_u_fwd(k, v, beta, A, gk, cu_seqlens, chunk_indices)

    torch.testing.assert_close(w, w_ref, rtol=1e-2, atol=2e-3)
    torch.testing.assert_close(u, u_ref, rtol=1e-2, atol=2e-3)
    torch.testing.assert_close(kg, kg_ref, rtol=1e-2, atol=2e-3)
