# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pathlib
import sys

import pytest
import torch
from fla.ops.utils import prepare_chunk_indices

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cula.cudac as cula_cuda
from cula.ops.kda.sm100.recompute_wu import recompute_w_u_from_preprocessed, recompute_w_u_fwd


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

    assert torch.equal(w, w_ref), f"w differs bitwise: max_abs={(w.float() - w_ref.float()).abs().max().item()}"
    assert torch.equal(u, u_ref), f"u differs bitwise: max_abs={(u.float() - u_ref.float()).abs().max().item()}"
    assert torch.equal(kg, kg_ref), f"kg differs bitwise: max_abs={(kg.float() - kg_ref.float()).abs().max().item()}"


def test_preprocessed_recompute_wu_matches_torch():
    _requires_sm100()
    torch.manual_seed(1)
    device = torch.device("cuda")
    batch, seqlen, heads, dim, chunk_size = 1, 256, 4, 128, 64

    k_scaled = torch.randn(batch, seqlen, heads, dim, device=device, dtype=torch.bfloat16) * 0.1
    v = torch.randn_like(k_scaled) * 0.1
    beta = torch.rand(batch, seqlen, heads, device=device, dtype=torch.bfloat16)
    A = torch.randn(batch, seqlen, heads, chunk_size, device=device, dtype=torch.bfloat16) * 0.02
    row = torch.arange(seqlen, device=device) % chunk_size
    col = torch.arange(chunk_size, device=device)
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
