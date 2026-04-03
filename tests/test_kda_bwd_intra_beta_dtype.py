# Copyright 2025-2026 Ant Group Co., Ltd.

import pathlib
import sys

import pytest
import torch

pytestmark = pytest.mark.sm100_only

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra  # noqa: E402
from fla.ops.utils import prepare_chunk_indices  # noqa: E402
from fla.utils import assert_close  # noqa: E402

from cula import cudac  # noqa: E402


@pytest.mark.parametrize("beta_dtype", [torch.bfloat16, torch.float32])
def test_kda_bwd_intra_beta_dtype_adaptive(beta_dtype: torch.dtype):
    torch.manual_seed(42)
    device = torch.device("cuda")

    B, T, H, D, BT = 1, 256, 4, 128, 64
    q = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
    g = torch.randn(B, T, H, D, device=device, dtype=torch.float32) / 10
    beta = torch.randn(B, T, H, device=device, dtype=beta_dtype)
    dAqk = torch.randn(B, T, H, BT, device=device, dtype=torch.float32)
    dAkk = torch.randn(B, T, H, BT, device=device, dtype=torch.float32)
    dq = torch.randn(B, T, H, D, device=device, dtype=torch.float32)
    dk = torch.randn(B, T, H, D, device=device, dtype=torch.float32)
    db = torch.randn(B, T, H, device=device, dtype=torch.float32)
    dg = torch.randn(B, T, H, D, device=device, dtype=torch.float32)

    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)
    chunk_indices = prepare_chunk_indices(cu_seqlens.to(torch.long), BT).to(torch.int32)

    dq_out = torch.empty_like(dq, dtype=torch.bfloat16)
    dk_out = torch.empty_like(dk, dtype=torch.bfloat16)
    db_out = torch.empty_like(db, dtype=torch.float32)
    dg_out = torch.empty_like(dg, dtype=torch.float32)
    tile_counter = torch.zeros(1, dtype=torch.int32, device=device)

    cudac.chunk_kda_bwd_intra_cuda(
        q,
        k,
        g,
        beta,
        dAqk,
        dAkk,
        dq,
        dk,
        db,
        dg,
        cu_seqlens,
        chunk_indices,
        dq_out,
        dk_out,
        db_out,
        dg_out,
        tile_counter,
        BT,
    )

    dq_ref, dk_ref, db_ref, dg_ref = chunk_kda_bwd_intra(
        q,
        k,
        g,
        beta,
        dAqk,
        dAkk,
        dq,
        dk,
        db,
        dg,
        cu_seqlens,
        chunk_indices,
        BT,
        True,
    )

    assert_close("dq", dq_ref, dq_out, 0.008)
    assert_close("dk", dk_ref, dk_out, 0.008)
    assert_close("db", db_ref, db_out, 0.02)
    assert_close("dg", dg_ref, dg_out, 0.02)
