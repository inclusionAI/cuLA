# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Persistent CUDA ``mma.sync`` backend for KDA intra-chunk backward.

The source remains under the SM90 directory to match the current repository
layout, but the low-level kernel is also compiled for and supported on SM100
and SM103.
"""

import torch

import cula.cudac as cula_cuda
from cula.utils import get_device_sm_version

_SUPPORTED_CAPABILITIES = {(9, 0), (10, 0), (10, 3)}
_CHUNK_SIZE = 64
_HEAD_DIM = 128


def kda_bwd_intra_mma(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    d_aq: torch.Tensor,
    d_ak: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    dq_out: torch.Tensor | None = None,
    dk_out: torch.Tensor | None = None,
    db_out: torch.Tensor | None = None,
    dg_out: torch.Tensor | None = None,
    chunk_size: int = _CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the portable SM90-style MMA kernel directly.

    This is the low-level validation/benchmark entry point. Unsupported
    production cases are handled by the higher-level Triton fallback in
    :func:`cula.kda.chunk_intra.chunk_kda_bwd_intra`.
    """

    if not q.is_cuda:
        raise ValueError("kda_bwd_intra_mma requires CUDA tensors")
    capability = get_device_sm_version(q.device)
    if capability not in _SUPPORTED_CAPABILITIES:
        raise RuntimeError(f"kda_bwd_intra_mma requires SM90, SM100, or SM103, got SM{capability[0]}{capability[1]}")
    if chunk_size != _CHUNK_SIZE:
        raise ValueError(f"kda_bwd_intra_mma supports only chunk_size={_CHUNK_SIZE}, got {chunk_size}")
    if q.shape[-1] != _HEAD_DIM:
        raise ValueError(f"kda_bwd_intra_mma supports only head dimension {_HEAD_DIM}, got {q.shape[-1]}")

    cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
    chunk_indices = chunk_indices.to(device=q.device, dtype=torch.int32).contiguous()
    beta_fp32 = beta.float().contiguous()
    dq_out = torch.empty_like(q) if dq_out is None else dq_out
    dk_out = torch.empty_like(k) if dk_out is None else dk_out
    db_out = torch.empty_like(db, dtype=torch.float32) if db_out is None else db_out
    dg_out = torch.empty_like(dg, dtype=torch.float32) if dg_out is None else dg_out

    cula_cuda.chunk_kda_bwd_intra_cuda(
        q,
        k,
        g,
        beta_fp32,
        d_aq,
        d_ak,
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
        chunk_size,
    )
    return dq_out, dk_out, db_out, dg_out


__all__ = ["kda_bwd_intra_mma"]
