# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Standalone CuTeDSL composition of the SM100 KDA intra/WU kernels."""

from __future__ import annotations

import torch

from cula.ops.kda.index import prepare_chunk_indices
from cula.ops.kda.sm100.full_port import chunk_kda_fwd_full_sm100_from_gk
from cula.ops.kda.sm100.intra_fused import chunk_kda_fwd_intra_sm100_from_gk
from cula.ops.kda.sm100.recompute_wu import recompute_w_u_fwd
from cula.utils import get_device_sm_version


def chunk_kda_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.IntTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.IntTensor | None = None,
    safe_gate: bool = False,
    disable_recompute: bool = False,
    use_tf32_inverse: bool = True,
    unified_gref: bool = False,
):
    """Compose the standalone SM100 intra and W/U preparation kernels.

    ``gk`` is the already activated, chunk-cumulative gate. Gate activation,
    cumsum, recurrent state propagation, and output projection are deliberately
    outside this atomic API.
    """

    if not safe_gate:
        raise ValueError("chunk_kda_fwd_intra requires safe_gate=True")
    major, minor = get_device_sm_version(q.device)
    if not (major == 10 and minor in (0, 3)):
        raise RuntimeError(
            f"Unsupported CUDA compute capability sm_{major}{minor}; the modular intra path requires SM100 or SM103"
        )
    if gk is None or beta is None:
        raise ValueError("gk and beta are required")
    if unified_gref:
        raise NotImplementedError("unified_gref=True is not supported by the exact CuTeDSL path")
    if chunk_size != 64:
        raise NotImplementedError(f"the SM100 modular kernels require chunk_size=64, got {chunk_size}")

    batch, length, qk_heads, key_dim = k.shape
    value_heads = v.size(2)
    if qk_heads <= 0 or value_heads <= 0 or value_heads % qk_heads:
        raise ValueError(f"value heads ({value_heads}) must be a positive multiple of Q/K heads ({qk_heads})")
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is not None and (cu_seqlens.dtype != torch.int32 or chunk_indices.dtype != torch.int32):
        raise TypeError("cu_seqlens and chunk_indices must be int32")

    resolved_scale = key_dim**-0.5 if scale is None else scale
    if (
        cu_seqlens is None
        and not disable_recompute
        and batch >= 2
        and length % (4 * chunk_size) == 0
        and key_dim == 128
        and v.shape[-1] == 128
    ):
        return chunk_kda_fwd_full_sm100_from_gk(q, k, v, gk, beta, resolved_scale)

    Aqk, Akk = chunk_kda_fwd_intra_sm100_from_gk(
        q=q,
        k=k,
        gk=gk,
        beta=beta,
        scale=resolved_scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        use_tf32_inverse=use_tf32_inverse,
    )
    w, u, qg, kg = recompute_w_u_fwd(
        k,
        v,
        beta,
        Akk,
        gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        q=q if disable_recompute else None,
    )
    return w, u, qg, kg, Aqk, Akk


__all__ = ["chunk_kda_fwd_intra"]
