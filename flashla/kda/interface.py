from typing import Tuple, Optional, List
import torch

from kda.kda_cuda import chunk_kda_bwd_intra_cuda, chunk_kda_bwd_wy_dqkg_fused_cuda

def kda_bwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    dq_out: torch.Tensor,
    dk_out: torch.Tensor,
    db_out: torch.Tensor,
    dg_out: torch.Tensor,
    chunk_size: int,
    tile_counter: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    if tile_counter is not None:
        # Zero the persistent tile counter before each kernel call
        tile_counter.zero_()
    else:
        tile_counter = torch.zeros(1, dtype=torch.int32, device=q.device)
    chunk_kda_bwd_intra_cuda(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, cu_seqlens, chunk_indices, dq_out, dk_out, db_out, dg_out, tile_counter, chunk_size)
    return dq_out, dk_out, db_out, dg_out


def kda_bwd_wy_dqkg_fused(
    q: torch.Tensor,           # [B, T, H, K] bf16
    k: torch.Tensor,           # [B, T, H, K] bf16
    v: torch.Tensor,           # [B, T, H, V] bf16
    v_new: torch.Tensor,       # [B, T, H, V] bf16
    g: torch.Tensor,           # [B, T, H, K] fp32
    beta: torch.Tensor,        # [B, T, H] bf16
    A: torch.Tensor,           # [B, T, H, BT] fp32
    h: torch.Tensor,           # [NT, H, K, V] fp32
    do: torch.Tensor,          # [B, T, H, V] bf16
    dh: torch.Tensor,          # [NT, H, K, V] fp32
    dv: torch.Tensor,          # [B, T, H, V] fp32
    cu_seqlens: torch.Tensor,  # [B + 1]
    chunk_indices: torch.Tensor, # [NT * 2]
    dq_out: torch.Tensor,      # [B, T, H, K] fp32
    dk_out: torch.Tensor,      # [B, T, H, K] fp32
    dv_out: torch.Tensor,      # [B, T, H, V] bf16
    db_out: torch.Tensor,      # [B, T, H] fp32
    dg_out: torch.Tensor,      # [B, T, H, K] fp32
    dA_out: torch.Tensor,      # [B, T, H, BT] fp32
    scale: float,
    chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    KDA backward WY dqkg fused kernel.
    
    This function fuses the WY backward computation with dq/dk/dg gradient computation.
    Corresponds to chunk_kda_bwd_wy_dqkg_fused in fla.ops.kda.chunk_bwd.
    
    Args:
        q: Query tensor [B, T, H, K] bf16
        k: Key tensor [B, T, H, K] bf16
        v: Value tensor [B, T, H, V] bf16
        v_new: New value tensor (after WY transform) [B, T, H, V] bf16
        g: Gate tensor [B, T, H, K] fp32
        beta: Beta tensor [B, T, H] bf16
        A: A matrix (Akk) [B, T, H, BT] bf16
        h: Hidden state tensor [NT, H, K, V] fp32
        do: Output gradient tensor [B, T, H, V] bf16
        dh: Hidden state gradient tensor [NT, H, K, V] fp32
        dv: Value gradient tensor [B, T, H, V] bf16
        cu_seqlens: Cumulative sequence lengths [B + 1]
        chunk_indices: Chunk indices [NT * 2]
        dq_out: Query gradient output [B, T, H, K] fp32
        dk_out: Key gradient output [B, T, H, K] fp32
        dv_out: Value gradient output [B, T, H, V] bf16
        db_out: Beta gradient output [B, T, H] fp32
        dg_out: Gate gradient output [B, T, H, K] fp32
        dA_out: A matrix gradient output [B, T, H, BT] fp32
        scale: Scale factor
        chunk_size: Chunk size
        
    Returns:
        Tuple of (dq_out, dk_out, dv_out, db_out, dg_out, dA_out)
    """
    chunk_kda_bwd_wy_dqkg_fused_cuda(
        q, k, v, v_new, g, beta, A, h, do, dh, dv,
        cu_seqlens, chunk_indices,
        dq_out, dk_out, dv_out, db_out, dg_out, dA_out,
        scale, chunk_size
    )
    return dq_out, dk_out, dv_out, db_out, dg_out, dA_out


