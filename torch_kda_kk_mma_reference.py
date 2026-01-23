#!/usr/bin/env python3
"""
PyTorch BF16 MMA Reference Implementation for KDA K*K^T Computation.

This script provides a pure PyTorch reference implementation for KDA's KK MMA 
computation with BF16 precision, useful for validation and testing.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.modules.l2norm import l2norm_fwd


class KDAKKMMAReference:
    """
    Reference implementation for KDA K*K^T MMA computation using BF16 precision.
    
    This class demonstrates the intended computation for the KDA KK MMA block,
    which computes the attention matrix K*K^T with exponential gating.
    """
    
    def __init__(
        self,
        chunk_size: int = 64,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        """
        Initialize the reference MMA implementation.
        
        Args:
            chunk_size: Size of each chunk (typically 64)
            dtype: Data type for K matrices (BF16 or FP32)
            device: Device to use ("cuda" or "cpu")
        """
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.device = device
    
    @torch.no_grad()
    def compute_full_kda_step1(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        use_cumsum: bool = False,
        chunk_size: int = 64,
        output_kk_mma: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Full KDA Step 1 computation (gate processing).
        
        Args:
            Q: Query tensor [L, D] (2D) or [B, T, H, D] (4D)
            K: Key tensor [L, D] (2D) or [B, T, H, D] (4D)
            g: Gate values [L, 1] (2D) or [B, T, H, D] (4D) in float32
            use_cumsum: Whether to apply chunk_local_cumsum to g
            chunk_size: Chunk size for cumsum
            output_kk_mma: Whether to output KK MMA result
        
        Returns:
            (Q_gated, K_inter, K_intra, kk_matrix, kk_mma): Processed tensors and optional KK MMA
        """
        # Step 1: Compute exp(g) and exp(-g) with optional cumsum
        g_f32 = g.to(torch.float32)
        
        exp_g = torch.exp2(g_f32)
        exp_neg_g = torch.exp2(-g_f32)

        # Step 2: Convert to FP32 for computation
        Q_f32 = Q.to(torch.float32)
        K_f32 = K.to(torch.float32)
        
        # Step 3: Apply gates
        Q_gated = Q_f32 * exp_g
        K_inter = K_f32 * exp_g
        K_intra = K_f32 * exp_neg_g

        # Step 4: Convert to BF16 (matching kernel)
        Q_gated_bf16 = Q_gated.to(torch.bfloat16)
        K_inter_bf16 = K_inter.to(torch.bfloat16)
        K_intra_bf16 = K_intra.to(torch.bfloat16)
        
        out = torch.zeros(64, 64, dtype=torch.bfloat16, device="cuda")
        torch.matmul(K_inter_bf16, K_intra_bf16.transpose(-2, -1), out=out)

        print(f"out:\n {out}")
        kk_matrix = out.to(torch.bfloat16)

        kk_matrix_f32 = torch.matmul(K_inter, K_intra.transpose(-2, -1))

        print(f"kk bf16:\n {kk_matrix[-2:, -5:]}")
        print(f"kk f32:\n {kk_matrix_f32}")
        print(f"beta:\n {beta}")

        kk_matrix = (torch.diag(beta) @ kk_matrix.float()).to(torch.bfloat16)
        kk_matrix_f32 = torch.diag(beta) @ kk_matrix_f32 
        
        # Step 6: Optional KK MMA output
        kk_mma = None
        if output_kk_mma:
            kk_mma = kk_matrix
        
        return Q_gated_bf16, K_inter_bf16, K_intra_bf16, kk_matrix, kk_mma, kk_matrix_f32

    def compute_inverse(
        self,
        kk_matrix: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> torch.Tensor:
        """
        Compute the inverse of the KK matrix with numerical stability.
        
        Args:
            kk_matrix: KK matrix tensor [L, L]
            epsilon: Small value to add to diagonal for stability
        
        Returns:
            Inverse of the KK matrix
        """
        L = kk_matrix.size(0)
        kk_matrix = kk_matrix.to(torch.float16)
        identity = torch.eye(L, dtype=kk_matrix.dtype, device=kk_matrix.device)
        kk_matrix_stable = torch.tril(kk_matrix, diagonal=-1) + identity

        print(f"I + STril(kk) before inverse:\n {kk_matrix_stable}")

        kk_inv_f32 = torch.inverse(kk_matrix_stable.to(torch.float32))
        # kk_inv_f16 = torch.inverse(kk_matrix_stable)

        print(f"kk_inv_f32:\n {kk_inv_f32}")
        #$ print(f"kk_inv_f16:\n {kk_inv_f16}")

        return kk_inv_f32

def test_bench_kda_aligned():
    """Test with configuration aligned to bench_kda.py."""
    print("\n" + "="*80)
    print("KDA KK MMA Test (Aligned to bench_kda.py)")
    print("="*80)
    
    # Configuration aligned with bench_kda.py
    B, H, D = 1, 1, 128
    S = T = 64
    CHUNK_SIZE = 64
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    print(f"\nConfiguration:")
    print(f"  B={B}, H={H}, D={D}")
    print(f"  S={S}, T={T}, CHUNK_SIZE={CHUNK_SIZE}")
    print(f"  Device: {device}, dtype: {dtype}")
    
    # Create test data aligned with bench_kda
    q = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    g_raw = torch.randn(B, T, H, D, dtype=dtype, device=device).requires_grad_(False)
    g = F.logsigmoid(g_raw)
    beta = torch.randn(B, T, H, dtype=torch.float32, device=device).requires_grad_(False)

    g = chunk_local_cumsum(
        g=g,
        chunk_size=CHUNK_SIZE,
        scale=RCP_LN2,
        cu_seqlens=None,
        chunk_indices=None
    )

    q, _ = l2norm_fwd(q)
    k, _ = l2norm_fwd(k)

    print(f"q: {q}")
    print(f"k: {k}")
    print(f"v: {v}")
    print(f"g: {g}")
    
    # Reshape for processing (B, T, H, D) -> (T, 1) for single sequence
    # Note: g needs to be (T, 1) for broadcasting
    q_seq = q[0, :, 0, :]
    k_seq = k[0, :, 0, :]
    g_seq = g[0, :, 0, :]
    beta_seq = beta[0, :, 0]
    
    print(f"\nInput shapes (after reshape):")
    print(f"  q_seq: {q_seq.shape}")
    print(f"  k_seq: {k_seq.shape}")
    print(f"  g_seq: {g_seq.shape}")
    print(f"  g_seq stats: min={g_seq.min():.6f}, max={g_seq.max():.6f}, mean={g_seq.mean():.6f}")
    
    ref = KDAKKMMAReference(chunk_size=CHUNK_SIZE, device=device)
    
    print(f"\n--- Test: WITH cumsum ---")
    Q_gated_cs, K_inter_cs, K_intra_cs, kk_matrix_cs, kk_mma_cs, kk_matrix_f32_cs = ref.compute_full_kda_step1(
        q_seq, k_seq, g_seq, beta_seq,
        use_cumsum=True,
        chunk_size=CHUNK_SIZE,
        output_kk_mma=True,
    )

    inv_f32 = ref.compute_inverse(kk_matrix_cs)
    
    print(f"\nKK MMA Result Statistics (WITH cumsum):")
    print(f"  Min: {kk_mma_cs.min():.6e}")
    print(f"  Max: {kk_mma_cs.max():.6e}")
    print(f"  Mean: {kk_mma_cs.mean():.6e}")
    print(f"  Std: {kk_mma_cs.std():.6e}")
    
    # Check for inf/nan
    has_inf = torch.isinf(kk_mma_cs).any()
    has_nan = torch.isnan(kk_mma_cs).any()
    print(f"  Has Inf: {has_inf}, Has NaN: {has_nan}")
    
    if has_inf or has_nan:
        print(f"\n  ⚠️ Warning: Cumsum caused numerical overflow!")
        print(f"  This is expected for logsigmoid-based gate values")
    else:
        print(f"\nKK MMA Matrix (first 5x5 elements):")
        print(kk_mma_cs[:5, :5])
        
        print(f"\nKK MMA Matrix (last 5x5 elements):")
        print(kk_mma_cs[-5:, -5:])

        print(f"\nKK MMA Matrix :")
        print(kk_mma_cs)

        print(f"\nKK MMA Matrix F32:")
        print(kk_matrix_f32_cs)
    
    return kk_mma_cs, Q_gated_cs, K_inter_cs, K_intra_cs


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PyTorch BF16 MMA Reference Implementation for KDA")
    print("="*80)
    
    # Generate KK results
    kk_mma, Q_gated, K_inter, K_intra = test_bench_kda_aligned()
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)
