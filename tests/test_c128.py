#!/usr/bin/env python3
"""
Test suite for the C=128 LinearAttentionChunkwiseDecay kernel.

Compares C=128 kernel output against PyTorch reference implementation.
"""

import sys
import warnings
import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32

warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, "/ossfs/workspace/flashla")
from flashla.lightning_attn_c128 import lightning_attn_fwd, lightning_attn_fwd_varlen


# ---------------------------------------------------------------------------
# PyTorch reference implementation (from original test)
# ---------------------------------------------------------------------------

def torch_ref_chunkwise_decay(Q, K, V, decay_s, scale, chunk_size=128,
                              initial_state=None, output_final_state=False):
    """
    Pure PyTorch reference: chunkwise linear attention with per-position exponential decay.
    
    Args:
        Q, K, V: (B, S, H, D) bfloat16
        decay_s: (H,) float32, s > 0
        scale: float
        chunk_size: int
        initial_state: (B, H, D, D) float32 or None
        output_final_state: bool
    Returns:
        O: (B, S, H, D) bfloat16
        ht: (B, H, D, D) float32 or None
    """
    B, S, H, D = Q.shape
    C = chunk_size
    num_chunks = (S + C - 1) // C

    Q_f = Q.float()
    K_f = K.float()
    V_f = V.float()

    O = torch.zeros(B, S, H, D, device=Q.device, dtype=torch.float32)

    # Per-head state: (B, H, D, D)
    state = torch.zeros(B, H, D, D, device=Q.device, dtype=torch.float32)
    if initial_state is not None:
        state = initial_state.clone()

    for c_idx in range(num_chunks):
        start = c_idx * C
        end = min(start + C, S)
        actual_len = end - start

        Qc = Q_f[:, start:end, :, :]  # (B, L, H, D)
        Kc = K_f[:, start:end, :, :]
        Vc = V_f[:, start:end, :, :]

        # Position indices within chunk: 0, 1, ..., L-1
        positions = torch.arange(actual_len, device=Q.device, dtype=torch.float32)

        # Decay matrix: D[i,j] = exp(-s * (i - j)) for i >= j, 0 for i < j
        # Shape: (H, L, L)
        s = decay_s.unsqueeze(-1).unsqueeze(-1)  # (H, 1, 1)
        pos_diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # (L, L)
        decay_matrix = torch.exp(-s * pos_diff) * (pos_diff >= 0).float()  # (H, L, L)

        # Intra-chunk: O_intra = decay_matrix @ (Q @ K^T) @ V
        # Actually: S = Q @ K^T, S_masked = S * decay_matrix, O_intra = S_masked @ V
        S_qk = torch.einsum('bshd,bthd->bhst', Qc, Kc)  # (B, H, L, L)
        S_masked = S_qk * decay_matrix.unsqueeze(0)  # (B, H, L, L)
        O_intra = torch.einsum('bhst,bthd->bshd', S_masked, Vc)  # (B, L, H, D)

        # Inter-chunk: O_inter = state @ Q^T with per-position decay
        # For position i in chunk: decay_factor = exp(-s * (i+1))
        pos_decay = torch.exp(-decay_s.unsqueeze(-1) * (positions + 1))  # (H, L)
        O_inter = torch.einsum('bhde,bshd->bshe', state, Qc)  # (B, L, H, D←E)
        # Wait, state is (B, H, D_out, D_in), Q is (B, L, H, D_in)
        # O_inter[b,i,h,d] = sum_e state[b,h,d,e] * Q[b,i,h,e] * decay(i)
        O_inter = torch.einsum('bhde,bihe->bihd', state, Qc) * pos_decay.permute(1, 0).unsqueeze(0).unsqueeze(-1)

        # Combine
        O[:, start:end, :, :] = (O_intra + O_inter) * scale

        # State update: state = state * block_decay + sum_j K_weighted[j] outer V[j]
        # K_weighted[j] = K[j] * exp(-s * (C-1-j))
        block_decay = torch.exp(-decay_s * C)  # (H,)
        k_weights = torch.exp(-decay_s.unsqueeze(-1) * (C - 1 - positions))  # (H, L)
        Kc_weighted = Kc * k_weights.permute(1, 0).unsqueeze(0).unsqueeze(-1)  # (B, L, H, D)
        # State += K_weighted^T @ V
        state_update = torch.einsum('bshd,bshe->bhde', Kc_weighted, Vc)
        state = state * block_decay.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) + state_update

    O_bf16 = O.to(torch.bfloat16)
    ht = state if output_final_state else None
    return O_bf16, ht


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_c128(B=2, S=1024, H=4, D=128, chunk_size=128):
    """Test C=128 kernel output matches PyTorch reference."""
    print(f"\n{'='*60}")
    print(f"test_basic_c128: B={B}, S={S}, H={H}, D={D}, C={chunk_size}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    Q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    K = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    V = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    decay = torch.rand(H, device="cuda", dtype=torch.float32) * 0.1 + 0.01
    scale = 1.0 / (D ** 0.5)

    # PyTorch reference
    O_ref, _ = torch_ref_chunkwise_decay(Q, K, V, decay, scale, chunk_size=chunk_size)

    # CuTeDSL C=128 kernel
    O_cute, _ = lightning_attn_fwd(Q, K, V, decay, scale=scale, chunk_size=chunk_size)
    torch.cuda.synchronize()

    # Compare
    max_diff = (O_cute.float() - O_ref.float()).abs().max().item()
    mean_diff = (O_cute.float() - O_ref.float()).abs().mean().item()
    
    print(f"  Max absolute diff:  {max_diff:.6e}")
    print(f"  Mean absolute diff: {mean_diff:.6e}")
    
    # Relative error (avoid div by zero)
    mask = O_ref.float().abs() > 1e-6
    if mask.any():
        rel_err = ((O_cute.float() - O_ref.float()).abs()[mask] / O_ref.float().abs()[mask]).mean().item()
        print(f"  Mean relative error: {rel_err:.6e}")
    
    # BF16 tolerance: 2^-7 ≈ 0.0078, allow a bit more for accumulated errors
    if max_diff < 0.05:
        print(f"  ✓ PASSED")
        return True
    else:
        print(f"  ✗ FAILED (max_diff={max_diff:.4f} > 0.05)")
        return False


def test_with_initial_state(B=1, S=512, H=4, D=128, chunk_size=128):
    """Test C=128 kernel with initial state."""
    print(f"\n{'='*60}")
    print(f"test_with_initial_state: B={B}, S={S}, H={H}, D={D}, C={chunk_size}")
    print(f"{'='*60}")

    torch.manual_seed(123)
    Q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    K = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    V = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    decay = torch.rand(H, device="cuda", dtype=torch.float32) * 0.1 + 0.01
    scale = 1.0 / (D ** 0.5)
    h0 = torch.randn(B, H, D, D, device="cuda", dtype=torch.float32) * 0.01

    O_ref, ht_ref = torch_ref_chunkwise_decay(
        Q, K, V, decay, scale, chunk_size=chunk_size,
        initial_state=h0, output_final_state=True,
    )
    O_cute, ht_cute = lightning_attn_fwd(
        Q, K, V, decay, scale=scale,
        initial_state=h0, output_final_state=True,
        chunk_size=chunk_size,
    )
    torch.cuda.synchronize()

    o_diff = (O_cute.float() - O_ref.float()).abs().max().item()
    ht_diff = (ht_cute.float() - ht_ref.float()).abs().max().item()
    
    print(f"  O max diff:  {o_diff:.6e}")
    print(f"  ht max diff: {ht_diff:.6e}")
    
    passed = o_diff < 0.05 and ht_diff < 0.1
    print(f"  {'✓ PASSED' if passed else '✗ FAILED'}")
    return passed


def test_single_chunk(B=1, S=128, H=2, D=128, chunk_size=128):
    """Test with exactly one chunk (S == C)."""
    print(f"\n{'='*60}")
    print(f"test_single_chunk: B={B}, S={S}, H={H}, D={D}, C={chunk_size}")
    print(f"{'='*60}")

    torch.manual_seed(0)
    Q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    K = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    V = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    decay = torch.full((H,), 0.05, device="cuda", dtype=torch.float32)
    scale = 1.0 / (D ** 0.5)

    O_ref, _ = torch_ref_chunkwise_decay(Q, K, V, decay, scale, chunk_size=chunk_size)
    O_cute, _ = lightning_attn_fwd(Q, K, V, decay, scale=scale, chunk_size=chunk_size)
    torch.cuda.synchronize()

    max_diff = (O_cute.float() - O_ref.float()).abs().max().item()
    print(f"  Max absolute diff: {max_diff:.6e}")
    passed = max_diff < 0.05
    print(f"  {'✓ PASSED' if passed else '✗ FAILED'}")
    return passed


if __name__ == "__main__":
    results = []
    results.append(("single_chunk", test_single_chunk()))
    results.append(("basic_c128", test_basic_c128()))
    results.append(("with_initial_state", test_with_initial_state()))
    
    print(f"\n{'='*60}")
    print("SUMMARY:")
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\nAll tests PASSED!")
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)
