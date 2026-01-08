"""
Test KDA Step 1 End-to-End: Gate processing with CUDA kernel

This test:
1. Creates Q, K, g_cumsum inputs
2. Runs torch reference
3. Calls KDA CUDA kernel
4. Compares outputs for correctness
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import cutlass
from cutlass.cute.typing import Int32
from cutlass.cute.runtime import from_dlpack
import cutlass.torch as cutlass_torch
import cutlass.cute as cute

from flashla.kda import KDAChunkwise
from test_kda_step1_gate import kda_gate_torch_reference


def test_kda_gate_e2e():
    """End-to-end test comparing CUDA kernel with torch reference"""
    
    torch.manual_seed(42)
    B, S, H, D = 1, 128, 1, 64
    chunk_size = 64
    device = 'cuda'
    
    print("=" * 80)
    print(f"Test Configuration: B={B}, S={S}, H={H}, D={D}, chunk_size={chunk_size}")
    print("=" * 80)
    
    # Create inputs
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device)
    
    # Create g and compute g_cumsum
    g = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device) * 0.01
    num_chunks = S // chunk_size
    g_chunked = g.view(B, num_chunks, chunk_size, H, D)
    g_cumsum_chunked = torch.cumsum(g_chunked, dim=2)
    g_cumsum = g_cumsum_chunked.view(B, S, H, D)
    
    print(f"\n[1] Created inputs")
    print(f"  Q: {q.shape}, dtype={q.dtype}")
    print(f"  K: {k.shape}, dtype={k.dtype}")
    print(f"  g_cumsum: {g_cumsum.shape}, range=[{g_cumsum.min():.4f}, {g_cumsum.max():.4f}]")
    
    # Torch reference (in float32 for precision)
    print(f"\n[2] Computing torch reference...")
    q_ref, k_ref, kt_ref = kda_gate_torch_reference(
        q.float(), k.float(), g_cumsum.float(), chunk_size=chunk_size
    )
    print(f"  ✓ Reference computed")
    print(f"    q_gated: range=[{q_ref.min():.4f}, {q_ref.max():.4f}]")
    print(f"    k_gated: range=[{k_ref.min():.4f}, {k_ref.max():.4f}]")
    print(f"    kt_exp_neg_g: range=[{kt_ref.min():.4f}, {kt_ref.max():.4f}]")
    
    # TODO: CUDA kernel implementation
    print(f"\n[3] Running CUDA kernel...")
    print(f"  ⚠ Kernel logic not yet implemented")
    print(f"  ⚠ Need to add:")
    print(f"    - TMA load g_cumsum to SMEM")
    print(f"    - CUDA cores compute exp(g), exp(-g)")
    print(f"    - Elementwise: Q*exp(g), K*exp(g), K^T*exp(-g)")
    print(f"    - Store outputs for validation")
    
    # For now, create KDA instance to test setup
    kda = KDAChunkwise(
        chunk_size=chunk_size,
        qk_acc_dtype=cutlass.Float32,
        kv_acc_dtype=cutlass.Float32,
        acc_dtype=cutlass.Float32,
        io_dtype=cutlass.BFloat16,
    )
    
    o = torch.zeros(B, S, H, D, dtype=torch.bfloat16, device=device)
    decay = torch.ones(H, dtype=torch.float32, device=device) * 0.99
    
    # Get cute tensors
    q_cute = from_dlpack(q)
    k_cute = from_dlpack(k)
    v_cute = from_dlpack(v)
    g_cute = from_dlpack(g_cumsum)  # Use g_cumsum as input
    o_cute = from_dlpack(o)
    decay_cute = from_dlpack(decay)
    
    problem_size = (Int32(B), Int32(S), Int32(H), Int32(D))
    stream = cutlass_torch.default_stream()
    
    try:
        compiled = cute.compile(
            kda,
            q_cute.iterator,
            k_cute.iterator,
            v_cute.iterator,
            g_cute.iterator,
            o_cute.iterator,
            decay_cute.iterator,
            problem_size,
            stream,
        )
        
        compiled(
            q_cute.iterator,
            k_cute.iterator,
            v_cute.iterator,
            g_cute.iterator,
            o_cute.iterator,
            decay_cute.iterator,
            problem_size,
            stream,
        )
        torch.cuda.synchronize()
        print(f"  ✓ Kernel executed (but logic not implemented yet)")
        
    except Exception as e:
        print(f"  ✗ Kernel error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n[4] Comparison")
    print(f"  ⚠ Skipped - kernel logic not implemented")
    print(f"  ⚠ Next step: Implement gate processing in kernel")
    
    return True


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("KDA Step 1 End-to-End Test")
    print("=" * 80)
    
    success = test_kda_gate_e2e()
    
    print("\n" + "=" * 80)
    if success:
        print("Status: Kernel compiles and runs ✓")
        print("TODO: Implement gate processing logic in kernel")
    else:
        print("Status: Test failed ✗")
    print("=" * 80)
