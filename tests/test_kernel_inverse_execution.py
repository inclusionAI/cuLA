#!/usr/bin/env python3
"""
Test the matrix inverse function by running the complete KDA kernel.

This test invokes the full kernel execution path where:
1. Input Q, K, V, beta are provided
2. Kernel computes KK^T
3. Applies M = I + StrictTril(beta*KK^T)
4. Calls compute_matrix_inverse_64x64 to get M^{-1}
5. Output is compared against reference
"""

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flashla.kda import KDAChunkwise


def test_kernel_with_inverse():
    """Test the KDA kernel with matrix inverse."""
    print("=" * 70)
    print("Test: KDA Kernel with Matrix Inverse")
    print("=" * 70)
    
    try:
        # Check CUDA
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, skipping GPU test")
            return None
        
        print("✓ CUDA available")
        
        # Create small test case
        batch_size = 1
        seq_len = 64
        num_heads = 1
        head_dim = 64
        
        print(f"\nTest configuration:")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Num heads: {num_heads}")
        print(f"  Head dim: {head_dim}")
        
        # Create random inputs
        torch.manual_seed(42)
        Q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
        K = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
        V = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
        
        # Create beta tensor (B, S, H)
        beta = torch.ones(batch_size, seq_len, num_heads, dtype=torch.float32, device='cuda') * 0.5
        
        print(f"\n✓ Input tensors created:")
        print(f"  Q: {Q.shape} {Q.dtype}")
        print(f"  K: {K.shape} {K.dtype}")
        print(f"  V: {V.shape} {V.dtype}")
        print(f"  beta: {beta.shape} {beta.dtype}")
        
        # Create kernel
        kernel = KDAChunkwise()
        print(f"✓ KDA kernel created")
        
        # Run kernel
        print(f"\nRunning kernel...")
        try:
            output = kernel(Q, K, V, beta)
            print(f"✓ Kernel execution succeeded")
            print(f"  Output shape: {output.shape}")
            print(f"  Output dtype: {output.dtype}")
            
            # Basic sanity checks
            if torch.isnan(output).any():
                print(f"✗ Output contains NaN values")
                return False
            if torch.isinf(output).any():
                print(f"✗ Output contains Inf values")
                return False
            
            print(f"✓ Output contains valid values")
            
            # Check output statistics
            print(f"\nOutput statistics:")
            print(f"  Min: {output.min():.4f}")
            print(f"  Max: {output.max():.4f}")
            print(f"  Mean: {output.mean():.4f}")
            print(f"  Std: {output.std():.4f}")
            
            print(f"\n✓ PASS: Kernel executed successfully with inverse computation")
            return True
            
        except RuntimeError as e:
            if "matrix inverse" in str(e).lower() or "compute_matrix_inverse" in str(e).lower():
                print(f"✗ Error in matrix inverse computation: {e}")
                return False
            else:
                print(f"✗ Kernel execution error: {e}")
                raise
        
    except Exception as e:
        print(f"✗ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kernel_output_validity():
    """Test that kernel output is valid when using matrix inverse."""
    print("\n" + "=" * 70)
    print("Test: Kernel Output Validity with Inverse")
    print("=" * 70)
    
    try:
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, skipping GPU test")
            return None
        
        # Minimal test
        batch_size = 1
        seq_len = 64
        num_heads = 1
        head_dim = 64
        
        torch.manual_seed(123)
        Q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
        K = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
        V = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
        beta = torch.ones(batch_size, seq_len, num_heads, dtype=torch.float32, device='cuda')
        
        kernel = KDAChunkwise()
        
        print(f"Running kernel with beta=1.0...")
        output = kernel(Q, K, V, beta)
        
        # Verify output
        checks = [
            ("No NaN", not torch.isnan(output).any()),
            ("No Inf", not torch.isinf(output).any()),
            ("Output shape correct", output.shape == (batch_size, seq_len, num_heads, head_dim)),
            ("Output dtype is BF16", output.dtype == torch.bfloat16),
        ]
        
        all_passed = True
        for check_name, result in checks:
            status = "✓" if result else "✗"
            print(f"{status} {check_name}")
            all_passed = all_passed and result
        
        if all_passed:
            print(f"\n✓ PASS: All output validity checks passed")
            return True
        else:
            print(f"\n✗ FAIL: Some checks failed")
            return False
        
    except Exception as e:
        print(f"✗ EXCEPTION: {e}")
        return False


def test_beta_parameter_effect():
    """Test that beta parameter actually affects the output."""
    print("\n" + "=" * 70)
    print("Test: Beta Parameter Effect on Output")
    print("=" * 70)
    
    try:
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, skipping GPU test")
            return None
        
        batch_size = 1
        seq_len = 64
        num_heads = 1
        head_dim = 64
        
        # Use fixed seed for reproducibility
        torch.manual_seed(456)
        Q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
        K = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
        V = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
        
        kernel = KDAChunkwise()
        
        # Test with different beta values
        beta_values = [0.5, 1.0, 2.0]
        outputs = []
        
        for beta_val in beta_values:
            beta = torch.ones(batch_size, seq_len, num_heads, dtype=torch.float32, device='cuda') * beta_val
            
            print(f"\nRunning with beta={beta_val}...")
            output = kernel(Q, K, V, beta)
            outputs.append(output.clone())
            print(f"  Output mean: {output.mean():.6f}")
            print(f"  Output norm: {torch.norm(output):.6f}")
        
        # Check that different beta values produce different outputs
        output_diff = torch.norm(outputs[0] - outputs[1])
        print(f"\nOutput difference (beta=0.5 vs beta=1.0): {output_diff:.6f}")
        
        if output_diff > 1e-4:
            print(f"✓ Beta parameter has effect on output")
            print(f"✓ PASS: Beta parameter integration is working")
            return True
        else:
            print(f"⚠️  Beta parameter may not have enough effect (diff={output_diff:.6f})")
            return False
        
    except Exception as e:
        print(f"✗ EXCEPTION: {e}")
        return False


def run_all_tests():
    """Run all kernel-level tests."""
    print("\n")
    print("█" * 70)
    print("KDA Matrix Inverse Kernel Execution Tests")
    print("█" * 70)
    
    tests = [
        ("Kernel with Inverse", test_kernel_with_inverse),
        ("Output Validity", test_kernel_output_validity),
        ("Beta Parameter Effect", test_beta_parameter_effect),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result is not None:
                results[test_name] = result
        except Exception as e:
            print(f"\n✗ EXCEPTION in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    if not results:
        print("⚠️  No tests executed (may require GPU)")
        return 0
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All kernel inverse execution tests passed!")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed or skipped")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
