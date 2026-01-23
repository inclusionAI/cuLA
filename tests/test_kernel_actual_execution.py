"""
Test for actual kernel execution of matrix inverse computation.

This test:
1. Creates a 64x64 lower triangular FP16 matrix
2. Calls the actual CUDA kernel to compute the inverse
3. Verifies the result using CPU reference and validation metrics
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

try:
    import cutlass.cute as cute
except ImportError:
    cute = None

try:
    import cutlass
except ImportError:
    cutlass = None

try:
    from flashla.inv import MatrixInverse64x64
except ImportError as e:
    print(f"Failed to import MatrixInverse64x64: {e}")
    MatrixInverse64x64 = None


def create_kernel_launcher(inv_kernel):
    """
    Create a JIT-compiled kernel launcher wrapper that directly calls the kernel.
    
    Args:
        inv_kernel: MatrixInverse64x64 kernel instance
        
    Returns:
        A wrapper class with __call__ method
    """
    
    class KernelLauncher:
        def __init__(self, kernel):
            self.kernel = kernel
            self.jit_wrapper = self._create_jit_wrapper()
        
        def _create_jit_wrapper(self):
            @cute.jit
            def jit_call(torch_tensor):
                # Inside JIT context, we can call the kernel directly
                # by getting data_ptr and creating tensor
                return self.kernel(torch_tensor, stream=None)
            return jit_call
        
        def __call__(self, torch_tensor):
            """Launch kernel with a PyTorch tensor."""
            # Call within the JIT context
            return self.jit_wrapper(torch_tensor)
    
    return KernelLauncher(inv_kernel) if cute is not None else None


def create_well_conditioned_lower_triangular(size=64, seed=None):
    """
    Create a well-conditioned lower triangular matrix.
    
    Args:
        size: Matrix dimension
        seed: Random seed for reproducibility
        
    Returns:
        Lower triangular matrix (FP32 for CPU reference)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Create random lower triangular matrix
    mat = torch.randn(size, size, dtype=torch.float32)
    mat = torch.tril(mat)
    
    # Make diagonal dominant for better conditioning
    # Set diagonal to be larger than sum of absolute values in each row
    for i in range(size):
        row_sum = torch.sum(torch.abs(mat[i, :i]))
        mat[i, i] = row_sum + 2.0
    
    return mat


def compute_cpu_inverse_reference(mat_fp32):
    """
    Compute CPU reference inverse using PyTorch.
    
    Args:
        mat_fp32: Lower triangular matrix in FP32
        
    Returns:
        Inverse matrix in FP32
    """
    try:
        inv_fp32 = torch.linalg.inv(mat_fp32)
        return inv_fp32
    except RuntimeError as e:
        print(f"Error computing CPU inverse: {e}")
        return None


def verify_matrix_inverse(A_orig, A_inv, tolerance=1e-3):
    """
    Verify that A_inv is indeed the inverse of A_orig.
    
    Checks:
    1. ||A * A_inv - I||_F < tolerance
    2. ||A_inv * A - I||_F < tolerance
    3. Lower triangular structure preserved
    
    Args:
        A_orig: Original matrix
        A_inv: Computed inverse
        tolerance: Frobenius norm tolerance
        
    Returns:
        Dictionary with verification results
    """
    results = {}
    
    # Convert to FP32 for numerical verification
    A_orig_f32 = A_orig.float()
    A_inv_f32 = A_inv.float()
    
    # Check A * A_inv = I
    product1 = torch.mm(A_orig_f32, A_inv_f32)
    identity = torch.eye(A_orig_f32.shape[0], device=A_orig_f32.device, dtype=torch.float32)
    error1 = torch.norm(product1 - identity, p='fro').item()
    results['error_A_inv'] = error1
    results['pass_A_inv'] = error1 < tolerance
    
    # Check A_inv * A = I
    product2 = torch.mm(A_inv_f32, A_orig_f32)
    error2 = torch.norm(product2 - identity, p='fro').item()
    results['error_inv_A'] = error2
    results['pass_inv_A'] = error2 < tolerance
    
    # Check lower triangular structure
    lower_mask = torch.tril(torch.ones_like(A_inv_f32, dtype=torch.bool))
    upper_part = A_inv_f32 * ~lower_mask
    upper_norm = torch.norm(upper_part, p='fro').item()
    results['upper_triangular_norm'] = upper_norm
    results['pass_lower_triangular'] = upper_norm < 1e-4
    
    # Overall pass
    results['pass'] = (results['pass_A_inv'] and 
                       results['pass_inv_A'] and 
                       results['pass_lower_triangular'])
    
    return results


def test_kernel_actual_execution():
    """Test actual kernel execution with real matrix inversion."""
    
    print("\n" + "="*70)
    print("TEST: Actual Kernel Execution with Matrix Inversion")
    print("="*70)
    
    if MatrixInverse64x64 is None or cute is None or cutlass is None:
        print("SKIPPED: Required modules not available")
        return False
    
    if not torch.cuda.is_available():
        print("SKIPPED: CUDA not available")
        return False
    
    try:
        # Step 1: Create test matrix
        print("\n[Step 1] Creating test matrix...")
        size = 64
        mat_cpu = create_well_conditioned_lower_triangular(size=size, seed=42)
        print(f"  ✓ Created {size}x{size} lower triangular matrix")
        print(f"    Condition number (CPU): {torch.linalg.cond(mat_cpu).item():.2f}")
        
        # Step 2: Compute CPU reference
        print("\n[Step 2] Computing CPU reference inverse...")
        inv_cpu_fp32 = compute_cpu_inverse_reference(mat_cpu)
        if inv_cpu_fp32 is None:
            print("  ✗ Failed to compute CPU reference")
            return False
        print(f"  ✓ CPU reference computed successfully")
        
        # Step 3: Prepare GPU data
        print("\n[Step 3] Preparing GPU data...")
        mat_gpu = mat_cpu.clone().to(torch.float16).cuda()
        print(f"  ✓ Matrix copied to GPU as FP16")
        print(f"    GPU tensor shape: {mat_gpu.shape}, dtype: {mat_gpu.dtype}, device: {mat_gpu.device}")
        
        # Step 4: Create and launch kernel
        print("\n[Step 4] Creating kernel instance...")
        inv_kernel = MatrixInverse64x64(acc_dtype=cutlass.Float32)
        print(f"  ✓ Kernel instance created")
        print(f"    - MATRIX_SIZE: {inv_kernel.MATRIX_SIZE}")
        print(f"    - THREADS_PER_CTA: {inv_kernel.THREADS_PER_CTA}")
        print(f"    - GRID_SIZE: {inv_kernel.GRID_SIZE}")
        print(f"    - SMEM_ALIGN_BYTES: {inv_kernel.SMEM_ALIGN_BYTES}")
        
        print("\n[Step 4b] Creating kernel launcher...")
        kernel_launcher = create_kernel_launcher(inv_kernel)
        if kernel_launcher is None:
            print(f"  ✗ Failed to create kernel launcher")
            return False
        print(f"  ✓ Kernel launcher created")
        
        print("\n[Step 5] Launching kernel...")
        # Call the kernel launcher with the PyTorch tensor
        try:
            kernel_launcher(mat_gpu)
            print(f"  ✓ Kernel launched successfully")
        except Exception as e:
            print(f"  ✗ Kernel launch failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Synchronize to ensure kernel completion
        torch.cuda.synchronize()
        print(f"  ✓ CUDA synchronization complete")
        
        # Step 5: Verify results
        print("\n[Step 6] Verifying results...")
        
        # Convert results back to FP32 for comparison
        inv_gpu_result = mat_gpu.float()
        
        # Verify using both CPU reference and direct validation
        results = verify_matrix_inverse(
            mat_cpu,
            inv_gpu_result,
            tolerance=1e-1  # More lenient for FP16 operations
        )
        
        print(f"  Verification Results:")
        print(f"    - ||A * A_inv - I||_F: {results['error_A_inv']:.6f}")
        print(f"      ✓ PASS" if results['pass_A_inv'] else f"      ✗ FAIL")
        print(f"    - ||A_inv * A - I||_F: {results['error_inv_A']:.6f}")
        print(f"      ✓ PASS" if results['pass_inv_A'] else f"      ✗ FAIL")
        print(f"    - Upper triangular norm: {results['upper_triangular_norm']:.6e}")
        print(f"      ✓ PASS" if results['pass_lower_triangular'] else f"      ✗ FAIL")
        
        # Step 6: Compare with CPU reference
        print("\n[Step 7] Comparing with CPU reference...")
        inv_cpu_f16 = inv_cpu_fp32.to(torch.float16)
        diff_from_ref = torch.norm(
            inv_gpu_result.float() - inv_cpu_fp32,
            p='fro'
        ).item()
        print(f"  ||GPU result - CPU reference||_F: {diff_from_ref:.6f}")
        
        # Summary
        print("\n" + "-"*70)
        if results['pass']:
            print("✓ KERNEL EXECUTION TEST PASSED")
            print("  - Matrix inverse computed successfully on GPU")
            print("  - Results satisfy A * A_inv ≈ I")
            print("  - Lower triangular structure preserved")
            return True
        else:
            print("✗ KERNEL EXECUTION TEST FAILED")
            print("  - Verification checks did not pass")
            return False
            
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kernel_batch_execution():
    """Test kernel execution with multiple matrices."""
    
    print("\n" + "="*70)
    print("TEST: Batch Kernel Execution")
    print("="*70)
    
    if MatrixInverse64x64 is None or not torch.cuda.is_available():
        print("SKIPPED: Required modules or CUDA not available")
        return False
    
    try:
        print("\n[Step 1] Creating kernel instance...")
        inv_kernel = MatrixInverse64x64(acc_dtype=cutlass.Float32)
        print(f"  ✓ Kernel instance created")
        
        print("\n[Step 2] Creating kernel launcher...")
        kernel_launcher = create_kernel_launcher(inv_kernel)
        if kernel_launcher is None:
            print(f"  ✗ Failed to create kernel launcher")
            return False
        print(f"  ✓ Kernel launcher created")
        
        num_matrices = 3
        size = 64
        tolerance = 1e-1
        all_passed = True
        
        for i in range(num_matrices):
            print(f"\n[Matrix {i+1}/{num_matrices}]")
            
            # Create test matrix
            mat_cpu = create_well_conditioned_lower_triangular(size=size, seed=42+i)
            inv_cpu_ref = torch.linalg.inv(mat_cpu)
            
            # Prepare GPU data
            mat_gpu = mat_cpu.clone().to(torch.float16).cuda()
            
            # Launch kernel
            try:
                kernel_launcher(mat_gpu)
                torch.cuda.synchronize()
                print(f"  ✓ Kernel executed")
            except Exception as e:
                print(f"  ✗ Kernel failed: {e}")
                all_passed = False
                continue
            
            # Verify
            inv_gpu_result = mat_gpu.float()
            results = verify_matrix_inverse(mat_cpu, inv_gpu_result, tolerance=tolerance)
            
            if results['pass']:
                print(f"  ✓ Verification passed")
                print(f"    Error A*A_inv: {results['error_A_inv']:.6f}")
            else:
                print(f"  ✗ Verification failed")
                print(f"    Error A*A_inv: {results['error_A_inv']:.6f}")
                all_passed = False
        
        print("\n" + "-"*70)
        if all_passed:
            print(f"✓ BATCH EXECUTION TEST PASSED ({num_matrices} matrices)")
            return True
        else:
            print(f"✗ BATCH EXECUTION TEST FAILED")
            return False
            
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all kernel execution tests."""
    print("="*70)
    print("ACTUAL KERNEL EXECUTION TESTS")
    print("="*70)
    
    results = []
    
    # Test 1: Actual kernel execution
    print("\n\n>>> Test 1: Actual Kernel Execution <<<")
    results.append(test_kernel_actual_execution())
    
    # Test 2: Batch execution
    print("\n\n>>> Test 2: Batch Kernel Execution <<<")
    results.append(test_kernel_batch_execution())
    
    # Summary
    print("\n" + "="*70)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("="*70)
    
    return all(results)


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
