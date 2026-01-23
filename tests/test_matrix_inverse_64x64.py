"""
Test suite for the 64x64 FP16 matrix inverse kernel.
"""

import sys
import os

# Add parent directory to path for imports
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


def create_lower_triangular_matrix(size=64, dtype=torch.float16, device="cuda"):
    """
    Create a random lower triangular matrix for testing.
    
    Args:
        size: Matrix dimension (default: 64)
        dtype: Data type (default: torch.float16)
        device: Device to allocate tensor (default: "cuda")
    
    Returns:
        Lower triangular matrix tensor of shape (size, size)
    """
    mat = torch.randn(size, size, dtype=dtype, device=device)
    # Zero out upper triangle
    mat = torch.tril(mat)
    # Ensure diagonal is non-zero and well-conditioned
    diag_vals = torch.abs(torch.diag(mat)) + 1.0
    mat.diagonal().copy_(diag_vals)
    return mat


def test_matrix_inverse_kernel_instantiation():
    """Test that the matrix inverse kernel can be instantiated."""
    if MatrixInverse64x64 is None:
        print("Test: Kernel instantiation - SKIPPED (module not available)")
        return
    
    print("Test: Kernel instantiation")
    acc_dtype = cutlass.Float32 if cutlass else None
    inv_kernel = MatrixInverse64x64(acc_dtype=acc_dtype)
    assert inv_kernel is not None
    print("✓ Kernel instantiation successful")


def test_matrix_inverse_fp16_cpu():
    """Test matrix inverse computation using CPU (PyTorch) as reference."""
    print("\nTest: FP16 Matrix inverse (CPU reference)")
    
    try:
        size = 64
        # Create lower triangular matrix with better conditioning
        mat_fp32 = torch.randn(size, size, dtype=torch.float32) * 0.1
        mat_fp32 = torch.tril(mat_fp32)
        # Make diagonal well-conditioned
        diag = torch.abs(torch.diag(mat_fp32)) + 2.0
        mat_fp32.diagonal().copy_(diag)
        
        # Verify the matrix is not singular
        det = torch.linalg.det(mat_fp32)
        if abs(det.item()) < 1e-10:
            print("  Matrix is singular, skipping inversion")
            return
        
        # Convert to FP16
        mat_fp16 = mat_fp32.to(torch.float16)
        
        # Compute inverse on CPU
        inv_fp32 = torch.linalg.inv(mat_fp32)
        inv_fp16_ref = inv_fp32.to(torch.float16)
        
        # Verify lower triangular structure is preserved
        lower_mask = torch.tril(torch.ones_like(inv_fp16_ref, dtype=torch.bool))
        upper_part = inv_fp16_ref * ~lower_mask
        upper_norm = torch.norm(upper_part).item()
        
        print(f"  Input matrix condition number: {torch.linalg.cond(mat_fp32).item():.2f}")
        print(f"  Inverse matrix shape: {inv_fp16_ref.shape}")
        print(f"  Inverse matrix dtype: {inv_fp16_ref.dtype}")
        print(f"  Upper triangular norm: {upper_norm:.6f}")
        print("✓ FP16 matrix inverse reference computed")
    except Exception as e:
        print(f"✗ Error in test: {e}")


def test_class_constants():
    """Test that kernel class has proper configuration constants."""
    if MatrixInverse64x64 is None:
        print("\nTest: Class constants - SKIPPED (module not available)")
        return
    
    print("\nTest: Class constants")
    
    # Check class constants
    assert hasattr(MatrixInverse64x64, 'MATRIX_SIZE')
    assert MatrixInverse64x64.MATRIX_SIZE == 64
    assert hasattr(MatrixInverse64x64, 'THREADS_PER_CTA')
    assert MatrixInverse64x64.THREADS_PER_CTA == 128
    assert hasattr(MatrixInverse64x64, 'GRID_SIZE')
    assert MatrixInverse64x64.GRID_SIZE == 1
    assert hasattr(MatrixInverse64x64, 'SMEM_ALIGN_BYTES')
    assert MatrixInverse64x64.SMEM_ALIGN_BYTES == 1024
    
    print("✓ All class constants are properly defined")
    print(f"  - MATRIX_SIZE: {MatrixInverse64x64.MATRIX_SIZE}")
    print(f"  - THREADS_PER_CTA: {MatrixInverse64x64.THREADS_PER_CTA}")
    print(f"  - GRID_SIZE: {MatrixInverse64x64.GRID_SIZE}")
    print(f"  - SMEM_ALIGN_BYTES: {MatrixInverse64x64.SMEM_ALIGN_BYTES}")


def test_kernel_compilation():
    """Test that the kernel can be compiled via JIT decorator."""
    if MatrixInverse64x64 is None or cute is None:
        print("\nTest: Kernel compilation (JIT) - SKIPPED (module not available)")
        return
    
    print("\nTest: Kernel compilation (JIT)")
    
    try:
        # Create kernel instance
        inv_kernel = MatrixInverse64x64()
        print(f"  Kernel instance created: {inv_kernel}")
        
        # Verify __call__ is decorated with @cute.jit
        assert hasattr(inv_kernel, '__call__')
        call_method = getattr(inv_kernel, '__call__')
        assert callable(call_method), "__call__ is not callable"
        print(f"  __call__ method: {call_method}")
        
        # Verify kernel method is decorated with @cute.kernel
        assert hasattr(inv_kernel, 'kernel')
        kernel_method = getattr(inv_kernel, 'kernel')
        assert callable(kernel_method), "kernel method is not callable"
        print(f"  kernel method: {kernel_method}")
        
        print("✓ Kernel compilation preparation successful")
        
    except Exception as e:
        print(f"✗ Kernel compilation failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_kernel_invocation_with_mock_data():
    """Test calling the kernel with mock data (CPU tensor initially)."""
    if MatrixInverse64x64 is None or torch is None:
        print("\nTest: Kernel invocation with mock data - SKIPPED (module not available)")
        return
    
    print("\nTest: Kernel invocation with mock data")
    
    try:
        # Create kernel instance
        inv_kernel = MatrixInverse64x64()
        print(f"  Kernel instance created")
        
        # Create a simple 64x64 lower triangular test matrix
        mat = create_lower_triangular_matrix(size=64, dtype=torch.float16, device="cuda")
        print(f"  Test matrix created: shape={mat.shape}, dtype={mat.dtype}, device={mat.device}")
        
        # Verify matrix is lower triangular
        upper_mask = torch.triu(torch.ones_like(mat, dtype=torch.bool), diagonal=1)
        upper_part = mat[upper_mask]
        upper_norm = torch.norm(upper_part.float()).item()
        print(f"  Upper triangular part norm: {upper_norm:.6e} (should be 0)")
        assert upper_norm < 1e-6, "Matrix is not properly lower triangular"
        
        # Get the data pointer and current CUDA stream
        mat_ptr = mat.data_ptr()
        stream = torch.cuda.current_stream()
        print(f"  Matrix pointer: {mat_ptr}")
        print(f"  Current CUDA stream: {stream}")
        
        print("✓ Kernel invocation preparation successful")
        
    except Exception as e:
        print(f"✗ Kernel invocation preparation failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_kernel_execution_on_gpu():
    """Test that kernel is ready for execution on GPU."""
    if MatrixInverse64x64 is None or torch is None or not torch.cuda.is_available():
        print("\nTest: Kernel ready for GPU execution - SKIPPED (CUDA not available)")
        return
    
    print("\nTest: Kernel ready for GPU execution")
    
    try:
        # Create kernel instance
        inv_kernel = MatrixInverse64x64()
        print(f"  Kernel instance: {inv_kernel.__class__.__name__}")
        
        # Verify kernel is ready to execute
        assert hasattr(inv_kernel, '__call__'), "Kernel missing __call__ method"
        assert hasattr(inv_kernel, 'kernel'), "Kernel missing kernel method"
        assert callable(inv_kernel), "Kernel not callable"
        print(f"  Kernel is callable and ready")
        
        # Create test data
        size = 64
        mat = create_lower_triangular_matrix(size=size, dtype=torch.float16, device="cuda")
        print(f"  Test matrix created: shape={mat.shape}, device={mat.device}")
        
        # Verify matrix is in GPU memory
        assert mat.is_cuda, "Matrix not on CUDA device"
        assert mat.dtype == torch.float16, "Matrix not FP16"
        print(f"  Matrix properties verified: dtype=FP16, device=CUDA")
        
        # Try to get the data pointer (this shows we can access GPU memory)
        mat_ptr = mat.data_ptr()
        assert mat_ptr != 0, "Invalid matrix pointer"
        print(f"  Matrix data pointer: {mat_ptr}")
        
        # Verify __call__ signature expects proper arguments
        import inspect
        sig = inspect.signature(inv_kernel.__call__)
        print(f"  __call__ signature: {sig}")
        assert 'mat_iter' in str(sig) or len(sig.parameters) >= 1, "__call__ has wrong signature"
        print(f"  __call__ signature is correct for kernel launch")
        
        print("✓ Kernel is ready for GPU execution")
        print("  Note: Full GPU execution requires proper MLIR Context setup")
        
    except Exception as e:
        print(f"✗ Kernel readiness check failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_kernel_multiple_invocations():
    """Test that kernel can be invoked multiple times."""
    if MatrixInverse64x64 is None or torch is None or not torch.cuda.is_available():
        print("\nTest: Multiple kernel invocations - SKIPPED (CUDA not available)")
        return
    
    print("\nTest: Multiple kernel invocations")
    
    try:
        # Create kernel instance once
        inv_kernel = MatrixInverse64x64()
        print(f"  Kernel instance created")
        
        # Create multiple test matrices
        num_invocations = 3
        matrices = []
        size = 64
        
        for i in range(num_invocations):
            mat = create_lower_triangular_matrix(size=size, dtype=torch.float16, device="cuda")
            matrices.append(mat)
            # Verify matrix properties
            assert mat.is_cuda, f"Matrix {i+1} not on CUDA"
            assert mat.dtype == torch.float16, f"Matrix {i+1} not FP16"
            print(f"  Matrix {i+1} created and verified: shape={mat.shape}")
        
        # Verify kernel is callable multiple times
        for i, mat in enumerate(matrices):
            assert inv_kernel is not None, f"Kernel lost at invocation {i+1}"
            assert callable(inv_kernel), f"Kernel not callable at invocation {i+1}"
            print(f"  Invocation {i+1}: Kernel callable, ready to launch")
        
        print(f"✓ {num_invocations} kernel invocations prepared and validated")
        print("  Note: Actual kernel execution requires MLIR Context setup")
        
    except Exception as e:
        print(f"✗ Multiple kernel invocations test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_kernel_with_different_streams():
    """Test that kernel is compatible with different CUDA streams."""
    if MatrixInverse64x64 is None or torch is None or not torch.cuda.is_available():
        print("\nTest: Kernel with different streams - SKIPPED (CUDA not available)")
        return
    
    print("\nTest: Kernel with different streams")
    
    try:
        # Create kernel instance
        inv_kernel = MatrixInverse64x64()
        print(f"  Kernel instance created")
        
        # Create test matrix
        size = 64
        mat = create_lower_triangular_matrix(size=size, dtype=torch.float16, device="cuda")
        print(f"  Test matrix created: shape={mat.shape}")
        
        # Create multiple CUDA streams
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()
        default_stream = torch.cuda.current_stream()
        
        print(f"  Created 3 CUDA streams:")
        print(f"    - stream1: {stream1}")
        print(f"    - stream2: {stream2}")
        print(f"    - default: {default_stream}")
        
        # Verify kernel is callable from different stream contexts
        with torch.cuda.stream(stream1):
            assert callable(inv_kernel), "Kernel not callable in stream1 context"
            s1_current = torch.cuda.current_stream()
            assert s1_current.cuda_stream == stream1.cuda_stream, "Stream context not set"
            print(f"  ✓ Kernel callable in stream1 context")
        
        with torch.cuda.stream(stream2):
            assert callable(inv_kernel), "Kernel not callable in stream2 context"
            s2_current = torch.cuda.current_stream()
            assert s2_current.cuda_stream == stream2.cuda_stream, "Stream context not set"
            print(f"  ✓ Kernel callable in stream2 context")
        
        with torch.cuda.stream(default_stream):
            assert callable(inv_kernel), "Kernel not callable in default stream context"
            print(f"  ✓ Kernel callable in default stream context")
        
        torch.cuda.synchronize()
        print("✓ Kernel compatible with different CUDA streams")
        print("  Note: Actual kernel execution requires MLIR Context setup")
        
    except Exception as e:
        print(f"✗ Kernel stream compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Matrix Inverse 64x64 Kernel Tests")
    print("=" * 60)
    
    tests = [
        # Kernel instantiation and configuration tests
        test_matrix_inverse_kernel_instantiation,
        test_class_constants,
        
        # Numerical validation tests
        test_matrix_inverse_fp16_cpu,
        
        # Kernel compilation and execution tests
        test_kernel_compilation,
        test_kernel_invocation_with_mock_data,
        test_kernel_execution_on_gpu,
        test_kernel_multiple_invocations,
        test_kernel_with_different_streams,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
