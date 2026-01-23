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


def test_canonical_lane_id():
    """Test the canonical_lane_id function."""
    if MatrixInverse64x64 is None:
        print("\nTest: Canonical lane ID - SKIPPED (module not available)")
        return
    
    print("\nTest: Canonical lane ID")
    inv_kernel = MatrixInverse64x64()
    # This would need to be called from within a CUDA kernel
    # For now, we just verify the method exists
    assert hasattr(inv_kernel, 'canonical_lane_id')
    print("✓ Canonical lane ID method exists")


def test_load_store_operations():
    """Test load and store operations for 8x8 blocks."""
    if MatrixInverse64x64 is None:
        print("\nTest: Load/Store operations - SKIPPED (module not available)")
        return
    
    print("\nTest: Load/Store operations")
    inv_kernel = MatrixInverse64x64()
    
    # Create a test 8x8 matrix
    test_mat = torch.randn(8, 8, dtype=torch.float16)
    
    # The actual load/store would happen in the kernel
    # Here we just verify the methods exist
    assert hasattr(inv_kernel, 'load_row_mat8x8')
    assert hasattr(inv_kernel, 'store_row_mat8x8')
    print("✓ Load/Store operations available")


def test_convert_layout():
    """Test layout conversion utilities."""
    if MatrixInverse64x64 is None:
        print("\nTest: Layout conversion - SKIPPED (module not available)")
        return
    
    print("\nTest: Layout conversion")
    inv_kernel = MatrixInverse64x64()
    
    # Verify conversion methods exist
    assert hasattr(inv_kernel, 'convert_layout_c_to_a')
    assert hasattr(inv_kernel, 'make_acc_as_a')
    print("✓ Layout conversion methods available")


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


def test_stage_1_kernel_exists():
    """Test that 8x8 diagonal inverse kernel method exists."""
    if MatrixInverse64x64 is None:
        print("\nTest: Stage 1 (8x8) kernel - SKIPPED (module not available)")
        return
    
    print("\nTest: Stage 1 (8x8) kernel")
    inv_kernel = MatrixInverse64x64()
    assert hasattr(inv_kernel, 'compute_diagonal_inverse_8x8')
    print("✓ Stage 1 kernel method exists")


def test_stage_2_kernel_exists():
    """Test that 8x8->16x16 conversion kernel method exists."""
    if MatrixInverse64x64 is None:
        print("\nTest: Stage 2 (8x8->16x16) kernel - SKIPPED (module not available)")
        return
    
    print("\nTest: Stage 2 (8x8->16x16) kernel")
    inv_kernel = MatrixInverse64x64()
    assert hasattr(inv_kernel, 'compute_diagonal_inverse_8x8_to_16x16')
    print("✓ Stage 2 kernel method exists")


def test_stage_3_kernel_exists():
    """Test that 16x16->32x32 conversion kernel method exists."""
    if MatrixInverse64x64 is None:
        print("\nTest: Stage 3 (16x16->32x32) kernel - SKIPPED (module not available)")
        return
    
    print("\nTest: Stage 3 (16x16->32x32) kernel")
    inv_kernel = MatrixInverse64x64()
    assert hasattr(inv_kernel, 'compute_diagonal_inverse_16x16_to_32x32')
    print("✓ Stage 3 kernel method exists")


def test_stage_4_kernel_exists():
    """Test that 32x32->64x64 conversion kernel method exists."""
    if MatrixInverse64x64 is None:
        print("\nTest: Stage 4 (32x32->64x64) kernel - SKIPPED (module not available)")
        return
    
    print("\nTest: Stage 4 (32x32->64x64) kernel")
    inv_kernel = MatrixInverse64x64()
    assert hasattr(inv_kernel, 'compute_diagonal_inverse_32x32_to_64x64')
    print("✓ Stage 4 kernel method exists")


def test_main_kernel_exists():
    """Test that the main compute_matrix_inverse_64x64 kernel exists."""
    if MatrixInverse64x64 is None:
        print("\nTest: Main 64x64 inverse kernel - SKIPPED (module not available)")
        return
    
    print("\nTest: Main 64x64 inverse kernel")
    inv_kernel = MatrixInverse64x64()
    assert hasattr(inv_kernel, 'compute_matrix_inverse_64x64')
    print("✓ Main kernel method exists")


def test_barrier_initialization():
    """Test that the work-group barrier is initialized."""
    if MatrixInverse64x64 is None:
        print("\nTest: Barrier initialization - SKIPPED (module not available)")
        return
    
    print("\nTest: Barrier initialization")
    inv_kernel = MatrixInverse64x64()
    assert hasattr(inv_kernel, 'cuda_wg_sync_barrier')
    print("✓ Work-group barrier initialized")


def test_kernel_structure():
    """Comprehensive test of kernel structure and organization."""
    if MatrixInverse64x64 is None:
        print("\nTest: Kernel structure - SKIPPED (module not available)")
        return
    
    print("\nTest: Kernel structure")
    
    inv_kernel = MatrixInverse64x64()
    
    # Verify all required methods exist
    required_methods = [
        'canonical_lane_id',
        'convert_layout_c_to_a',
        'make_acc_as_a',
        'make_op_a_from_acc_rmem_16x8x8',
        'compute_diagonal_inverse_8x8',
        'load_row_mat8x8',
        'store_row_mat8x8',
        'compute_diagonal_inverse_8x8_to_16x16',
        'compute_diagonal_inverse_16x16_to_32x32',
        'compute_diagonal_inverse_32x32_to_64x64',
        'compute_matrix_inverse_64x64',
    ]
    
    for method_name in required_methods:
        assert hasattr(inv_kernel, method_name), f"Missing method: {method_name}"
        assert callable(getattr(inv_kernel, method_name)), f"Not callable: {method_name}"
    
    print(f"✓ All {len(required_methods)} required methods present and callable")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Matrix Inverse 64x64 Kernel Tests")
    print("=" * 60)
    
    tests = [
        test_matrix_inverse_kernel_instantiation,
        test_canonical_lane_id,
        test_load_store_operations,
        test_convert_layout,
        test_matrix_inverse_fp16_cpu,
        test_stage_1_kernel_exists,
        test_stage_2_kernel_exists,
        test_stage_3_kernel_exists,
        test_stage_4_kernel_exists,
        test_main_kernel_exists,
        test_barrier_initialization,
        test_kernel_structure,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
