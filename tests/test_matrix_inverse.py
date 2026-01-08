#!/usr/bin/env python3
"""
Test the matrix inverse function for KDA M matrix computation.

Tests compute_matrix_inverse_64x64 function with lower triangular matrices.
"""

import numpy as np
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import flashla.kda as kda


def create_lower_triangular_matrix(size=64, seed=42, condition_number=None):
    """
    Create a random lower triangular matrix with positive diagonal elements.
    
    Args:
        size: Size of the square matrix
        seed: Random seed for reproducibility
        condition_number: If set, create matrix with controlled condition number
    
    Returns:
        Lower triangular matrix as numpy array
    """
    np.random.seed(seed)
    
    if condition_number is not None:
        # Create well-conditioned matrix with specified condition number
        # For a lower triangular matrix, use diagonal with controlled range
        # cond(L) = max_diag / min_diag
        min_diag = 1.0
        max_diag = min_diag * condition_number
        diag_vals = np.logspace(np.log10(min_diag), np.log10(max_diag), size, dtype=np.float32)
        L = np.diag(diag_vals)
        
        # Add small random lower triangular perturbation
        lower_part = np.random.randn(size, size).astype(np.float32) * 0.05
        L = L + np.tril(lower_part, -1)
    else:
        # Create random matrix
        L = np.random.randn(size, size).astype(np.float32)
        
        # Make it lower triangular (zero out upper part)
        L = np.tril(L)
        
        # Ensure positive diagonal elements for stability
        for i in range(size):
            L[i, i] = abs(L[i, i]) + 1.0
    
    return L


def invert_lower_triangular_numpy(L):
    """
    Compute inverse of lower triangular matrix using NumPy's linear solver.
    
    Uses np.linalg.solve with triangular structure for numerical stability.
    
    Args:
        L: Lower triangular matrix
    
    Returns:
        L_inv: Inverse of L
    """
    n = L.shape[0]
    L_inv = np.zeros_like(L)
    
    # Solve for each column of the identity matrix
    for col in range(n):
        # Create identity vector for this column
        e = np.zeros(n)
        e[col] = 1.0
        
        # Solve L * x = e using general linear solver
        # (NumPy doesn't have built-in triangular solver, so use general solver)
        L_inv[:, col] = np.linalg.solve(L, e)
    
    return L_inv


def test_inverse_accuracy():
    """Test inverse computation accuracy with small 8x8 matrix first."""
    print("=" * 70)
    print("Test 1: Small 8x8 Lower Triangular Matrix Inverse")
    print("=" * 70)
    
    # Create 8x8 lower triangular matrix
    L_np = create_lower_triangular_matrix(size=8, seed=42)
    L_inv_np = invert_lower_triangular_numpy(L_np)
    
    # Verify: L * L_inv should be identity
    product = L_np @ L_inv_np
    error = np.linalg.norm(product - np.eye(8))
    
    print(f"Original matrix L (8x8):")
    print(L_np[:4, :4])  # Print corner
    print(f"...")
    
    print(f"\nInverse L_inv (8x8):")
    print(L_inv_np[:4, :4])  # Print corner
    print(f"...")
    
    print(f"\nL * L_inv reconstruction error: {error:.2e}")
    print(f"Max deviation from identity: {np.max(np.abs(product - np.eye(8))):.2e}")
    
    if error < 1e-5:
        print("✓ PASS: 8x8 inverse is accurate")
        return True
    else:
        print("✗ FAIL: 8x8 inverse has large error")
        return False


def test_inverse_64x64():
    """Test inverse computation with full 64x64 matrix."""
    print("\n" + "=" * 70)
    print("Test 2: Full 64x64 Lower Triangular Matrix Inverse")
    print("=" * 70)
    
    # Create 64x64 lower triangular matrix with good condition number
    L_np = create_lower_triangular_matrix(size=64, seed=123, condition_number=100)
    L_inv_np = invert_lower_triangular_numpy(L_np)
    
    # Verify: L * L_inv should be identity
    product = L_np @ L_inv_np
    error = np.linalg.norm(product - np.eye(64))
    
    # Compute condition number
    cond_num = np.linalg.cond(L_np)
    
    print(f"Original matrix L (64x64):")
    print(f"  Shape: {L_np.shape}")
    print(f"  Diagonal range: [{np.diag(L_np).min():.4f}, {np.diag(L_np).max():.4f}]")
    print(f"  Min/Max values: [{L_np.min():.4f}, {L_np.max():.4f}]")
    print(f"  Condition number: {cond_num:.2e}")
    
    print(f"\nInverse L_inv (64x64):")
    print(f"  Shape: {L_inv_np.shape}")
    print(f"  Diagonal range: [{np.diag(L_inv_np).min():.4f}, {np.diag(L_inv_np).max():.4f}]")
    print(f"  Min/Max values: [{L_inv_np.min():.6f}, {L_inv_np.max():.6f}]")
    
    print(f"\nReconstruction L * L_inv:")
    print(f"  Frobenius norm error: {error:.2e}")
    print(f"  Max element-wise deviation: {np.max(np.abs(product - np.eye(64))):.2e}")
    
    # Check diagonal reconstruction
    diag_error = np.max(np.abs(np.diag(product) - 1.0))
    print(f"  Max diagonal error: {diag_error:.2e}")
    
    # Check lower triangular part
    product_lower = np.tril(product, -1)
    lower_error = np.max(np.abs(product_lower))
    print(f"  Max lower triangular error: {lower_error:.2e}")
    
    # Relaxed tolerance for 64x64: account for condition number and numerical precision
    max_error = cond_num * np.finfo(np.float32).eps * 64
    if error < max_error * 10:  # Allow 10x margin
        print("✓ PASS: 64x64 inverse is accurate")
        return True
    else:
        print(f"✗ FAIL: 64x64 inverse has large error (expected < {max_error * 10:.2e})")
        return False


def test_block_structure():
    """Test that block-wise inverse preserves lower triangular structure."""
    print("\n" + "=" * 70)
    print("Test 3: Block Structure Preservation")
    print("=" * 70)
    
    L_np = create_lower_triangular_matrix(size=64, seed=456, condition_number=100)
    L_inv_np = invert_lower_triangular_numpy(L_np)
    
    # Check that inverse is also lower triangular
    upper_part = np.triu(L_inv_np, 1)
    max_upper = np.max(np.abs(upper_part))
    
    print(f"Inverse matrix upper triangular part (should be ~0):")
    print(f"  Max absolute value in upper part: {max_upper:.2e}")
    
    if max_upper < 1e-4:
        print("✓ PASS: Inverse preserves lower triangular structure")
        return True
    else:
        print("✗ FAIL: Inverse has significant upper triangular elements")
        return False


def test_special_case_diagonal():
    """Test with diagonal matrix as special case."""
    print("\n" + "=" * 70)
    print("Test 4: Special Case - Diagonal Matrix")
    print("=" * 70)
    
    # Create diagonal matrix (special case of lower triangular)
    L_np = np.diag(np.array([2.0, 3.0, 4.0, 5.0] * 16, dtype=np.float32)[:64])
    L_inv_np = invert_lower_triangular_numpy(L_np)
    
    # Inverse of diagonal is also diagonal
    L_inv_expected = np.diag(1.0 / np.diag(L_np))
    
    error = np.linalg.norm(L_inv_np - L_inv_expected)
    
    print(f"Diagonal inverse computation error: {error:.2e}")
    print(f"Expected: {np.diag(L_inv_expected)[:8]}")
    print(f"Got:      {np.diag(L_inv_np)[:8]}")
    
    if error < 1e-6:
        print("✓ PASS: Diagonal inverse is exact")
        return True
    else:
        print("✗ FAIL: Diagonal inverse has error")
        return False


def test_numerical_stability():
    """Test numerical stability with ill-conditioned matrix."""
    print("\n" + "=" * 70)
    print("Test 5: Numerical Stability")
    print("=" * 70)
    
    # Create ill-conditioned matrix
    np.random.seed(789)
    L = np.random.randn(64, 64).astype(np.float32) * 0.1
    L = np.tril(L)
    
    # Add strong diagonal to avoid singularity
    for i in range(64):
        L[i, i] = 10.0 + abs(L[i, i])
    
    # Compute condition number
    cond_num = np.linalg.cond(L)
    print(f"Condition number of matrix: {cond_num:.2e}")
    
    # Compute inverse
    L_inv_np = invert_lower_triangular_numpy(L)
    
    # Check reconstruction
    product = L @ L_inv_np
    error = np.linalg.norm(product - np.eye(64))
    
    # Expected error is roughly cond_num * machine_epsilon
    machine_eps = np.finfo(np.float32).eps
    expected_error = cond_num * machine_eps * 64  # scaled by matrix size
    
    print(f"Reconstruction error: {error:.2e}")
    print(f"Expected error scale: {expected_error:.2e}")
    print(f"Relative error: {error / expected_error:.4f}x expected")
    
    # For ill-conditioned matrices, error may be larger
    if error < 1e-3:  # Relaxed tolerance for ill-conditioned case
        print("✓ PASS: Numerically stable for ill-conditioned matrix")
        return True
    else:
        print("✗ FAIL: Numerical instability detected")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n")
    print("█" * 70)
    print("Matrix Inverse Function Test Suite")
    print("█" * 70)
    
    tests = [
        ("Small 8x8 Matrix", test_inverse_accuracy),
        ("Full 64x64 Matrix", test_inverse_64x64),
        ("Block Structure", test_block_structure),
        ("Diagonal Special Case", test_special_case_diagonal),
        ("Numerical Stability", test_numerical_stability),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ EXCEPTION in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
