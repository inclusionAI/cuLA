"""
根据Kernel代码直接翻译的CPU版本的Schur补集算法

对于分块下三角矩阵的Schur补集求逆：
对于 2x2 分块矩阵：
[A  0]
[C  D]

Schur补集是D（因为是下三角矩阵）。

算法：
1. 对A求逆 -> A_inv
2. 对D求逆 -> D_inv  
3. 更新C的位置为: -D_inv @ C @ A_inv

关键是对角线上已经有了逆，只需要更新下三角部分。

4个阶段分别处理：
- Stage 1: 8 个 8x8 块
- Stage 2: 4 个 16x16 块（每个由 2x2 个 8x8 块组成）
- Stage 3: 2 个 32x32 块（每个由 2x2 个 16x16 块组成）
- Stage 4: 1 个 64x64 块（由 2x2 个 32x32 块组成）
"""

import torch
import numpy as np
import sys
import time

# Try to import kernel for comparison
try:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    import cutlass.torch as cutlass_torch
    from flashla.inv import MatrixInverse64x64
    KERNEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import kernel modules: {e}")
    KERNEL_AVAILABLE = False


def cpu_stage1(A):
    """
    Stage 1: 计算8个8x8对角块的逆
    """
    A_inv = A.clone()  # 保持原矩阵的下三角结构，更新为逆
    
    # 对每个8x8对角块求逆
    for i in range(8):
        start = i * 8
        end = start + 8
        block = A[start:end, start:end]
        block_inv = torch.linalg.inv(block)
        A_inv[start:end, start:end] = block_inv
    
    return A_inv


def cpu_stage2(A, A_inv_stage1):
    """
    Stage 2: 处理4个16x16对角块
    每个16x16块分为2x2个8x8块
    
    对于第i个16x16块：
    [A_inv_00    0      ]  (已经是逆)
    [C          D_inv   ]  (需要更新D_inv和C)
    
    D_inv = inv(D)
    C_new = -D_inv @ C @ A_inv_00
    """
    A_inv = A_inv_stage1.clone()
    
    # 处理4个16x16对角块
    for block_idx in range(4):
        # 16x16块的左上角在（block_idx*16, block_idx*16）
        # 分为4个8x8块
        start_16 = block_idx * 16
        
        # 左上 8x8 块 (0,0)
        r00, c00 = start_16, start_16
        r10, c10 = start_16 + 8, start_16
        r11, c11 = start_16 + 8, start_16 + 8
        
        A_inv_00 = A_inv[r00:r00+8, c00:c00+8]      # 已是逆（Stage 1的结果）
        C_orig = A[r10:r10+8, c10:c10+8]            # 原矩阵的左下块
        D_orig = A[r11:r11+8, c11:c11+8]            # 原矩阵的右下块
        
        # 右下块求逆（Schur补集）
        D_inv = torch.linalg.inv(D_orig)
        
        # 更新
        A_inv[r11:r11+8, c11:c11+8] = D_inv
        A_inv[r10:r10+8, c10:c10+8] = -D_inv @ C_orig @ A_inv_00
    
    return A_inv


def cpu_stage3(A, A_inv_stage2):
    """
    Stage 3: 处理2个32x32对角块
    每个32x32块分为2x2个16x16块
    逻辑同Stage 2
    """
    A_inv = A_inv_stage2.clone()
    
    # 处理2个32x32对角块
    for block_idx in range(2):
        start_32 = block_idx * 32
        
        r00, c00 = start_32, start_32
        r10, c10 = start_32 + 16, start_32
        r11, c11 = start_32 + 16, start_32 + 16
        
        A_inv_00 = A_inv[r00:r00+16, c00:c00+16]
        C_orig = A[r10:r10+16, c10:c10+16]
        D_orig = A[r11:r11+16, c11:c11+16]
        
        D_inv = torch.linalg.inv(D_orig)
        
        A_inv[r11:r11+16, c11:c11+16] = D_inv
        A_inv[r10:r10+16, c10:c10+16] = -D_inv @ C_orig @ A_inv_00
    
    return A_inv


def cpu_stage4(A, A_inv_stage3):
    """
    Stage 4: 最后的64x64块
    分为2x2个32x32块
    """
    A_inv = A_inv_stage3.clone()
    
    r00, c00 = 0, 0
    r10, c10 = 32, 0
    r11, c11 = 32, 32
    
    A_inv_00 = A_inv[r00:r00+32, c00:c00+32]
    C_orig = A[r10:r10+32, c10:c10+32]
    D_orig = A[r11:r11+32, c11:c11+32]
    
    D_inv = torch.linalg.inv(D_orig)
    
    A_inv[r11:r11+32, c11:c11+32] = D_inv
    A_inv[r10:r10+32, c10:c10+32] = -D_inv @ C_orig @ A_inv_00
    
    return A_inv




def verify_inverse(A, A_inv, name=""):
    """验证A * A_inv是否接近单位矩阵"""
    I_recon = A @ A_inv
    I_true = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    error = torch.norm(I_recon - I_true)
    print(f"{name:40s} | Reconstruction Error: {error:.6e}")
    return error


# ============================================================================
# 主测试
# ============================================================================

torch.manual_seed(42)
device = "cuda"
dtype = torch.float32  # 用FP32来验证CPU baseline的正确性
size = 64

# Method 1: Simple random matrix (original test)
print("="*90)
print("Test 1: Simple random lower triangular matrix")
print("="*90)

# 创建测试矩阵
mat = torch.eye(size, dtype=dtype, device=device)
strict_lower = torch.tril(
    torch.randn(size, size, dtype=dtype, device=device) * 0.1,
    diagonal=-1
)
A = mat + strict_lower

# 获取真实逆矩阵
A_cpu_full_inv = torch.linalg.inv(A)

print()
print("CPU Baseline Schur补集 4-Stage 验证")
print("-"*90)

# 验证每个stage
A_inv_stage1 = cpu_stage1(A)
err1 = verify_inverse(A, A_inv_stage1, "Stage 1 (8 blocks of 8x8)")

A_inv_stage2 = cpu_stage2(A, A_inv_stage1)
err2 = verify_inverse(A, A_inv_stage2, "Stage 2 (4 blocks of 16x16)")

A_inv_stage3 = cpu_stage3(A, A_inv_stage2)
err3 = verify_inverse(A, A_inv_stage3, "Stage 3 (2 blocks of 32x32)")

A_inv_stage4 = cpu_stage4(A, A_inv_stage3)
err4 = verify_inverse(A, A_inv_stage4, "Stage 4 (1 block of 64x64)")

# 验证最终结果
print()
final_diff = torch.norm(A_inv_stage4 - A_cpu_full_inv)
print(f"Stage 4 vs torch.linalg.inv: {final_diff:.6e}")
verify_inverse(A, A_cpu_full_inv, "torch.linalg.inv (Ground Truth)")

print()
if final_diff < 1e-5:
    print("✓ CPU Baseline Schur补集实现完全正确!")
else:
    print("✗ CPU Baseline 实现有问题" if final_diff > 1e-2 else "⚠ CPU Baseline 精度问题")


# Method 2: KDA-style M matrix (from k, g, beta)
print()
print()
print("="*90)
print("Test 2: KDA-style M matrix (from k, g, beta)")
print("="*90)

# Import necessary functions for KDA inputs
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from fla.ops.utils import chunk_local_cumsum
    from fla.modules.l2norm import l2norm_fwd
    from fla.ops.utils.constant import RCP_LN2
    KDA_AVAILABLE = True
except ImportError as e:
    print(f"⚠ fla modules not available: {e}")
    KDA_AVAILABLE = False

if KDA_AVAILABLE:
    # Setup parameters matching validate_inverse_v2.py
    B, S, H, D = 1, 64, 1, 128
    chunk_size = 64
    dtype_kda = torch.bfloat16
    
    # Generate inputs (same seed as validate_inverse_v2.py)
    torch.manual_seed(42)
    q = torch.randn(B, S, H, D, dtype=dtype_kda, device=device)
    k = torch.randn(B, S, H, D, dtype=dtype_kda, device=device)
    v = torch.randn(B, S, H, D, dtype=dtype_kda, device=device)
    g_raw = torch.randn(B, S, H, D, dtype=dtype_kda, device=device)
    beta_raw = torch.randn(B, S, H, dtype=torch.float, device=device)
    
    # Process inputs
    g = F.logsigmoid(g_raw.float())
    beta = beta_raw.sigmoid()
    
    # Compute g_cumsum
    g_cumsum = chunk_local_cumsum(
        g=g,
        chunk_size=chunk_size,
        scale=RCP_LN2,
        cu_seqlens=None,
        chunk_indices=None
    )
    
    # Apply l2norm
    q, _ = l2norm_fwd(q)
    k, _ = l2norm_fwd(k)
    
    # Build M matrix (single chunk, covering all 64 elements)
    k_f32 = k.float()
    g_cumsum_f32 = g_cumsum.float()
    beta_f32 = beta.float()
    
    # Compute k*exp2(g) and k*exp2(-g)
    k_exp_g = k_f32 * torch.exp2(g_cumsum_f32)  # [B, S, H, D]
    k_exp_neg_g = k_f32 * torch.exp2(-g_cumsum_f32)  # [B, S, H, D]
    
    # Reshape to [B, H, S, D] for matrix multiplication
    k_exp_g_bh = k_exp_g.permute(0, 2, 1, 3)
    k_exp_neg_g_bh = k_exp_neg_g.permute(0, 2, 1, 3)
    
    # Compute KK = [B, H, S, S]
    KK = torch.matmul(k_exp_g_bh, k_exp_neg_g_bh.transpose(-2, -1))
    
    # Build M matrix: diagonal=1, strict_lower=beta*KK
    M_kda = torch.zeros(B, H, 64, 64, dtype=torch.float32, device=device)
    M_kda[:, :, torch.arange(64), torch.arange(64)] = 1.0
    
    for i in range(64):
        for j in range(i):
            M_kda[:, :, i, j] = beta_f32[:, i, :] * KK[:, :, i, j]
    
    # Extract single matrix for testing [64, 64]
    A_kda = M_kda[0, 0]
    
    print()
    print("M matrix from KDA inputs:")
    print(f"  Shape: {A_kda.shape}")
    print(f"  Dtype: {A_kda.dtype}")
    print(f"  Diagonal values: {A_kda.diag()[:8]}")
    print(f"  Max off-diagonal: {torch.tril(A_kda, -1).abs().max():.6e}")
    
    # Compute ground truth inverse
    A_kda_full_inv = torch.linalg.inv(A_kda)
    
    print()
    print("CPU Baseline Schur补集 4-Stage 验证 (KDA M matrix)")
    print("-"*90)
    
    # 验证每个stage
    A_kda_inv_stage1 = cpu_stage1(A_kda)
    err1 = verify_inverse(A_kda, A_kda_inv_stage1, "Stage 1 (8 blocks of 8x8)")
    
    A_kda_inv_stage2 = cpu_stage2(A_kda, A_kda_inv_stage1)
    err2 = verify_inverse(A_kda, A_kda_inv_stage2, "Stage 2 (4 blocks of 16x16)")
    
    A_kda_inv_stage3 = cpu_stage3(A_kda, A_kda_inv_stage2)
    err3 = verify_inverse(A_kda, A_kda_inv_stage3, "Stage 3 (2 blocks of 32x32)")
    
    A_kda_inv_stage4 = cpu_stage4(A_kda, A_kda_inv_stage3)
    err4 = verify_inverse(A_kda, A_kda_inv_stage4, "Stage 4 (1 block of 64x64)")
    
    # 验证最终结果
    print()
    final_diff_kda = torch.norm(A_kda_inv_stage4 - A_kda_full_inv)
    print(f"Stage 4 vs torch.linalg.inv: {final_diff_kda:.6e}")
    verify_inverse(A_kda, A_kda_full_inv, "torch.linalg.inv (Ground Truth)")
    
    # Detailed error analysis with KDA matrix
    print()
    print("Detailed Error Analysis (Stage 4 vs Ground Truth):")
    print("-"*90)
    abs_err = (A_kda_inv_stage4 - A_kda_full_inv).abs()
    rel_err = abs_err / (A_kda_full_inv.abs() + 1e-8)
    
    print(f"  Absolute error:")
    print(f"    Max:    {abs_err.max():.6e}")
    print(f"    Mean:   {abs_err.mean():.6e}")
    print(f"    Median: {abs_err.median():.6e}")
    
    print(f"  Relative error (all elements):")
    print(f"    Max:    {rel_err.max():.6e}")
    print(f"    Mean:   {rel_err.mean():.6e}")
    print(f"    Median: {rel_err.median():.6e}")
    
    # Distribution
    high_rel_10 = (rel_err > 0.1).sum().item()
    high_rel_100 = (rel_err > 1.0).sum().item()
    total = rel_err.numel()
    
    print(f"  Relative error distribution:")
    print(f"    > 10%:  {high_rel_10}/{total} ({100*high_rel_10/total:.2f}%)")
    print(f"    > 100%: {high_rel_100}/{total} ({100*high_rel_100/total:.2f}%)")
    
    print()
    if final_diff_kda < 1e-5:
        print("✓ CPU Baseline with KDA M matrix: 完全正确!")
    else:
        print("✗ CPU Baseline 实现有问题" if final_diff_kda > 1e-2 else "⚠ CPU Baseline 精度问题")


# ============================================================================
# Kernel vs CPU Baseline 对比
# ============================================================================

if KERNEL_AVAILABLE and KDA_AVAILABLE:
    print()
    print()
    print("="*90)
    print("inv.py Kernel vs CPU Baseline 对比 (KDA M matrix)")
    print("="*90)
    print()
    
    # Prepare FP16 input for kernel
    A_kda_fp16 = A_kda.to(torch.float16).contiguous()
    
    # Compile kernel
    print("Compiling kernel...")
    try:
        inv_kernel = MatrixInverse64x64(acc_dtype=cutlass.Float32)
        print("✓ Kernel compiled successfully")
    except Exception as e:
        print(f"✗ Kernel compilation failed: {e}")
        import traceback
        traceback.print_exc()
        KERNEL_AVAILABLE = False
    
    if KERNEL_AVAILABLE:
        print()
        print("Testing Stage 4 (full 64x64 matrix)...")
        print("-"*90)
        
        try:
            # Prepare input
            mat_input = A_kda_fp16.clone()
            mat_cute = from_dlpack(mat_input)
            stream = cutlass_torch.default_stream()
            
            # Compile and run
            compiled = cute.compile(inv_kernel, mat_cute, stream)
            
            # Execute
            start = time.time()
            compiled(mat_cute, stream)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            # Get result
            inv_kernel_fp16 = mat_input
            inv_kernel_fp32 = inv_kernel_fp16.float()
            
            print(f"✓ Kernel executed in {elapsed*1000:.3f} ms")
            
            # Compare with CPU Stage 4 result
            cpu_result = A_kda_inv_stage4
            
            abs_err = (inv_kernel_fp32 - cpu_result).abs()
            rel_err = abs_err / (cpu_result.abs() + 1e-8)
            
            print()
            print("Error Analysis (Kernel vs CPU Schur):")
            print(f"  Absolute error:")
            print(f"    Max:    {abs_err.max():.6e}")
            print(f"    Mean:   {abs_err.mean():.6e}")
            print(f"    Median: {abs_err.median():.6e}")
            
            print(f"  Relative error (all elements):")
            print(f"    Max:    {rel_err.max():.6e}")
            print(f"    Mean:   {rel_err.mean():.6e}")
            print(f"    Median: {rel_err.median():.6e}")
            
            # Distribution analysis
            high_rel_10 = (rel_err > 0.1).sum().item()
            high_rel_100 = (rel_err > 1.0).sum().item()
            total = rel_err.numel()
            
            print(f"  Relative error distribution:")
            print(f"    > 10%:  {high_rel_10}/{total} ({100*high_rel_10/total:.2f}%)")
            print(f"    > 100%: {high_rel_100}/{total} ({100*high_rel_100/total:.2f}%)")
            
            # Show samples with high relative error
            if high_rel_100 > 0:
                high_mask = rel_err > 1.0
                indices = torch.nonzero(high_mask)[:5]
                print()
                print(f"  Sample elements with rel_err > 100%:")
                print(f"    {'Row':<6} {'Col':<6} {'Kernel':<14} {'CPU':<14} {'Abs Err':<14} {'Rel Err':<10}")
                for idx in indices:
                    r, c = idx[0].item(), idx[1].item()
                    k_val = inv_kernel_fp32[r, c].item()
                    c_val = cpu_result[r, c].item()
                    a_err = abs_err[r, c].item()
                    r_err = rel_err[r, c].item()
                    print(f"    {r:<6} {c:<6} {k_val:<14.6e} {c_val:<14.6e} {a_err:<14.6e} {r_err:<10.2%}")
            
            # Validation
            abs_pass = abs_err.max() <= 1e-2
            rel_pass = rel_err.max() <= 0.1
            
            print()
            print("Validation:")
            print(f"  Abs error <= 1e-2:  {'✓ PASS' if abs_pass else '✗ FAIL'}")
            print(f"  Rel error <= 10%:   {'✓ PASS' if rel_pass else '✗ FAIL'}")
            
            # Summary
            print()
            print("="*90)
            if abs_pass:
                print("✓ Kernel vs CPU Schur: Absolute error acceptable (< 1e-2)")
                if not rel_pass:
                    print("⚠ Note: High relative errors occur on very small baseline values")
            else:
                print("✗ Kernel vs CPU Schur: Absolute error too large")
            print("="*90)
            
        except Exception as e:
            print(f"✗ Kernel execution failed: {e}")
            import traceback
            traceback.print_exc()
elif KERNEL_AVAILABLE and not KDA_AVAILABLE:
    print()
    print("="*90)
    print("⚠ Kernel comparison with KDA matrix skipped (flashla.utils not available)")
    print("="*90)
elif not KERNEL_AVAILABLE:
    print()
    print("="*90)
    print("⚠ Kernel comparison skipped (kernel modules not available)")
    print("="*90)

