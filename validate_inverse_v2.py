#!/usr/bin/env python3
"""
Validate inverse matrix output from KDA kernel by comparing with torch baseline.

The inverse is computed as: M^-1 where
M[i,j] = 1.0 if i == j (diagonal)
M[i,j] = beta[i] * KK[i,j] if i > j (strict lower triangular)
M[i,j] = 0.0 if i < j (upper triangular)

where KK = (k*exp(g)) @ (k*exp(-g))^T
"""

import torch
import torch.nn.functional as F
import sys

sys.path.insert(0, '/ossfs/workspace/flashla')

from flashla.kda import KDAChunkwise
from fla.modules.l2norm import l2norm_fwd
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_inverse_baseline(k, g, beta, chunk_size=64):
    """
    Compute M^-1 baseline on CPU where M follows KDA's apply_M_transform logic.
    
    Args:
        k: [B, S, H, D] in FP32
        g: [B, S, H, D] in FP32 (logsigmoid of raw gate)
        beta: [B, S, H] in FP32
    
    Returns:
        inverse: [num_chunks, B, H, 64, 64] in FP32
    """
    B, S, H, D = k.shape
    device = k.device
    num_chunks = (S + chunk_size - 1) // chunk_size
    
    inverse_list = []
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, S)
        chunk_len = end_idx - start_idx
        
        # Extract chunk
        k_chunk = k[:, start_idx:end_idx]  # [B, chunk_len, H, D]
        g_chunk = g[:, start_idx:end_idx]  # [B, chunk_len, H, D]
        beta_chunk = beta[:, start_idx:end_idx]  # [B, chunk_len, H]
        
        # Pad to 64 if needed
        if chunk_len < 64:
            pad_len = 64 - chunk_len
            k_chunk = F.pad(k_chunk, (0, 0, 0, 0, 0, pad_len))  # Pad sequence dim
            g_chunk = F.pad(g_chunk, (0, 0, 0, 0, 0, pad_len))
            beta_chunk = F.pad(beta_chunk, (0, 0, 0, pad_len))
        
        # Compute k*exp2(g) and k*exp2(-g)
        k_exp_g = k_chunk * torch.exp2(g_chunk)  # [B, 64, H, D]
        k_exp_neg_g = k_chunk * torch.exp2(-g_chunk)  # [B, 64, H, D]
        
        # Reshape to [B, H, 64, D] for matrix multiplication
        k_exp_g_bh = k_exp_g.permute(0, 2, 1, 3)
        k_exp_neg_g_bh = k_exp_neg_g.permute(0, 2, 1, 3)
        
        # Compute KK = (k*exp2(g)) @ (k*exp2(-g))^T = [B, H, 64, 64]
        KK = torch.matmul(k_exp_g_bh, k_exp_neg_g_bh.transpose(-2, -1))
        
        # Build M matrix following apply_M_transform:
        # Diagonal = 1.0
        # Strict lower triangular = beta[i] * KK[i,j]
        # Upper triangular = 0.0
        M = torch.zeros(B, H, 64, 64, dtype=torch.float32, device=device)
        
        # Set diagonal to 1
        M[:, :, torch.arange(64), torch.arange(64)] = 1.0
        
        # Set strict lower triangular: M[i,j] = beta[i] * KK[i,j] for i > j
        for i in range(64):
            for j in range(i):
                # beta is [B, 64, H], need to get beta[i] for all B and H
                M[:, :, i, j] = beta_chunk[:, i, :] * KK[:, :, i, j]
        
        # Compute inverse
        M_inv = torch.linalg.inv(M)
        
        inverse_list.append(M_inv)
    
    return torch.stack(inverse_list, dim=0)  # [num_chunks, B, H, 64, 64]


def run_kernel(q, k, v, g_cumsum, beta, chunk_size=64, max_inverse_stage=4):
    """Run KDA kernel and return inverse output
    
    Args:
        max_inverse_stage: Maximum inverse stage to execute (1-4)
            1: Only 8x8 diagonal blocks
            2: Up to 16x16 diagonal blocks
            3: Up to 32x32 diagonal blocks
            4: Full 64x64 matrix (default)
    """
    B, S, H, D = q.shape
    dtype = q.dtype
    device = q.device
    
    
    # Allocate outputs
    o = torch.zeros_like(q)
    fstate = torch.zeros(B, H, D, D, dtype=dtype, device=device)
    
    num_chunks = (S + chunk_size - 1) // chunk_size
    inverse = torch.zeros(B, num_chunks, H, 64, 64, dtype=torch.float16, device=device)
    
    # Setup kernel
    scale = D ** (-0.5)
    attn_kernel = KDAChunkwise(
        chunk_size=chunk_size,
        qk_acc_dtype=cutlass.Float32,
        kv_acc_dtype=cutlass.Float32,
        io_dtype=cutlass.BFloat16,
        scale=scale,
        max_inverse_stage=max_inverse_stage,  # NEW: Control inverse stage
    )
    
    # Convert to CuTe
    q_cute = from_dlpack(q)
    k_cute = from_dlpack(k)
    v_cute = from_dlpack(v)
    g_cute = from_dlpack(g_cumsum)
    o_cute = from_dlpack(o)
    beta_cute = from_dlpack(beta)
    inverse_cute = from_dlpack(inverse)
    
    stream = cutlass_torch.default_stream()
    
    # Compile
    print("[*] Compiling kernel...")
    compiled = cute.compile(
        attn_kernel,
        q_cute.iterator,
        k_cute.iterator,
        v_cute.iterator,
        g_cute.iterator,
        o_cute.iterator,
        beta_cute.iterator,
        inverse_cute.iterator,
        (B, S, H, D),
        stream,
    )
    
    # Execute
    print("[*] Executing kernel...")
    compiled(
        q_cute.iterator,
        k_cute.iterator,
        v_cute.iterator,
        g_cute.iterator,
        o_cute.iterator,
        beta_cute.iterator,
        inverse_cute.iterator,
        (B, S, H, D),
        stream,
    )
    torch.cuda.synchronize()
    
    return inverse


def main():
    print("=" * 70)
    print("KDA Kernel Inverse Matrix Validation - Stage-by-Stage")
    print("=" * 70)
    
    B, S, H, D = 1, 64, 1, 128
    chunk_size = 64
    dtype = torch.bfloat16
    device = torch.device('cuda')
    
    set_seed(42)
    
    # Create inputs
    q = torch.randn(B, S, H, D, dtype=dtype, device=device)
    k = torch.randn(B, S, H, D, dtype=dtype, device=device)
    v = torch.randn(B, S, H, D, dtype=dtype, device=device)
    g_raw = torch.randn(B, S, H, D, dtype=dtype, device=device)
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
    
    print(f"\nConfiguration:")
    print(f"  B={B}, S={S}, H={H}, D={D}")
    print(f"  Expected inverse shape: [{B}, {(S + chunk_size - 1) // chunk_size}, {H}, 64, 64]")
    
    # Compute baseline once
    print(f"\n[Step 1] Computing CPU baseline...")
    try:
        k_f32 = k.float()
        g_cumsum_f32 = g_cumsum.float()
        beta_f32 = beta.float()
        inverse_baseline = compute_inverse_baseline(k_f32, g_cumsum_f32, beta_f32, chunk_size)
        print(f"✓ Baseline computed, shape: {inverse_baseline.shape}")
    except Exception as e:
        print(f"✗ Baseline failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    inverse_baseline_reshaped = inverse_baseline.permute(1, 0, 2, 3, 4)  # [B, num_chunks, H, 64, 64]
    
    # Test each stage
    stages_to_test = [1, 2, 3, 4]
    results = []
    
    for stage in stages_to_test:
        print(f"\n{'='*70}")
        print(f"Testing Stage {stage}")
        print(f"{'='*70}")
        
        # Run kernel with specific max stage
        print(f"[Step 2.{stage}] Running kernel with max_inverse_stage={stage}...")
        try:
            inverse_kernel = run_kernel(q, k, v, g_cumsum, beta, chunk_size, max_inverse_stage=stage)
            print(f"✓ Kernel successful, shape: {inverse_kernel.shape}")
        except Exception as e:
            print(f"✗ Kernel failed: {e}")
            results.append({'stage': stage, 'success': False, 'error': str(e)})
            continue
        
        # Process kernel output
        inverse_kernel_rowmajor = inverse_kernel.transpose(-2, -1).contiguous()
        
        # Save Stage 4 kernel output to M.txt in FP16
        if stage == 4:
            output_file = '/ossfs/workspace/flashla/M.txt'
            inv_matrix_fp16 = inverse_kernel_rowmajor[0, 0, 0].cpu().numpy()  # Keep FP16
            with open(output_file, 'w') as f:
                f.write(f"# KDAChunkwise GPU Kernel Inverse Matrix Output (Stage 4, FP16)\n")
                f.write(f"# Shape: {inv_matrix_fp16.shape}\n")
                f.write(f"# Format: Each row is one line, space-separated values\n")
                f.write("#\n")
                for row in inv_matrix_fp16:
                    f.write(' '.join(f'{float(val):.5e}' for val in row) + '\n')
            print(f"  ✓ Saved kernel inverse output to {output_file}")
        
        # Convert to FP32 for validation
        inverse_kernel_f32 = inverse_kernel_rowmajor.float()
        
        # Create mask for which blocks to compare based on stage
        # Stage 1: Only 8 diagonal 8x8 blocks
        # Stage 2: Only 4 diagonal 16x16 blocks
        # Stage 3: Only 2 diagonal 32x32 blocks
        # Stage 4: Full 64x64 matrix
        comparison_mask = torch.zeros(64, 64, dtype=torch.bool, device=inverse_kernel.device)
        
        if stage == 1:
            # 8 diagonal 8x8 blocks
            block_size = 8
            num_blocks = 8
            for i in range(num_blocks):
                row_start = i * block_size
                row_end = row_start + block_size
                col_start = i * block_size
                col_end = col_start + block_size
                comparison_mask[row_start:row_end, col_start:col_end] = True
            print(f"  Comparing only 8 diagonal 8x8 blocks ({comparison_mask.sum().item()}/4096 elements)")
        elif stage == 2:
            # 4 diagonal 16x16 blocks
            block_size = 16
            num_blocks = 4
            for i in range(num_blocks):
                row_start = i * block_size
                row_end = row_start + block_size
                col_start = i * block_size
                col_end = col_start + block_size
                comparison_mask[row_start:row_end, col_start:col_end] = True
            print(f"  Comparing only 4 diagonal 16x16 blocks ({comparison_mask.sum().item()}/4096 elements)")
        elif stage == 3:
            # 2 diagonal 32x32 blocks
            block_size = 32
            num_blocks = 2
            for i in range(num_blocks):
                row_start = i * block_size
                row_end = row_start + block_size
                col_start = i * block_size
                col_end = col_start + block_size
                comparison_mask[row_start:row_end, col_start:col_end] = True
            print(f"  Comparing only 2 diagonal 32x32 blocks ({comparison_mask.sum().item()}/4096 elements)")
        else:  # stage == 4
            # Full matrix
            comparison_mask[:, :] = True
            print(f"  Comparing full 64x64 matrix ({comparison_mask.sum().item()}/4096 elements)")
        
        # Apply mask to both kernel and baseline
        kernel_masked = inverse_kernel_f32[0, 0, 0][comparison_mask]
        baseline_masked = inverse_baseline_reshaped[0, 0, 0][comparison_mask]
        
        # Compute errors on masked region only
        abs_err = (kernel_masked - baseline_masked).abs()
        rel_err = abs_err / (baseline_masked.abs() + 1e-8)
        
        print(f"\nAbsolute error (masked region):")
        print(f"  Max: {abs_err.max():.6e}")
        print(f"  Mean: {abs_err.mean():.6e}")
        print(f"  Median: {abs_err.median():.6e}")
        
        print(f"\nRelative error (all elements):")
        print(f"  Max: {rel_err.max():.6e}")
        print(f"  Mean: {rel_err.mean():.6e}")
        print(f"  Median: {rel_err.median():.6e}")
        
        # Check thresholds
        abs_tol = 1e-2
        rel_tol = 1e-1
        abs_pass = abs_err.max() <= abs_tol
        rel_pass = rel_err.max() <= rel_tol
        
        print(f"\nError checks:")
        print(f"  Absolute: {abs_err.max():.6e} <= {abs_tol:.0e} {'✓' if abs_pass else '✗'}")
        print(f"  Relative: {rel_err.max():.6e} <= {rel_tol:.0e} {'✓' if rel_pass else '✗'}")
        
        overall_pass = abs_pass and rel_pass
        print(f"\n{'✓ PASSED' if overall_pass else '✗ FAILED'}")
        
        results.append({
            'stage': stage,
            'success': True,
            'abs_max': abs_err.max().item(),
            'abs_mean': abs_err.mean().item(),
            'rel_max': rel_err.max().item(),
            'rel_mean': rel_err.mean().item(),
            'passed': overall_pass
        })
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Stage-by-Stage Validation Summary")
    print(f"{'='*70}")
    print(f"\n{'Stage':<8} {'Status':<10} {'Max Abs Err':<15} {'Max Rel Err':<15} {'Result':<10}")
    print(f"{'-'*70}")
    
    for res in results:
        if res['success']:
            status = "✓ OK"
            result = "PASS" if res['passed'] else "FAIL"
            print(f"{res['stage']:<8} {status:<10} {res['abs_max']:<15.6e} {res['rel_max']:<15.6e} {result:<10}")
        else:
            print(f"{res['stage']:<8} {'✗ ERROR':<10} {res['error']}")
    
    all_passed = all(r.get('passed', False) for r in results if r['success'])
    print(f"\n{'='*70}")
    print(f"Overall: {'✓ ALL STAGES PASSED' if all_passed else '✗ SOME STAGES FAILED'}")
    print(f"{'='*70}")
    
    return all_passed
    print(f"  Max: {abs_err.max():.6e}")
    print(f"  Mean: {abs_err.mean():.6e}")
    print(f"  Median: {abs_err.median():.6e}")
    
    print(f"\nRelative error (all elements):")
    print(f"  Max: {rel_err.max():.6e}")
    print(f"  Mean: {rel_err.mean():.6e}")
    print(f"  Median: {rel_err.median():.6e}")
    
    if rel_err_significant.numel() > 0:
        print(f"\nRelative error (|baseline| > 1e-3 only, {rel_err_significant.numel()}/{rel_err.numel()} elements):")
        print(f"  Max: {rel_err_significant.max():.6e}")
        print(f"  Mean: {rel_err_significant.mean():.6e}")
        print(f"  Median: {rel_err_significant.median():.6e}")
    
    # Check error thresholds
    abs_tol = 1e-2
    rel_tol = 1e-1
    abs_pass = abs_err.max() <= abs_tol
    # Use filtered relative error for pass criterion
    rel_pass = rel_err_significant.max() <= rel_tol if rel_err_significant.numel() > 0 else True
    
    print(f"\nError threshold checks:")
    print(f"  Absolute error: {abs_err.max():.6e} <= {abs_tol:.6e} {'✓' if abs_pass else '✗'}")
    if rel_err_significant.numel() > 0:
        print(f"  Relative error (significant): {rel_err_significant.max():.6e} <= {rel_tol:.6e} {'✓' if rel_pass else '✗'}")
    else:
        print(f"  Relative error: N/A (no significant baseline values)")
    
    # Debug: show samples
    print(f"\nDebug - Diagonal comparison:")
    kernel_diag = torch.diagonal(inverse_kernel_f32[0, 0, 0])
    baseline_diag = torch.diagonal(inverse_baseline_reshaped[0, 0, 0])
    print(f"  Kernel diagonal:   {kernel_diag[:10]}")
    print(f"  Baseline diagonal: {baseline_diag[:10]}")
    
    print(f"\nDebug - First few rows:")
    for i in range(min(3, 64)):
        print(f"  Row {i}:")
        print(f"    Kernel:   {inverse_kernel_f32[0, 0, 0, i, :8]}")
        print(f"    Baseline: {inverse_baseline_reshaped[0, 0, 0, i, :8]}")
    
    # Find max error location
    max_err_idx = abs_err.argmax()
    max_err_loc = torch.unravel_index(max_err_idx, abs_err.shape)
    print(f"\nMax error location: {max_err_loc}")
    print(f"  Kernel value:   {inverse_kernel_f32[max_err_loc]:.6e}")
    print(f"  Baseline value: {inverse_baseline_reshaped[max_err_loc]:.6e}")
    print(f"  Absolute error: {abs_err[max_err_loc]:.6e}")
    
    # Block-wise analysis: divide 64x64 into 64 8x8 blocks
    print(f"\n{'='*70}")
    print(f"Block-wise Error Analysis (64x64 → 8x8 blocks)")
    print(f"{'='*70}")
    
    # First analyze by inverse computation stages
    print(f"\n{'='*70}")
    print(f"Stage-wise Error Analysis")
    print(f"{'='*70}")
    print(f"\nInverse computation stages:")
    print(f"  Stage 1: 8x8 diagonal blocks   (8 blocks)")
    print(f"  Stage 2: 16x16 diagonal blocks (4 blocks)")
    print(f"  Stage 3: 32x32 diagonal blocks (2 blocks)")
    print(f"  Stage 4: 64x64 full matrix     (1 block)")
    
    # Define stage regions
    stage_regions = [
        {
            'name': 'Stage 1 (8x8 diagonal blocks)',
            'blocks': [(i, i) for i in range(8)],  # 8 diagonal 8x8 blocks
            'block_size': 8
        },
        {
            'name': 'Stage 2 (16x16 diagonal blocks)',
            'blocks': [(i, i) for i in range(0, 8, 2)],  # 4 diagonal 16x16 blocks
            'block_size': 16
        },
        {
            'name': 'Stage 3 (32x32 diagonal blocks)',
            'blocks': [(i, i) for i in range(0, 8, 4)],  # 2 diagonal 32x32 blocks
            'block_size': 32
        },
        {
            'name': 'Stage 4 (64x64 full matrix)',
            'blocks': [(0, 0)],  # Full matrix
            'block_size': 64
        }
    ]
    
    print(f"\n{'Stage':<30} {'Max Abs Err':<14} {'Mean Abs Err':<14} {'Max Rel Err':<14} {'Mean Rel Err':<14} {'#Significant':<12}")
    print(f"{'-'*100}")
    
    for stage in stage_regions:
        stage_abs_errs = []
        stage_rel_errs = []
        stage_num_sig = 0
        
        for block_row, block_col in stage['blocks']:
            bs = stage['block_size']
            row_start = block_row * bs
            row_end = min(row_start + bs, 64)
            col_start = block_col * bs
            col_end = min(col_start + bs, 64)
            
            block_abs_err = abs_err[0, 0, 0, row_start:row_end, col_start:col_end]
            block_baseline = inverse_baseline_reshaped[0, 0, 0, row_start:row_end, col_start:col_end]
            block_rel_err = block_abs_err / (block_baseline.abs() + 1e-8)
            
            stage_abs_errs.append(block_abs_err)
            stage_rel_errs.append(block_rel_err)
            
            block_mask = block_baseline.abs() > 1e-3
            stage_num_sig += block_mask.sum().item()
        
        # Concatenate all blocks for this stage
        stage_abs_err_combined = torch.cat([e.flatten() for e in stage_abs_errs])
        stage_rel_err_combined = torch.cat([e.flatten() for e in stage_rel_errs])
        
        max_abs = stage_abs_err_combined.max().item()
        mean_abs = stage_abs_err_combined.mean().item()
        max_rel = stage_rel_err_combined.max().item()
        mean_rel = stage_rel_err_combined.mean().item()
        
        print(f"{stage['name']:<30} {max_abs:<14.6e} {mean_abs:<14.6e} {max_rel:<14.6e} {mean_rel:<14.6e} {stage_num_sig:<12}")
    
    # Detailed Stage 3 analysis
    print(f"\n{'='*70}")
    print(f"Stage 3 Detailed Analysis (32x32 blocks)")
    print(f"{'='*70}")
    
    # Stage 3 has 2 diagonal 32x32 blocks
    for block_idx in range(2):
        row_start = block_idx * 32
        row_end = row_start + 32
        col_start = block_idx * 32
        col_end = col_start + 32
        
        print(f"\n32x32 Block [{block_idx},{block_idx}] (rows {row_start}-{row_end-1}, cols {col_start}-{col_end-1}):")
        
        block_32_abs_err = abs_err[0, 0, 0, row_start:row_end, col_start:col_end]
        block_32_baseline = inverse_baseline_reshaped[0, 0, 0, row_start:row_end, col_start:col_end]
        block_32_kernel = inverse_kernel_f32[0, 0, 0, row_start:row_end, col_start:col_end]
        block_32_rel_err = block_32_abs_err / (block_32_baseline.abs() + 1e-8)
        
        # Overall stats for this 32x32 block
        mask_32_sig = block_32_baseline.abs() > 1e-3
        num_sig = mask_32_sig.sum().item()
        
        print(f"  Overall: Max abs err = {block_32_abs_err.max():.6e}, Mean abs err = {block_32_abs_err.mean():.6e}")
        print(f"  Overall: Max rel err = {block_32_rel_err.max():.6e}, Mean rel err = {block_32_rel_err.mean():.6e}")
        print(f"  Significant elements (|baseline| > 1e-3): {num_sig}/1024")
        
        # Subdivide into 4 16x16 blocks
        print(f"\n  Subdivision into 16x16 blocks:")
        print(f"  {'Block':<10} {'Position':<20} {'Max Abs Err':<14} {'Max Rel Err':<14} {'#Significant':<12}")
        print(f"  {'-'*75}")
        
        for sub_row in range(2):
            for sub_col in range(2):
                sub_row_start = row_start + sub_row * 16
                sub_row_end = sub_row_start + 16
                sub_col_start = col_start + sub_col * 16
                sub_col_end = sub_col_start + 16
                
                sub_abs_err = abs_err[0, 0, 0, sub_row_start:sub_row_end, sub_col_start:sub_col_end]
                sub_baseline = inverse_baseline_reshaped[0, 0, 0, sub_row_start:sub_row_end, sub_col_start:sub_col_end]
                sub_rel_err = sub_abs_err / (sub_baseline.abs() + 1e-8)
                
                sub_mask = sub_baseline.abs() > 1e-3
                sub_num_sig = sub_mask.sum().item()
                
                block_name = f"[{sub_row},{sub_col}]"
                position = f"rows {sub_row_start}-{sub_row_end-1}, cols {sub_col_start}-{sub_col_end-1}"
                
                print(f"  {block_name:<10} {position:<20} {sub_abs_err.max():<14.6e} {sub_rel_err.max():<14.6e} {sub_num_sig:<12}")
        
        # Find elements with high relative error in this 32x32 block
        # Check both significant (>1e-3) and moderately small (>1e-5) elements
        for threshold, desc in [(1e-3, "significant (|baseline| > 1e-3)"), (1e-5, "moderately small (1e-5 < |baseline| < 1e-3)")]:
            if threshold == 1e-3:
                high_rel_err_mask = (block_32_rel_err > 0.5) & (block_32_baseline.abs() > threshold)
            else:
                high_rel_err_mask = (block_32_rel_err > 0.5) & (block_32_baseline.abs() > threshold) & (block_32_baseline.abs() <= 1e-3)
            
            if high_rel_err_mask.any():
                num_high = high_rel_err_mask.sum().item()
                print(f"\n  Elements with rel err > 0.5 in {desc}: {num_high}")
                
                # Find locations
                high_err_indices = torch.nonzero(high_rel_err_mask, as_tuple=False)
                print(f"  {'Local Row':<10} {'Local Col':<10} {'Kernel':<14} {'Baseline':<14} {'Abs Err':<14} {'Rel Err':<14}")
                print(f"  {'-'*85}")
                for idx in high_err_indices[:10]:
                    local_row, local_col = idx[0].item(), idx[1].item()
                    kernel_val = block_32_kernel[local_row, local_col].item()
                    baseline_val = block_32_baseline[local_row, local_col].item()
                    abs_err_val = block_32_abs_err[local_row, local_col].item()
                    rel_err_val = block_32_rel_err[local_row, local_col].item()
                    print(f"  {local_row:<10} {local_col:<10} {kernel_val:<14.6e} {baseline_val:<14.6e} {abs_err_val:<14.6e} {rel_err_val:<14.6e}")
                
                if num_high > 10:
                    print(f"  ... and {num_high - 10} more")
    
    print(f"\n{'='*70}")
    print(f"8x8 Block-level Analysis")
    print(f"{'='*70}")
    
    block_size = 8
    num_blocks = 64 // block_size  # 8x8 grid of blocks
    
    # Store block statistics
    block_stats = []
    
    for block_row in range(num_blocks):
        for block_col in range(num_blocks):
            # Extract 8x8 block
            row_start = block_row * block_size
            row_end = row_start + block_size
            col_start = block_col * block_size
            col_end = col_start + block_size
            
            block_abs_err = abs_err[0, 0, 0, row_start:row_end, col_start:col_end]
            block_baseline = inverse_baseline_reshaped[0, 0, 0, row_start:row_end, col_start:col_end]
            
            # Calculate relative error for all elements in this block
            block_rel_err = block_abs_err / (block_baseline.abs() + 1e-8)
            
            max_rel = block_rel_err.max().item()
            mean_rel = block_rel_err.mean().item()
            max_abs = block_abs_err.max().item()
            mean_abs = block_abs_err.mean().item()
            
            # Also track significant elements separately
            block_mask = block_baseline.abs() > 1e-3
            num_significant = block_mask.sum().item()
            
            block_stats.append({
                'block_row': block_row,
                'block_col': block_col,
                'row_range': (row_start, row_end-1),
                'col_range': (col_start, col_end-1),
                'max_abs_err': max_abs,
                'mean_abs_err': mean_abs,
                'max_rel_err': max_rel,
                'mean_rel_err': mean_rel,
                'num_significant': num_significant
            })
    
    # Sort by max relative error
    block_stats_sorted = sorted(block_stats, key=lambda x: x['max_rel_err'], reverse=True)
    
    print(f"\nTop 10 blocks with highest relative error:")
    print(f"{'Block':<12} {'Row Range':<12} {'Col Range':<12} {'Max Rel Err':<14} {'Mean Rel Err':<14} {'Max Abs Err':<14} {'#Total/#Sig':<15}")
    print(f"{'-'*105}")
    
    for i, stats in enumerate(block_stats_sorted[:10]):
        block_id = f"[{stats['block_row']},{stats['block_col']}]"
        row_range = f"[{stats['row_range'][0]}:{stats['row_range'][1]}]"
        col_range = f"[{stats['col_range'][0]}:{stats['col_range'][1]}]"
        elem_counts = f"64/{stats['num_significant']}"
        print(f"{block_id:<12} {row_range:<12} {col_range:<12} {stats['max_rel_err']:<14.6e} {stats['mean_rel_err']:<14.6e} {stats['max_abs_err']:<14.6e} {elem_counts:<15}")
    
    # Summary statistics
    blocks_with_high_rel_err = sum(1 for s in block_stats if s['max_rel_err'] > rel_tol)
    blocks_with_high_abs_err = sum(1 for s in block_stats if s['max_abs_err'] > abs_tol)
    
    print(f"\nBlock summary:")
    print(f"  Total blocks: {len(block_stats)}")
    print(f"  Blocks with max_rel_err > {rel_tol}: {blocks_with_high_rel_err}")
    print(f"  Blocks with max_abs_err > {abs_tol}: {blocks_with_high_abs_err}")
    
    # Visualize error distribution
    print(f"\nRelative error heatmap (8x8 blocks, showing max rel err per block):")
    print(f"  '.' = <1%   'o' = 1-5%   'O' = 5-10%   'X' = >10%")
    print(f"\n    ", end="")
    for col in range(num_blocks):
        print(f"{col:>5}", end="")
    print()
    
    for row in range(num_blocks):
        print(f"  {row} ", end="")
        for col in range(num_blocks):
            idx = row * num_blocks + col
            max_rel = block_stats[idx]['max_rel_err']
            if max_rel < 0.01:
                symbol = '.'
            elif max_rel < 0.05:
                symbol = 'o'
            elif max_rel < 0.1:
                symbol = 'O'
            else:
                symbol = 'X'
            print(f"{symbol:>5}", end="")
        print()
    
    is_close = torch.allclose(inverse_kernel_f32, inverse_baseline_reshaped, atol=abs_tol, rtol=rel_tol)
    overall_pass = abs_pass and rel_pass and is_close
    
    print(f"\n{'✓' if overall_pass else '✗'} Results {'match' if overall_pass else 'DO NOT match'} baseline (atol={abs_tol:.0e}, rtol={rel_tol:.0e})")
    
    return overall_pass


if __name__ == '__main__':
    success = main()
    print("\n" + "=" * 70)
    print(f"{'✓ PASSED' if success else '✗ FAILED'}")
    print("=" * 70)
    sys.exit(0 if success else 1)
