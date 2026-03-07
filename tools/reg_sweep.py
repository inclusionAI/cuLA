#!/usr/bin/env python
"""Sweep through register configurations to find optimal settings."""

import torch
import time
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from fla.modules.l2norm import l2norm_fwd
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

from flashla.kda import KDAChunkwise

CHUNK_SIZE = 64

def test_config(mma_regs, cuda_regs, q, k, v, g, beta, scale):
    """Test a single register configuration."""
    B, S, H, D = q.shape
    
    # Prepare data
    g_cumsum = chunk_local_cumsum(
        g=g,
        chunk_size=CHUNK_SIZE,
        scale=RCP_LN2,
        cu_seqlens=None,
        chunk_indices=None
    )
    
    q_norm, _ = l2norm_fwd(q)
    k_norm, _ = l2norm_fwd(k)
    
    q_cute = from_dlpack(q_norm)
    k_cute = from_dlpack(k_norm)
    v_cute = from_dlpack(v)
    g_cute = from_dlpack(g_cumsum)
    beta_cute = from_dlpack(beta)
    
    o = torch.zeros_like(q)
    o_cute = from_dlpack(o)
    
    stream = cutlass_torch.default_stream()
    
    try:
        # Create kernel with specific register config
        attn_kernel = KDAChunkwise(
            chunk_size=CHUNK_SIZE,
            qk_acc_dtype=cutlass.Float32,
            kv_acc_dtype=cutlass.Float32,
            io_dtype=cutlass.BFloat16,
            scale=scale,
            num_regs_cuda=cuda_regs,
            num_regs_others=mma_regs,
        )
        
        # Compile
        compiled = cute.compile(
            attn_kernel,
            q_cute.iterator,
            k_cute.iterator,
            v_cute.iterator,
            g_cute.iterator,
            o_cute.iterator,
            beta_cute.iterator,
            (B, S, H, D),
            stream,
        )
        
        # Warmup
        for _ in range(3):
            compiled(
                q_cute.iterator,
                k_cute.iterator,
                v_cute.iterator,
                g_cute.iterator,
                o_cute.iterator,
                beta_cute.iterator,
                (B, S, H, D),
                stream,
            )
        torch.cuda.synchronize()
        
        # Benchmark - Mode 2: sync at end
        n_iters = 10
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            compiled(
                q_cute.iterator,
                k_cute.iterator,
                v_cute.iterator,
                g_cute.iterator,
                o_cute.iterator,
                beta_cute.iterator,
                (B, S, H, D),
                stream,
            )
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_iters * 1000
        
        return elapsed, None
        
    except Exception as e:
        return None, str(e)

def main():
    # Test configuration
    B, H, S, D = 2, 64, 4096, 128
    torch.manual_seed(42)
    
    # Prepare inputs
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
    k = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
    v = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
    g = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda').abs() * 0.1
    beta = torch.ones(B, S, H, dtype=torch.bfloat16, device='cuda')
    scale = float(D) ** -0.5
    
    # Register configurations to test
    mma_regs_list = [32, 64, 96, 120]
    cuda_regs_list = [160, 192, 224, 248]
    
    print("=" * 80)
    print("Register Configuration Sweep")
    print("=" * 80)
    print(f"Config: B={B}, H={H}, S={S}, D={D}")
    print(f"Testing {len(mma_regs_list)} MMA configs × {len(cuda_regs_list)} CUDA configs = {len(mma_regs_list) * len(cuda_regs_list)} total")
    print()
    
    results = []
    
    for mma_regs in mma_regs_list:
        for cuda_regs in cuda_regs_list:
            print(f"Testing mma={mma_regs:3d}, cuda={cuda_regs:3d}... ", end="", flush=True)
            
            elapsed, error = test_config(mma_regs, cuda_regs, q, k, v, g, beta, scale)
            
            if elapsed is not None:
                print(f"{elapsed:.3f} ms")
                results.append((mma_regs, cuda_regs, elapsed))
            else:
                print(f"FAILED: {error[:60]}")
    
    print()
    print("=" * 80)
    print("Results Summary")
    print("=" * 80)
    
    if results:
        # Sort by performance
        results.sort(key=lambda x: x[2])
        
        print(f"\n{'Rank':<6} {'MMA':>5} {'CUDA':>5} {'Time(ms)':>10} {'vs Best':>8}")
        print("-" * 80)
        
        best_time = results[0][2]
        for i, (mma, cuda, elapsed) in enumerate(results, 1):
            speedup = elapsed / best_time
            print(f"{i:<6} {mma:>5} {cuda:>5} {elapsed:>10.3f} {speedup:>8.2f}x")
        
        print()
        print(f"Best configuration: MMA={results[0][0]}, CUDA={results[0][1]} ({results[0][2]:.3f} ms)")
    else:
        print("No successful configurations!")

if __name__ == "__main__":
    main()
