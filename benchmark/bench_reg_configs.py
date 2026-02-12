#!/usr/bin/env python3
"""
Benchmark different register allocation configurations for KDA kernel.

Tests various combinations of:
- num_regs_mma: Registers for MMA warp (currently 32)
- num_regs_cuda: Registers for CUDA warpgroups (currently 248)
- num_regs_epilogue_warps: Registers for epilogue warps (currently 24)
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
import cutlass

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flashla.kda import KDAChunkwise
from fla.modules.l2norm import l2norm_fwd
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.torch as cutlass_torch


def run_single_config(
    config: Dict[str, int],
    B: int, S: int, H: int, D: int,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    G: torch.Tensor,
    beta: torch.Tensor,
    warmup: int = 3,
    iterations: int = 10,
) -> Tuple[float, bool]:
    """
    Run benchmark for a single register configuration.
    
    Returns:
        (avg_time_ms, success): Average execution time in ms and whether it succeeded
    """
    try:
        # Create kernel with specified register configuration
        kernel = KDAChunkwise(
            chunk_size=64,
            qk_acc_dtype=cutlass.Float32,
            kv_acc_dtype=cutlass.Float32,
            io_dtype=cutlass.BFloat16,
            scale=D ** -0.5,
            num_regs_mma=config['num_regs_mma'],
            num_regs_cuda=config['num_regs_cuda'],
            num_regs_epilogue_warps=config['num_regs_epilogue_warps'],
        )
        
        # Convert to dlpack
        q_cute = from_dlpack(Q)
        k_cute = from_dlpack(K)
        v_cute = from_dlpack(V)
        g_cute = from_dlpack(G)
        beta_cute = from_dlpack(beta)
        o_cute = from_dlpack(torch.zeros_like(Q))
        
        stream = cutlass_torch.default_stream()
        
        # Compile
        print(f"  Compiling config: mma={config['num_regs_mma']}, "
              f"cuda={config['num_regs_cuda']}, epi={config['num_regs_epilogue_warps']}...", 
              end=' ', flush=True)
        
        compile_start = time.time()
        compiled = cute.compile(
            kernel,
            q_cute.iterator,
            k_cute.iterator,
            v_cute.iterator,
            g_cute.iterator,
            o_cute.iterator,
            beta_cute.iterator,
            (B, S, H, D),
            stream,
        )
        compile_time = time.time() - compile_start
        print(f"compiled in {compile_time:.2f}s", flush=True)
        
        # Warmup
        for _ in range(warmup):
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
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
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
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms
        
        avg_time = sum(times) / len(times)
        return avg_time, True
        
    except Exception as e:
        print(f"  FAILED: {str(e)}")
        return float('inf'), False


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark different register configurations for KDA"
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--output", type=str, default="reg_config_results.json",
                       help="Output JSON file")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Register Configuration Benchmark for KDA Kernel")
    print("=" * 80)
    print(f"Problem size: B={args.batch_size}, S={args.seq_len}, "
          f"H={args.num_heads}, D={args.head_dim}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return 1
    
    # Set random seed
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    B, S, H, D = args.batch_size, args.seq_len, args.num_heads, args.head_dim
    
    # Create input tensors
    print("\nPreparing input tensors...")
    Q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    G = torch.nn.functional.logsigmoid(
        torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    )
    beta = torch.randn(B, S, H, device="cuda", dtype=torch.float32).sigmoid()
    
    # Apply cumsum for G
    chunk_size = 64
    num_chunks = S // chunk_size
    G = G.float().view(B, num_chunks, chunk_size, H, D).cumsum(dim=2).view(B, S, H, D) * 1.4426950216
    
    # L2 Norm
    Q, _ = l2norm_fwd(Q)
    K, _ = l2norm_fwd(K)
    
    print("Input tensors prepared.")
    
    # Define configurations to test
    # Current baseline: mma=32, cuda=248, epi=24
    configs = [
        # Baseline
        {"num_regs_mma": 32, "num_regs_cuda": 248, "num_regs_epilogue_warps": 24, "name": "baseline"},
        
        # Increase MMA registers
        {"num_regs_mma": 48, "num_regs_cuda": 248, "num_regs_epilogue_warps": 24, "name": "mma_48"},
        {"num_regs_mma": 64, "num_regs_cuda": 248, "num_regs_epilogue_warps": 24, "name": "mma_64"},
        {"num_regs_mma": 80, "num_regs_cuda": 248, "num_regs_epilogue_warps": 24, "name": "mma_80"},
        {"num_regs_mma": 96, "num_regs_cuda": 248, "num_regs_epilogue_warps": 24, "name": "mma_96"},
        {"num_regs_mma": 128, "num_regs_cuda": 248, "num_regs_epilogue_warps": 24, "name": "mma_128"},
        
        # Decrease CUDA registers slightly to give more headroom
        {"num_regs_mma": 64, "num_regs_cuda": 232, "num_regs_epilogue_warps": 24, "name": "mma_64_cuda_232"},
        {"num_regs_mma": 80, "num_regs_cuda": 224, "num_regs_epilogue_warps": 24, "name": "mma_80_cuda_224"},
        {"num_regs_mma": 96, "num_regs_cuda": 216, "num_regs_epilogue_warps": 24, "name": "mma_96_cuda_216"},
        
        # Increase epilogue registers
        {"num_regs_mma": 64, "num_regs_cuda": 240, "num_regs_epilogue_warps": 32, "name": "mma_64_epi_32"},
        {"num_regs_mma": 80, "num_regs_cuda": 232, "num_regs_epilogue_warps": 32, "name": "mma_80_epi_32"},
        
        # Balanced configurations
        {"num_regs_mma": 72, "num_regs_cuda": 224, "num_regs_epilogue_warps": 32, "name": "balanced_72_224_32"},
        {"num_regs_mma": 88, "num_regs_cuda": 216, "num_regs_epilogue_warps": 32, "name": "balanced_88_216_32"},
        
        # More aggressive MMA allocation
        {"num_regs_mma": 104, "num_regs_cuda": 208, "num_regs_epilogue_warps": 32, "name": "aggressive_104"},
        {"num_regs_mma": 120, "num_regs_cuda": 200, "num_regs_epilogue_warps": 32, "name": "aggressive_120"},
    ]
    
    print(f"\nTesting {len(configs)} configurations...\n")
    
    results = []
    for i, config in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] Testing: {config['name']}")
        avg_time, success = run_single_config(
            config, B, S, H, D, Q, K, V, G, beta,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        
        result = {
            "config": config,
            "avg_time_ms": avg_time if success else None,
            "success": success,
        }
        results.append(result)
        
        if success:
            print(f"  Result: {avg_time:.2f} ms\n")
        print()
    
    # Save results
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "problem_size": {"B": B, "S": S, "H": H, "D": D},
        "warmup_iterations": args.warmup,
        "benchmark_iterations": args.iterations,
        "results": results,
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    # Sort by time
    successful = [(r["config"]["name"], r["avg_time_ms"]) 
                  for r in results if r["success"]]
    successful.sort(key=lambda x: x[1])
    
    if successful:
        print(f"\nTop 5 configurations:")
        for i, (name, time_ms) in enumerate(successful[:5], 1):
            speedup = successful[0][1] / time_ms if time_ms > 0 else 0
            print(f"  {i}. {name:30s} {time_ms:8.2f} ms  "
                  f"(speedup: {speedup:.2f}x vs best)")
        
        # Find baseline
        baseline_time = None
        for r in results:
            if r["config"]["name"] == "baseline" and r["success"]:
                baseline_time = r["avg_time_ms"]
                break
        
        if baseline_time:
            print(f"\nBaseline: {baseline_time:.2f} ms")
            best_time = successful[0][1]
            speedup = baseline_time / best_time
            print(f"Best:     {successful[0][1]:.2f} ms ({successful[0][0]})")
            print(f"Speedup:  {speedup:.2f}x")
    else:
        print("\nNo successful configurations!")
    
    failed = [r["config"]["name"] for r in results if not r["success"]]
    if failed:
        print(f"\nFailed configurations ({len(failed)}):")
        for name in failed:
            print(f"  - {name}")
    
    print(f"\nResults saved to: {args.output}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
