#!/usr/bin/env python3
"""Quick test for different register configurations."""

import os
import sys
import time
import pathlib

# Avoid torch import issues
os.environ['LD_LIBRARY_PATH'] = '/usr/local/lib:/usr/lib/x86_64-linux-gnu'

import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from flashla.kda import KDAChunkwise
from fla.modules.l2norm import l2norm_fwd
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.torch as cutlass_torch


def test_config(mma_regs, cuda_regs, epi_regs, B=2, S=2048, H=8, D=128):
    """Test a single configuration."""
    print(f"\nTesting: mma={mma_regs}, cuda={cuda_regs}, epi={epi_regs}")
    
    try:
        # Create input tensors
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
        
        # Create kernel
        kernel = KDAChunkwise(
            chunk_size=64,
            qk_acc_dtype=cutlass.Float32,
            kv_acc_dtype=cutlass.Float32,
            io_dtype=cutlass.BFloat16,
            scale=D ** -0.5,
            num_regs_mma=mma_regs,
            num_regs_cuda=cuda_regs,
            num_regs_epilogue_warps=epi_regs,
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
        print("  Compiling...", end=' ', flush=True)
        t0 = time.time()
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
        compile_time = time.time() - t0
        print(f"OK ({compile_time:.1f}s)")
        
        # Warmup
        print("  Warmup...", end=' ', flush=True)
        for _ in range(2):
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
        print("OK")
        
        # Benchmark
        print("  Benchmark...", end=' ', flush=True)
        times = []
        for _ in range(5):
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
            times.append((time.perf_counter() - start) * 1000)
        
        avg_time = sum(times) / len(times)
        print(f"OK")
        print(f"  Result: {avg_time:.2f} ms (avg)")
        return avg_time, True
        
    except Exception as e:
        print(f"  FAILED: {e}")
        return None, False


def main():
    print("=" * 80)
    print("Quick Register Configuration Test")
    print("=" * 80)
    
    configs = [
        # (mma, cuda, epi, name)
        (32, 248, 24, "baseline"),
        (64, 248, 24, "mma_64"),
        (80, 248, 24, "mma_80"),
        (96, 248, 24, "mma_96"),
        (128, 248, 24, "mma_128"),
        (64, 232, 24, "mma_64_cuda_232"),
        (80, 224, 32, "mma_80_cuda_224_epi_32"),
        (96, 216, 32, "mma_96_cuda_216_epi_32"),
    ]
    
    results = []
    for mma, cuda, epi, name in configs:
        print(f"\n[{name}]")
        time_ms, success = test_config(mma, cuda, epi)
        results.append((name, mma, cuda, epi, time_ms, success))
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    # Print successful results sorted by time
    successful = [(n, m, c, e, t) for n, m, c, e, t, s in results if s]
    successful.sort(key=lambda x: x[4])
    
    if successful:
        baseline_time = None
        for n, m, c, e, t in successful:
            if n == "baseline":
                baseline_time = t
                break
        
        print(f"\n{'Config':<25} {'MMA':>5} {'CUDA':>5} {'EPI':>5} {'Time(ms)':>10} {'Speedup':>8}")
        print("-" * 80)
        for name, mma, cuda, epi, time_ms in successful:
            speedup = baseline_time / time_ms if baseline_time else 1.0
            marker = " *" if name == "baseline" else ""
            print(f"{name:<25} {mma:>5} {cuda:>5} {epi:>5} {time_ms:>10.2f} {speedup:>7.2f}x{marker}")
        
        if baseline_time:
            best = successful[0]
            print(f"\nBest: {best[0]} ({best[4]:.2f} ms, {baseline_time/best[4]:.2f}x faster)")
    
    # Print failed configs
    failed = [n for n, m, c, e, t, s in results if not s]
    if failed:
        print(f"\nFailed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
