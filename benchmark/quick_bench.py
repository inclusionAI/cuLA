#!/usr/bin/env python
"""Quick benchmark for single configuration to test optimizations."""

import torch
import time
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from fla.ops.kda import chunk_kda
from fla.modules.l2norm import l2norm_fwd
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

from flashla.kda import KDAChunkwise

# Config
CHUNK_SIZE = 64

# Global kernel cache
compiled_kernel = None

def flashkda_prefill_impl(q, k, v, g, beta, scale, chunk_size=CHUNK_SIZE):
    """FlashKDA prefill implementation using KDAChunkwise kernel.
    
    Note: Does NOT synchronize internally - caller must handle synchronization.
    """
    global compiled_kernel
    
    B, S, H, D = q.shape
    
    g_cumsum = chunk_local_cumsum(
        g=g,
        chunk_size=chunk_size,
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
    
    if compiled_kernel is None:
        attn_kernel = KDAChunkwise(
            chunk_size=chunk_size,
            qk_acc_dtype=cutlass.Float32,
            kv_acc_dtype=cutlass.Float32,
            io_dtype=cutlass.BFloat16,
            scale=scale,
        )
        compiled_kernel = cute.compile(
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
    
    compiled_kernel(
        q_cute.iterator,
        k_cute.iterator,
        v_cute.iterator,
        g_cute.iterator,
        o_cute.iterator,
        beta_cute.iterator,
        (B, S, H, D),
        stream,
    )
    
    return o

def run_benchmark():
    B, H, S, D = 4, 64, 4096, 128
    torch.manual_seed(42)
    
    # Both FLA and FlashKDA use [B, S, H, D] format
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
    k = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
    v = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
    # g needs to be float and apply logsigmoid
    g = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
    g = torch.nn.functional.logsigmoid(g)
    # beta needs to be float and apply sigmoid
    beta = torch.randn(B, S, H, dtype=torch.float, device='cuda').sigmoid()
    
    scale = float(D) ** -0.5
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        o1 = flashkda_prefill_impl(q, k, v, g, beta, scale)
        o2 = chunk_kda(q, k, v, g, beta)
        torch.cuda.synchronize()
    
    print("Benchmarking...")
    n_iters = 10
    
    # Mode 1: Sync after each call (measure kernel + sync overhead)
    print("\n=== Mode 1: Sync after each call ===")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        o1 = flashkda_prefill_impl(q, k, v, g, beta, scale)
        torch.cuda.synchronize()
    flashkda_time_sync_each = (time.perf_counter() - start) / n_iters * 1000
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        o2 = chunk_kda(q, k, v, g, beta)
        torch.cuda.synchronize()
    fla_time_sync_each = (time.perf_counter() - start) / n_iters * 1000
    
    # Mode 2: Sync only at the end (measure pure kernel launch time)
    print("\n=== Mode 2: Sync only at the end ===")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        o1 = flashkda_prefill_impl(q, k, v, g, beta, scale)
    torch.cuda.synchronize()
    flashkda_time_sync_end = (time.perf_counter() - start) / n_iters * 1000
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        o2 = chunk_kda(q, k, v, g, beta)
    torch.cuda.synchronize()
    fla_time_sync_end = (time.perf_counter() - start) / n_iters * 1000
    
    print(f'\n=== Summary ===')
    print(f'Config: B={B}, H={H}, S={S}, D={D}')
    print(f'\nMode 1 (sync each call):')
    print(f'  FlashKDA: {flashkda_time_sync_each:.3f}ms')
    print(f'  FLA:      {fla_time_sync_each:.3f}ms')
    print(f'  Speedup:  {fla_time_sync_each/flashkda_time_sync_each:.2f}x')
    print(f'\nMode 2 (sync at end):')
    print(f'  FlashKDA: {flashkda_time_sync_end:.3f}ms')
    print(f'  FLA:      {fla_time_sync_end:.3f}ms')
    print(f'  Speedup:  {fla_time_sync_end/flashkda_time_sync_end:.2f}x')
    print(f'\nSync overhead:')
    print(f'  FlashKDA: {flashkda_time_sync_each - flashkda_time_sync_end:.3f}ms')
    print(f'  FLA:      {fla_time_sync_each - fla_time_sync_end:.3f}ms')

if __name__ == "__main__":
    run_benchmark()
