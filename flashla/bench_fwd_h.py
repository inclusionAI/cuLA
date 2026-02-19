#!/usr/bin/env python3
"""
Performance comparison: Our SM100 CuTe DSL kernel vs FLA Triton kernel
for chunk_gated_delta_rule_fwd_h.
"""

import os
os.environ["CUTLASS_DSL_DEBUG_LEVEL"] = "0"

import sys
import time
import argparse
import torch
import torch.nn.functional as F

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

from chunk_delta_h import ChunkDeltaRuleFwdH

# FLA reference - import just the function we need, avoiding heavy deps
import importlib
import types
# Bypass fla.__init__ which imports transformers
_fla_mod = types.ModuleType('fla')
_fla_mod.__path__ = ["/ossfs/workspace/flash-linear-attention/fla"]
sys.modules['fla'] = _fla_mod

# Also need fla.ops, fla.ops.common
for sub in ['fla.ops', 'fla.ops.common', 'fla.ops.utils']:
    mod = importlib.import_module(sub)

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h as fla_chunk_fwd_h


CHUNK_SIZE = 64


# Cache compiled kernels
_compiled_cache = {}


def our_kernel_run(k, w, u, g, gk, h0, B, T, H, K, V,
                   use_g=False, use_gk=False, use_h0=False):
    """Run our SM100 CuTe DSL kernel."""
    BT = CHUNK_SIZE
    NT = (T + BT - 1) // BT

    h_out = torch.zeros(B, NT, H, K, V, device="cuda", dtype=torch.bfloat16)
    v_new = torch.zeros(B, T, H, V, device="cuda", dtype=torch.bfloat16)
    ht = torch.zeros(B, H, K, V, device="cuda", dtype=torch.float32)

    g_t = g if g is not None else torch.zeros(B, T, H, device="cuda", dtype=torch.float32)
    gk_t = gk if gk is not None else torch.zeros(B, T, H, K, device="cuda", dtype=torch.float32)
    h0_t = h0 if h0 is not None else torch.zeros(B, H, K, V, device="cuda", dtype=torch.bfloat16)

    k_cute = from_dlpack(k)
    w_cute = from_dlpack(w)
    u_cute = from_dlpack(u)
    g_cute = from_dlpack(g_t)
    gk_cute = from_dlpack(gk_t)
    h0_cute = from_dlpack(h0_t)
    h_out_cute = from_dlpack(h_out)
    v_new_cute = from_dlpack(v_new)
    ht_cute = from_dlpack(ht)
    stream = cutlass_torch.default_stream()
    problem_size = (B, T, H, K, V)

    cache_key = (B, T, H, K, V, use_g, use_gk, use_h0)
    if cache_key not in _compiled_cache:
        kernel = ChunkDeltaRuleFwdH(chunk_size=BT, head_dim_k=K, head_dim_v=V)
        compiled = cute.compile(
            kernel,
            k_cute.iterator, w_cute.iterator, u_cute.iterator,
            g_cute.iterator, gk_cute.iterator,
            h_out_cute.iterator, v_new_cute.iterator,
            h0_cute.iterator, ht_cute.iterator,
            problem_size,
            use_g, use_gk, use_h0, True, True,
            stream,
        )
        _compiled_cache[cache_key] = compiled

    compiled = _compiled_cache[cache_key]
    compiled(
        k_cute.iterator, w_cute.iterator, u_cute.iterator,
        g_cute.iterator, gk_cute.iterator,
        h_out_cute.iterator, v_new_cute.iterator,
        h0_cute.iterator, ht_cute.iterator,
        problem_size,
        use_g, use_gk, use_h0, True, True,
        stream,
    )
    return h_out, v_new


def fla_kernel_run(k, w, u, g, gk, h0_fp32, use_h0=False):
    """Run FLA's Triton kernel."""
    h, v_new, ht = fla_chunk_fwd_h(
        k=k, w=w, u=u,
        g=g, gk=gk,
        initial_state=h0_fp32 if use_h0 else None,
        output_final_state=True,
        chunk_size=CHUNK_SIZE,
        save_new_value=True,
    )
    return h, v_new


def benchmark_fn(fn, warmup=10, repeat=50):
    """Benchmark with CUDA events for accurate GPU timing."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Measure with CUDA events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    for i in range(repeat):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    # Use median
    median_ms = times[len(times) // 2]
    mean_ms = sum(times) / len(times)
    min_ms = times[0]
    return median_ms, mean_ms, min_ms


def run_benchmark(B, T, H, K, V, use_g=False, use_gk=False, use_h0=False, warmup=10, repeat=50):
    """Run benchmark for a single configuration."""
    torch.manual_seed(42)

    k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16) * 0.1
    w = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16) * 0.1
    u = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16) * 0.1

    g = None
    if use_g:
        g_raw = torch.randn(B, T, H, device="cuda", dtype=torch.float32) * 0.01
        g = g_raw.cumsum(dim=1)

    gk = None
    if use_gk:
        gk_raw = torch.randn(B, T, H, K, device="cuda", dtype=torch.float32) * 0.01
        gk = gk_raw.cumsum(dim=1)

    h0_fp32 = None
    h0_bf16 = None
    if use_h0:
        h0_fp32 = torch.randn(B, H, K, V, device="cuda", dtype=torch.float32) * 0.1
        h0_bf16 = h0_fp32.to(torch.bfloat16)

    # Pre-compile our kernel
    our_kernel_run(k, w, u, g, gk, h0_bf16, B, T, H, K, V, use_g, use_gk, use_h0)
    torch.cuda.synchronize()

    # Benchmark FLA
    fla_median, fla_mean, fla_min = benchmark_fn(
        lambda: fla_kernel_run(k, w, u, g, gk, h0_fp32, use_h0),
        warmup=warmup, repeat=repeat,
    )

    # Benchmark ours
    ours_median, ours_mean, ours_min = benchmark_fn(
        lambda: our_kernel_run(k, w, u, g, gk, h0_bf16, B, T, H, K, V, use_g, use_gk, use_h0),
        warmup=warmup, repeat=repeat,
    )

    speedup = fla_median / ours_median if ours_median > 0 else float('inf')

    return {
        'fla_median': fla_median,
        'fla_mean': fla_mean,
        'fla_min': fla_min,
        'ours_median': ours_median,
        'ours_mean': ours_mean,
        'ours_min': ours_min,
        'speedup': speedup,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark chunk_gated_delta_rule_fwd_h")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer configs")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()

    warmup_iters = args.warmup
    repeat_iters = args.repeat

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Warmup: {warmup_iters}, Repeat: {repeat_iters}")
    print()

    K = 128
    V = 128

    if args.quick:
        configs = [
            # (B, T, H, use_g, use_h0)
            (1, 256,  1, False, False),
            (1, 1024, 1, False, False),
            (1, 4096, 1, False, False),
            (1, 1024, 4, False, False),
            (4, 1024, 4, False, False),
            (1, 1024, 1, True,  False),
            (1, 1024, 1, False, True),
        ]
    else:
        configs = [
            # Vary sequence length (B=1, H=1, no gates)
            (1, 128,   1, False, False),
            (1, 256,   1, False, False),
            (1, 512,   1, False, False),
            (1, 1024,  1, False, False),
            (1, 2048,  1, False, False),
            (1, 4096,  1, False, False),
            (1, 8192,  1, False, False),
            # Vary heads (B=1, T=1024)
            (1, 1024,  4, False, False),
            (1, 1024, 16, False, False),
            (1, 1024, 32, False, False),
            # Vary batch (T=1024, H=4)
            (2, 1024,  4, False, False),
            (4, 1024,  4, False, False),
            (8, 1024,  4, False, False),
            # With gates
            (1, 1024,  1, True,  False),
            (1, 4096,  1, True,  False),
            (4, 1024,  4, True,  False),
            # With initial state
            (1, 1024,  1, False, True),
            (1, 4096,  1, False, True),
            (4, 1024,  4, False, True),
            # With both
            (1, 1024,  1, True,  True),
            (4, 1024,  4, True,  True),
        ]

    # Header
    header = (
        f"{'B':>3} {'T':>6} {'H':>3} {'K':>4} {'V':>4} "
        f"{'g':>2} {'h0':>3}  "
        f"{'FLA(ms)':>9} {'Ours(ms)':>9} {'Speedup':>8}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for B, T, H, use_g, use_h0 in configs:
        label = f"B={B},T={T},H={H},g={use_g},h0={use_h0}"
        try:
            r = run_benchmark(B, T, H, K, V, use_g=use_g, use_h0=use_h0, warmup=warmup_iters, repeat=repeat_iters)
            tag_g = "Y" if use_g else "N"
            tag_h0 = "Y" if use_h0 else "N"
            speedup_str = f"{r['speedup']:.2f}x"
            print(
                f"{B:>3} {T:>6} {H:>3} {K:>4} {V:>4} "
                f"{tag_g:>2} {tag_h0:>3}  "
                f"{r['fla_median']:>9.3f} {r['ours_median']:>9.3f} {speedup_str:>8}"
            )
            results.append((B, T, H, use_g, use_h0, r))
        except Exception as e:
            print(f"  {label}: ERROR - {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if results:
        speedups = [r['speedup'] for *_, r in results]
        print()
        print(f"Geometric mean speedup: {(torch.tensor(speedups).prod() ** (1/len(speedups))).item():.2f}x")
        print(f"Min speedup: {min(speedups):.2f}x")
        print(f"Max speedup: {max(speedups):.2f}x")


if __name__ == "__main__":
    main()
