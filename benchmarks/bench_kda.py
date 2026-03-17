#!/usr/bin/env python3
"""
bench_kda.py — Benchmark: flashla CuTe DSL vs FLA Triton baseline
               for chunk_kda (KDA forward)

Compares:
  - Accuracy: RMSE, relative max diff between flashla and FLA outputs
  - Performance: kernel execution time (ms) with CUDA events

Modes:
  - Fixed-length: B=1, B=2 with various T
  - Varlen: ~20 seqs with 2-3x length variation

Usage:
  python bench_kda.py [--mode fixed|varlen|both] [--ncu]

With --ncu, warmup=1 and iters=1 for ncu profiling:
  ncu --set full -o report python bench_kda.py --mode varlen --ncu
"""

import sys
import pathlib
import argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from fla.ops.kda import chunk_kda as fla_chunk_kda
from flashla.kda.chunk import chunk_kda as flashla_chunk_kda
from benchmarks.utils import (
    set_seed, exclusive_cumsum, prepare_safe_gate_inputs, SEED,
)

# ============================================================
# Constants
# ============================================================
H, D = 64, 128
WARMUP = 10
N_ITERS = 100
NCU_MODE = False
SANITIZER_MODE = False


# ============================================================
# Helpers
# ============================================================
def time_kernel(fn, warmup=None, n_iters=None):
    if warmup is None:
        warmup = 1 if (NCU_MODE or SANITIZER_MODE) else WARMUP
    if n_iters is None:
        n_iters = 1 if (NCU_MODE or SANITIZER_MODE) else N_ITERS
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(n_iters):
        fn()
    end_evt.record()
    torch.cuda.synchronize()
    return start_evt.elapsed_time(end_evt) / n_iters


def accuracy_stats(ref, out):
    """Compute RMSE, relative max diff, and mean absolute difference."""
    ref_f = ref.float()
    out_f = out.float()
    diff = (ref_f - out_f).abs()
    rmse = diff.pow(2).mean().sqrt().item()
    max_diff = diff.max().item()
    denom = ref_f.abs().max().item()
    rel_max = max_diff / denom if denom > 0 else 0.0
    mean_diff = diff.mean().item()
    return rmse, rel_max, mean_diff


def gen_varlen_seqs(target_total, n_seqs, seed=0):
    """Generate n_seqs random seq lengths summing to target_total.
    Lengths vary ~2-3x (log-uniform-ish), each rounded up to multiple of 2."""
    import random
    rng = random.Random(seed)
    raw = [rng.uniform(0.4, 1.0) for _ in range(n_seqs)]
    s = sum(raw)
    lens = [max(2, round(r / s * target_total / 2) * 2) for r in raw]
    diff = target_total - sum(lens)
    lens[-1] += diff
    if lens[-1] < 2:
        lens[-1] = 2
    return lens


def run_kda(q, k, v, g, beta, scale, A_log, dt_bias,
            init_state, cu_seqlens, lower_bound, fn):
    return fn(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        A_log=A_log, dt_bias=dt_bias,
        initial_state=init_state, output_final_state=True,
        use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens,
        use_gate_in_kernel=True, safe_gate=True, lower_bound=lower_bound,
    )


# ============================================================
# Fixed-length benchmark
# ============================================================
def bench_fixed(configs):
    print("\n" + "=" * 100)
    print(" Fixed-Length Benchmark: flashla CuTe DSL vs FLA Triton")
    print("=" * 100)
    results = []

    for B, T in configs:
        set_seed(SEED)
        device = torch.device("cuda")
        torch.cuda.empty_cache()

        seq_lens = [T] * B
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

        inputs = prepare_safe_gate_inputs(B, T, H, D, device, cu_seqlens=cu_seqlens)
        q, k, v, g, beta = inputs['q'], inputs['k'], inputs['v'], inputs['g'], inputs['beta']
        A_log, dt_bias = inputs['A_log'], inputs['dt_bias']
        scale, init_state, lower_bound = inputs['scale'], inputs['init_state'], inputs['lower_bound']

        common = dict(q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                      A_log=A_log, dt_bias=dt_bias, init_state=init_state,
                      cu_seqlens=cu_seqlens, lower_bound=lower_bound)

        # Accuracy: compare outputs
        o_fla, _ = run_kda(**common, fn=fla_chunk_kda)
        o_flashla, _ = run_kda(**common, fn=flashla_chunk_kda)
        torch.cuda.synchronize()

        rmse, rel_max, mean_diff = accuracy_stats(o_fla, o_flashla)

        # Performance
        def fn_fla(**common_kw):
            return lambda: run_kda(**common_kw, fn=fla_chunk_kda)

        def fn_flashla(**common_kw):
            return lambda: run_kda(**common_kw, fn=flashla_chunk_kda)

        ms_fla = time_kernel(fn_fla(**common))
        ms_flashla = time_kernel(fn_flashla(**common))
        speedup = ms_fla / ms_flashla if ms_flashla > 0 else float('inf')

        r = {
            'B': B, 'T': T,
            'rmse': rmse, 'rel_max': rel_max, 'mean_diff': mean_diff,
            'ms_fla': ms_fla, 'ms_flashla': ms_flashla, 'speedup': speedup,
        }
        results.append(r)
        print(f"  B={B:2d} T={T:5d} | "
              f"RMSE={rmse:.6f} rel_max={rel_max:.6f} mean_diff={mean_diff:.8f} | "
              f"FLA={ms_fla:.4f}ms flashla={ms_flashla:.4f}ms | "
              f"speedup={speedup:.2f}x")

        del o_fla, o_flashla, q, k, v, g, beta, A_log, dt_bias, inputs
        torch.cuda.empty_cache()

    return results


# ============================================================
# Varlen benchmark
# ============================================================
def bench_varlen(configs):
    print("\n" + "=" * 100)
    print(" Varlen Benchmark: flashla CuTe DSL vs FLA Triton")
    print("=" * 100)
    results = []

    for seq_lens, total_len in configs:
        set_seed(SEED)
        device = torch.device("cuda")
        torch.cuda.empty_cache()

        T = total_len
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

        inputs = prepare_safe_gate_inputs(1, T, H, D, device, cu_seqlens=cu_seqlens)
        q, k, v, g, beta = inputs['q'], inputs['k'], inputs['v'], inputs['g'], inputs['beta']
        A_log, dt_bias = inputs['A_log'], inputs['dt_bias']
        scale, init_state, lower_bound = inputs['scale'], inputs['init_state'], inputs['lower_bound']

        common = dict(q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                      A_log=A_log, dt_bias=dt_bias, init_state=init_state,
                      cu_seqlens=cu_seqlens, lower_bound=lower_bound)

        # Accuracy
        o_fla, _ = run_kda(**common, fn=fla_chunk_kda)
        o_flashla, _ = run_kda(**common, fn=flashla_chunk_kda)
        torch.cuda.synchronize()

        rmse, rel_max, mean_diff = accuracy_stats(o_fla, o_flashla)

        # Performance
        def fn_fla(**common_kw):
            return lambda: run_kda(**common_kw, fn=fla_chunk_kda)

        def fn_flashla(**common_kw):
            return lambda: run_kda(**common_kw, fn=flashla_chunk_kda)

        ms_fla = time_kernel(fn_fla(**common))
        ms_flashla = time_kernel(fn_flashla(**common))
        speedup = ms_fla / ms_flashla if ms_flashla > 0 else float('inf')

        n_seqs = len(seq_lens)
        min_l, max_l = min(seq_lens), max(seq_lens)
        avg_l = T // n_seqs
        tag = f"{n_seqs}seqs T={T} [{min_l}..{max_l}] avg={avg_l}"

        r = {
            'tag': tag, 'T_total': T, 'n_seqs': n_seqs,
            'rmse': rmse, 'rel_max': rel_max, 'mean_diff': mean_diff,
            'ms_fla': ms_fla, 'ms_flashla': ms_flashla, 'speedup': speedup,
        }
        results.append(r)
        print(f"  {tag:45s} | "
              f"RMSE={rmse:.6f} rel_max={rel_max:.6f} mean_diff={mean_diff:.8f} | "
              f"FLA={ms_fla:.4f}ms flashla={ms_flashla:.4f}ms | "
              f"speedup={speedup:.2f}x")

        del o_fla, o_flashla, q, k, v, g, beta, A_log, dt_bias, inputs
        torch.cuda.empty_cache()

    return results


# ============================================================
# Report
# ============================================================
def print_report(fixed_results, varlen_results):
    sep = "=" * 110
    print(f"\n\n{sep}")
    print("                       BENCHMARK REPORT: chunk_kda")
    print("                       flashla CuTe DSL vs FLA Triton")
    print(f"                       H={H}  D={D}  dtype=bf16  safe_gate=True")
    wu = 1 if (NCU_MODE or SANITIZER_MODE) else WARMUP
    ni = 1 if (NCU_MODE or SANITIZER_MODE) else N_ITERS
    mode_tag = "  [NCU mode]" if NCU_MODE else ("  [Sanitizer mode]" if SANITIZER_MODE else "")
    print(f"                       Warmup={wu}  Iters={ni}{mode_tag}")
    print(sep)

    if fixed_results:
        print("\n  [Fixed-Length]")
        print(f"  {'─' * 100}")
        print(f"  {'B':>3s}  {'T':>5s}  │  {'RMSE':>10s}  {'rel_max':>10s}  {'mean_diff':>12s}"
              f"  │  {'FLA(ms)':>9s}  {'flashla(ms)':>11s}  {'Speedup':>8s}")
        print(f"  {'─' * 100}")
        for r in fixed_results:
            print(f"  {r['B']:3d}  {r['T']:5d}  │  "
                  f"{r['rmse']:10.6f}  {r['rel_max']:10.6f}  {r['mean_diff']:12.8f}  │  "
                  f"{r['ms_fla']:9.4f}  {r['ms_flashla']:11.4f}  {r['speedup']:7.2f}x")
        print(f"  {'─' * 100}")

    if varlen_results:
        print("\n  [Varlen]")
        print(f"  {'─' * 115}")
        print(f"  {'Config':>45s}  │  {'RMSE':>10s}  {'rel_max':>10s}  {'mean_diff':>12s}"
              f"  │  {'FLA(ms)':>9s}  {'flashla(ms)':>11s}  {'Speedup':>8s}")
        print(f"  {'─' * 115}")
        for r in varlen_results:
            print(f"  {r['tag']:>45s}  │  "
                  f"{r['rmse']:10.6f}  {r['rel_max']:10.6f}  {r['mean_diff']:12.8f}  │  "
                  f"{r['ms_fla']:9.4f}  {r['ms_flashla']:11.4f}  {r['speedup']:7.2f}x")
        print(f"  {'─' * 115}")

    print(f"\n{sep}\n")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="bench_kda: flashla CuTe DSL vs FLA Triton baseline"
    )
    parser.add_argument(
        "--mode", type=str, default="both",
        choices=["fixed", "varlen", "both"],
        help="Which benchmark mode to run (default: both)",
    )
    parser.add_argument(
        "--ncu", action="store_true",
        help="NCU profiling mode: warmup=1, iters=1",
    )
    parser.add_argument(
        "--sanitizer", action="store_true",
        help="Sanitizer mode: warmup=1, iters=1 (avoid Triton memory leak under compute-sanitizer)",
    )
    args = parser.parse_args()

    global NCU_MODE, SANITIZER_MODE
    if args.ncu:
        NCU_MODE = True
        print("[NCU mode] warmup=1, iters=1")
    if args.sanitizer:
        SANITIZER_MODE = True
        print("[Sanitizer mode] warmup=1, iters=1")

    fixed_configs = [
        # (B, T)
        (1, 128), (1, 256), (1, 512), (1, 1024), (1, 2048),
        (1, 4096), (1, 8192), (1, 16384),
        (2, 128), (2, 256), (2, 512), (2, 1024), (2, 2048),
        (2, 4096), (2, 8192), (2, 16384),
    ]

    varlen_configs = [
        # (seq_lens, total_len)
        # Single sequence
        ([4096], 4096),
        ([8192], 8192),
        ([16384], 16384),
        # Normal varlen (~20-25 seqs, 2-3x variation)
        (gen_varlen_seqs(4096, 20, seed=1), 4096),
        (gen_varlen_seqs(8192, 20, seed=2), 8192),
        (gen_varlen_seqs(8192, 25, seed=3), 8192),
        (gen_varlen_seqs(16384, 20, seed=4), 16384),
        (gen_varlen_seqs(16384, 25, seed=5), 16384),
        # Extreme varlen: 1 long seq + many short seqs
        ([4096 - 19 * 64] + [64] * 19, 4096),
        ([8192 - 19 * 64] + [64] * 19, 8192),
        ([16384 - 19 * 64] + [64] * 19, 16384),
        # Extreme varlen: many tiny seqs + 1 huge seq
        ([64] * 19 + [4096 - 19 * 64], 4096),
        ([64] * 19 + [8192 - 19 * 64], 8192),
        ([64] * 19 + [16384 - 19 * 64], 16384),
    ]

    fixed_res, varlen_res = [], []

    if args.mode in ("fixed", "both"):
        fixed_res = bench_fixed(fixed_configs)

    if args.mode in ("varlen", "both"):
        varlen_res = bench_varlen(varlen_configs)

    print_report(fixed_res, varlen_res)


if __name__ == "__main__":
    main()