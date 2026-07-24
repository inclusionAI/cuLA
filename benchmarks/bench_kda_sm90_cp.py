#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
bench_kda_sm90_cp.py — Benchmark: SM90 FlashKDA CP-auto vs CP-off

Compares flashkda_prefill(use_intracard_cp="auto") against
flashkda_prefill(use_intracard_cp=False) on varlen configs. Reports whether
the auto planner actually engages CP for each shape (auto may stay serial).

Usage:
  python bench_kda_sm90_cp.py [--ncu] [--sanitizer]

With --ncu, warmup=1 and iters=1 for ncu profiling:
  ncu --set full -o report python bench_kda_sm90_cp.py --ncu
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from benchmarks.utils import (
    SEED,
    benchmark_cuda_mode_fn,
    exclusive_cumsum,
    prepare_safe_gate_inputs,
    resolve_benchmark_repeats,
    set_seed,
)
from cula.kda import flashkda_prefill as cula_kda_prefill
from cula.ops.kda.sm90.cp.plan import CHUNK, plan_auto
from cula.utils import assert_hopper, get_device_sm_count

D = 128
H_VALUES = [4, 8]
WARMUP = 10
N_ITERS = 100
NCU_MODE = False
SANITIZER_MODE = False

# (tag, seq_lens) — each entry is tested at every H in H_VALUES.
CONFIGS = [
    ("T=1023", [1023]),
    ("T=1025", [1025]),
    ("T=4K", [4096]),
    ("T=8K", [8192]),
    ("T=16K", [16384]),
    ("T=32K", [32768]),
    ("T=64K", [65536]),
    ("T=64K+1", [65537]),
    ("2x16K", [16384, 16384]),
    ("32K+4K", [32768, 4096]),
    ("32K+1K", [32768, 1024]),
    ("32K+1023+1025", [32768, 1023, 1025]),
    ("64K+1K", [65536, 1024]),
    ("64K+2x1K", [65536, 1024, 1024]),
    ("64K+5x1K", [65536] + [1024] * 5),
    ("64K+1+1023+1025", [65537, 1023, 1025]),
]


def time_kernel(fn):
    return benchmark_cuda_mode_fn(
        fn,
        default_warmup=WARMUP,
        default_rep=N_ITERS,
        ncu_mode=NCU_MODE,
        sanitizer_mode=SANITIZER_MODE,
    )


def run_kernel(q, k, v, g, beta, scale, A_log, dt_bias, cu_seqlens, lower_bound, *, use_auto_cp):
    cula_kda_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        A_log=A_log,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        output_final_state=False,
        safe_gate=True,
        lower_bound=lower_bound,
        use_intracard_cp="auto" if use_auto_cp else False,
    )


def _auto_engages(seq_lens, H, sm_count) -> bool:
    seq_tiles = [(sl + CHUNK - 1) // CHUNK for sl in seq_lens]
    return not plan_auto(seq_tiles, H, sm_count).trivial


def bench_cp(h_values, configs):
    print("\n" + "=" * 100)
    print(" SM90 FlashKDA Benchmark: CP-auto vs CP-off")
    print("=" * 100)

    device = torch.device("cuda")
    assert_hopper(device)
    num_sms = get_device_sm_count(device)
    print(f" [Device] {torch.cuda.get_device_name(0)}  SMs={num_sms}")
    results = []

    for H in h_values:
        for tag, seq_lens in configs:
            set_seed(SEED)
            torch.cuda.empty_cache()

            total_T = sum(seq_lens)
            engaged = _auto_engages(seq_lens, H, num_sms)
            cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)
            inputs = prepare_safe_gate_inputs(1, total_T, H, D, device, cu_seqlens=cu_seqlens, seed=SEED)
            q, k, v, g, beta = inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"]
            A_log, dt_bias = inputs["A_log"], inputs["dt_bias"]
            scale, lower_bound = inputs["scale"], inputs["lower_bound"]

            common = dict(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                A_log=A_log,
                dt_bias=dt_bias,
                cu_seqlens=cu_seqlens,
                lower_bound=lower_bound,
            )

            ms_off = time_kernel(lambda: run_kernel(**common, use_auto_cp=False))
            ms_auto = time_kernel(lambda: run_kernel(**common, use_auto_cp=True))

            if ms_off <= 0 or ms_auto <= 0:
                raise RuntimeError(f"invalid benchmark timing: CP-off={ms_off} ms, CP-auto={ms_auto} ms")
            speedup = ms_off / ms_auto
            r = dict(
                tag=tag,
                H=H,
                total_T=total_T,
                ms_off=ms_off,
                ms_auto=ms_auto,
                speedup=speedup,
                engaged=engaged,
            )
            results.append(r)

            del q, k, v, g, beta, A_log, dt_bias, inputs
            torch.cuda.empty_cache()

    return results


def print_report(results, h_values):
    sep = "=" * 105
    print(f"\n\n{sep}")
    print("               BENCHMARK REPORT: SM90 FlashKDA")
    print('               CP-auto (use_intracard_cp="auto") vs CP-off (False)')
    print(f"               D={D}  dtype=bf16  safe_gate=True")
    wu, ni = resolve_benchmark_repeats(WARMUP, N_ITERS, ncu_mode=NCU_MODE, sanitizer_mode=SANITIZER_MODE)
    mode_tag = "  [NCU mode]" if NCU_MODE else ("  [Sanitizer mode]" if SANITIZER_MODE else "")
    print(f"               Warmup={wu}  Iters={ni}{mode_tag}")
    print(sep)

    for H_val in h_values:
        h_results = [r for r in results if r["H"] == H_val]
        if not h_results:
            continue
        print(f"\n  [H={H_val}]")
        print(f"  {'─' * 90}")
        print(f"  {'config':<20s} {'T':>7s}  {'engage':>7s}  │  {'CP_off(ms)':>10s}  {'CP_auto(ms)':>11s}  {'Speedup':>8s}")
        print(f"  {'─' * 90}")
        for r in h_results:
            eng = "yes" if r["engaged"] else "no"
            print(
                f"  {r['tag']:<20s} {r['total_T']:>7d}  {eng:>7s}  │  "
                f"{r['ms_off']:>10.4f}  {r['ms_auto']:>11.4f}  {r['speedup']:>7.2f}x"
            )
        print(f"  {'─' * 90}")

    engaged = [r for r in results if r["engaged"]]
    speedups = [r["speedup"] for r in engaged]
    if speedups:
        geo = 1.0
        for s in speedups:
            geo *= s
        geo = geo ** (1 / len(speedups))
        print(
            f"\n  Engaged configs only ({len(engaged)}/{len(results)}): "
            f"geo-mean={geo:.2f}x  best={max(speedups):.2f}x  worst={min(speedups):.2f}x"
        )
    else:
        print("\n  No config engaged CP-auto on this device; speedups are serial-vs-serial.")

    print(f"\n{sep}\n")


def main():
    parser = argparse.ArgumentParser(description="bench_kda_sm90_cp: FlashKDA CP-auto vs CP-off")
    parser.add_argument("--ncu", action="store_true", help="NCU profiling mode: warmup=1, iters=1")
    parser.add_argument("--sanitizer", action="store_true", help="Sanitizer mode: warmup=1, iters=1")
    args = parser.parse_args()

    global NCU_MODE, SANITIZER_MODE
    if args.ncu:
        NCU_MODE = True
        print("[NCU mode] warmup=1, iters=1")
    if args.sanitizer:
        SANITIZER_MODE = True
        print("[Sanitizer mode] warmup=1, iters=1")

    results = bench_cp(H_VALUES, CONFIGS)
    print_report(results, H_VALUES)
    return results


if __name__ == "__main__":
    main()
