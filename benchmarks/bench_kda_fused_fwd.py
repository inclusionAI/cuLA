#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
bench_kda_fused_fwd.py — Benchmark: cuLA fully-fused KDA forward vs FLA Triton baseline

Automatically selects the cuLA fully-fused implementation based on the current
GPU architecture:
  - sm100 (Blackwell) → cula.kda.blackwell_fused_fwd.flash_kda_prefill
  - sm90  (Hopper)    → cula.kda.hopper_fused_fwd.cula_kda_prefill

Compares:
  - Accuracy: RMSE, relative max diff between cuLA fully-fused and FLA Triton
  - Performance: kernel execution time (ms) with CUDA events

Modes:
  - Fixed-length: various (B, T) configs
  - Varlen: sequences with 2-3x length variation

Usage:
  python bench_kda_fused_fwd.py [--mode fixed|varlen|both] [--ncu]

With --ncu, warmup=1 and iters=1 for ncu profiling:
  ncu --set full -o report python bench_kda_fused_fwd.py --mode varlen --ncu
"""

import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))  # Enable fast ops in FLA for fair comparison

import torch
from fla.ops.kda import chunk_kda as fla_chunk_kda

from benchmarks.utils import (
    SEED,
    build_varlen_configs,
    exclusive_cumsum,
    prepare_safe_gate_inputs,
    set_seed,
)
from cula.utils import get_device_sm_version, get_kda_fused_fwd

# ============================================================
# Resolve cuLA fully-fused implementation at import time
# ============================================================
_device = torch.device("cuda")
_major, _minor = get_device_sm_version(_device)
_SM_TAG = f"sm{_major}{_minor}"
cula_kda_fused_fwd = get_kda_fused_fwd(_device)

# ============================================================
# Constants
# ============================================================
H, D = 64, 128
WARMUP = 25
N_ITERS = 100
NCU_MODE = False
SANITIZER_MODE = False
HAS_INIT_STATE = False


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


def run_fla(q, k, v, g, beta, scale, A_log, dt_bias, init_state, cu_seqlens, lower_bound):
    return fla_chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=init_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=lower_bound,
        transpose_state_layout=True,
    )


def run_cula(q, k, v, g, beta, scale, A_log, dt_bias, init_state, cu_seqlens, lower_bound):
    return cula_kda_fused_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=init_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=lower_bound,
    )


# ============================================================
# Fixed-length benchmark
# ============================================================
def bench_fixed(configs):
    print("\n" + "=" * 100)
    print(f" Fixed-Length Benchmark: cuLA fully-fused ({_SM_TAG}) vs FLA Triton")
    print("=" * 100)
    results = []

    for B, T in configs:
        set_seed(SEED)
        device = torch.device("cuda")
        torch.cuda.empty_cache()

        seq_lens = [T] * B
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

        inputs = prepare_safe_gate_inputs(B, T, H, D, device, cu_seqlens=cu_seqlens, has_init_state=HAS_INIT_STATE)
        q, k, v, g, beta = inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"]
        A_log, dt_bias = inputs["A_log"], inputs["dt_bias"]
        scale, init_state, lower_bound = inputs["scale"], inputs["init_state"], inputs["lower_bound"]

        common = dict(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            A_log=A_log,
            dt_bias=dt_bias,
            init_state=init_state,
            cu_seqlens=cu_seqlens,
            lower_bound=lower_bound,
        )

        # Accuracy
        o_fla, _ = run_fla(**common)
        o_cula, _ = run_cula(**common)
        torch.cuda.synchronize()

        rmse, rel_max, mean_diff = accuracy_stats(o_fla, o_cula)

        # Performance
        ms_fla = time_kernel(lambda: run_fla(**common))
        ms_cula = time_kernel(lambda: run_cula(**common))
        speedup = ms_fla / ms_cula if ms_cula > 0 else float("inf")

        results.append(
            {
                "B": B,
                "T": T,
                "rmse": rmse,
                "rel_max": rel_max,
                "mean_diff": mean_diff,
                "ms_fla": ms_fla,
                "ms_cula": ms_cula,
                "speedup": speedup,
            }
        )

        del o_fla, o_cula, q, k, v, g, beta, A_log, dt_bias, inputs
        torch.cuda.empty_cache()

    return results


# ============================================================
# Varlen benchmark
# ============================================================
def bench_varlen(configs):
    print("\n" + "=" * 100)
    print(f" Varlen Benchmark: cuLA fully-fused ({_SM_TAG}) vs FLA Triton")
    print("=" * 100)
    results = []

    for seq_lens, total_len, dist in configs:
        set_seed(SEED)
        device = torch.device("cuda")
        torch.cuda.empty_cache()

        T = total_len
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

        inputs = prepare_safe_gate_inputs(1, T, H, D, device, cu_seqlens=cu_seqlens, has_init_state=HAS_INIT_STATE)
        q, k, v, g, beta = inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"]
        A_log, dt_bias = inputs["A_log"], inputs["dt_bias"]
        scale, init_state, lower_bound = inputs["scale"], inputs["init_state"], inputs["lower_bound"]

        common = dict(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            A_log=A_log,
            dt_bias=dt_bias,
            init_state=init_state,
            cu_seqlens=cu_seqlens,
            lower_bound=lower_bound,
        )

        # Accuracy
        o_fla, _ = run_fla(**common)
        o_cula, _ = run_cula(**common)
        torch.cuda.synchronize()

        rmse, rel_max, mean_diff = accuracy_stats(o_fla, o_cula)

        # Performance
        ms_fla = time_kernel(lambda: run_fla(**common))
        ms_cula = time_kernel(lambda: run_cula(**common))
        speedup = ms_fla / ms_cula if ms_cula > 0 else float("inf")

        n_seqs = len(seq_lens)
        min_l, max_l = min(seq_lens), max(seq_lens)
        avg_l = T // n_seqs
        tag = f"{dist:>7s} {n_seqs:>2d}seqs T={T} [{min_l}..{max_l}] avg={avg_l}"

        results.append(
            {
                "tag": tag,
                "dist": dist,
                "T_total": T,
                "n_seqs": n_seqs,
                "rmse": rmse,
                "rel_max": rel_max,
                "mean_diff": mean_diff,
                "ms_fla": ms_fla,
                "ms_cula": ms_cula,
                "speedup": speedup,
            }
        )

        del o_fla, o_cula, q, k, v, g, beta, A_log, dt_bias, inputs
        torch.cuda.empty_cache()

    return results


# ============================================================
# Report
# ============================================================
def print_report(fixed_results, varlen_results):
    sep = "=" * 110
    print(f"\n\n{sep}")
    print("                  BENCHMARK REPORT: cula_kda_fused_fwd (fully-fused)")
    print(f"                  cuLA {_SM_TAG} fully-fused vs FLA Triton")
    print(f"                  H={H}  D={D}  dtype=bf16  safe_gate=True  has_init_state={HAS_INIT_STATE}")
    wu = 1 if (NCU_MODE or SANITIZER_MODE) else WARMUP
    ni = 1 if (NCU_MODE or SANITIZER_MODE) else N_ITERS
    mode_tag = "  [NCU mode]" if NCU_MODE else ("  [Sanitizer mode]" if SANITIZER_MODE else "")
    print(f"                  Warmup={wu}  Iters={ni}{mode_tag}")
    print(sep)

    if fixed_results:
        print("\n  [Fixed-Length]")
        print(f"  {'─' * 90}")
        print(
            f"  {'B':>3s}  {'T':>6s}  │  {'RMSE':>10s}  {'rel_max':>10s}  {'mean_diff':>10s}"
            f"  │  {'FLA(ms)':>9s}  {'cuLA(ms)':>10s}  {'Speedup':>8s}"
        )
        print(f"  {'─' * 90}")
        for r in fixed_results:
            print(
                f"  {r['B']:3d}  {r['T']:6d}  │  "
                f"{r['rmse']:10.6f}  {r['rel_max']:10.6f}  {r['mean_diff']:10.6f}  │  "
                f"{r['ms_fla']:9.4f}  {r['ms_cula']:10.4f}  {r['speedup']:7.2f}x"
            )
        print(f"  {'─' * 90}")

    if varlen_results:
        print("\n  [Varlen]")
        print(f"  {'─' * 105}")
        print(
            f"  {'Config':>45s}  │  {'RMSE':>10s}  {'rel_max':>10s}  {'mean_diff':>10s}"
            f"  │  {'FLA(ms)':>9s}  {'cuLA(ms)':>10s}  {'Speedup':>8s}"
        )
        print(f"  {'─' * 105}")
        for r in varlen_results:
            print(
                f"  {r['tag']:>45s}  │  "
                f"{r['rmse']:10.6f}  {r['rel_max']:10.6f}  {r['mean_diff']:10.6f}  │  "
                f"{r['ms_fla']:9.4f}  {r['ms_cula']:10.4f}  {r['speedup']:7.2f}x"
            )
        print(f"  {'─' * 105}")

    print(f"\n{sep}\n")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="bench_kda_fused_fwd: cuLA fully-fused KDA forward vs FLA Triton")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["fixed", "varlen", "both"],
        help="Which benchmark mode to run (default: both)",
    )
    parser.add_argument(
        "--ncu",
        action="store_true",
        help="NCU profiling mode: warmup=1, iters=1",
    )
    parser.add_argument(
        "--sanitizer",
        action="store_true",
        help="Sanitizer mode: warmup=1, iters=1",
    )
    parser.add_argument(
        "--init_state",
        action="store_true",
        help="Use non-zero initial state (default: False)",
    )
    args = parser.parse_args()

    global NCU_MODE, SANITIZER_MODE, HAS_INIT_STATE
    if args.ncu:
        NCU_MODE = True
        print("[NCU mode] warmup=1, iters=1")
    if args.sanitizer:
        SANITIZER_MODE = True
        print("[Sanitizer mode] warmup=1, iters=1")
    if args.init_state:
        HAS_INIT_STATE = True
        print("[init_state] using non-zero initial state")

    print(
        f"[Device] {torch.cuda.get_device_name(0)}  compute capability {_SM_TAG}  →  using {cula_kda_fused_fwd.__module__}.{cula_kda_fused_fwd.__name__}"
    )

    fixed_configs = [
        # (B, T)
        (1, 512),
        (1, 1024),
        (1, 4096),
        (1, 8192),
        (1, 16384),
        (2, 512),
        (2, 1024),
        (2, 4096),
        (2, 8192),
        (2, 16384),
    ]

    varlen_configs = build_varlen_configs(
        num_seqs_list=(10, 20),
        total_lens=(4096, 8192, 16384),
        dists=("uniform", "random", "skewed"),
    )

    fixed_res, varlen_res = [], []

    if args.mode in ("fixed", "both"):
        fixed_res = bench_fixed(fixed_configs)

    if args.mode in ("varlen", "both"):
        varlen_res = bench_varlen(varlen_configs)

    print_report(fixed_res, varlen_res)

    return fixed_res, varlen_res


if __name__ == "__main__":
    main()
