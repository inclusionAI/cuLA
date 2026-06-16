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
    - Accuracy: relative_rms_error, relative max diff between cuLA fully-fused and FLA Triton
  - Performance: kernel execution time (ms) with CUDA events

Modes:
  - Fixed-length: various (B, T) configs
  - Varlen: sequences with 2-3x length variation

H (number of Q/K heads) is a module-level constant; HV (number of V heads)
defaults to H and can be overridden globally via --hv to run every config in
GVA (Grouped Value Attention) mode. HV must be a positive multiple of H.

Usage:
  python bench_kda_fused_fwd.py [--mode fixed|varlen|both] [--heads H] [--hv HV] [--ncu]

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
    benchmark_cuda_mode_fn,
    build_varlen_configs,
    exclusive_cumsum,
    prepare_safe_gate_inputs,
    relative_rms_error_rel_max_mean_abs,
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
# Default number of Q/K heads (H) and V heads (HV). When HV > H the run is in
# GVA mode (the kernel sees HV expanded q/k heads, prepared internally by
# prepare_safe_gate_inputs). HV is overridable globally via --hv.
H, D = 64, 128
HV = H
WARMUP = 25
N_ITERS = 100
NCU_MODE = False
SANITIZER_MODE = False
HAS_INIT_STATE = False


# ============================================================
# Helpers
# ============================================================


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

    for cfg in configs:
        B, T = cfg
        set_seed(SEED)
        device = torch.device("cuda")
        torch.cuda.empty_cache()

        seq_lens = [T] * B
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

        inputs = prepare_safe_gate_inputs(
            B,
            T,
            H,
            D,
            device,
            cu_seqlens=cu_seqlens,
            has_init_state=HAS_INIT_STATE,
            num_v_heads=HV,
        )
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

        relative_rms_error, rel_max, mean_diff = relative_rms_error_rel_max_mean_abs(o_fla, o_cula)

        # Performance
        ms_fla = benchmark_cuda_mode_fn(
            lambda: run_fla(**common),
            default_warmup=WARMUP,
            default_rep=N_ITERS,
            ncu_mode=NCU_MODE,
            sanitizer_mode=SANITIZER_MODE,
        )
        ms_cula = benchmark_cuda_mode_fn(
            lambda: run_cula(**common),
            default_warmup=WARMUP,
            default_rep=N_ITERS,
            ncu_mode=NCU_MODE,
            sanitizer_mode=SANITIZER_MODE,
        )
        speedup = ms_fla / ms_cula if ms_cula > 0 else float("inf")

        results.append(
            {
                "B": B,
                "T": T,
                "H": H,
                "HV": HV,
                "relative_rms_error": relative_rms_error,
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

    for cfg in configs:
        seq_lens, total_len, dist = cfg
        set_seed(SEED)
        device = torch.device("cuda")
        torch.cuda.empty_cache()

        T = total_len
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

        inputs = prepare_safe_gate_inputs(
            1,
            T,
            H,
            D,
            device,
            cu_seqlens=cu_seqlens,
            has_init_state=HAS_INIT_STATE,
            num_v_heads=HV,
        )
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

        relative_rms_error, rel_max, mean_diff = relative_rms_error_rel_max_mean_abs(o_fla, o_cula)

        # Performance
        ms_fla = benchmark_cuda_mode_fn(
            lambda: run_fla(**common),
            default_warmup=WARMUP,
            default_rep=N_ITERS,
            ncu_mode=NCU_MODE,
            sanitizer_mode=SANITIZER_MODE,
        )
        ms_cula = benchmark_cuda_mode_fn(
            lambda: run_cula(**common),
            default_warmup=WARMUP,
            default_rep=N_ITERS,
            ncu_mode=NCU_MODE,
            sanitizer_mode=SANITIZER_MODE,
        )
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
                "H": H,
                "HV": HV,
                "relative_rms_error": relative_rms_error,
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
    sep = "=" * 120
    print(f"\n\n{sep}")
    print("                  BENCHMARK REPORT: cula_kda_fused_fwd (fully-fused)")
    print(f"                  cuLA {_SM_TAG} fully-fused vs FLA Triton")
    print(f"                  D={D}  dtype=bf16  safe_gate=True  has_init_state={HAS_INIT_STATE}")
    gva_note = f"GVA enabled (HV={HV} > H={H}, ratio={HV // H}x)" if HV > H else f"MHA (HV=H={H})"
    print(f"                  {gva_note}")
    wu = 1 if (NCU_MODE or SANITIZER_MODE) else WARMUP
    ni = 1 if (NCU_MODE or SANITIZER_MODE) else N_ITERS
    mode_tag = "  [NCU mode]" if NCU_MODE else ("  [Sanitizer mode]" if SANITIZER_MODE else "")
    print(f"                  Warmup={wu}  Iters={ni}{mode_tag}")
    print(sep)

    if fixed_results:
        print("\n  [Fixed-Length]")
        print(f"  {'─' * 110}")
        print(
            f"  {'B':>3s}  {'T':>6s}  {'H':>3s}  {'HV':>3s}  {'GVA':>4s}  │  "
            f"{'rel_rmse':>10s}  {'rel_max':>10s}  {'mean_diff':>10s}  │  "
            f"{'FLA(ms)':>9s}  {'cuLA(ms)':>10s}  {'Speedup':>8s}"
        )
        print(f"  {'─' * 110}")
        for r in fixed_results:
            gva_tag = f"{r['HV'] // r['H']}x" if r["HV"] > r["H"] else "no"
            print(
                f"  {r['B']:3d}  {r['T']:6d}  {r['H']:3d}  {r['HV']:3d}  {gva_tag:>4s}  │  "
                f"{r['relative_rms_error']:10.6f}  {r['rel_max']:10.6f}  {r['mean_diff']:10.6f}  │  "
                f"{r['ms_fla']:9.4f}  {r['ms_cula']:10.4f}  {r['speedup']:7.2f}x"
            )
        print(f"  {'─' * 110}")

    if varlen_results:
        print("\n  [Varlen]")
        print(f"  {'─' * 120}")
        print(
            f"  {'Config':>45s}  {'H':>3s}  {'HV':>3s}  {'GVA':>4s}  │  "
            f"{'rel_rmse':>10s}  {'rel_max':>10s}  {'mean_diff':>10s}  │  "
            f"{'FLA(ms)':>9s}  {'cuLA(ms)':>10s}  {'Speedup':>8s}"
        )
        print(f"  {'─' * 120}")
        for r in varlen_results:
            gva_tag = f"{r['HV'] // r['H']}x" if r["HV"] > r["H"] else "no"
            print(
                f"  {r['tag']:>45s}  {r['H']:3d}  {r['HV']:3d}  {gva_tag:>4s}  │  "
                f"{r['relative_rms_error']:10.6f}  {r['rel_max']:10.6f}  {r['mean_diff']:10.6f}  │  "
                f"{r['ms_fla']:9.4f}  {r['ms_cula']:10.4f}  {r['speedup']:7.2f}x"
            )
        print(f"  {'─' * 120}")

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
        help="Which benchmark mode to run (default: both).",
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
    global H
    parser.add_argument(
        "--heads",
        type=int,
        default=H,
        help=f"Number of Q/K heads (H). Default: {H}",
    )
    parser.add_argument(
        "--hv",
        type=int,
        default=None,
        help=f"Override number of V heads (HV). Default: H ({H}, no GVA). Set HV > H to run all configs in GVA mode.",
    )
    args = parser.parse_args()

    global NCU_MODE, SANITIZER_MODE, HAS_INIT_STATE, HV
    H = args.heads
    if args.ncu:
        NCU_MODE = True
        print("[NCU mode] warmup=1, iters=1")
    if args.sanitizer:
        SANITIZER_MODE = True
        print("[Sanitizer mode] warmup=1, iters=1")
    if args.init_state:
        HAS_INIT_STATE = True
        print("[init_state] using non-zero initial state")
    if args.hv is not None:
        if args.hv < H or args.hv % H != 0:
            raise ValueError(f"--hv must be a positive multiple of H ({H}), got {args.hv}")
        HV = args.hv
        if HV > H:
            print(f"[GVA] HV={HV} (H={H}, ratio={HV // H}x)")

    print(
        f"[Device] {torch.cuda.get_device_name(0)}  compute capability {_SM_TAG}  →  using {cula_kda_fused_fwd.__module__}.{cula_kda_fused_fwd.__name__}"
    )

    # ------------------------------------------------------------------
    # Fixed-length configs — (B, T). Per-row H/HV defaults to global H/HV
    # (HV overridable via --hv to switch all rows into GVA mode).
    # ------------------------------------------------------------------
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

    # Varlen configs — same layout as fixed; HV is controlled globally via --hv.
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
