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
bench_kda_sm90_prefill.py — Benchmark: SM90 CuTeDSL KDA prefill vs FLA Triton baseline

Hopper-only. Calls cula.kda.flashkda (cula_kda_prefill, the K1+K2
two-kernel prefill) directly -- no get_kda_fused_fwd dispatcher.

Compares:
  - Accuracy: relative_rms_error, rel_max, mean_diff of the output o vs FLA Triton
  - Performance: kernel execution time (ms) via CUDA events

Modes:
  - Fixed-length: various (B, T) configs
  - Varlen: packed sequences with 2-3x length variation

The SM90 prefill is MHA only (HV == H); grouped-value attention is not yet
supported on this backend, so there is no --hv option.

Usage:
  python bench_kda_sm90_prefill.py [--mode fixed|varlen|both] [--heads H] [--init_state] [--ncu]

With --ncu, warmup=1 and iters=1 for ncu profiling:
  ncu --set full -o report python bench_kda_sm90_prefill.py --mode varlen --ncu
"""

import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))  # fair comparison with FLA

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
from cula.kda import flashkda_prefill as cula_kda_prefill
from cula.utils import assert_hopper

_SM_TAG = "sm90"

H, D = 64, 128
WARMUP = 25
N_ITERS = 100
NCU_MODE = False
SANITIZER_MODE = False
HAS_INIT_STATE = False


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
        transpose_state_layout=True,  # match the SM90 VK-transposed state layout
    )


def run_cula(q, k, v, g, beta, scale, A_log, dt_bias, init_state, cu_seqlens, lower_bound):
    return cula_kda_prefill(
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
        cu_seqlens=cu_seqlens,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=lower_bound,
    )


def _bench_one(common):
    """Accuracy (on o) + perf (ms) for a single config."""
    o_fla, _ = run_fla(**common)
    o_cula, _ = run_cula(**common)
    torch.cuda.synchronize()
    rel_rmse, rel_max, mean_diff = relative_rms_error_rel_max_mean_abs(o_fla, o_cula)
    # iqr_mean: a stray transient iteration must not poison the reported mean.
    ms_fla = benchmark_cuda_mode_fn(
        lambda: run_fla(**common),
        default_warmup=WARMUP,
        default_rep=N_ITERS,
        ncu_mode=NCU_MODE,
        sanitizer_mode=SANITIZER_MODE,
        aggregate="iqr_mean",
    )
    ms_cula = benchmark_cuda_mode_fn(
        lambda: run_cula(**common),
        default_warmup=WARMUP,
        default_rep=N_ITERS,
        ncu_mode=NCU_MODE,
        sanitizer_mode=SANITIZER_MODE,
        aggregate="iqr_mean",
    )
    if ms_fla <= 0 or ms_cula <= 0:
        raise RuntimeError(f"invalid benchmark timing: FLA={ms_fla} ms, cuLA={ms_cula} ms")
    speedup = ms_fla / ms_cula
    del o_fla, o_cula
    return rel_rmse, rel_max, mean_diff, ms_fla, ms_cula, speedup


def _make_common(B, T, cu_seqlens, device):
    inputs = prepare_safe_gate_inputs(B, T, H, D, device, cu_seqlens=cu_seqlens, has_init_state=HAS_INIT_STATE, num_v_heads=H)
    return dict(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        g=inputs["g"],
        beta=inputs["beta"],
        scale=inputs["scale"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        init_state=inputs["init_state"],
        cu_seqlens=cu_seqlens,
        lower_bound=inputs["lower_bound"],
    )


def bench_fixed(configs):
    print("\n" + "=" * 100)
    print(f" Fixed-Length Benchmark: SM90 cula_kda_prefill ({_SM_TAG}) vs FLA Triton")
    print("=" * 100)
    results = []
    for B, T in configs:
        set_seed(SEED)
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        common = _make_common(B, T, None, device)
        rel_rmse, rel_max, mean_diff, ms_fla, ms_cula, speedup = _bench_one(common)
        results.append(
            {
                "B": B,
                "T": T,
                "relative_rms_error": rel_rmse,
                "rel_max": rel_max,
                "mean_diff": mean_diff,
                "ms_fla": ms_fla,
                "ms_cula": ms_cula,
                "speedup": speedup,
            }
        )
        del common
        torch.cuda.empty_cache()
    return results


def bench_varlen(configs):
    print("\n" + "=" * 100)
    print(f" Varlen Benchmark: SM90 cula_kda_prefill ({_SM_TAG}) vs FLA Triton")
    print("=" * 100)
    results = []
    for seq_lens, total_len, dist in configs:
        set_seed(SEED)
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        T = total_len
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)
        common = _make_common(1, T, cu_seqlens, device)
        rel_rmse, rel_max, mean_diff, ms_fla, ms_cula, speedup = _bench_one(common)
        n_seqs = len(seq_lens)
        tag = f"{dist:>7s} {n_seqs:>2d}seqs T={T} [{min(seq_lens)}..{max(seq_lens)}] avg={T // n_seqs}"
        results.append(
            {
                "tag": tag,
                "relative_rms_error": rel_rmse,
                "rel_max": rel_max,
                "mean_diff": mean_diff,
                "ms_fla": ms_fla,
                "ms_cula": ms_cula,
                "speedup": speedup,
            }
        )
        del common
        torch.cuda.empty_cache()
    return results


def print_report(fixed_results, varlen_results):
    sep = "=" * 110
    print(f"\n\n{sep}")
    print("                  BENCHMARK REPORT: SM90 cula_kda_prefill (K1+K2 two-kernel)")
    print(f"                  cuLA {_SM_TAG} vs FLA Triton")
    print(f"                  H={H}  D={D}  dtype=bf16  safe_gate=True  has_init_state={HAS_INIT_STATE}")
    wu = 1 if (NCU_MODE or SANITIZER_MODE) else WARMUP
    ni = 1 if (NCU_MODE or SANITIZER_MODE) else N_ITERS
    mode_tag = "  [NCU mode]" if NCU_MODE else ("  [Sanitizer mode]" if SANITIZER_MODE else "")
    print(f"                  Warmup={wu}  Iters={ni}{mode_tag}")
    print(sep)

    if fixed_results:
        print("\n  [Fixed-Length]")
        print(f"  {'─' * 96}")
        print(
            f"  {'B':>3s}  {'T':>6s}  │  {'rel_rmse':>10s}  {'rel_max':>10s}  {'mean_diff':>10s}  │  "
            f"{'FLA(ms)':>9s}  {'cuLA(ms)':>10s}  {'Speedup':>8s}"
        )
        print(f"  {'─' * 96}")
        for r in fixed_results:
            print(
                f"  {r['B']:3d}  {r['T']:6d}  │  "
                f"{r['relative_rms_error']:10.6f}  {r['rel_max']:10.6f}  {r['mean_diff']:10.6f}  │  "
                f"{r['ms_fla']:9.4f}  {r['ms_cula']:10.4f}  {r['speedup']:7.2f}x"
            )
        print(f"  {'─' * 96}")

    if varlen_results:
        print("\n  [Varlen]")
        print(f"  {'─' * 106}")
        print(
            f"  {'Config':>45s}  │  {'rel_rmse':>10s}  {'rel_max':>10s}  {'mean_diff':>10s}  │  "
            f"{'FLA(ms)':>9s}  {'cuLA(ms)':>10s}  {'Speedup':>8s}"
        )
        print(f"  {'─' * 106}")
        for r in varlen_results:
            print(
                f"  {r['tag']:>45s}  │  "
                f"{r['relative_rms_error']:10.6f}  {r['rel_max']:10.6f}  {r['mean_diff']:10.6f}  │  "
                f"{r['ms_fla']:9.4f}  {r['ms_cula']:10.4f}  {r['speedup']:7.2f}x"
            )
        print(f"  {'─' * 106}")

    print(f"\n{sep}\n")


def main():
    parser = argparse.ArgumentParser(description="bench_kda_sm90_prefill: SM90 cula_kda_prefill vs FLA Triton")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["fixed", "varlen", "both"],
        help="Which benchmark mode to run (default: both).",
    )
    parser.add_argument("--ncu", action="store_true", help="NCU profiling mode: warmup=1, iters=1")
    parser.add_argument("--sanitizer", action="store_true", help="Sanitizer mode: warmup=1, iters=1")
    parser.add_argument("--init_state", action="store_true", help="Use non-zero initial state (default: False)")
    global H
    parser.add_argument("--heads", type=int, default=H, help=f"Number of heads (H == HV, MHA). Default: {H}")
    args = parser.parse_args()

    assert_hopper(torch.device("cuda"))

    global NCU_MODE, SANITIZER_MODE, HAS_INIT_STATE
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

    print(f"[Device] {torch.cuda.get_device_name(0)}  compute capability {_SM_TAG}  →  cula_kda_prefill (SM90 K1+K2)")

    fixed_configs = [
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
        num_seqs_list=(10, 20), total_lens=(4096, 8192, 16384), dists=("uniform", "random", "skewed")
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
