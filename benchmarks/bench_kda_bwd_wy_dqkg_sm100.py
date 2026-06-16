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
bench_kda_bwd_wy_dqkg_sm100.py — Benchmark: cuLA CuTe DSL vs FLA Triton baseline
                                  for chunk_kda_bwd_wy_dqkg_fused kernel

Compares:
  - Accuracy: relative_rms_error, relative max diff between cuLA and FLA outputs
  - Performance: kernel execution time (ms) with CUDA events

Modes:
  - Fixed-length: B=1,2 with various T
  - Varlen: variable-length sequences with different distributions

Usage:
  python bench_kda_bwd_wy_dqkg_sm100.py [--mode fixed|varlen|both] [--ncu] [--heads 32 64]

With --ncu, warmup=1 and iters=1 for ncu profiling:
  ncu --set full -o report python bench_kda_bwd_wy_dqkg_sm100.py --mode fixed --ncu
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
from fla.ops.kda.chunk_bwd import chunk_kda_bwd_wy_dqkg_fused as fla_chunk_kda_bwd_wy_dqkg_fused

from benchmarks.utils import (
    SEED,
    benchmark_cuda_mode_fn,
    build_varlen_configs,
    exclusive_cumsum,
    prepare_bwd_wy_dqkg_fused_inputs,
    relative_rms_error_rel_max_mean_abs,
    set_seed,
)
from cula.ops.chunk_wy_dqkg_sm100 import chunk_kda_bwd_wy_dqkg_fused as cula_chunk_kda_bwd_wy_dqkg_fused

torch.backends.cuda.matmul.allow_tf32 = True

# ============================================================
# Constants
# ============================================================
H_DEFAULT = 32
K = 128
V = 128
BT = 64
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")
WARMUP = 25
N_ITERS = 100
NCU_MODE = False


def generate_balanced_seqlens(total_tokens, num_seqs):
    base = total_tokens // num_seqs
    remainder = total_tokens % num_seqs
    return [base] * (num_seqs - 1) + [base + remainder]


# ============================================================
# Runners
# ============================================================
def run_fla_triton(inputs: dict):
    """Run the FLA Triton baseline."""
    return fla_chunk_kda_bwd_wy_dqkg_fused(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        v_new=inputs["v_new"],
        g=inputs["g"],
        beta=inputs["beta"],
        A=inputs["A"],
        h=inputs["h"],
        do=inputs["do"],
        dh=inputs["dh"],
        dv=inputs["dv"],
        scale=inputs["scale"],
        cu_seqlens=inputs["cu_seqlens"],
        chunk_size=BT,
        chunk_indices=inputs["chunk_indices"],
        transpose_state_layout=False,
    )


def run_cula(inputs: dict):
    """Run the CuTe DSL Blackwell kernel."""
    return cula_chunk_kda_bwd_wy_dqkg_fused(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        v_new=inputs["v_new"],
        g=inputs["g"],
        beta=inputs["beta"],
        A=inputs["A"],
        h=inputs["h"],
        do=inputs["do"],
        dh=inputs["dh"],
        dv=inputs["dv"],
        scale=inputs["scale"],
        cu_seqlens=inputs["cu_seqlens"],
        chunk_size=BT,
        chunk_indices=inputs["chunk_indices"],
    )


def check_determinism(H=4, HV=None, total_T=2001, num_seqs=4, iters=1000):
    """Verify deterministic outputs across repeated runs."""
    if HV is None:
        HV = H
    torch.manual_seed(42)
    seq_lens = generate_balanced_seqlens(total_T, num_seqs)
    cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=DEVICE)
    inputs = prepare_bwd_wy_dqkg_fused_inputs(
        B=1,
        T=total_T,
        H=H,
        K=K,
        V=V,
        HV=HV,
        chunk_size=BT,
        device=DEVICE,
        seed=SEED,
        cu_seqlens=cu_seqlens,
    )

    ref_dq, ref_dk, ref_dv, ref_db, ref_dg, ref_dA = run_cula(inputs)
    for i in range(iters):
        dq_out, dk_out, dv_out, db_out, dg_out, dA_out = run_cula(inputs)
        assert torch.isnan(dq_out).sum() == 0, f"dq contains NaNs at iter {i}"
        assert torch.isnan(dk_out).sum() == 0, f"dk contains NaNs at iter {i}"
        assert torch.isnan(dv_out).sum() == 0, f"dv contains NaNs at iter {i}"
        assert torch.isnan(db_out).sum() == 0, f"db contains NaNs at iter {i}"
        assert torch.isnan(dg_out).sum() == 0, f"dg contains NaNs at iter {i}"
        assert torch.isnan(dA_out).sum() == 0, f"dA contains NaNs at iter {i}"
        assert torch.isfinite(dq_out).all(), f"dq contains infs at iter {i}"
        assert torch.isfinite(dk_out).all(), f"dk contains infs at iter {i}"
        assert torch.isfinite(dv_out).all(), f"dv contains infs at iter {i}"
        assert torch.isfinite(db_out).all(), f"db contains infs at iter {i}"
        assert torch.isfinite(dg_out).all(), f"dg contains infs at iter {i}"
        assert torch.isfinite(dA_out).all(), f"dA contains infs at iter {i}"
        assert torch.equal(dq_out, ref_dq), f"dq mismatch at iter {i}"
        assert torch.equal(dk_out, ref_dk), f"dk mismatch at iter {i}"
        assert torch.equal(dv_out, ref_dv), f"dv mismatch at iter {i}"
        assert torch.equal(dg_out, ref_dg), f"dg mismatch at iter {i}"
        assert torch.equal(dA_out, ref_dA), f"dA mismatch at iter {i}"
        assert torch.equal(db_out, ref_db), f"db mismatch at iter {i}"
    return True


# ============================================================
# Fixed-length benchmark
# ============================================================
def bench_fixed(configs, H: int, HV: int | None = None):
    if HV is None:
        HV = H
    print("\n" + "=" * 120)
    print(f" Fixed-Length Benchmark: cuLA CuTe DSL vs FLA Triton  (H={H}, HV={HV}, K={K}, V={V}, BT={BT})")
    print("=" * 120)
    results = []

    for B, T in configs:
        set_seed(SEED)
        torch.cuda.empty_cache()

        seq_lens = [T] * B
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=DEVICE)

        inputs = prepare_bwd_wy_dqkg_fused_inputs(
            B=B,
            T=T,
            H=H,
            K=K,
            V=V,
            HV=HV,
            chunk_size=BT,
            device=DEVICE,
            seed=SEED,
            cu_seqlens=cu_seqlens,
        )

        # Accuracy
        ref = run_fla_triton(inputs)  # (dq, dk, dv, db, dg, dA)
        out = run_cula(inputs)  # (dq, dk, dv, db, dg, dA)
        torch.cuda.synchronize()

        acc = {}
        names = ["dq", "dk", "dv", "db", "dg", "dA"]
        for name, r, o in zip(names, ref, out):
            rel_rmse, rel_max, mean_diff = relative_rms_error_rel_max_mean_abs(r, o)
            acc[name] = {"rel_rmse": rel_rmse, "rel_max": rel_max, "mean_diff": mean_diff}

        # Performance
        ms_fla = benchmark_cuda_mode_fn(
            lambda: run_fla_triton(inputs),
            default_warmup=WARMUP,
            default_rep=N_ITERS,
            ncu_mode=NCU_MODE,
        )
        ms_cula = benchmark_cuda_mode_fn(
            lambda: run_cula(inputs),
            default_warmup=WARMUP,
            default_rep=N_ITERS,
            ncu_mode=NCU_MODE,
        )
        speedup = ms_fla / ms_cula if ms_cula > 0 else float("inf")

        r = {
            "B": B,
            "T": T,
            "accuracy": acc,
            "ms_fla": ms_fla,
            "ms_cula": ms_cula,
            "speedup": speedup,
        }
        results.append(r)

        torch.cuda.empty_cache()

    return results


# ============================================================
# Varlen benchmark
# ============================================================
def bench_varlen(configs, H: int, HV: int | None = None):
    if HV is None:
        HV = H
    print("\n" + "=" * 120)
    print(f" Varlen Benchmark: cuLA CuTe DSL vs FLA Triton  (H={H}, HV={HV}, K={K}, V={V}, BT={BT})")
    print("=" * 120)
    results = []

    for seq_lens, total_len, dist in configs:
        set_seed(SEED)
        torch.cuda.empty_cache()

        T = total_len
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=DEVICE)

        inputs = prepare_bwd_wy_dqkg_fused_inputs(
            B=1,
            T=T,
            H=H,
            K=K,
            V=V,
            HV=HV,
            chunk_size=BT,
            device=DEVICE,
            seed=SEED,
            cu_seqlens=cu_seqlens,
        )

        # Accuracy
        ref = run_fla_triton(inputs)
        out = run_cula(inputs)
        torch.cuda.synchronize()

        acc = {}
        names = ["dq", "dk", "dv", "db", "dg", "dA"]
        for name, r, o in zip(names, ref, out):
            rel_rmse, rel_max, mean_diff = relative_rms_error_rel_max_mean_abs(r, o)
            acc[name] = {"rel_rmse": rel_rmse, "rel_max": rel_max, "mean_diff": mean_diff}

        # Performance
        ms_fla = benchmark_cuda_mode_fn(
            lambda: run_fla_triton(inputs),
            default_warmup=WARMUP,
            default_rep=N_ITERS,
            ncu_mode=NCU_MODE,
        )
        ms_cula = benchmark_cuda_mode_fn(
            lambda: run_cula(inputs),
            default_warmup=WARMUP,
            default_rep=N_ITERS,
            ncu_mode=NCU_MODE,
        )
        speedup = ms_fla / ms_cula if ms_cula > 0 else float("inf")

        n_seqs = len(seq_lens)
        min_l, max_l = min(seq_lens), max(seq_lens)
        avg_l = T // n_seqs
        tag = f"{dist:>7s} {n_seqs:>2d}seqs T={T} [{min_l}..{max_l}] avg={avg_l}"

        r = {
            "tag": tag,
            "dist": dist,
            "T_total": T,
            "n_seqs": n_seqs,
            "accuracy": acc,
            "ms_fla": ms_fla,
            "ms_cula": ms_cula,
            "speedup": speedup,
        }
        results.append(r)

        torch.cuda.empty_cache()

    return results


# ============================================================
# Report
# ============================================================
def print_report(fixed_results, varlen_results, H: int):
    sep = "=" * 130
    print(f"\n\n{sep}")
    print("                       BENCHMARK REPORT: chunk_kda_bwd_wy_dqkg_fused")
    print("                       cuLA CuTe DSL vs FLA Triton")
    wu = 1 if NCU_MODE else WARMUP
    ni = 1 if NCU_MODE else N_ITERS
    mode_tag = "  [NCU mode]" if NCU_MODE else ""
    print(f"                       H={H}  K={K}  V={V}  BT={BT}  dtype=bf16{mode_tag}")
    print(f"                       Warmup={wu}  Iters={ni}")
    print(sep)

    acc_keys = ["dq", "dk", "dv", "db", "dg", "dA"]
    acc_header = "  ".join(f"{k:>10s}" for k in acc_keys)

    if fixed_results:
        print("\n  [Fixed-Length]")
        print(f"  {'─' * 125}")
        print(f"  {'B':>3s}  {'T':>5s}  │  {'FLA(ms)':>9s}  {'cuLA(ms)':>9s}  {'Speedup':>8s}  │  {'':>10s}{acc_header}")
        print(f"  {'─' * 125}")

        for r in fixed_results:
            rel_max_vals = "  ".join(f"{r['accuracy'].get(k, {}).get('rel_max', 0.0):10.6f}" for k in acc_keys)
            rel_rmse_vals = "  ".join(f"{r['accuracy'].get(k, {}).get('rel_rmse', 0.0):10.6f}" for k in acc_keys)
            print(
                f"  {r['B']:3d}  {r['T']:5d}  │  "
                f"{r['ms_fla']:9.4f}  {r['ms_cula']:9.4f}  {r['speedup']:7.2f}x  │  "
                f"{'rel_max:':>10s}{rel_max_vals}"
            )
            print(f"  {'':3s}  {'':5s}  │  {'':9s}  {'':9s}  {'':8s}  │  {'rel_rmse:':>10s}{rel_rmse_vals}")
        print(f"  {'─' * 125}")

    if varlen_results:
        print("\n  [Varlen]")
        print(f"  {'─' * 140}")
        print(f"  {'Config':>45s}  │  {'FLA(ms)':>9s}  {'cuLA(ms)':>9s}  {'Speedup':>8s}  │  {'':>10s}{acc_header}")
        print(f"  {'─' * 140}")

        for r in varlen_results:
            rel_max_vals = "  ".join(f"{r['accuracy'].get(k, {}).get('rel_max', 0.0):10.6f}" for k in acc_keys)
            rel_rmse_vals = "  ".join(f"{r['accuracy'].get(k, {}).get('rel_rmse', 0.0):10.6f}" for k in acc_keys)
            print(
                f"  {r['tag']:>45s}  │  "
                f"{r['ms_fla']:9.4f}  {r['ms_cula']:9.4f}  {r['speedup']:7.2f}x  │  "
                f"{'rel_max:':>10s}{rel_max_vals}"
            )
            print(f"  {'':>45s}  │  {'':9s}  {'':9s}  {'':8s}  │  {'rel_rmse:':>10s}{rel_rmse_vals}")
        print(f"  {'─' * 140}")

    print(f"\n{sep}\n")


# ============================================================
# Main
# ============================================================
def main():
    global NCU_MODE

    parser = argparse.ArgumentParser(description="Benchmark chunk_kda_bwd_wy_dqkg_fused: cuLA CuTe DSL vs FLA Triton")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["fixed", "varlen", "both"],
        help="Which benchmark mode to run (default: both)",
    )
    parser.add_argument(
        "--heads",
        nargs="+",
        type=int,
        default=[H_DEFAULT],
        help=f"Head counts to benchmark (default: [{H_DEFAULT}])",
    )
    parser.add_argument(
        "--hv",
        type=int,
        default=None,
        help="Number of value heads HV (default: same as H, i.e. no GVA). Must be a multiple of H.",
    )
    parser.add_argument("--ncu", action="store_true", help="NCU profiling mode: warmup=1, iters=1")
    args = parser.parse_args()

    if args.ncu:
        NCU_MODE = True
        print("[NCU mode] warmup=1, iters=1")

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    wu = 1 if NCU_MODE else WARMUP
    ni = 1 if NCU_MODE else N_ITERS
    print(f"K={K}, V={V}, BT={BT}, dtype={DTYPE}, warmup={wu}, rep={ni}")

    fixed_configs = [
        (1, 256),
        (1, 512),
        (1, 1024),
        (1, 2048),
        (1, 4096),
        (1, 8192),
        (2, 512),
        (2, 1024),
        (2, 2048),
        (2, 4096),
        (2, 8192),
    ]

    varlen_configs = build_varlen_configs(
        num_seqs_list=(10, 20),
        total_lens=(4096, 8192, 16384),
        dists=("uniform", "random", "skewed"),
    )

    for H in args.heads:
        HV = args.hv if args.hv is not None else H
        if not args.ncu:
            check_determinism(H=H, HV=HV, iters=10000)

        fixed_res, varlen_res = [], []

        if args.mode in ("fixed", "both"):
            fixed_res = bench_fixed(fixed_configs, H, HV)

        if args.mode in ("varlen", "both"):
            varlen_res = bench_varlen(varlen_configs, H, HV)

        print_report(fixed_res, varlen_res, H)

    print(f"\n{'=' * 130}")
    print("  All benchmarks done.")
    print(f"{'=' * 130}")


if __name__ == "__main__":
    main()
