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
bench_fwd_o.py — Benchmark: CuTe DSL kernel vs FLA Triton baseline
                  for chunk_gla_fwd_o (KDA forward output)

Compares:
    - Accuracy: relative_rms_error, max_diff, mean_diff between CuTe DSL and FLA Triton outputs
  - Performance: kernel execution time (ms) with CUDA events

Both non-varlen and varlen modes are supported.
K=128, V=128, BT=64, dtype=bf16, use_exp2=True.

Note: FLA internally allocates output tensor per call (torch.zeros_like).
      CuTe reuses a pre-allocated buffer. This makes the comparison
      favor CuTe slightly for small configs, but reflects real-world
      usage where CuTe output is pre-allocated in the fused pipeline.

Usage:
  python bench_fwd_o.py [--mode non-varlen|varlen|both] [--ncu]

With --ncu, warmup=1 and iters=1 for ncu profiling:
  ncu --set full -o report python bench_fwd_o.py --mode varlen --ncu
"""

import argparse
import importlib
import os
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# ─── CuTe DSL wrapper (TVM-FFI compile cache) ───
_fwd_o_mod = importlib.import_module("cula.ops.fwd_o_sm100")
chunk_gla_fwd_o = _fwd_o_mod.chunk_gla_fwd_o
build_chunk_indices = _fwd_o_mod.build_chunk_indices

# ─── FLA baseline imports ───
os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))  # Enable fast ops in FLA for fair comparison
from fla.ops.gla.chunk import chunk_gla_fwd_o_gk  # noqa: E402

from benchmarks.utils import benchmark_cuda_mode_fn, relative_rms_error_max_rel_mean_abs  # noqa: E402

# ============================================================
# Constants
# ============================================================
K, V, BT = 128, 128, 64
dtype = torch.bfloat16
device = "cuda"

WARMUP = 10
N_ITERS = 100
NCU_MODE = False


# ============================================================
# Helpers
# ============================================================


# ============================================================
# Non-varlen benchmark
# ============================================================
def bench_non_varlen(configs):
    print("\n" + "=" * 80)
    print(" Non-Varlen Benchmark: CuTe DSL (SM100a) vs FLA Triton")
    print("=" * 80)
    results = []

    for B, T, H, HV in configs:
        scale = K**-0.5
        NT = (T + BT - 1) // BT
        torch.manual_seed(42)
        torch.cuda.empty_cache()

        q = torch.randn(B, T, H, K, dtype=dtype, device=device)
        v = torch.randn(B, T, HV, V, dtype=dtype, device=device)
        g = torch.randn(B, T, HV, K, dtype=torch.float32, device=device) * 0.1
        h = torch.randn(B, NT, HV, K, V, dtype=dtype, device=device) * 0.01
        A = torch.randn(B, T, HV, BT, dtype=dtype, device=device) * 0.1

        # ---- FLA baseline (accuracy) ----
        o_fla = chunk_gla_fwd_o_gk(
            q=q,
            v=v,
            g=g,
            A=A,
            h=h.flatten(0, 1),
            scale=scale,
            chunk_size=BT,
            use_exp2=True,
        )

        # ---- CuTe DSL (accuracy) ----
        o_cute_t = torch.zeros(B, T, HV, V, dtype=dtype, device=device)

        # Warmup / first call triggers compilation via cache
        chunk_gla_fwd_o(
            q=q,
            v=v,
            g=g,
            h=h,
            o=o_cute_t,
            A=A,
            scale=scale,
            chunk_size=BT,
            is_varlen=False,
            persistent=True,
        )
        torch.cuda.synchronize()

        relative_rms_error, max_diff, rel_max_diff, mean_diff = relative_rms_error_max_rel_mean_abs(o_fla, o_cute_t)

        # ---- Performance timing ----
        def run_fla(q=q, v=v, g=g, A=A, h=h, scale=scale):
            chunk_gla_fwd_o_gk(
                q=q,
                v=v,
                g=g,
                A=A,
                h=h.flatten(0, 1),
                scale=scale,
                chunk_size=BT,
                use_exp2=True,
            )

        def run_cute(q=q, v=v, g=g, h=h, o=o_cute_t, A=A, scale=scale):
            chunk_gla_fwd_o(
                q=q,
                v=v,
                g=g,
                h=h,
                o=o,
                A=A,
                scale=scale,
                chunk_size=BT,
                is_varlen=False,
                persistent=True,
            )

        ms_fla = benchmark_cuda_mode_fn(run_fla, default_warmup=WARMUP, default_rep=N_ITERS, ncu_mode=NCU_MODE)
        ms_cute = benchmark_cuda_mode_fn(run_cute, default_warmup=WARMUP, default_rep=N_ITERS, ncu_mode=NCU_MODE)
        speedup = ms_fla / ms_cute if ms_cute > 0 else float("inf")

        hv_str = f"/{HV}" if HV != H else ""
        r = {
            "B": B,
            "T": T,
            "H": H,
            "HV": HV,
            "hv_str": hv_str,
            "relative_rms_error": relative_rms_error,
            "max_diff": max_diff,
            "rel_max_diff": rel_max_diff,
            "mean_diff": mean_diff,
            "ms_fla": ms_fla,
            "ms_cute": ms_cute,
            "speedup": speedup,
        }
        results.append(r)
        print(
            f"  B={B:2d} T={T:5d} H={H:2d}{hv_str:<4s} | "
            f"relative_rms_error={relative_rms_error:.6f} max_diff={max_diff:.6f} rel_max={rel_max_diff:.6f} mean_diff={mean_diff:.8f} | "
            f"FLA={ms_fla:.4f}ms CuTe={ms_cute:.4f}ms | "
            f"speedup={speedup:.2f}x"
        )

    return results


# ============================================================
# Varlen benchmark
# ============================================================
def gen_varlen_seqs(target_total, n_seqs, seed=0):
    """Generate n_seqs random seq lengths summing to target_total.
    Lengths vary ~2-3x (log-uniform-ish), each rounded up to multiple of 2."""
    import random

    rng = random.Random(seed)
    # Sample raw weights with 2-3x spread via log-uniform
    raw = [rng.uniform(0.4, 1.0) for _ in range(n_seqs)]
    s = sum(raw)
    # Scale to target, round to even, fix rounding error on last
    lens = [max(2, round(r / s * target_total / 2) * 2) for r in raw]
    diff = target_total - sum(lens)
    lens[-1] += diff
    if lens[-1] < 2:
        lens[-1] = 2
    return lens


def bench_varlen(configs):
    print("\n" + "=" * 80)
    print(" Varlen Benchmark: CuTe DSL (SM100a) vs FLA Triton")
    print("=" * 80)
    results = []

    for seq_lens, H, HV in configs:
        scale = K**-0.5
        T_total = sum(seq_lens)
        cu_seqlens_list = [0]
        for sl in seq_lens:
            cu_seqlens_list.append(cu_seqlens_list[-1] + sl)
        total_nt_val = sum((sl + BT - 1) // BT for sl in seq_lens)

        torch.manual_seed(42)
        torch.cuda.empty_cache()

        # Flat token-indexed tensors (shared data for both kernels)
        # q uses H (QK heads), g/v/h/A/o use HV (value heads)
        q_flat = torch.randn(1, T_total, H, K, dtype=dtype, device=device)
        v_flat = torch.randn(1, T_total, HV, V, dtype=dtype, device=device)
        g_flat = torch.randn(1, T_total, HV, K, dtype=torch.float32, device=device) * 0.1
        h_flat = torch.randn(1, total_nt_val, HV, K, V, dtype=dtype, device=device) * 0.01
        A_flat = torch.randn(1, T_total, HV, BT, dtype=dtype, device=device) * 0.1

        # ---- FLA baseline (needs [1, T_total, H, *] + cu_seqlens int64) ----
        cu_fla = torch.tensor(cu_seqlens_list, dtype=torch.long, device=device)

        o_fla = chunk_gla_fwd_o_gk(
            q=q_flat,
            v=v_flat,
            g=g_flat,
            A=A_flat,
            h=h_flat.flatten(0, 1),
            scale=scale,
            cu_seqlens=cu_fla,
            chunk_size=BT,
            use_exp2=True,
        )

        # ---- CuTe DSL varlen ----
        o_cute_flat = torch.zeros(1, T_total, HV, V, dtype=dtype, device=device)
        cu_cute = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=device)
        ci_cute = build_chunk_indices(seq_lens, BT=BT, device=device)

        # Warmup / first call triggers compilation via cache
        chunk_gla_fwd_o(
            q=q_flat,
            v=v_flat,
            g=g_flat,
            h=h_flat,
            o=o_cute_flat,
            A=A_flat,
            scale=scale,
            chunk_size=BT,
            cu_seqlens=cu_cute,
            chunk_indices=ci_cute,
            is_varlen=True,
            persistent=True,
        )
        torch.cuda.synchronize()

        # Both outputs are [1, T_total, H, V]; squeeze to [T_total, H, V] for comparison
        relative_rms_error, max_diff, rel_max_diff, mean_diff = relative_rms_error_max_rel_mean_abs(
            o_fla.squeeze(0), o_cute_flat.squeeze(0)
        )

        # ---- Performance timing ----
        def run_fla(q_flat=q_flat, v_flat=v_flat, g_flat=g_flat, A_flat=A_flat, h_flat=h_flat, cu_fla=cu_fla, scale=scale):
            chunk_gla_fwd_o_gk(
                q=q_flat,
                v=v_flat,
                g=g_flat,
                A=A_flat,
                h=h_flat.flatten(0, 1),
                scale=scale,
                cu_seqlens=cu_fla,
                chunk_size=BT,
                use_exp2=True,
            )

        def run_cute(
            q_flat=q_flat,
            v_flat=v_flat,
            g_flat=g_flat,
            h_flat=h_flat,
            o=o_cute_flat,
            A_flat=A_flat,
            cu_cute=cu_cute,
            ci_cute=ci_cute,
            scale=scale,
        ):
            chunk_gla_fwd_o(
                q=q_flat,
                v=v_flat,
                g=g_flat,
                h=h_flat,
                o=o,
                A=A_flat,
                scale=scale,
                chunk_size=BT,
                cu_seqlens=cu_cute,
                chunk_indices=ci_cute,
                is_varlen=True,
                persistent=True,
            )

        ms_fla = benchmark_cuda_mode_fn(run_fla, default_warmup=WARMUP, default_rep=N_ITERS, ncu_mode=NCU_MODE)
        ms_cute = benchmark_cuda_mode_fn(run_cute, default_warmup=WARMUP, default_rep=N_ITERS, ncu_mode=NCU_MODE)
        speedup = ms_fla / ms_cute if ms_cute > 0 else float("inf")

        n_seqs = len(seq_lens)
        min_l, max_l = min(seq_lens), max(seq_lens)
        avg_l = T_total // n_seqs
        tag = f"{n_seqs}seqs T={T_total} [{min_l}..{max_l}] avg={avg_l}"
        hv_str = f"/{HV}" if HV != H else ""
        r = {
            "tag": tag,
            "T_total": T_total,
            "H": H,
            "HV": HV,
            "hv_str": hv_str,
            "n_seqs": n_seqs,
            "relative_rms_error": relative_rms_error,
            "max_diff": max_diff,
            "rel_max_diff": rel_max_diff,
            "mean_diff": mean_diff,
            "ms_fla": ms_fla,
            "ms_cute": ms_cute,
            "speedup": speedup,
        }
        results.append(r)
        print(
            f"  {tag:45s} H={H:2d}{hv_str:<4s} | "
            f"relative_rms_error={relative_rms_error:.6f} max_diff={max_diff:.6f} rel_max={rel_max_diff:.6f} mean_diff={mean_diff:.8f} | "
            f"FLA={ms_fla:.4f}ms CuTe={ms_cute:.4f}ms | "
            f"speedup={speedup:.2f}x"
        )

    return results


# ============================================================
# Report table
# ============================================================
def print_report(nv_results, vl_results):
    sep = "=" * 100
    print(f"\n\n{sep}")
    print("                     BENCHMARK REPORT: chunk_gla_fwd_o")
    print("                     CuTe DSL (Blackwell SM100a) vs FLA Triton")
    print(f"                     K={K}  V={V}  BT={BT}  dtype=bf16  use_exp2=True")
    wu = 1 if NCU_MODE else WARMUP
    ni = 1 if NCU_MODE else N_ITERS
    ncu_tag = "  [NCU mode]" if NCU_MODE else ""
    print(f"                     Warmup={wu}  Iters={ni}{ncu_tag}")
    print(sep)

    if nv_results:
        print("\n  [Non-Varlen]")
        hdr = (
            f"  {'B':>3s}  {'T':>5s}  {'H':>7s}  │  {'rel_rmse':>10s}  {'max_diff':>10s}  {'rel_max':>10s}  {'mean_diff':>12s}"
            f"  │  {'FLA(ms)':>9s}  {'CuTe(ms)':>9s}  {'Speedup':>8s}"
        )
        print(f"  {'─' * 90}")
        print(hdr)
        print(f"  {'─' * 90}")
        for r in nv_results:
            h_label = f"{r['H']}{r['hv_str']}"
            print(
                f"  {r['B']:3d}  {r['T']:5d}  {h_label:>7s}  │  "
                f"{r['relative_rms_error']:10.6f}  {r['max_diff']:10.6f}  {r['rel_max_diff']:10.6f}  {r['mean_diff']:12.8f}  │  "
                f"{r['ms_fla']:9.4f}  {r['ms_cute']:9.4f}  {r['speedup']:7.2f}x"
            )
        print(f"  {'─' * 90}")

    if vl_results:
        print("\n  [Varlen]")
        hdr = (
            f"  {'Config':>45s}  {'H':>7s}  │  {'rel_rmse':>10s}  {'max_diff':>10s}  {'rel_max':>10s}  {'mean_diff':>12s}"
            f"  │  {'FLA(ms)':>9s}  {'CuTe(ms)':>9s}  {'Speedup':>8s}"
        )
        print(f"  {'─' * 117}")
        print(hdr)
        print(f"  {'─' * 117}")
        for r in vl_results:
            h_label = f"{r['H']}{r['hv_str']}"
            print(
                f"  {r['tag']:>45s}  {h_label:>7s}  │  "
                f"{r['relative_rms_error']:10.6f}  {r['max_diff']:10.6f}  {r['rel_max_diff']:10.6f}  {r['mean_diff']:12.8f}  │  "
                f"{r['ms_fla']:9.4f}  {r['ms_cute']:9.4f}  {r['speedup']:7.2f}x"
            )
        print(f"  {'─' * 117}")

    print(f"\n{sep}\n")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="bench_fwd_o: CuTe DSL (SM100a) vs FLA Triton baseline")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["non-varlen", "varlen", "both"],
        help="Which benchmark mode to run (default: both)",
    )
    parser.add_argument(
        "--ncu",
        action="store_true",
        help="NCU profiling mode: warmup=1, iters=1",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=64,
        help="Number of QK heads H (default: 64)",
    )
    parser.add_argument(
        "--hv",
        type=int,
        default=None,
        help="Number of value heads HV (default: same as --heads, i.e. no GVA)",
    )
    args = parser.parse_args()

    global NCU_MODE
    if args.ncu:
        NCU_MODE = True
        print("[NCU mode] warmup=1, iters=1")

    H = args.heads
    HV = args.hv if args.hv is not None else H
    assert HV >= H and HV % H == 0, f"HV ({HV}) must be >= H ({H}) and divisible by H"

    non_varlen_configs = [
        # (B, T, H, HV)
        (2, 8192, H, HV),
        (2, 32768, H, HV),
        (4, 8192, H, HV),
        (4, 32768, H, HV),
    ]

    varlen_configs = [
        # (seq_lens, H, HV)
        (gen_varlen_seqs(8192, 20, seed=1), H, HV),
        (gen_varlen_seqs(8192, 25, seed=2), H, HV),
        (gen_varlen_seqs(32768, 20, seed=3), H, HV),
        (gen_varlen_seqs(32768, 25, seed=4), H, HV),
    ]

    nv_res, vl_res = [], []

    if args.mode in ("non-varlen", "both"):
        nv_res = bench_non_varlen(non_varlen_configs)

    if args.mode in ("varlen", "both"):
        vl_res = bench_varlen(varlen_configs)

    print_report(nv_res, vl_res)


if __name__ == "__main__":
    main()
