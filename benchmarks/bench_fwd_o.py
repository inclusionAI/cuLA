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
  - Accuracy: max_diff, mean_diff between CuTe DSL and FLA Triton outputs
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

# ─── CuTe DSL wrapper (TVM-FFI compile cache) ───
_fwd_o_mod = importlib.import_module("cula.ops.fwd_o")
chunk_gla_fwd_o = _fwd_o_mod.chunk_gla_fwd_o
build_chunk_indices = _fwd_o_mod.build_chunk_indices

# ─── FLA baseline imports ───
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))  # Enable fast ops in FLA for fair comparison
from fla.ops.gla.chunk import chunk_gla_fwd_o_gk  # noqa: E402

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
def time_kernel(fn, warmup=None, n_iters=None):
    if warmup is None:
        warmup = 1 if NCU_MODE else WARMUP
    if n_iters is None:
        n_iters = 1 if NCU_MODE else N_ITERS
    """Time a kernel using CUDA events. Returns ms/call."""
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
    """Compute max, relative max, and mean absolute difference."""
    ref_f = ref.float()
    diff = (ref_f - out.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    denom = ref_f.abs().max().item()
    rel_max_diff = max_diff / denom if denom > 0 else 0.0
    return max_diff, rel_max_diff, mean_diff


# ============================================================
# Non-varlen benchmark
# ============================================================
def bench_non_varlen(configs):
    print("\n" + "=" * 80)
    print(" Non-Varlen Benchmark: CuTe DSL (SM100a) vs FLA Triton")
    print("=" * 80)
    results = []

    for B, T, H in configs:
        scale = K**-0.5
        NT = (T + BT - 1) // BT
        torch.manual_seed(42)
        torch.cuda.empty_cache()

        q = torch.randn(B, T, H, K, dtype=dtype, device=device)
        v = torch.randn(B, T, H, V, dtype=dtype, device=device)
        g = torch.randn(B, T, H, K, dtype=torch.float32, device=device) * 0.1
        h = torch.randn(B, NT, H, K, V, dtype=dtype, device=device) * 0.01
        A = torch.randn(B, T, H, BT, dtype=dtype, device=device) * 0.1

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
        o_cute_t = torch.zeros(B, T, H, V, dtype=dtype, device=device)

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

        max_diff, rel_max_diff, mean_diff = accuracy_stats(o_fla, o_cute_t)

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

        ms_fla = time_kernel(run_fla)
        ms_cute = time_kernel(run_cute)
        speedup = ms_fla / ms_cute if ms_cute > 0 else float("inf")

        r = {
            "B": B,
            "T": T,
            "H": H,
            "max_diff": max_diff,
            "rel_max_diff": rel_max_diff,
            "mean_diff": mean_diff,
            "ms_fla": ms_fla,
            "ms_cute": ms_cute,
            "speedup": speedup,
        }
        results.append(r)
        print(
            f"  B={B:2d} T={T:5d} H={H:2d} | "
            f"max_diff={max_diff:.6f} rel_max={rel_max_diff:.6f} mean_diff={mean_diff:.8f} | "
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

    for seq_lens, H in configs:
        scale = K**-0.5
        T_total = sum(seq_lens)
        cu_seqlens_list = [0]
        for sl in seq_lens:
            cu_seqlens_list.append(cu_seqlens_list[-1] + sl)
        total_nt_val = sum((sl + BT - 1) // BT for sl in seq_lens)

        torch.manual_seed(42)
        torch.cuda.empty_cache()

        # Flat token-indexed tensors (shared data for both kernels)
        # 4D with B=1: [1, T_total, H, *]
        q_flat = torch.randn(1, T_total, H, K, dtype=dtype, device=device)
        v_flat = torch.randn(1, T_total, H, V, dtype=dtype, device=device)
        g_flat = torch.randn(1, T_total, H, K, dtype=torch.float32, device=device) * 0.1
        h_flat = torch.randn(1, total_nt_val, H, K, V, dtype=dtype, device=device) * 0.01
        A_flat = torch.randn(1, T_total, H, BT, dtype=dtype, device=device) * 0.1

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
        o_cute_flat = torch.zeros(1, T_total, H, V, dtype=dtype, device=device)
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
        max_diff, rel_max_diff, mean_diff = accuracy_stats(o_fla.squeeze(0), o_cute_flat.squeeze(0))

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

        ms_fla = time_kernel(run_fla)
        ms_cute = time_kernel(run_cute)
        speedup = ms_fla / ms_cute if ms_cute > 0 else float("inf")

        n_seqs = len(seq_lens)
        min_l, max_l = min(seq_lens), max(seq_lens)
        avg_l = T_total // n_seqs
        tag = f"{n_seqs}seqs T={T_total} [{min_l}..{max_l}] avg={avg_l}"
        r = {
            "tag": tag,
            "T_total": T_total,
            "H": H,
            "n_seqs": n_seqs,
            "max_diff": max_diff,
            "rel_max_diff": rel_max_diff,
            "mean_diff": mean_diff,
            "ms_fla": ms_fla,
            "ms_cute": ms_cute,
            "speedup": speedup,
        }
        results.append(r)
        print(
            f"  {tag:45s} H={H:2d} | "
            f"max_diff={max_diff:.6f} rel_max={rel_max_diff:.6f} mean_diff={mean_diff:.8f} | "
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
            f"  {'B':>3s}  {'T':>5s}  {'H':>3s}  │  {'max_diff':>10s}  {'rel_max':>10s}  {'mean_diff':>12s}"
            f"  │  {'FLA(ms)':>9s}  {'CuTe(ms)':>9s}  {'Speedup':>8s}"
        )
        print(f"  {'─' * 90}")
        print(hdr)
        print(f"  {'─' * 90}")
        for r in nv_results:
            print(
                f"  {r['B']:3d}  {r['T']:5d}  {r['H']:3d}  │  "
                f"{r['max_diff']:10.6f}  {r['rel_max_diff']:10.6f}  {r['mean_diff']:12.8f}  │  "
                f"{r['ms_fla']:9.4f}  {r['ms_cute']:9.4f}  {r['speedup']:7.2f}x"
            )
        print(f"  {'─' * 90}")

    if vl_results:
        print("\n  [Varlen]")
        hdr = (
            f"  {'Config':>45s}  {'H':>3s}  │  {'max_diff':>10s}  {'rel_max':>10s}  {'mean_diff':>12s}"
            f"  │  {'FLA(ms)':>9s}  {'CuTe(ms)':>9s}  {'Speedup':>8s}"
        )
        print(f"  {'─' * 117}")
        print(hdr)
        print(f"  {'─' * 117}")
        for r in vl_results:
            print(
                f"  {r['tag']:>45s}  {r['H']:3d}  │  "
                f"{r['max_diff']:10.6f}  {r['rel_max_diff']:10.6f}  {r['mean_diff']:12.8f}  │  "
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
    args = parser.parse_args()

    global NCU_MODE
    if args.ncu:
        NCU_MODE = True
        print("[NCU mode] warmup=1, iters=1")

    non_varlen_configs = [
        # (B, T, H)
        (2, 8192, 64),
        (2, 32768, 64),
        (4, 8192, 64),
        (4, 32768, 64),
    ]

    varlen_configs = [
        # (seq_lens, H) — realistic serving scenarios
        # ~20-25 seqs, total 8k/32k, lengths vary 2-3x, H=64
        (gen_varlen_seqs(8192, 20, seed=1), 64),
        (gen_varlen_seqs(8192, 25, seed=2), 64),
        (gen_varlen_seqs(32768, 20, seed=3), 64),
        (gen_varlen_seqs(32768, 25, seed=4), 64),
    ]

    nv_res, vl_res = [], []

    if args.mode in ("non-varlen", "both"):
        nv_res = bench_non_varlen(non_varlen_configs)

    if args.mode in ("varlen", "both"):
        vl_res = bench_varlen(varlen_configs)

    print_report(nv_res, vl_res)


if __name__ == "__main__":
    main()
