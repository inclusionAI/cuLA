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
bench_kda_bwd_wy_dqkg_sm90.py — Benchmark: SM90 CuTe DSL vs FLA Triton
                                 for chunk_kda_bwd_wy_dqkg kernel (Hopper)

Modes:
  - Fixed-length: B=1,2 with various T
  - Varlen: variable-length sequences with different distributions

Usage:
  python benchmarks/bench_kda_bwd_wy_dqkg_sm90.py \
    [--mode fixed|varlen|both] [--heads 4 32]

  /usr/local/cuda-12.9/bin/ncu --profile-from-start off --set full \
    -o ncu_reports/<name> \
    python benchmarks/bench_kda_bwd_wy_dqkg_sm90.py --ncu \
      --mode varlen --heads 32 --total-len 16384 --num-seqs 20 --dist random
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
from einops import rearrange
from fla.modules.l2norm import l2norm_fwd
from fla.ops.kda.chunk_bwd import chunk_kda_bwd_wy_dqkg_fused as fla_bwd
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.index import prepare_chunk_indices

from benchmarks.utils import (
    SEED,
    build_varlen_configs,
    exclusive_cumsum,
    gen_random,
    gen_skewed,
    gen_uniform,
    set_seed,
)
from cula.ops.kda.sm90.bwd_wy_dqkg import chunk_kda_bwd_wy_dqkg_fused as sm90_bwd

torch.backends.cuda.matmul.allow_tf32 = True

H_DEFAULT = 32
K = 128
V = 128
BT = 64
BK = 32
BV = 64
MIN_OCC = 2
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")
WARMUP = 25
N_ITERS = 100
NCU_MODE = False

FIXED_CONFIGS = [
    (1, 256),
    (1, 512),
    (1, 1024),
    (1, 2048),
    (1, 4096),
    (1, 8192),
    (1, 16384),
    (1, 32768),
    (2, 512),
    (2, 1024),
    (2, 2048),
    (2, 4096),
    (2, 8192),
    (2, 16384),
    (2, 32768),
]
VARLEN_NUM_SEQS_LIST = (10, 20)
VARLEN_TOTAL_LENS = (4096, 8192, 16384)
VARLEN_DISTS = ("uniform", "random", "skewed")


def benchmark_fixed_configs():
    return list(FIXED_CONFIGS)


def benchmark_varlen_configs():
    return build_varlen_configs(
        num_seqs_list=VARLEN_NUM_SEQS_LIST,
        total_lens=VARLEN_TOTAL_LENS,
        dists=VARLEN_DISTS,
    )


# ============================================================
# Runners
# ============================================================
def prepare_bwd_wy_dqkg_fused_inputs(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    chunk_size: int = BT,
    device: torch.device | str = DEVICE,
    seed: int = SEED,
    cu_seqlens: torch.Tensor | None = None,
    dtype: torch.dtype = DTYPE,
) -> dict:
    """Prepare inputs for FLA and SM90 WY-DqKG fused backward runners."""
    scale = K**-0.5
    set_seed(seed)

    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    g_raw = torch.randn(B, T, H, K, dtype=dtype, device=device)
    beta = torch.randn(B, T, H, dtype=torch.float32, device=device).sigmoid()

    q, _ = l2norm_fwd(q)
    k, _ = l2norm_fwd(k)

    A_log = torch.randn(H, dtype=torch.float32, device=device)
    dt_bias = torch.randn(H * K, dtype=torch.float32, device=device)

    v_new = torch.randn(B, T, H, V, dtype=dtype, device=device)
    do = torch.randn(B, T, H, V, dtype=dtype, device=device)
    dv = torch.randn(B, T, H, V, dtype=dtype, device=device)
    A = torch.randn(B, T, H, chunk_size, dtype=dtype, device=device) * 0.1

    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.int()
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        NT = chunk_indices.shape[0]

        # Explicit cu_seqlens means the input is represented as one flattened
        # varlen stream. Fixed-length B>1 is just uniform varlen here.
        if B != 1:
            q, k, v, g_raw, beta = map(
                lambda x: rearrange(x, "b t ... -> 1 (b t) ..."),
                (q, k, v, g_raw, beta),
            )
            v_new, do, dv, A = map(
                lambda x: rearrange(x, "b t ... -> 1 (b t) ..."),
                (v_new, do, dv, A),
            )

        h = torch.randn(1, NT, H, K, V, dtype=dtype, device=device) * 0.01
        dh = torch.randn(1, NT, H, K, V, dtype=dtype, device=device) * 0.01
    else:
        NT = (T + chunk_size - 1) // chunk_size
        chunk_indices = None
        h = torch.randn(B, NT, H, K, V, dtype=dtype, device=device) * 0.01
        dh = torch.randn(B, NT, H, K, V, dtype=dtype, device=device) * 0.01

    g = kda_gate_chunk_cumsum(
        g=g_raw,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=RCP_LN2,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        lower_bound=-5.0,
    )

    return dict(
        q=q,
        k=k,
        v=v,
        v_new=v_new,
        g=g,
        beta=beta,
        A=A,
        h=h,
        dh=dh,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )


def run_sm90(inputs: dict):
    """Run the SM90 CuTe DSL kernel."""
    return sm90_bwd(
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
        bk=BK,
        bv=BV,
        min_occupancy=MIN_OCC,
    )


def run_fla(inputs: dict):
    """Run the FLA Triton baseline."""
    return fla_bwd(
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


# ============================================================
# Helpers
# ============================================================
def time_kernel(fn, warmup=None, n_iters=None):
    if warmup is None:
        warmup = 1 if NCU_MODE else WARMUP
    if n_iters is None:
        n_iters = 1 if NCU_MODE else N_ITERS
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iters


def accuracy_stats(ref, out):
    """Compute err_ratio, relative max diff, and mean absolute difference."""
    ref_f = ref.float()
    out_f = out.float()
    diff = (ref_f - out_f).abs()
    err = diff.flatten().pow(2).mean().sqrt().item()
    base = ref_f.flatten().pow(2).mean().sqrt().item()
    err_ratio = err / (base + 1e-8)
    max_diff = diff.max().item()
    denom = ref_f.abs().max().item()
    rel_max = max_diff / denom if denom > 0 else 0.0
    mean_diff = diff.mean().item()
    return err_ratio, rel_max, mean_diff


# Both SM90 and FLA return (dq, dk, dv, db, dg, dA).
OUT_MAP = {"dq": 0, "dk": 1, "dv": 2, "db": 3, "dg": 4, "dA": 5}
ACC_KEYS = ["dq", "dk", "dv", "db", "dg", "dA"]


def compute_accuracy(sm90_out, fla_out):
    """Compute per-output accuracy stats."""
    acc = {}
    for name in ACC_KEYS:
        s = sm90_out[OUT_MAP[name]]
        f = fla_out[OUT_MAP[name]]
        if s.shape != f.shape:
            f = f.reshape(s.shape)
        err_ratio, rel_max, mean_diff = accuracy_stats(f, s)
        acc[name] = {"err_ratio": err_ratio, "rel_max": rel_max, "mean_diff": mean_diff}
    return acc


def make_profile_seq_lens(args):
    profile_mode = "varlen" if args.mode == "both" else args.mode
    if profile_mode == "fixed":
        return [args.total_len] * args.batch, args.batch, args.total_len, profile_mode
    if args.dist == "uniform":
        seq_lens = gen_uniform(args.num_seqs, args.total_len)
    elif args.dist == "random":
        seq_lens = gen_random(args.num_seqs, args.total_len, seed=SEED)
    elif args.dist == "skewed":
        seq_lens = gen_skewed(args.num_seqs, args.total_len)
    else:
        raise ValueError(f"unknown dist: {args.dist}")
    return seq_lens, 1, args.total_len, profile_mode


def run_ncu_profile(args):
    """Run one SM90-only config bracketed by cudaProfilerStart/Stop."""
    seq_lens, batch, T, profile_mode = make_profile_seq_lens(args)
    cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=DEVICE)
    inputs = prepare_bwd_wy_dqkg_fused_inputs(
        B=batch,
        T=T,
        H=args.heads[0],
        K=K,
        V=V,
        chunk_size=BT,
        device=DEVICE,
        seed=SEED,
        cu_seqlens=cu_seqlens,
    )

    print(
        f"[NCU profiler] mode={profile_mode} dist={args.dist} H={args.heads[0]} "
        f"T={T} seqs={len(seq_lens)} min={min(seq_lens)} max={max(seq_lens)} "
        f"BK={BK} BV={BV} OCC={MIN_OCC} warmup={args.profile_warmup} "
        f"profile_iters={args.profile_iters}",
        flush=True,
    )

    for _ in range(args.profile_warmup):
        run_sm90(inputs)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(args.profile_iters):
        run_sm90(inputs)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("[NCU profiler] done")


# ============================================================
# Fixed-length benchmark
# ============================================================
def bench_fixed(configs, H: int):
    print(f"\n{'=' * 120}")
    print(f" Fixed-Length Benchmark: SM90 CuTe DSL vs FLA Triton  (H={H}, K={K}, V={V}, BT={BT})")
    print(f"{'=' * 120}")
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
            chunk_size=BT,
            device=DEVICE,
            seed=SEED,
            cu_seqlens=cu_seqlens,
        )

        sm90_out = run_sm90(inputs)
        fla_out = run_fla(inputs)
        torch.cuda.synchronize()

        acc = compute_accuracy(sm90_out, fla_out)

        ms_fla = time_kernel(lambda inp=inputs: run_fla(inp))
        ms_sm90 = time_kernel(lambda inp=inputs: run_sm90(inp))
        speedup = ms_fla / ms_sm90 if ms_sm90 > 0 else float("inf")

        results.append(
            {
                "B": B,
                "T": T,
                "accuracy": acc,
                "ms_fla": ms_fla,
                "ms_sm90": ms_sm90,
                "speedup": speedup,
            }
        )

        del inputs
        torch.cuda.empty_cache()

    return results


# ============================================================
# Varlen benchmark
# ============================================================
def bench_varlen(configs, H: int):
    print(f"\n{'=' * 120}")
    print(f" Varlen Benchmark: SM90 CuTe DSL vs FLA Triton  (H={H}, K={K}, V={V}, BT={BT})")
    print(f"{'=' * 120}")
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
            chunk_size=BT,
            device=DEVICE,
            seed=SEED,
            cu_seqlens=cu_seqlens,
        )

        sm90_out = run_sm90(inputs)
        fla_out = run_fla(inputs)
        torch.cuda.synchronize()

        acc = compute_accuracy(sm90_out, fla_out)

        ms_fla = time_kernel(lambda inp=inputs: run_fla(inp))
        ms_sm90 = time_kernel(lambda inp=inputs: run_sm90(inp))
        speedup = ms_fla / ms_sm90 if ms_sm90 > 0 else float("inf")

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
                "accuracy": acc,
                "ms_fla": ms_fla,
                "ms_sm90": ms_sm90,
                "speedup": speedup,
            }
        )

        del inputs
        torch.cuda.empty_cache()

    return results


# ============================================================
# Report
# ============================================================
def print_report(fixed_results, varlen_results, H: int):
    sep = "=" * 160
    wu = 1 if NCU_MODE else WARMUP
    ni = 1 if NCU_MODE else N_ITERS
    print(f"\n{sep}")
    print("  BENCHMARK REPORT: chunk_kda_bwd SM90 vs FLA Triton")
    print(f"  H={H}  K={K}  V={V}  BT={BT}  BK={BK}  BV={BV}  OCC={MIN_OCC}  dtype=bf16  Warmup={wu}  Iters={ni}")
    print(sep)

    acc_header = "  ".join(f"{k:>10s}" for k in ACC_KEYS)

    if fixed_results:
        print("\n  [Fixed-Length]")
        print(f"  {'─' * 145}")
        print(f"  {'B':>3s}  {'T':>5s}  │  {'FLA(ms)':>9s}  {'SM90(ms)':>9s}  {'Speedup':>8s}  │  {'':>10s}{acc_header}")
        print(f"  {'─' * 145}")

        for r in fixed_results:
            rel_max_vals = "  ".join(f"{r['accuracy'].get(k, {}).get('rel_max', 0.0):10.6f}" for k in ACC_KEYS)
            err_ratio_vals = "  ".join(f"{r['accuracy'].get(k, {}).get('err_ratio', 0.0):10.6f}" for k in ACC_KEYS)
            print(
                f"  {r['B']:3d}  {r['T']:5d}  │  "
                f"{r['ms_fla']:9.4f}  {r['ms_sm90']:9.4f}  {r['speedup']:7.2f}x  │  "
                f"{'rel_max:':>10s}{rel_max_vals}"
            )
            print(f"  {'':3s}  {'':5s}  │  {'':9s}  {'':9s}  {'':8s}  │  {'err_ratio:':>10s}{err_ratio_vals}")
        print(f"  {'─' * 145}")

    if varlen_results:
        print("\n  [Varlen]")
        print(f"  {'─' * 160}")
        print(f"  {'Config':>45s}  │  {'FLA(ms)':>9s}  {'SM90(ms)':>9s}  {'Speedup':>8s}  │  {'':>10s}{acc_header}")
        print(f"  {'─' * 160}")

        for r in varlen_results:
            rel_max_vals = "  ".join(f"{r['accuracy'].get(k, {}).get('rel_max', 0.0):10.6f}" for k in ACC_KEYS)
            err_ratio_vals = "  ".join(f"{r['accuracy'].get(k, {}).get('err_ratio', 0.0):10.6f}" for k in ACC_KEYS)
            print(
                f"  {r['tag']:>45s}  │  "
                f"{r['ms_fla']:9.4f}  {r['ms_sm90']:9.4f}  {r['speedup']:7.2f}x  │  "
                f"{'rel_max:':>10s}{rel_max_vals}"
            )
            print(f"  {'':>45s}  │  {'':9s}  {'':9s}  {'':8s}  │  {'err_ratio:':>10s}{err_ratio_vals}")
        print(f"  {'─' * 160}")

    print(f"\n{sep}\n")


# ============================================================
# Main
# ============================================================
def main():
    global NCU_MODE, BK, BV, MIN_OCC

    parser = argparse.ArgumentParser(description="Benchmark SM90 bwd kernel vs FLA Triton")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["fixed", "varlen", "both"],
        help="Which benchmark mode to run (default: both)",
    )
    parser.add_argument("--heads", nargs="+", type=int, default=[H_DEFAULT])
    parser.add_argument("--bk", type=int, default=32, choices=[32, 64])
    parser.add_argument("--bv", type=int, default=64, choices=[32, 64])
    parser.add_argument("--occ", type=int, default=2, choices=[1, 2])
    parser.add_argument(
        "--ncu",
        action="store_true",
        help="Run one SM90-only workload inside cudaProfilerStart/Stop for NCU.",
    )
    parser.add_argument("--batch", type=int, default=1, help="Fixed-length profiling batch size.")
    parser.add_argument("--total-len", type=int, default=16384, help="Profiling sequence/token length.")
    parser.add_argument("--num-seqs", type=int, default=20, help="Varlen profiling sequence count.")
    parser.add_argument(
        "--dist",
        type=str,
        default="random",
        choices=["uniform", "random", "skewed"],
        help="Varlen profiling length distribution.",
    )
    parser.add_argument(
        "--profile-warmup",
        "--warmup",
        dest="profile_warmup",
        type=int,
        default=1,
        help="Warmup iterations before the NCU profiler region.",
    )
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=1,
        help="SM90 iterations inside the NCU profiler region.",
    )
    args = parser.parse_args()

    BK = args.bk
    BV = args.bv
    MIN_OCC = args.occ

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"K={K}, V={V}, BT={BT}, BK={BK}, BV={BV}, OCC={MIN_OCC}, dtype={DTYPE}")

    if args.ncu:
        NCU_MODE = True
        run_ncu_profile(args)
        return

    fixed_configs = benchmark_fixed_configs()
    varlen_configs = benchmark_varlen_configs()

    for H in args.heads:
        fixed_res, varlen_res = [], []

        if args.mode in ("fixed", "both"):
            fixed_res = bench_fixed(fixed_configs, H)

        if args.mode in ("varlen", "both"):
            varlen_res = bench_varlen(varlen_configs, H)

        print_report(fixed_res, varlen_res, H)

    print("All benchmarks done.")


if __name__ == "__main__":
    main()
