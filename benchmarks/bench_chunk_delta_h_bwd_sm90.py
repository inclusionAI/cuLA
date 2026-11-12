#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
bench_chunk_delta_h_bwd_sm90.py - Benchmark: SM90 CuTe DSL bwd_dhu kernel
                                  vs FLA Triton baseline.

This mirrors benchmarks/bench_chunk_delta_h.py as closely as the backward API
allows:
  - non-varlen and varlen modes
  - K=128, V=128, BT=64, dtype=bf16
  - same default B/T/H and varlen sequence-count ranges as fwd
  - dht/dh0 map to fwd initial_state/output_final_state

Usage:
  python benchmarks/bench_chunk_delta_h_bwd_sm90.py --mode both
  python benchmarks/bench_chunk_delta_h_bwd_sm90.py --preset focused --mode non-varlen
"""

import argparse
import math
import os
import pathlib
import sys

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu as fla_bwd_dhu
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets

import cula.ops.chunk_delta_h_bwd as bwd_mod

chunk_gated_delta_rule_bwd_dhu_sm90 = bwd_mod.chunk_gated_delta_rule_bwd_dhu_sm90

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


K, V, BT = 128, 128, 64
dtype = torch.bfloat16
device = "cuda"

WARMUP = 10
N_ITERS = 100
NCU_MODE = False


def time_kernel(fn, warmup=None, n_iters=None):
    if warmup is None:
        warmup = 1 if NCU_MODE else WARMUP
    if n_iters is None:
        n_iters = 1 if NCU_MODE else N_ITERS
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
    diff = (ref.float() - out.float()).abs()
    max_abs = diff.max().item()
    rel_linf = max_abs / max(ref.float().abs().max().item(), 1e-6)
    return max_abs, diff.mean().item(), rel_linf


def bwd_accuracy_stats(ref_result, cute_result):
    ref_dh, ref_dh0, ref_dv2 = ref_result
    got_dh, got_dh0, got_dv2 = cute_result
    dh_max, dh_mean, dh_rel = accuracy_stats(ref_dh, got_dh)
    dv2_max, dv2_mean, dv2_rel = accuracy_stats(ref_dv2, got_dv2)
    dh0_max, dh0_mean, dh0_rel = 0.0, 0.0, 0.0
    if ref_dh0 is not None:
        dh0_max, dh0_mean, dh0_rel = accuracy_stats(ref_dh0, got_dh0)
    return {
        "dh_max": dh_max,
        "dh_mean": dh_mean,
        "dh_rel": dh_rel,
        "dh0_max": dh0_max,
        "dh0_mean": dh0_mean,
        "dh0_rel": dh0_rel,
        "dv2_max": dv2_max,
        "dv2_mean": dv2_mean,
        "dv2_rel": dv2_rel,
        "max_diff": max(dh_max, dh0_max, dv2_max),
        "mean_diff": max(dh_mean, dh0_mean, dv2_mean),
        "max_rel": max(dh_rel, dh0_rel, dv2_rel),
    }


def make_non_varlen_inputs(B, T, H, use_g, use_gk, use_dht, use_dh0, transpose_state=False, seed=42):
    torch.manual_seed(seed)
    torch.cuda.empty_cache()

    q = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
    k = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
    w = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
    do = torch.randn(B, T, H, V, device=device, dtype=dtype) * 0.1
    dv = torch.randn(B, T, H, V, device=device, dtype=dtype) * 0.1

    g = None
    if use_g:
        g = -torch.abs(torch.randn(B, T, H, device=device, dtype=torch.float32) * 0.01).cumsum(dim=1)

    gk = None
    if use_gk:
        gk = -torch.abs(torch.randn(B, T, H, K, device=device, dtype=torch.float32) * 0.01).cumsum(dim=1)

    state_shape = (B, H, V, K) if transpose_state else (B, H, K, V)
    dht = torch.randn(state_shape, device=device, dtype=torch.float32) * 0.01 if use_dht else None
    dh0 = torch.empty(state_shape, device=device, dtype=torch.float32) if use_dh0 else None
    return q, k, w, do, dv, g, gk, dht, dh0


def generate_seq_lens(num_seqs, total_T, ratio, seed=42):
    rng = np.random.RandomState(seed)
    log_weights = rng.uniform(0, np.log(ratio), num_seqs)
    weights = np.exp(log_weights)
    raw_lens = weights / weights.sum() * total_T
    seq_lens = np.maximum(np.round(raw_lens).astype(int), 1)
    diff = total_T - seq_lens.sum()
    if diff > 0:
        indices = np.argsort(seq_lens)
        for i in range(abs(diff)):
            seq_lens[indices[i % num_seqs]] += 1
    elif diff < 0:
        indices = np.argsort(-seq_lens)
        for i in range(abs(diff)):
            seq_lens[indices[i % num_seqs]] -= 1
    assert seq_lens.sum() == total_T
    return list(seq_lens)


def make_varlen_inputs(num_seqs, total_T, H, ratio, use_g, use_gk, use_dht, use_dh0, seed=42):
    seq_lens = generate_seq_lens(num_seqs, total_T, ratio, seed=seed)
    cu_seqlens_list = [0]
    for seq_len in seq_lens:
        cu_seqlens_list.append(cu_seqlens_list[-1] + seq_len)
    cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=device)
    cu_seqlens_long = cu_seqlens.long()

    chunk_indices = prepare_chunk_indices(cu_seqlens_long, BT)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens_long, BT).int()

    torch.manual_seed(seed)
    torch.cuda.empty_cache()

    q = torch.randn(1, total_T, H, K, device=device, dtype=dtype) * 0.1
    k = torch.randn(1, total_T, H, K, device=device, dtype=dtype) * 0.1
    w = torch.randn(1, total_T, H, K, device=device, dtype=dtype) * 0.1
    do = torch.randn(1, total_T, H, V, device=device, dtype=dtype) * 0.1
    dv = torch.randn(1, total_T, H, V, device=device, dtype=dtype) * 0.1

    g = None
    if use_g:
        g_raw = torch.randn(1, total_T, H, device=device, dtype=torch.float32) * 0.01
        g = torch.zeros_like(g_raw)
        for i in range(num_seqs):
            bos = cu_seqlens[i].item()
            eos = cu_seqlens[i + 1].item()
            g[:, bos:eos] = -torch.abs(g_raw[:, bos:eos]).cumsum(dim=1)

    gk = None
    if use_gk:
        gk_raw = torch.randn(1, total_T, H, K, device=device, dtype=torch.float32) * 0.01
        gk = torch.zeros_like(gk_raw)
        for i in range(num_seqs):
            bos = cu_seqlens[i].item()
            eos = cu_seqlens[i + 1].item()
            gk[:, bos:eos] = -torch.abs(gk_raw[:, bos:eos]).cumsum(dim=1)

    state_shape = (num_seqs, H, K, V)
    dht = torch.randn(state_shape, device=device, dtype=torch.float32) * 0.01 if use_dht else None
    dh0 = torch.empty(state_shape, device=device, dtype=torch.float32) if use_dh0 else None
    return seq_lens, cu_seqlens, cu_seqlens_long, chunk_indices, chunk_offsets, q, k, w, do, dv, g, gk, dht, dh0


def run_fla(q, k, w, do, dv, g, gk, dht, dh0, cu_seqlens_long=None):
    return fla_bwd_dhu(
        q=q,
        k=k,
        w=w,
        do=do,
        dv=dv,
        g=g,
        gk=gk,
        h0=dh0,
        dht=dht,
        scale=K**-0.5,
        cu_seqlens=cu_seqlens_long,
        chunk_size=BT,
        use_exp2=True,
    )


def run_cute(q, k, w, do, dv, g, gk, dht, dh0, cu_seqlens=None, chunk_indices=None, chunk_offsets=None):
    return chunk_gated_delta_rule_bwd_dhu_sm90(
        q=q,
        k=k,
        w=w,
        do=do,
        dv=dv,
        g=g,
        gk=gk,
        h0=dh0,
        dht=dht,
        scale=K**-0.5,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        chunk_size=BT,
        use_exp2=True,
    )


def flags_str(use_g, use_gk, use_dht, use_dh0):
    flags = []
    if use_g:
        flags.append("g")
    if use_gk:
        flags.append("gk")
    if use_dht:
        flags.append("dht")
    if use_dh0:
        flags.append("dh0")
    return f" [{','.join(flags)}]" if flags else ""


FOCUSED_FEATURE_MODES = {
    "A": (False, False, False, False),
    "B": (False, False, True, True),
    "C": (True, False, False, False),
    "D": (False, True, False, False),
    "E": (True, True, False, False),
}


def focused_feature_label(use_g, use_gk, use_dht, use_dh0):
    for name, flags in FOCUSED_FEATURE_MODES.items():
        if flags == (use_g, use_gk, use_dht, use_dh0):
            return name
    return "-"


def build_focused_non_varlen_configs(feature_mode="all"):
    modes = FOCUSED_FEATURE_MODES.items()
    if feature_mode != "all":
        modes = [(feature_mode, FOCUSED_FEATURE_MODES[feature_mode])]

    configs = []
    for B in (1, 2, 4):
        for H in (16, 32):
            for T in (2048, 4096, 8192, 16384):
                for _, flags in modes:
                    use_g, use_gk, use_dht, use_dh0 = flags
                    configs.append((B, T, H, use_g, use_gk, use_dht, use_dh0))
    return configs


def filter_non_varlen_configs(configs, only_b=None, only_h=None, only_t=None):
    if only_b is not None:
        configs = [cfg for cfg in configs if cfg[0] == only_b]
    if only_t is not None:
        configs = [cfg for cfg in configs if cfg[1] == only_t]
    if only_h is not None:
        configs = [cfg for cfg in configs if cfg[2] == only_h]
    return configs


def _compile_cache_misses():
    return bwd_mod._compile_bwd_dhu_sm90.cache_info().misses


def bench_non_varlen(configs):
    print("\n" + "=" * 80)
    print(" Non-Varlen Benchmark: CuTe DSL (SM90) bwd_dhu vs FLA Triton")
    print("=" * 80)
    results = []

    for B, T, H, use_g, use_gk, use_dht, use_dh0 in configs:
        q, k, w, do, dv, g, gk, dht, dh0 = make_non_varlen_inputs(B, T, H, use_g, use_gk, use_dht, use_dh0)

        ref = run_fla(q, k, w, do, dv, g, gk, dht, dh0)
        misses_before = _compile_cache_misses()
        got = run_cute(q, k, w, do, dv, g, gk, dht, dh0)
        compiled_new = _compile_cache_misses() > misses_before
        torch.cuda.synchronize()
        acc = bwd_accuracy_stats(ref, got)

        def run_fla_case(q=q, k=k, w=w, do=do, dv=dv, g=g, gk=gk, dht=dht, dh0=dh0):
            run_fla(q, k, w, do, dv, g, gk, dht, dh0)

        def run_cute_case(q=q, k=k, w=w, do=do, dv=dv, g=g, gk=gk, dht=dht, dh0=dh0):
            run_cute(q, k, w, do, dv, g, gk, dht, dh0)

        ms_fla = time_kernel(run_fla_case)
        ms_cute = time_kernel(run_cute_case)
        speedup = ms_fla / ms_cute if ms_cute > 0 else float("inf")
        flag_str = flags_str(use_g, use_gk, use_dht, use_dh0)
        feature_mode = focused_feature_label(use_g, use_gk, use_dht, use_dh0)

        r = {
            "B": B,
            "T": T,
            "H": H,
            "feature_mode": feature_mode,
            "flags": flag_str,
            "compiled_new": compiled_new,
            "ms_fla": ms_fla,
            "ms_cute": ms_cute,
            "speedup": speedup,
            **acc,
        }
        results.append(r)
        print(
            f"  B={B:2d} T={T:5d} H={H:3d} mode={feature_mode}{flag_str:<18s} | "
            f"abs(dh={acc['dh_max']:.6f} dh0={acc['dh0_max']:.6f} dv2={acc['dv2_max']:.6f}) "
            f"rel(dh={acc['dh_rel']:.3e} dh0={acc['dh0_rel']:.3e} dv2={acc['dv2_rel']:.3e}) | "
            f"FLA={ms_fla:.4f}ms CuTe={ms_cute:.4f}ms speedup={speedup:.2f}x | "
            f"compiled={'yes' if compiled_new else 'no'}"
        )

    return results


def bench_varlen(configs):
    print("\n" + "=" * 80)
    print(" Varlen Benchmark: CuTe DSL (SM90) bwd_dhu vs FLA Triton")
    print("=" * 80)
    results = []

    for num_seqs, total_T, H, ratio, use_g, use_gk, use_dht, use_dh0 in configs:
        (
            seq_lens,
            cu_seqlens,
            cu_seqlens_long,
            chunk_indices,
            chunk_offsets,
            q,
            k,
            w,
            do,
            dv,
            g,
            gk,
            dht,
            dh0,
        ) = make_varlen_inputs(num_seqs, total_T, H, ratio, use_g, use_gk, use_dht, use_dh0)

        ref = run_fla(q, k, w, do, dv, g, gk, dht, dh0, cu_seqlens_long=cu_seqlens_long)
        got = run_cute(
            q,
            k,
            w,
            do,
            dv,
            g,
            gk,
            dht,
            dh0,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
        )
        torch.cuda.synchronize()
        acc = bwd_accuracy_stats(ref, got)

        def run_fla_case(q=q, k=k, w=w, do=do, dv=dv, g=g, gk=gk, dht=dht, dh0=dh0, cu=cu_seqlens_long):
            run_fla(q, k, w, do, dv, g, gk, dht, dh0, cu_seqlens_long=cu)

        def run_cute_case(
            q=q,
            k=k,
            w=w,
            do=do,
            dv=dv,
            g=g,
            gk=gk,
            dht=dht,
            dh0=dh0,
            cu=cu_seqlens,
            ci=chunk_indices,
            co=chunk_offsets,
        ):
            run_cute(q, k, w, do, dv, g, gk, dht, dh0, cu_seqlens=cu, chunk_indices=ci, chunk_offsets=co)

        ms_fla = time_kernel(run_fla_case)
        ms_cute = time_kernel(run_cute_case)
        speedup = ms_fla / ms_cute if ms_cute > 0 else float("inf")

        min_l, max_l = min(seq_lens), max(seq_lens)
        avg_l = total_T // num_seqs
        tag = f"{num_seqs}seqs T={total_T} [{min_l}..{max_l}] avg={avg_l}"
        flag_str = flags_str(use_g, use_gk, use_dht, use_dh0)

        r = {
            "tag": tag,
            "T_total": total_T,
            "H": H,
            "n_seqs": num_seqs,
            "feature_mode": focused_feature_label(use_g, use_gk, use_dht, use_dh0),
            "flags": flag_str,
            "compiled_new": False,
            "ms_fla": ms_fla,
            "ms_cute": ms_cute,
            "speedup": speedup,
            **acc,
        }
        results.append(r)
        print(
            f"  {tag:40s} H={H:3d}{flag_str:<18s} | "
            f"max={acc['max_diff']:.6f} mean={acc['mean_diff']:.8f} "
            f"(dh={acc['dh_max']:.6f} dh0={acc['dh0_max']:.6f} dv2={acc['dv2_max']:.6f}) | "
            f"FLA={ms_fla:.4f}ms CuTe={ms_cute:.4f}ms | speedup={speedup:.2f}x"
        )

    return results


def print_report(nv_results, vl_results):
    sep = "=" * 120
    print(f"\n\n{sep}")
    print("                     BENCHMARK REPORT: chunk_delta_rule_bwd_dhu")
    print("                     CuTe DSL (Hopper SM90) vs FLA Triton")
    print(f"                     K={K}  V={V}  BT={BT}  dtype=bf16")
    wu = 1 if NCU_MODE else WARMUP
    ni = 1 if NCU_MODE else N_ITERS
    ncu_tag = "  [NCU mode]" if NCU_MODE else ""
    print(f"                     Warmup={wu}  Iters={ni}{ncu_tag}")
    print(sep)

    if nv_results:
        print("\n  [Non-Varlen]")
        print(f"  {'-' * 132}")
        print(
            f"  {'Config':<45s} | {'max_abs':>10s} {'max_rel':>10s} | "
            f"{'FLA(ms)':>9s} {'CuTe(ms)':>9s} {'Speedup':>8s} {'Compiled':>8s}"
        )
        print(f"  {'-' * 132}")
        for r in nv_results:
            label = f"B={r['B']:2d} T={r['T']:5d} H={r['H']:3d} mode={r.get('feature_mode', '-')}{r['flags']}"
            print(
                f"  {label:<45s} | {r['max_diff']:10.6f} {r['max_rel']:10.3e} | "
                f"{r['ms_fla']:9.4f} {r['ms_cute']:9.4f} {r['speedup']:7.2f}x "
                f"{'yes' if r.get('compiled_new') else 'no':>8s}"
            )
        print(f"  {'-' * 132}")
        speedups = [r["speedup"] for r in nv_results]
        geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        print(f"  {'Geometric mean':<45s} | {'':>10s} {'':>10s} | {'':>9s} {'':>9s} {geo:7.2f}x {'':>8s}")

    if vl_results:
        print("\n  [Varlen]")
        print(f"  {'-' * 120}")
        print(f"  {'Config':>60s} | {'max_diff':>10s} {'mean_diff':>12s} | {'FLA(ms)':>9s} {'CuTe(ms)':>9s} {'Speedup':>8s}")
        print(f"  {'-' * 120}")
        for r in vl_results:
            label = f"{r['tag']} H={r['H']:3d}{r['flags']}"
            print(
                f"  {label:>60s} | {r['max_diff']:10.6f} {r['mean_diff']:12.8f} | "
                f"{r['ms_fla']:9.4f} {r['ms_cute']:9.4f} {r['speedup']:7.2f}x"
            )
        print(f"  {'-' * 120}")
        speedups = [r["speedup"] for r in vl_results]
        geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        print(f"  {'Geometric mean':>60s} | {'':>10s} {'':>12s} | {'':>9s} {'':>9s} {geo:7.2f}x")

    print(f"\n{sep}\n")


def main():
    parser = argparse.ArgumentParser(description="bench_chunk_delta_h_bwd_sm90: CuTe DSL (SM90) vs FLA Triton")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["non-varlen", "varlen", "both"],
        help="Which benchmark mode to run (default: both)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="fwd",
        choices=["representative", "fwd", "focused"],
        help="fwd mirrors bench_chunk_delta_h.py; representative runs a short subset; focused runs the long non-varlen matrix",
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="all",
        choices=["all", "A", "B", "C", "D", "E"],
        help="For --preset focused: A=no gates/state, B=dht+dh0, C=g, D=gk, E=g+gk",
    )
    parser.add_argument("--max-configs", type=int, default=None, help="Run only the first N configs from the selected preset")
    parser.add_argument("--filter-b", type=int, default=None, help="Only run non-varlen configs with this B")
    parser.add_argument("--filter-h", type=int, default=None, help="Only run non-varlen configs with this H")
    parser.add_argument("--filter-t", type=int, default=None, help="Only run non-varlen configs with this T")
    parser.add_argument("--warmup", type=int, default=None, help="Override warmup iterations")
    parser.add_argument("--iters", type=int, default=None, help="Override timed iterations")
    parser.add_argument("--ncu", action="store_true", help="NCU profiling mode: warmup=1, iters=1")
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("This benchmark requires an SM90/Hopper GPU.")

    global NCU_MODE, WARMUP, N_ITERS
    if args.ncu:
        NCU_MODE = True
        print("[NCU mode] warmup=1, iters=1")
    if args.warmup is not None:
        WARMUP = args.warmup
    if args.iters is not None:
        N_ITERS = args.iters

    if args.preset == "focused":
        # Focused non-varlen long-token matrix requested for SM90 bwd_dhu tuning.
        # Tuple: (B, T, H, use_g, use_gk, use_dht, use_dh0)
        non_varlen_configs = build_focused_non_varlen_configs(args.feature_mode)
        varlen_configs = []
    elif args.preset == "fwd":
        # Matches bench_chunk_delta_h.py's default dimensions.
        # Tuple: (B, T, H, use_g, use_gk, use_dht, use_dh0)
        non_varlen_configs = [
            (1, 8192, 64, False, True, True, True),
            (2, 8192, 64, False, True, True, True),
            (4, 8192, 64, False, True, True, True),
            (8, 8192, 64, False, True, True, True),
        ]

        # Tuple: (num_seqs, total_T, H, ratio, use_g, use_gk, use_dht, use_dh0)
        varlen_configs = [
            (20, 8192, 64, 2.0, False, True, True, True),
            (25, 8192, 64, 3.0, False, True, True, True),
            (20, 8192, 64, 4.0, False, True, True, True),
            (20, 32768, 64, 2.0, False, True, True, True),
            (25, 32768, 64, 3.0, False, True, True, True),
        ]
    else:
        # Short representative subset for day-to-day iteration.
        # Tuple: (B, T, H, use_g, use_gk, use_dht, use_dh0)
        non_varlen_configs = [
            (1, 512, 4, False, True, True, False),
            (1, 512, 4, True, False, True, False),
            (2, 1024, 64, False, True, True, True),
            (1, 2048, 64, False, True, True, False),
        ]

        # Tuple: (num_seqs, total_T, H, ratio, use_g, use_gk, use_dht, use_dh0)
        varlen_configs = [
            (3, 512, 2, 3.0, False, True, True, False),
            (4, 768, 2, 4.0, True, False, True, True),
        ]

    non_varlen_configs = filter_non_varlen_configs(
        non_varlen_configs,
        only_b=args.filter_b,
        only_h=args.filter_h,
        only_t=args.filter_t,
    )

    if args.max_configs is not None:
        non_varlen_configs = non_varlen_configs[: args.max_configs]
        varlen_configs = varlen_configs[: args.max_configs]

    nv_res, vl_res = [], []
    if args.mode in ("non-varlen", "both"):
        nv_res = bench_non_varlen(non_varlen_configs)
    if args.mode in ("varlen", "both"):
        vl_res = bench_varlen(varlen_configs)
    print_report(nv_res, vl_res)


if __name__ == "__main__":
    main()
