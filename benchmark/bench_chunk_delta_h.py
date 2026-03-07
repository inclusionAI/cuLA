#!/usr/bin/env python3
"""
bench_chunk_delta_h.py — Benchmark: CuTe DSL kernel vs FLA Triton baseline
                         for chunk_delta_rule_fwd_h (inter-chunk recurrent state)

Compares:
  - Accuracy: max_diff, mean_diff between CuTe DSL and FLA Triton outputs
  - Performance: kernel execution time (ms) with CUDA events

Both non-varlen and varlen modes are supported.
K=128, V=128, BT=64, dtype=bf16.

Usage:
  python bench_chunk_delta_h.py [--mode non-varlen|varlen|both] [--ncu]

With --ncu, warmup=1 and iters=1 for ncu profiling:
  ncu --set full -o report python bench_chunk_delta_h.py --mode varlen --ncu
"""

import os
import sys
import math
import argparse
import pathlib

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import importlib

# ─── CuTe DSL wrapper (TVM-FFI compile cache) ───
_delta_h_mod = importlib.import_module("flashla.chunk_delta_h")
chunk_delta_rule_fwd_h = _delta_h_mod.chunk_delta_rule_fwd_h

# ─── FLA baseline imports ───
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h as fla_fwd_h


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
    """Compute max and mean absolute difference."""
    diff = (ref.float() - out.float()).abs()
    return diff.max().item(), diff.mean().item()


def make_chunk_offsets(seq_lens, BT, device="cuda"):
    """Create chunk_offsets from sequence lengths."""
    NTs = [(int(l) + BT - 1) // BT for l in seq_lens]
    co = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=device)
    for i, nt in enumerate(NTs):
        co[i + 1] = co[i] + nt
    return co, int(sum(NTs))


# ============================================================
# Non-varlen benchmark
# ============================================================
def bench_non_varlen(configs):
    print("\n" + "=" * 80)
    print(" Non-Varlen Benchmark: CuTe DSL (SM100a) vs FLA Triton")
    print("=" * 80)
    results = []

    for B, T, H, use_gk, use_h0, store_ht, save_vnew in configs:
        NT = T // BT
        torch.manual_seed(42)
        torch.cuda.empty_cache()

        k = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
        w = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
        u = torch.randn(B, T, H, V, device=device, dtype=dtype) * 0.1

        gk = None
        h0 = None
        if use_gk:
            gk = -torch.abs(torch.randn(B, T, H, K, device=device, dtype=torch.float32) * 0.1).cumsum(dim=1)
        if use_h0:
            h0 = torch.randn(B, H, K, V, device=device, dtype=torch.float32) * 0.01

        # ---- FLA baseline ----
        fla_result = fla_fwd_h(
            k=k, w=w, u=u,
            g=None, gk=gk,
            initial_state=h0,
            output_final_state=store_ht,
            chunk_size=BT,
            save_new_value=save_vnew,
        )
        h_fla = fla_result[0]  # h_out

        # ---- CuTe DSL ----
        g_tensor = torch.zeros(B, T, H, device=device, dtype=torch.float32)
        gk_tensor = gk if gk is not None else torch.zeros(B, T, H, K, device=device, dtype=torch.float32)
        h0_tensor = h0 if h0 is not None else torch.zeros(B, H, K, V, device=device, dtype=torch.float32)

        h_out = torch.zeros(B, NT, H, K, V, device=device, dtype=dtype)
        v_new_out = torch.zeros(B, T, H, V, device=device, dtype=dtype)
        ht_out = torch.zeros(B, H, K, V, device=device, dtype=torch.float32)

        chunk_delta_rule_fwd_h(
            k=k, w=w, u=u, g=g_tensor, gk=gk_tensor,
            h_out=h_out, v_new=v_new_out, h0=h0_tensor, ht=ht_out,
            use_g=0, use_gk=int(use_gk), use_initial_state=int(use_h0),
            store_final_state=int(store_ht), save_v_new=int(save_vnew),
            chunk_size=BT, is_varlen=False,
        )
        torch.cuda.synchronize()

        max_diff, mean_diff = accuracy_stats(h_fla, h_out)

        # ---- Performance timing ----
        def run_fla(k=k, w=w, u=u, gk=gk, h0=h0):
            fla_fwd_h(
                k=k, w=w, u=u, g=None, gk=gk,
                initial_state=h0, output_final_state=store_ht,
                chunk_size=BT, save_new_value=save_vnew,
            )

        def run_cute(k=k, w=w, u=u, g_tensor=g_tensor, gk_tensor=gk_tensor,
                     h_out=h_out, v_new_out=v_new_out, h0_tensor=h0_tensor,
                     ht_out=ht_out):
            chunk_delta_rule_fwd_h(
                k=k, w=w, u=u, g=g_tensor, gk=gk_tensor,
                h_out=h_out, v_new=v_new_out, h0=h0_tensor, ht=ht_out,
                use_g=0, use_gk=int(use_gk), use_initial_state=int(use_h0),
                store_final_state=int(store_ht), save_v_new=int(save_vnew),
                chunk_size=BT, is_varlen=False,
            )

        ms_fla = time_kernel(run_fla)
        ms_cute = time_kernel(run_cute)
        speedup = ms_fla / ms_cute if ms_cute > 0 else float('inf')

        flags = []
        if use_gk: flags.append("gk")
        if use_h0: flags.append("h0")
        if store_ht: flags.append("ht")
        if save_vnew: flags.append("vn")
        flag_str = f" [{','.join(flags)}]" if flags else ""

        r = {
            'B': B, 'T': T, 'H': H, 'flags': flag_str,
            'max_diff': max_diff, 'mean_diff': mean_diff,
            'ms_fla': ms_fla, 'ms_cute': ms_cute, 'speedup': speedup,
        }
        results.append(r)
        print(f"  B={B:2d} T={T:5d} H={H:3d}{flag_str:<16s} | "
              f"max_diff={max_diff:.6f} mean_diff={mean_diff:.8f} | "
              f"FLA={ms_fla:.4f}ms CuTe={ms_cute:.4f}ms | "
              f"speedup={speedup:.2f}x")

    return results


# ============================================================
# Varlen benchmark
# ============================================================
def generate_seq_lens(num_seqs, total_T, ratio, seed=42):
    """Generate variable-length sequences with given total and ratio."""
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


def bench_varlen(configs):
    print("\n" + "=" * 80)
    print(" Varlen Benchmark: CuTe DSL (SM100a) vs FLA Triton")
    print("=" * 80)
    results = []

    for num_seqs, total_T, H, ratio, use_gk, use_h0, store_ht, save_vnew in configs:
        seq_lens = generate_seq_lens(num_seqs, total_T, ratio)
        cu_seqlens_list = [0]
        for sl in seq_lens:
            cu_seqlens_list.append(cu_seqlens_list[-1] + sl)
        cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=device)
        chunk_offsets, total_NT = make_chunk_offsets(seq_lens, BT, device)

        torch.manual_seed(42)
        torch.cuda.empty_cache()

        # FLA uses [1, total_T, H, ...] (4D with B=1)
        k_4d = torch.randn(1, total_T, H, K, device=device, dtype=dtype) * 0.1
        w_4d = torch.randn(1, total_T, H, K, device=device, dtype=dtype) * 0.1
        u_4d = torch.randn(1, total_T, H, V, device=device, dtype=dtype) * 0.1

        gk = None
        gk_4d = None
        h0 = None
        if use_gk:
            gk_raw = torch.randn(1, total_T, H, K, device=device, dtype=torch.float32) * 0.1
            gk_4d = torch.zeros_like(gk_raw)
            for i in range(num_seqs):
                bos = cu_seqlens[i].item()
                eos = cu_seqlens[i + 1].item()
                gk_4d[:, bos:eos] = -torch.abs(gk_raw[:, bos:eos]).cumsum(dim=1)
            gk = gk_4d
        if use_h0:
            h0 = torch.randn(num_seqs, H, K, V, device=device, dtype=torch.float32) * 0.01

        # CuTe DSL uses [T_total, H, ...] (3D, flat)
        k_3d = k_4d.squeeze(0)
        w_3d = w_4d.squeeze(0)
        u_3d = u_4d.squeeze(0)

        # ---- FLA baseline ----
        cu_seqlens_long = cu_seqlens.long()
        fla_result = fla_fwd_h(
            k=k_4d, w=w_4d, u=u_4d, g=None, gk=gk_4d,
            initial_state=h0, output_final_state=store_ht,
            chunk_size=BT, save_new_value=save_vnew,
            cu_seqlens=cu_seqlens_long,
        )
        h_fla = fla_result[0]

        # ---- CuTe DSL varlen ----
        g_tensor = torch.zeros(total_T, H, device=device, dtype=torch.float32)
        gk_3d = gk_4d.squeeze(0) if gk_4d is not None else torch.zeros(total_T, H, K, device=device, dtype=torch.float32)
        h0_tensor = h0 if h0 is not None else torch.zeros(num_seqs, H, K, V, device=device, dtype=torch.float32)

        h_out = torch.zeros(total_NT, H, K, V, device=device, dtype=dtype)
        v_new_out = torch.zeros(total_T, H, V, device=device, dtype=dtype)
        ht_out = torch.zeros(num_seqs, H, K, V, device=device, dtype=torch.float32)
        workspace = torch.zeros(num_seqs * 128, dtype=torch.uint8, device=device)

        chunk_delta_rule_fwd_h(
            k=k_3d, w=w_3d, u=u_3d, g=g_tensor, gk=gk_3d,
            h_out=h_out, v_new=v_new_out, h0=h0_tensor, ht=ht_out,
            use_g=0, use_gk=int(use_gk), use_initial_state=int(use_h0),
            store_final_state=int(store_ht), save_v_new=int(save_vnew),
            chunk_size=BT, cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets,
            workspace=workspace, is_varlen=True, persistent=True,
        )
        torch.cuda.synchronize()

        # FLA returns [1, NT_total, H, K, V], CuTe DSL returns [NT_total, H, K, V]
        max_diff, mean_diff = accuracy_stats(h_fla.squeeze(0), h_out)

        # ---- Performance timing ----
        def run_fla(k=k_4d, w=w_4d, u=u_4d, gk=gk_4d, h0=h0, cu=cu_seqlens_long):
            fla_fwd_h(
                k=k, w=w, u=u, g=None, gk=gk,
                initial_state=h0, output_final_state=store_ht,
                chunk_size=BT, save_new_value=save_vnew,
                cu_seqlens=cu,
            )

        def run_cute(k=k_3d, w=w_3d, u=u_3d, g_tensor=g_tensor, gk_3d=gk_3d,
                     h_out=h_out, v_new_out=v_new_out, h0_tensor=h0_tensor,
                     ht_out=ht_out, cu_seqlens=cu_seqlens,
                     chunk_offsets=chunk_offsets, workspace=workspace):
            chunk_delta_rule_fwd_h(
                k=k, w=w, u=u, g=g_tensor, gk=gk_3d,
                h_out=h_out, v_new=v_new_out, h0=h0_tensor, ht=ht_out,
                use_g=0, use_gk=int(use_gk), use_initial_state=int(use_h0),
                store_final_state=int(store_ht), save_v_new=int(save_vnew),
                chunk_size=BT, cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets,
                workspace=workspace, is_varlen=True, persistent=True,
            )

        ms_fla = time_kernel(run_fla)
        ms_cute = time_kernel(run_cute)
        speedup = ms_fla / ms_cute if ms_cute > 0 else float('inf')

        min_l, max_l = min(seq_lens), max(seq_lens)
        avg_l = total_T // num_seqs
        actual_ratio = max_l / min_l
        tag = f"{num_seqs}seqs T={total_T} [{min_l}..{max_l}] avg={avg_l}"

        flags = []
        if use_gk: flags.append("gk")
        if use_h0: flags.append("h0")
        if store_ht: flags.append("ht")
        if save_vnew: flags.append("vn")
        flag_str = f" [{','.join(flags)}]" if flags else ""

        r = {
            'tag': tag, 'T_total': total_T, 'H': H, 'n_seqs': num_seqs,
            'flags': flag_str,
            'max_diff': max_diff, 'mean_diff': mean_diff,
            'ms_fla': ms_fla, 'ms_cute': ms_cute, 'speedup': speedup,
        }
        results.append(r)
        print(f"  {tag:40s} H={H:3d}{flag_str:<16s} | "
              f"max_diff={max_diff:.6f} mean_diff={mean_diff:.8f} | "
              f"FLA={ms_fla:.4f}ms CuTe={ms_cute:.4f}ms | "
              f"speedup={speedup:.2f}x")

    return results


# ============================================================
# Report table
# ============================================================
def print_report(nv_results, vl_results):
    sep = "=" * 110
    print(f"\n\n{sep}")
    print("                     BENCHMARK REPORT: chunk_delta_rule_fwd_h")
    print("                     CuTe DSL (Blackwell SM100a) vs FLA Triton")
    print(f"                     K={K}  V={V}  BT={BT}  dtype=bf16")
    wu = 1 if NCU_MODE else WARMUP
    ni = 1 if NCU_MODE else N_ITERS
    ncu_tag = "  [NCU mode]" if NCU_MODE else ""
    print(f"                     Warmup={wu}  Iters={ni}{ncu_tag}")
    print(sep)

    if nv_results:
        print("\n  [Non-Varlen]")
        print(f"  {'─' * 100}")
        print(f"  {'Config':<35s}  │  {'max_diff':>10s}  {'mean_diff':>12s}"
              f"  │  {'FLA(ms)':>9s}  {'CuTe(ms)':>9s}  {'Speedup':>8s}")
        print(f"  {'─' * 100}")
        for r in nv_results:
            label = f"B={r['B']:2d} T={r['T']:5d} H={r['H']:3d}{r['flags']}"
            print(f"  {label:<35s}  │  "
                  f"{r['max_diff']:10.6f}  {r['mean_diff']:12.8f}  │  "
                  f"{r['ms_fla']:9.4f}  {r['ms_cute']:9.4f}  {r['speedup']:7.2f}x")
        print(f"  {'─' * 100}")
        speedups = [r['speedup'] for r in nv_results]
        geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        print(f"  {'Geometric mean':<35s}  │  {'':>10s}  {'':>12s}  │  "
              f"{'':>9s}  {'':>9s}  {geo:7.2f}x")

    if vl_results:
        print("\n  [Varlen]")
        print(f"  {'─' * 115}")
        print(f"  {'Config':>55s}  │  {'max_diff':>10s}  {'mean_diff':>12s}"
              f"  │  {'FLA(ms)':>9s}  {'CuTe(ms)':>9s}  {'Speedup':>8s}")
        print(f"  {'─' * 115}")
        for r in vl_results:
            label = f"{r['tag']} H={r['H']:3d}{r['flags']}"
            print(f"  {label:>55s}  │  "
                  f"{r['max_diff']:10.6f}  {r['mean_diff']:12.8f}  │  "
                  f"{r['ms_fla']:9.4f}  {r['ms_cute']:9.4f}  {r['speedup']:7.2f}x")
        print(f"  {'─' * 115}")
        speedups = [r['speedup'] for r in vl_results]
        geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        print(f"  {'Geometric mean':>55s}  │  {'':>10s}  {'':>12s}  │  "
              f"{'':>9s}  {'':>9s}  {geo:7.2f}x")

    print(f"\n{sep}\n")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="bench_chunk_delta_h: CuTe DSL (SM100a) vs FLA Triton baseline"
    )
    parser.add_argument(
        "--mode", type=str, default="both",
        choices=["non-varlen", "varlen", "both"],
        help="Which benchmark mode to run (default: both)",
    )
    parser.add_argument(
        "--ncu", action="store_true",
        help="NCU profiling mode: warmup=1, iters=1",
    )
    args = parser.parse_args()

    global NCU_MODE
    if args.ncu:
        NCU_MODE = True
        print("[NCU mode] warmup=1, iters=1")

    # (B, T, H, use_gk, use_h0, store_ht, save_vnew)
    non_varlen_configs = [
        # Sweep B × H with all features (gk, h0, ht, vnew)
        (1,  8192,  64, True, True, True, True),
        (1,  8192, 128, True, True, True, True),
        (2,  8192,  64, True, True, True, True),
        (2,  8192, 128, True, True, True, True),
        (4,  8192,  64, True, True, True, True),
        (4,  8192, 128, True, True, True, True),
        (8,  8192,  64, True, True, True, True),
        (8,  8192, 128, True, True, True, True),
    ]

    # (num_seqs, total_T, H, ratio, use_gk, use_h0, store_ht, save_vnew)
    varlen_configs = [
        (20, 8192,  64, 2.0, True, True, True, True),
        (25, 8192,  64, 3.0, True, True, True, True),
        (20, 8192,  64, 4.0, True, True, True, True),
        (20, 32768, 64, 2.0, True, True, True, True),
        (25, 32768, 64, 3.0, True, True, True, True),
    ]

    nv_res, vl_res = [], []

    if args.mode in ("non-varlen", "both"):
        nv_res = bench_non_varlen(non_varlen_configs)

    if args.mode in ("varlen", "both"):
        vl_res = bench_varlen(varlen_configs)

    print_report(nv_res, vl_res)


if __name__ == "__main__":
    main()
