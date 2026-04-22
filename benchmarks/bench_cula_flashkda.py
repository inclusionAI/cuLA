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
bench_cula_flashkda.py — Benchmark: cuLA fully-fused KDA forward vs FlashKDA

Automatically selects the cuLA fully-fused implementation based on the current
GPU architecture:
  - sm100 (Blackwell) → cula.kda.blackwell_fused_fwd.flash_kda_prefill
  - sm90  (Hopper)    → cula.kda.hopper_fused_fwd.cula_kda_prefill

Compares:
  - Accuracy: err_ratio (FLA convention), relative max diff between cuLA fully-fused and FlashKDA
  - Performance: kernel execution time (ms) with CUDA events

Input dtype notes:
  - q, k, v, g : bfloat16  (q/k are L2-normalised before input)
  - beta        : bfloat16  (required)
  - A_log, dt_bias, scale : float32
  - initial_state for cuLA : float32
  - initial_state for FlashKDA : float32

Modes:
  - Fixed-length: various (B, T) configs
  - Varlen: sequences with length variation

Usage:
  python bench_cula_flashkda.py [--mode fixed|varlen|both] [--ncu]

With --ncu, warmup=1 and iters=1 for ncu profiling:
  ncu --set full -o report python bench_cula_flashkda.py --mode varlen --ncu
"""

import argparse
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import flash_kda
import torch
import torch.nn.functional as F
from fla.utils import get_err_ratio

from benchmarks.utils import (
    SEED,
    build_varlen_configs,
    exclusive_cumsum,
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
LOWER_BOUND = -5.0


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
    """Compute err_ratio (FLA convention), relative max diff, and mean absolute difference."""
    ref_f = ref.float()
    out_f = out.float()
    diff = (ref_f - out_f).abs()
    err_ratio = get_err_ratio(ref_f, out_f)
    max_diff = diff.max().item()
    denom = ref_f.abs().max().item()
    rel_max = max_diff / denom if denom > 0 else 0.0
    mean_diff = diff.mean().item()
    return err_ratio, rel_max, mean_diff


# ============================================================
# Input preparation
# ============================================================
def prepare_inputs(B, T, H, D, device, cu_seqlens, has_init_state):
    """Prepare inputs with beta in bfloat16 for both cuLA and FlashKDA.

    All sequence data is packed as (1, B*T, H, D) when cu_seqlens is provided.
    Two variants of the initial state are produced:
      - init_state_bf16 : bfloat16 (kept for reference, not used)
      - init_state_fp32 : for both cuLA and FlashKDA (float32)
    """
    N = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else B
    total_T = B * T  # tokens already flattened into dim-1 position

    scale = float(D) ** (-0.5)

    # ---- main tensors (bfloat16) ----
    q = F.normalize(
        torch.randn(1, total_T, H, D, dtype=torch.bfloat16, device=device), p=2, dim=-1
    )
    k = F.normalize(
        torch.randn(1, total_T, H, D, dtype=torch.bfloat16, device=device), p=2, dim=-1
    )
    v = torch.randn(1, total_T, H, D, dtype=torch.bfloat16, device=device)
    g = torch.randn(1, total_T, H, D, dtype=torch.bfloat16, device=device)
    # beta: raw bfloat16 logits — FlashKDA applies sigmoid internally
    beta = torch.randn(1, total_T, H, dtype=torch.bfloat16, device=device)
    # cuLA and FLA expect post-sigmoid beta; upstream hasn't implemented use_beta_sigmoid_in_kernel
    beta_activated = torch.sigmoid(beta.float()).to(torch.bfloat16)

    # ---- gate parameters (float32) ----
    A_log = torch.rand(H, dtype=torch.float32, device=device)
    dt_bias = torch.rand(H, D, dtype=torch.float32, device=device)

    # ---- initial states ----
    init_state_bf16 = None
    init_state_fp32 = None
    if has_init_state:
        init_state_bf16 = torch.randn(N, H, D, D, dtype=torch.bfloat16, device=device)
        init_state_fp32 = init_state_bf16.float()

    # ---- pre-allocated output buffers for FlashKDA (in-place write) ----
    out_buf = torch.zeros(1, total_T, H, D, dtype=torch.bfloat16, device=device)
    final_state_buf = torch.zeros(N, H, D, D, dtype=torch.float32, device=device)

    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        beta_activated=beta_activated,
        scale=scale,
        A_log=A_log,
        dt_bias=dt_bias,
        init_state_bf16=init_state_bf16,
        init_state_fp32=init_state_fp32,
        out_buf=out_buf,
        final_state_buf=final_state_buf,
        lower_bound=LOWER_BOUND,
        N=N,
    )


# ============================================================
# Kernel wrappers
# ============================================================
def run_flash_kda(inp, cu_seqlens_long):
    """Run FlashKDA forward pass. Output is written into inp['out_buf'] in-place."""
    extra = {} if cu_seqlens_long is None else {"cu_seqlens": cu_seqlens_long}
    kwargs = dict(
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        lower_bound=inp["lower_bound"],
        **extra,
    )
    flash_kda.fwd(
        inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"],  # raw pre-sigmoid beta
        inp["scale"], inp["out_buf"],
        initial_state=inp["init_state_fp32"],  # None is accepted (no initial state)
        final_state=inp["final_state_buf"],    # always capture final state
        **kwargs,
    )
    return inp["out_buf"]


def run_cula(inp, cu_seqlens_i32):
    """Run cuLA fully-fused forward pass."""
    return cula_kda_fused_fwd(
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        g=inp["g"],
        beta=inp["beta_activated"],  # post-sigmoid beta
        scale=inp["scale"],
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        initial_state=inp["init_state_fp32"],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens_i32,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=inp["lower_bound"],
    )


# ============================================================
# Fixed-length benchmark
# ============================================================
def bench_fixed(configs):
    print("\n" + "=" * 100)
    print(f" Fixed-Length Benchmark: cuLA fully-fused ({_SM_TAG}) vs FlashKDA")
    print("=" * 100)
    results = []

    for B, T in configs:
        set_seed(SEED)
        device = torch.device("cuda")
        torch.cuda.empty_cache()

        seq_lens = [T] * B
        # cuLA uses int32 cu_seqlens; FlashKDA uses int64 (long)
        cu_seqlens_i32 = torch.tensor(
            exclusive_cumsum(seq_lens), dtype=torch.int32, device=device
        )
        cu_seqlens_long = cu_seqlens_i32.long()

        inp = prepare_inputs(B, T, H, D, device, cu_seqlens_i32, HAS_INIT_STATE)

        # Accuracy
        o_flash = run_flash_kda(inp, cu_seqlens_long).clone()
        o_cula, _ = run_cula(inp, cu_seqlens_i32)
        torch.cuda.synchronize()

        err_ratio, rel_max, mean_diff = accuracy_stats(o_flash, o_cula)

        # Performance
        ms_flash = time_kernel(lambda: run_flash_kda(inp, cu_seqlens_long))
        ms_cula  = time_kernel(lambda: run_cula(inp, cu_seqlens_i32))
        speedup = ms_flash / ms_cula if ms_cula > 0 else float("inf")

        results.append(
            {
                "B": B,
                "T": T,
                "err_ratio": err_ratio,
                "rel_max": rel_max,
                "mean_diff": mean_diff,
                "ms_flash": ms_flash,
                "ms_cula": ms_cula,
                "speedup": speedup,
            }
        )

        del o_flash, o_cula, inp
        torch.cuda.empty_cache()

    return results


# ============================================================
# Varlen benchmark
# ============================================================
def bench_varlen(configs):
    print("\n" + "=" * 100)
    print(f" Varlen Benchmark: cuLA fully-fused ({_SM_TAG}) vs FlashKDA")
    print("=" * 100)
    results = []

    for seq_lens, total_len, dist in configs:
        set_seed(SEED)
        device = torch.device("cuda")
        torch.cuda.empty_cache()

        T = total_len
        # cuLA uses int32 cu_seqlens; FlashKDA uses int64 (long)
        cu_seqlens_i32 = torch.tensor(
            exclusive_cumsum(seq_lens), dtype=torch.int32, device=device
        )
        cu_seqlens_long = cu_seqlens_i32.long()

        # For varlen, B=1 and all sequences are packed into dim-1
        inp = prepare_inputs(1, T, H, D, device, cu_seqlens_i32, HAS_INIT_STATE)

        # Accuracy
        o_flash = run_flash_kda(inp, cu_seqlens_long).clone()
        o_cula, _ = run_cula(inp, cu_seqlens_i32)
        torch.cuda.synchronize()

        err_ratio, rel_max, mean_diff = accuracy_stats(o_flash, o_cula)

        # Performance
        ms_flash = time_kernel(lambda: run_flash_kda(inp, cu_seqlens_long))
        ms_cula  = time_kernel(lambda: run_cula(inp, cu_seqlens_i32))
        speedup = ms_flash / ms_cula if ms_cula > 0 else float("inf")

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
                "err_ratio": err_ratio,
                "rel_max": rel_max,
                "mean_diff": mean_diff,
                "ms_flash": ms_flash,
                "ms_cula": ms_cula,
                "speedup": speedup,
            }
        )

        del o_flash, o_cula, inp
        torch.cuda.empty_cache()

    return results


# ============================================================
# Report
# ============================================================
def print_report(fixed_results, varlen_results):
    sep = "=" * 115
    print(f"\n\n{sep}")
    print("                  BENCHMARK REPORT: cula_kda_fused_fwd (fully-fused) vs FlashKDA")
    print(f"                  cuLA {_SM_TAG} fully-fused vs FlashKDA")
    print(f"                  H={H}  D={D}  dtype=bf16  beta=bf16  safe_gate=True  has_init_state={HAS_INIT_STATE}")
    wu = 1 if (NCU_MODE or SANITIZER_MODE) else WARMUP
    ni = 1 if (NCU_MODE or SANITIZER_MODE) else N_ITERS
    mode_tag = "  [NCU mode]" if NCU_MODE else ("  [Sanitizer mode]" if SANITIZER_MODE else "")
    print(f"                  Warmup={wu}  Iters={ni}{mode_tag}")
    print(sep)

    if fixed_results:
        print("\n  [Fixed-Length]")
        print(f"  {'─' * 95}")
        print(
            f"  {'B':>3s}  {'T':>6s}  │  {'err_ratio':>10s}  {'rel_max':>10s}  {'mean_diff':>10s}"
            f"  │  {'FlashKDA(ms)':>13s}  {'cuLA(ms)':>10s}  {'Speedup':>8s}"
        )
        print(f"  {'─' * 95}")
        for r in fixed_results:
            print(
                f"  {r['B']:3d}  {r['T']:6d}  │  "
                f"{r['err_ratio']:10.6f}  {r['rel_max']:10.6f}  {r['mean_diff']:10.6f}  │  "
                f"{r['ms_flash']:13.4f}  {r['ms_cula']:10.4f}  {r['speedup']:7.2f}x"
            )
        print(f"  {'─' * 95}")

    if varlen_results:
        print("\n  [Varlen]")
        print(f"  {'─' * 110}")
        print(
            f"  {'Config':>45s}  │  {'err_ratio':>10s}  {'rel_max':>10s}  {'mean_diff':>10s}"
            f"  │  {'FlashKDA(ms)':>13s}  {'cuLA(ms)':>10s}  {'Speedup':>8s}"
        )
        print(f"  {'─' * 110}")
        for r in varlen_results:
            print(
                f"  {r['tag']:>45s}  │  "
                f"{r['err_ratio']:10.6f}  {r['rel_max']:10.6f}  {r['mean_diff']:10.6f}  │  "
                f"{r['ms_flash']:13.4f}  {r['ms_cula']:10.4f}  {r['speedup']:7.2f}x"
            )
        print(f"  {'─' * 110}")

    print(f"\n{sep}\n")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="bench_cula_flashkda: cuLA fully-fused KDA forward vs FlashKDA"
    )
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
    parser.add_argument("--H", type=int, default=64, help="Number of heads")
    parser.add_argument("--D", type=int, default=128, help="Head dimension")
    args = parser.parse_args()

    global NCU_MODE, SANITIZER_MODE, HAS_INIT_STATE, H, D
    if args.ncu:
        NCU_MODE = True
        print("[NCU mode] warmup=1, iters=1")
    if args.sanitizer:
        SANITIZER_MODE = True
        print("[Sanitizer mode] warmup=1, iters=1")
    if args.init_state:
        HAS_INIT_STATE = True
        print("[init_state] using non-zero initial state")
    H = args.H
    D = args.D

    print(
        f"[Device] {torch.cuda.get_device_name(0)}  compute capability {_SM_TAG}"
        f"  →  cuLA: {cula_kda_fused_fwd.__module__}.{cula_kda_fused_fwd.__name__}"
        f"  |  FlashKDA: flash_kda.fwd"
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
        num_seqs_list=(5, 10, 20),
        total_lens=(4096, 8192, 16384, 32768),
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
