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
bench_kda_fused_fwd_gqa.py — Benchmark: cuLA fully-fused KDA forward in the
"multi-value" GQA flavor (Q and K share fewer physical heads than V/state).

This benchmark targets the multi-value grouped variant used by e.g.
Qwen3.5-A3B Gated DeltaNet (linear_num_key_heads=16, linear_num_value_heads=32).
Three paths are compared, all producing the same output:

  1. cuLA GQA native    — Q: [L,num_k_heads,D], V: [L,num_heads,D] passed
                          straight through. Kernel indexes each state head's
                          Q/K via q_head_idx / k_group_size.
  2. cuLA MHA expanded  — Host-side repeat_interleave on Q/K to match
                          num_heads, then call the existing MHA path of the
                          same fused kernel. This is the workaround users
                          have today without native GQA support.
  3. FLA Triton expanded — Upstream flash-linear-attention's chunk_kda is
                           MHA-only, so the same host-side Q/K expansion is
                           required to run Qwen3.5 shapes through it.

Modes:
  - Fixed-length: various (B, T) configs
  - Varlen: sequences with 2-3x length variation

Usage:
  python bench_kda_fused_fwd_gqa.py [--mode fixed|varlen|both] [--ncu]
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
    prepare_safe_gate_inputs_gqa,
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
# Constants — Qwen3.5-A3B Gated DeltaNet shape by default
# ============================================================
H = 32        # linear_num_value_heads (= state head count, grid dim)
HK = 16       # linear_num_key_heads   (Q and K share this)
D = 128       # head_k_dim == head_v_dim
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


def run_cula_gqa(q, k, v, g, beta, scale, A_log, dt_bias, init_state, cu_seqlens, lower_bound):
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


def run_cula_mha_expanded(q, k, v, g, beta, scale, A_log, dt_bias, init_state, cu_seqlens, lower_bound, k_group_size):
    # Real user-facing cost: callers arrive with GQA-shaped q/k and must
    # materialise an MHA-shaped copy before hitting a MHA-only kernel. Time
    # the expansion together with the kernel so the "workaround" path shows
    # its true end-to-end cost.
    q_exp = q.repeat_interleave(k_group_size, dim=-2).contiguous()
    k_exp = k.repeat_interleave(k_group_size, dim=-2).contiguous()
    return cula_kda_fused_fwd(
        q=q_exp,
        k=k_exp,
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


def run_fla_expanded(q, k, v, g, beta, scale, A_log, dt_bias, init_state, cu_seqlens, lower_bound, k_group_size):
    # Upstream FLA's chunk_kda requires q/k/v to share a head count, so the
    # same host-side expansion is on the critical path here.
    q_exp = q.repeat_interleave(k_group_size, dim=-2).contiguous()
    k_exp = k.repeat_interleave(k_group_size, dim=-2).contiguous()
    return fla_chunk_kda(
        q=q_exp,
        k=k_exp,
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


# ============================================================
# Fixed-length benchmark
# ============================================================
def bench_fixed(configs):
    print("\n" + "=" * 120)
    print(f" Fixed-Length GQA Benchmark: cuLA fully-fused ({_SM_TAG}) native GQA vs MHA-expanded paths")
    print("=" * 120)
    results = []

    k_group_size = H // HK
    for B, T in configs:
        set_seed(SEED)
        device = torch.device("cuda")
        torch.cuda.empty_cache()

        seq_lens = [T] * B
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

        inputs = prepare_safe_gate_inputs_gqa(
            B, T, H, HK, D, device, cu_seqlens=cu_seqlens, has_init_state=HAS_INIT_STATE,
        )
        q, k, v, g, beta = inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"]
        A_log, dt_bias = inputs["A_log"], inputs["dt_bias"]
        scale, init_state, lower_bound = inputs["scale"], inputs["init_state"], inputs["lower_bound"]

        gqa_kwargs = dict(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale,
            A_log=A_log, dt_bias=dt_bias, init_state=init_state,
            cu_seqlens=cu_seqlens, lower_bound=lower_bound,
        )
        expanded_kwargs = dict(
            **gqa_kwargs,
            k_group_size=k_group_size,
        )

        # Accuracy — all three should produce the same output up to dtype rounding.
        # The MHA/FLA runners materialise the host-side expansion inside the
        # call, so the timed measurement reflects what an end-user pays when
        # they have GQA-shaped tensors and hit a MHA-only kernel.
        o_gqa, _ = run_cula_gqa(**gqa_kwargs)
        o_mha, _ = run_cula_mha_expanded(**expanded_kwargs)
        o_fla, _ = run_fla_expanded(**expanded_kwargs)
        torch.cuda.synchronize()

        rmse_mha, rel_mha, _ = accuracy_stats(o_mha, o_gqa)
        rmse_fla, rel_fla, _ = accuracy_stats(o_fla, o_gqa)

        # Performance
        ms_gqa = time_kernel(lambda: run_cula_gqa(**gqa_kwargs))
        ms_mha = time_kernel(lambda: run_cula_mha_expanded(**expanded_kwargs))
        ms_fla = time_kernel(lambda: run_fla_expanded(**expanded_kwargs))
        sp_vs_mha = ms_mha / ms_gqa if ms_gqa > 0 else float("inf")
        sp_vs_fla = ms_fla / ms_gqa if ms_gqa > 0 else float("inf")

        results.append(
            {
                "B": B,
                "T": T,
                "rmse_mha": rmse_mha,
                "rel_mha": rel_mha,
                "rmse_fla": rmse_fla,
                "rel_fla": rel_fla,
                "ms_gqa": ms_gqa,
                "ms_mha": ms_mha,
                "ms_fla": ms_fla,
                "sp_vs_mha": sp_vs_mha,
                "sp_vs_fla": sp_vs_fla,
            }
        )

        del o_gqa, o_mha, o_fla, q, k, v, g, beta, A_log, dt_bias, inputs
        torch.cuda.empty_cache()

    return results


# ============================================================
# Varlen benchmark
# ============================================================
def bench_varlen(configs):
    print("\n" + "=" * 120)
    print(f" Varlen GQA Benchmark: cuLA fully-fused ({_SM_TAG}) native GQA vs MHA-expanded paths")
    print("=" * 120)
    results = []

    k_group_size = H // HK
    for seq_lens, total_len, dist in configs:
        set_seed(SEED)
        device = torch.device("cuda")
        torch.cuda.empty_cache()

        T = total_len
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

        inputs = prepare_safe_gate_inputs_gqa(
            1, T, H, HK, D, device, cu_seqlens=cu_seqlens, has_init_state=HAS_INIT_STATE,
        )
        q, k, v, g, beta = inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"]
        A_log, dt_bias = inputs["A_log"], inputs["dt_bias"]
        scale, init_state, lower_bound = inputs["scale"], inputs["init_state"], inputs["lower_bound"]

        gqa_kwargs = dict(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale,
            A_log=A_log, dt_bias=dt_bias, init_state=init_state,
            cu_seqlens=cu_seqlens, lower_bound=lower_bound,
        )
        expanded_kwargs = dict(
            **gqa_kwargs,
            k_group_size=k_group_size,
        )

        # Accuracy
        o_gqa, _ = run_cula_gqa(**gqa_kwargs)
        o_mha, _ = run_cula_mha_expanded(**expanded_kwargs)
        o_fla, _ = run_fla_expanded(**expanded_kwargs)
        torch.cuda.synchronize()

        rmse_mha, rel_mha, _ = accuracy_stats(o_mha, o_gqa)
        rmse_fla, rel_fla, _ = accuracy_stats(o_fla, o_gqa)

        # Performance
        ms_gqa = time_kernel(lambda: run_cula_gqa(**gqa_kwargs))
        ms_mha = time_kernel(lambda: run_cula_mha_expanded(**expanded_kwargs))
        ms_fla = time_kernel(lambda: run_fla_expanded(**expanded_kwargs))
        sp_vs_mha = ms_mha / ms_gqa if ms_gqa > 0 else float("inf")
        sp_vs_fla = ms_fla / ms_gqa if ms_gqa > 0 else float("inf")

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
                "rmse_mha": rmse_mha,
                "rel_mha": rel_mha,
                "rmse_fla": rmse_fla,
                "rel_fla": rel_fla,
                "ms_gqa": ms_gqa,
                "ms_mha": ms_mha,
                "ms_fla": ms_fla,
                "sp_vs_mha": sp_vs_mha,
                "sp_vs_fla": sp_vs_fla,
            }
        )

        del o_gqa, o_mha, o_fla, q, k, v, g, beta, A_log, dt_bias, inputs
        torch.cuda.empty_cache()

    return results


# ============================================================
# Report
# ============================================================
def print_report(fixed_results, varlen_results):
    sep = "=" * 120
    print(f"\n\n{sep}")
    print("                 BENCHMARK REPORT: cula_kda_fused_fwd multi-value GQA")
    print(f"                 cuLA {_SM_TAG} native GQA  vs  cuLA MHA + host expand  vs  FLA Triton + host expand")
    print(f"                 H={H}  HK={HK}  D={D}  dtype=bf16  safe_gate=True  has_init_state={HAS_INIT_STATE}")
    wu = 1 if (NCU_MODE or SANITIZER_MODE) else WARMUP
    ni = 1 if (NCU_MODE or SANITIZER_MODE) else N_ITERS
    mode_tag = "  [NCU mode]" if NCU_MODE else ("  [Sanitizer mode]" if SANITIZER_MODE else "")
    print(f"                 Warmup={wu}  Iters={ni}{mode_tag}")
    print(sep)

    if fixed_results:
        print("\n  [Fixed-Length]")
        print(f"  {'─' * 115}")
        print(
            f"  {'B':>3s}  {'T':>6s}  │  "
            f"{'RMSE(mha)':>10s}  {'rel(mha)':>10s}  {'RMSE(fla)':>10s}  {'rel(fla)':>10s}  │  "
            f"{'GQA(ms)':>9s}  {'MHA(ms)':>9s}  {'FLA(ms)':>9s}  │  "
            f"{'GQA↑MHA':>8s}  {'GQA↑FLA':>8s}"
        )
        print(f"  {'─' * 115}")
        for r in fixed_results:
            print(
                f"  {r['B']:3d}  {r['T']:6d}  │  "
                f"{r['rmse_mha']:10.6f}  {r['rel_mha']:10.6f}  {r['rmse_fla']:10.6f}  {r['rel_fla']:10.6f}  │  "
                f"{r['ms_gqa']:9.4f}  {r['ms_mha']:9.4f}  {r['ms_fla']:9.4f}  │  "
                f"{r['sp_vs_mha']:7.2f}x  {r['sp_vs_fla']:7.2f}x"
            )
        print(f"  {'─' * 115}")

    if varlen_results:
        print("\n  [Varlen]")
        print(f"  {'─' * 130}")
        print(
            f"  {'Config':>45s}  │  "
            f"{'RMSE(mha)':>10s}  {'rel(mha)':>10s}  {'RMSE(fla)':>10s}  {'rel(fla)':>10s}  │  "
            f"{'GQA(ms)':>9s}  {'MHA(ms)':>9s}  {'FLA(ms)':>9s}  │  "
            f"{'GQA↑MHA':>8s}  {'GQA↑FLA':>8s}"
        )
        print(f"  {'─' * 130}")
        for r in varlen_results:
            print(
                f"  {r['tag']:>45s}  │  "
                f"{r['rmse_mha']:10.6f}  {r['rel_mha']:10.6f}  {r['rmse_fla']:10.6f}  {r['rel_fla']:10.6f}  │  "
                f"{r['ms_gqa']:9.4f}  {r['ms_mha']:9.4f}  {r['ms_fla']:9.4f}  │  "
                f"{r['sp_vs_mha']:7.2f}x  {r['sp_vs_fla']:7.2f}x"
            )
        print(f"  {'─' * 130}")

    print(f"\n{sep}\n")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="bench_kda_fused_fwd_gqa: multi-value GQA KDA forward benchmark")
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
        (2, 1024),
        (2, 4096),
        (2, 8192),
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
