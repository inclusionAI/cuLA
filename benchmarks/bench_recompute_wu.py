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
bench_recompute_wu.py — Benchmark: cuLA vs FLA Triton for recompute_w_u

Supports both standard (HV=H) and GVA (HV > H) modes.
In GVA mode both FLA (v0.5.0+) and cuLA accept compact q/k in HQK space natively.

Usage:
  python bench_recompute_wu.py [--heads H] [--hv HV] [--disable_recompute]
"""

import argparse
import os
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))  # Enable fast ops in FLA for fair comparison

from fla.ops.kda.wy_fast import recompute_w_u_fwd as fla_recompute_w_u_fwd

import cula.cudac as cula_cuda
from benchmarks.utils import (
    SEED,
    exclusive_cumsum,
    generate_random_seq_lens,
    prepare_intra_inputs,
    relative_rms_error,
    relative_rms_error_rel_max_mean_abs_rhs,
    triton_bench_fn,
)
from cula.kda.chunk_intra import chunk_kda_fwd_intra as cula_chunk_kda_fwd_intra

# Constant params
B, H, D = 2, 64, 128
HV = H  # overridable via --hv; HV > H enables GVA mode
BT = 64  # chunk size

# Varlen benchmark params
NUM_SEQS = 8
TOTAL_LEN = 8192
MIN_SEQ_LEN = 63
VARIANCE = 1.0

DISABLE_RECOMPUTE = False  # Whether to disable recompute (compute QG in forward)


def get_abs_err(ref: torch.Tensor, out: torch.Tensor) -> float:
    return (ref.float() - out.float()).abs().max().item()


def prepare_recompute_wu_inputs(B, T, device, cu_seqlens=None, chunk_size=BT):
    """Prepare inputs for recompute_w_u benchmarking (handles both MHA and GVA).

    Uses cuLA's GVA-aware chunk_kda_fwd_intra to produce Akk in HV head space,
    which is valid for both MHA (HV=H) and GVA (HV>H) layouts.
    """
    q, k, v, g, beta, scale, cu_seqlens, chunk_indices = prepare_intra_inputs(
        B, T, H, D, device, cu_seqlens=cu_seqlens, chunk_size=chunk_size, num_v_heads=HV
    )

    _, _, _, _, _, Akk = cula_chunk_kda_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=True,
        disable_recompute=False,
    )

    return q, k, v, g, beta, Akk, cu_seqlens, chunk_indices


def run_fla_recompute_wu(k, v, beta, Akk, q, gk, cu_seqlens, chunk_indices, disable_recompute):
    """FLA recompute_w_u reference (handles both MHA and GVA natively as of v0.5.0)."""
    return fla_recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=Akk,
        q=q if disable_recompute else None,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )


def run_cula_recompute_wu(k, v, beta, Akk, q, gk, cu_seqlens, chunk_indices, chunk_size, disable_recompute):
    """cuLA recompute_w_u (handles both MHA and GVA; w/u/qg/kg allocated in HV head space)."""
    B_flat, T, HV_out, Dv = v.shape
    w = torch.empty_like(v)
    u = torch.empty_like(v)
    qg = torch.empty(B_flat, T, HV_out, Dv, device=q.device, dtype=q.dtype) if disable_recompute else None
    kg = torch.empty_like(v) if gk is not None else None

    cula_cuda.recompute_w_u_cuda(
        k, v, beta, Akk, gk, cu_seqlens, chunk_indices, w, u, kg, chunk_size, q if disable_recompute else None, qg
    )
    return w, u, qg, kg


# ==============================================================================
# Unified uniform seqlen benchmark (handles both standard and GVA)
# ==============================================================================
def benchmark_recompute_wu_uniform():
    device = torch.device("cuda")
    chunk_size = BT
    gva_mode = HV > H
    gva_note = f"HQK={H} HV={HV} (group_size={HV // H})" if gva_mode else f"H={H}"
    T_vals = [512, 1024, 4096, 8192, 16384, 32768]

    print("=" * 100)
    print(
        f"  Uniform-Length RecomputeWU Benchmark: cuLA vs FLA Triton  "
        f"B={B} {gva_note} D={D}  disable_recompute={DISABLE_RECOMPUTE}"
    )
    print("=" * 100)
    print(
        f"{'B':>4} {'T':>7} │ {'rel_rmse':>10} {'rel_max':>10} {'mean_diff':>12} │ {'FLA(ms)':>9} {'cuLA(ms)':>9} {'Speedup':>8}"
    )
    print("─" * 100)

    for T in T_vals:
        seq_lens = [T] * B
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

        q, k, v, g, beta, Akk, cu_seqlens, chunk_indices = prepare_recompute_wu_inputs(
            B, T, device, cu_seqlens=cu_seqlens, chunk_size=chunk_size
        )

        # Accuracy: run once and compare
        w_fla, u_fla, qg_fla, kg_fla = run_fla_recompute_wu(
            k, v, beta, Akk, q, g, cu_seqlens, chunk_indices, DISABLE_RECOMPUTE
        )
        w_cula, u_cula, qg_cula, kg_cula = run_cula_recompute_wu(
            k, v, beta, Akk, q, g, cu_seqlens, chunk_indices, chunk_size, DISABLE_RECOMPUTE
        )

        stats = {}
        for name, t_fla, t_cula in [
            ("w", w_fla, w_cula),
            ("u", u_fla, u_cula),
            ("qg", qg_fla, qg_cula),
            ("kg", kg_fla, kg_cula),
        ]:
            if t_fla is not None and t_cula is not None:
                stats[name] = relative_rms_error_rel_max_mean_abs_rhs(t_fla, t_cula)
        # Use max across all outputs for display.
        relative_rms_error_value = max(s[0] for s in stats.values())
        rel_max = max(s[1] for s in stats.values())
        mean_diff = max(s[2] for s in stats.values())

        # Performance
        ms_fla = triton_bench_fn(
            lambda: run_fla_recompute_wu(k, v, beta, Akk, q, g, cu_seqlens, chunk_indices, DISABLE_RECOMPUTE),
        )
        ms_cula = triton_bench_fn(
            lambda: run_cula_recompute_wu(k, v, beta, Akk, q, g, cu_seqlens, chunk_indices, chunk_size, DISABLE_RECOMPUTE),
        )
        speedup = ms_fla / ms_cula if ms_cula > 0 else float("inf")

        print(
            f"{B:>4} {T:>7} │ {relative_rms_error_value:>10.6f} {rel_max:>10.6f} {mean_diff:>12.8f} │ {ms_fla:>9.4f} {ms_cula:>9.4f} {speedup:>7.2f}x"
        )

    print("─" * 100)


# ==============================================================================
# Unified varlen benchmark (handles both standard and GVA)
# ==============================================================================
def benchmark_recompute_wu_varlen():
    device = torch.device("cuda")
    chunk_size = BT
    gva_mode = HV > H
    gva_note = f"HQK={H} HV={HV} (group_size={HV // H})" if gva_mode else f"H={H}"
    total_len_vals = [8192, 16384, 32768, 65536]

    print()
    print("=" * 110)
    print(
        f"  Varlen RecomputeWU Benchmark: cuLA vs FLA Triton  "
        f"NUM_SEQS={NUM_SEQS} {gva_note} D={D}  disable_recompute={DISABLE_RECOMPUTE}"
    )
    print("=" * 110)
    print(
        f"{'total_len':>10} │ {'rel_rmse':>10} {'rel_max':>10} {'mean_diff':>12} │ {'FLA(ms)':>9} {'cuLA(ms)':>9} {'Speedup':>8}"
    )
    print("─" * 110)

    for total_len in total_len_vals:
        seq_lens = generate_random_seq_lens(NUM_SEQS, total_len, MIN_SEQ_LEN, VARIANCE, SEED)
        T = total_len
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

        q, k, v, g, beta, Akk, cu_seqlens, chunk_indices = prepare_recompute_wu_inputs(
            1, T, device, cu_seqlens=cu_seqlens, chunk_size=chunk_size
        )

        # Accuracy
        w_fla, u_fla, qg_fla, kg_fla = run_fla_recompute_wu(
            k, v, beta, Akk, q, g, cu_seqlens, chunk_indices, DISABLE_RECOMPUTE
        )
        w_cula, u_cula, qg_cula, kg_cula = run_cula_recompute_wu(
            k, v, beta, Akk, q, g, cu_seqlens, chunk_indices, chunk_size, DISABLE_RECOMPUTE
        )

        stats = {}
        for name, t_fla, t_cula in [
            ("w", w_fla, w_cula),
            ("u", u_fla, u_cula),
            ("qg", qg_fla, qg_cula),
            ("kg", kg_fla, kg_cula),
        ]:
            if t_fla is not None and t_cula is not None:
                stats[name] = relative_rms_error_rel_max_mean_abs_rhs(t_fla, t_cula)
        # Use max across all outputs for display.
        relative_rms_error_value = max(s[0] for s in stats.values())
        rel_max = max(s[1] for s in stats.values())
        mean_diff = max(s[2] for s in stats.values())

        # Performance
        ms_fla = triton_bench_fn(
            lambda: run_fla_recompute_wu(k, v, beta, Akk, q, g, cu_seqlens, chunk_indices, DISABLE_RECOMPUTE),
        )
        ms_cula = triton_bench_fn(
            lambda: run_cula_recompute_wu(k, v, beta, Akk, q, g, cu_seqlens, chunk_indices, chunk_size, DISABLE_RECOMPUTE),
        )
        speedup = ms_fla / ms_cula if ms_cula > 0 else float("inf")

        print(
            f"{total_len:>10} │ {relative_rms_error_value:>10.6f} {rel_max:>10.6f} {mean_diff:>12.8f} │ {ms_fla:>9.4f} {ms_cula:>9.4f} {speedup:>7.2f}x"
        )

    print("─" * 110)


def check_determinism(num_seqs=NUM_SEQS, T=2001, H=H, iters=1000):
    """Run the recompute_w_u kernel multiple times and check for deterministic outputs."""
    device = torch.device("cuda")
    chunk_size = BT

    seq_lens = generate_random_seq_lens(num_seqs, T, MIN_SEQ_LEN, VARIANCE, SEED)
    cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

    q, k, v, g, beta, Akk, cu_seqlens, chunk_indices = prepare_recompute_wu_inputs(
        B=1, T=T, device=device, cu_seqlens=cu_seqlens, chunk_size=chunk_size
    )

    ref_w, ref_u, ref_qg, ref_kg = run_cula_recompute_wu(
        k, v, beta, Akk, q, g, cu_seqlens, chunk_indices, chunk_size, disable_recompute=DISABLE_RECOMPUTE
    )

    for i in range(iters):
        w, u, qg, kg = run_cula_recompute_wu(
            k, v, beta, Akk, q, g, cu_seqlens, chunk_indices, chunk_size, disable_recompute=DISABLE_RECOMPUTE
        )

        if not torch.equal(w, ref_w):
            print(f"Iteration {i}: w mismatch")
            print(f"{get_abs_err(ref_w, w):.6f} absolute error, {relative_rms_error(ref_w, w):.6f} relative_rms_error")
            raise AssertionError("Non-deterministic output detected in w")
        if not torch.equal(u, ref_u):
            print(f"Iteration {i}: u mismatch")
            print(f"{get_abs_err(ref_u, u):.6f} absolute error, {relative_rms_error(ref_u, u):.6f} relative_rms_error")
            raise AssertionError("Non-deterministic output detected in u")
        if kg is not None and not torch.equal(kg, ref_kg):
            print(f"Iteration {i}: kg mismatch")
            print(f"{get_abs_err(ref_kg, kg):.6f} absolute error, {relative_rms_error(ref_kg, kg):.6f} relative_rms_error")
            raise AssertionError("Non-deterministic output detected in kg")
        if qg is not None and not torch.equal(qg, ref_qg):
            print(f"Iteration {i}: qg mismatch")
            print(f"{get_abs_err(ref_qg, qg):.6f} absolute error, {relative_rms_error(ref_qg, qg):.6f} relative_rms_error")
            raise AssertionError("Non-deterministic output detected in qg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="bench_recompute_wu: cuLA vs FLA Triton for recompute_w_u")
    parser.add_argument(
        "--disable_recompute",
        action="store_true",
        help="Disable recompute in both FLA and cuLA (pre-compute QG)",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=None,
        help=f"Override number of Q/K heads (H). Default: {H}.",
    )
    parser.add_argument(
        "--hv",
        type=int,
        default=None,
        help="Override number of V heads (HV). Default: H (no GVA). Set HV > H to enable GVA mode.",
    )
    args = parser.parse_args()

    if args.disable_recompute:
        DISABLE_RECOMPUTE = True
        print("[Disable recompute] pre-compute QG in forward")

    if args.heads is not None:
        if args.heads <= 0:
            raise ValueError(f"--heads must be a positive integer, got {args.heads}")
        H = args.heads
        HV = H  # reset HV to new H before --hv override

    if args.hv is not None:
        if args.hv < H or args.hv % H != 0:
            raise ValueError(f"--hv must be a positive multiple of H ({H}), got {args.hv}")
        HV = args.hv

    if HV > H:
        print(f"[GVA] HV={HV} (H={H}, group_size={HV // H}x)")

    check_determinism(iters=100000)

    benchmark_recompute_wu_uniform()
    benchmark_recompute_wu_varlen()
