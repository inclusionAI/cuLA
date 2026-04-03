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

import argparse
import os
import pathlib
import sys

import torch
import triton

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))  # Enable fast ops in FLA for fair comparison

from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra as fla_chunk_kda_fwd_intra

from benchmarks.utils import SEED, exclusive_cumsum, generate_random_seq_lens, prepare_intra_inputs
from cula.kda.chunk_intra import chunk_kda_fwd_intra as cula_chunk_kda_fwd_intra

# Constant params
B, H, D = 2, 64, 128
BT = 64  # chunk size

# Varlen benchmark params
NUM_SEQS = 8
TOTAL_LEN = 8192
MIN_SEQ_LEN = 63
VARIANCE = 1.0

DISABLE_RECOMPUTE = False  # Whether to disable recompute (compute QG in forward)


def accuracy_stats(a, b):
    """Compute RMSE, relative max diff, and mean absolute difference."""
    a, b = a.float(), b.float()
    diff = a - b
    rmse = diff.pow(2).mean().sqrt().item()
    max_diff = diff.abs().max().item()
    denom = b.abs().max().item()
    rel_max = max_diff / denom if denom > 0 else 0.0
    mean_diff = diff.abs().mean().item()
    return rmse, rel_max, mean_diff


# ==============================================================================
# Uniform seqlen benchmark
# ==============================================================================
def benchmark_chunk_intra_uniform():
    device = torch.device("cuda")
    chunk_size = BT
    T_vals = [512, 1024, 4096, 8192, 16384, 32768]

    print("=" * 90)
    print(
        f"  Uniform-Length ChunkIntra Benchmark: cuLA vs FLA Triton  B={B} H={H} D={D}  disable_recompute={DISABLE_RECOMPUTE}"
    )
    print("=" * 90)
    print(
        f"{'B':>4} {'T':>7} │ {'RMSE':>10} {'rel_max':>10} {'mean_diff':>12} │ {'FLA(ms)':>9} {'cuLA(ms)':>9} {'Speedup':>8}"
    )
    print("─" * 90)

    for T in T_vals:
        seq_lens = [T] * B
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

        q, k, v, g, beta, scale, cu_seqlens, chunk_indices = prepare_intra_inputs(B, T, H, D, device, cu_seqlens=cu_seqlens)

        # Accuracy: run once and compare
        out_fla = fla_chunk_kda_fwd_intra(
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
            disable_recompute=DISABLE_RECOMPUTE,
        )
        out_cula = cula_chunk_kda_fwd_intra(
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
            disable_recompute=DISABLE_RECOMPUTE,
        )
        # Compare the first output tensor (o)
        o_fla = out_fla[0] if isinstance(out_fla, (tuple, list)) else out_fla
        o_cula = out_cula[0] if isinstance(out_cula, (tuple, list)) else out_cula
        rmse, rel_max, mean_diff = accuracy_stats(o_fla, o_cula)

        # Performance
        ms_fla = triton.testing.do_bench(
            lambda: fla_chunk_kda_fwd_intra(
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
                disable_recompute=DISABLE_RECOMPUTE,
            ),
        )
        ms_cula = triton.testing.do_bench(
            lambda: cula_chunk_kda_fwd_intra(
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
                disable_recompute=DISABLE_RECOMPUTE,
            ),
        )
        speedup = ms_fla / ms_cula if ms_cula > 0 else float("inf")

        print(
            f"{B:>4} {T:>7} │ {rmse:>10.6f} {rel_max:>10.6f} {mean_diff:>12.8f} │ {ms_fla:>9.4f} {ms_cula:>9.4f} {speedup:>7.2f}x"
        )

    print("─" * 90)


# ==============================================================================
# Varlen benchmark
# ==============================================================================
def benchmark_chunk_intra_varlen():
    device = torch.device("cuda")
    chunk_size = BT
    total_len_vals = [8192, 16384, 32768, 65536]

    print()
    print("=" * 100)
    print(
        f"  Varlen ChunkIntra Benchmark: cuLA vs FLA Triton  NUM_SEQS={NUM_SEQS} H={H} D={D}  disable_recompute={DISABLE_RECOMPUTE}"
    )
    print("=" * 100)
    print(
        f"{'total_len':>10} │ {'RMSE':>10} {'rel_max':>10} {'mean_diff':>12} │ {'FLA(ms)':>9} {'cuLA(ms)':>9} {'Speedup':>8}"
    )
    print("─" * 100)

    for total_len in total_len_vals:
        seq_lens = generate_random_seq_lens(NUM_SEQS, total_len, MIN_SEQ_LEN, VARIANCE, SEED)
        T = total_len
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)

        q, k, v, g, beta, scale, cu_seqlens, chunk_indices = prepare_intra_inputs(1, T, H, D, device, cu_seqlens=cu_seqlens)

        # Accuracy
        out_fla = fla_chunk_kda_fwd_intra(
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
            disable_recompute=DISABLE_RECOMPUTE,
        )
        out_cula = cula_chunk_kda_fwd_intra(
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
            disable_recompute=DISABLE_RECOMPUTE,
        )
        o_fla = out_fla[0] if isinstance(out_fla, (tuple, list)) else out_fla
        o_cula = out_cula[0] if isinstance(out_cula, (tuple, list)) else out_cula
        rmse, rel_max, mean_diff = accuracy_stats(o_fla, o_cula)

        # Performance
        ms_fla = triton.testing.do_bench(
            lambda: fla_chunk_kda_fwd_intra(
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
                disable_recompute=DISABLE_RECOMPUTE,
            ),
        )
        ms_cula = triton.testing.do_bench(
            lambda: cula_chunk_kda_fwd_intra(
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
                disable_recompute=DISABLE_RECOMPUTE,
            ),
        )
        speedup = ms_fla / ms_cula if ms_cula > 0 else float("inf")

        print(
            f"{total_len:>10} │ {rmse:>10.6f} {rel_max:>10.6f} {mean_diff:>12.8f} │ {ms_fla:>9.4f} {ms_cula:>9.4f} {speedup:>7.2f}x"
        )

    print("─" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="bench_kda_chunk_intra: cuLA vs FLA Triton for chunk_kda_fwd_intra")
    parser.add_argument(
        "--disable_recompute",
        action="store_true",
        help="Disable recompute in both FLA and cuLA (pre-compute QG)",
    )
    args = parser.parse_args()

    if args.disable_recompute:
        DISABLE_RECOMPUTE = True
        print("[Disable recompute] pre-compute QG in forward")

    benchmark_chunk_intra_uniform()
    benchmark_chunk_intra_varlen()
