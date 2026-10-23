#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Stress the pure CuTeDSL raw-gate varlen KDA forward."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from fla.ops.utils import prepare_chunk_indices

from benchmarks.utils import exclusive_cumsum, set_seed
from cula.ops.kda.sm100.intra_fused import BT, K_DIM, chunk_kda_fwd_intra_sm100_varlen
from cula.ops.kda.sm100.recompute_wu import recompute_w_u_from_preprocessed

OUTPUT_NAMES = ("k_scaled", "kg", "q_scaled", "gk_last_exp", "Aqk", "Akk", "w", "u")


def _parse_seq_lens(text: str) -> list[int]:
    seq_lens = [int(value) for value in text.replace(",", " ").split()]
    if len(seq_lens) < 2 or any(length <= 0 for length in seq_lens):
        raise ValueError("--seq-lens must contain at least two positive lengths")
    return seq_lens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10_000_000)
    parser.add_argument("--checkpoint", type=int, default=1_000_000)
    parser.add_argument("--seq-lens", default="65,127,193,255")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--report-json")
    args = parser.parse_args()

    if args.iterations <= 0 or args.checkpoint <= 0:
        raise ValueError("--iterations and --checkpoint must be positive")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("an SM100 CUDA device is required")

    seq_lens = _parse_seq_lens(args.seq_lens)
    total_t = sum(seq_lens)
    device = torch.device("cuda")
    set_seed(args.seed)
    q = F.normalize(torch.randn(1, total_t, args.heads, K_DIM, device=device).float(), dim=-1).bfloat16()
    k = F.normalize(torch.randn(1, total_t, args.heads, K_DIM, device=device).float(), dim=-1).bfloat16()
    g = torch.randn(1, total_t, args.heads, K_DIM, device=device, dtype=torch.bfloat16)
    beta = torch.rand(1, total_t, args.heads, device=device, dtype=torch.bfloat16)
    a_log = torch.randn(args.heads, device=device)
    dt_bias = torch.randn(args.heads * K_DIM, device=device)
    cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

    def run_cutedsl() -> tuple[torch.Tensor, ...]:
        intra = chunk_kda_fwd_intra_sm100_varlen(
            q=q,
            k=k,
            g=g,
            beta=beta,
            A_log=a_log,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            scale=K_DIM**-0.5,
            dt_bias=dt_bias,
            safe_gate=True,
            lower_bound=-5.0,
            seq_lens=seq_lens,
            fp32_akk_inv=True,
        )
        w, u = recompute_w_u_from_preprocessed(
            intra[0],
            k,
            beta,
            intra[5],
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        return (*intra, w, u)

    print(
        f"device={torch.cuda.get_device_name(0)} seq_lens={seq_lens} total_t={total_t} "
        f"heads={args.heads} iterations={args.iterations}",
        flush=True,
    )
    golden = tuple(output.clone() for output in run_cutedsl())
    torch.cuda.synchronize()
    if not all(torch.isfinite(output).all().item() for output in golden):
        raise AssertionError("golden CuTeDSL output contains NaN or Inf")

    for _ in range(3):
        run_cutedsl()
    torch.cuda.synchronize()

    reference_flat = torch.cat([tensor.reshape(-1) for tensor in golden])
    mismatch_count = torch.zeros((), dtype=torch.int64, device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run_cutedsl()
        captured_flat = torch.cat([tensor.reshape(-1) for tensor in captured])
        mismatch_count.add_(torch.count_nonzero(captured_flat != reference_flat))

    torch.cuda.synchronize()
    if mismatch_count.item():
        raise AssertionError(f"graph capture differed from golden output: mismatches={mismatch_count.item()}")

    started = time.perf_counter()
    completed = 0
    while completed < args.iterations:
        stop = min(completed + args.checkpoint, args.iterations)
        for _ in range(completed, stop):
            graph.replay()
        torch.cuda.synchronize()
        completed = stop
        mismatches = mismatch_count.item()
        elapsed = time.perf_counter() - started
        print(
            f"progress={completed}/{args.iterations} mismatches={mismatches} "
            f"elapsed_s={elapsed:.3f} iterations_per_s={completed / elapsed:.1f}",
            flush=True,
        )
        if mismatches:
            raise AssertionError(f"non-deterministic output after {completed} iterations: mismatches={mismatches}")

    elapsed = time.perf_counter() - started
    report = {
        "status": "passed",
        "seq_lens": seq_lens,
        "heads": args.heads,
        "iterations": args.iterations,
        "mismatches": mismatch_count.item(),
        "elapsed_seconds": elapsed,
        "iterations_per_second": args.iterations / elapsed,
        "outputs_checked": OUTPUT_NAMES,
    }
    print("RESULT_JSON=" + json.dumps(report, sort_keys=True), flush=True)
    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")


if __name__ == "__main__":
    main()
