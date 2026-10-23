#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Replay the pure SM100 CuTeDSL KDA forward and compare every output bit."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from cula.kda.chunk_intra import chunk_kda_fwd_intra

OUTPUT_NAMES = ("W", "U", "KG", "Aqk", "Akk")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10_000_000)
    parser.add_argument("--checkpoint", type=int, default=1_000_000)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--beta-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--report-json")
    args = parser.parse_args()

    if args.iterations <= 0 or args.checkpoint <= 0:
        raise ValueError("--iterations and --checkpoint must be positive")
    if args.seqlen <= 0:
        raise ValueError("--seqlen must be positive")
    if args.batch <= 0 or args.heads <= 0:
        raise ValueError("--batch and --heads must be positive")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("an SM100 CUDA device is required")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dim = 128
    shape = (args.batch, args.seqlen, args.heads, dim)
    q = F.normalize(torch.randn(*shape, device=device).float(), dim=-1).bfloat16()
    k = F.normalize(torch.randn(*shape, device=device).float(), dim=-1).bfloat16()
    v = torch.randn(*shape, device=device, dtype=torch.bfloat16)
    gk = torch.randn(*shape, device=device, dtype=torch.float32) * 0.02
    beta_dtype = torch.bfloat16 if args.beta_dtype == "bfloat16" else torch.float32
    beta = torch.rand(*shape[:-1], device=device, dtype=beta_dtype)

    def run_cutedsl() -> tuple[torch.Tensor, ...]:
        w, u, _, kg, aqk, akk = chunk_kda_fwd_intra(
            q,
            k,
            v,
            gk,
            beta,
            scale=dim**-0.5,
            safe_gate=True,
        )
        return w, u, kg, aqk, akk

    print(
        f"device={torch.cuda.get_device_name(0)} shape={shape} beta_dtype={args.beta_dtype} iterations={args.iterations}",
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
    if mismatch_count.item() != 0:
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
        "shape": shape,
        "beta_dtype": args.beta_dtype,
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
