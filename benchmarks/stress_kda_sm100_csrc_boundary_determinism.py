#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Stress the bitwise-aligned SM100 CuTeDSL KDA forward path.

The script first requires complete tensor equality with the csrc boundary for
Aqk, Akk, KG, W, and U. It then captures the CuTeDSL intra, Akk inverse,
recompute-WU, and an exact comparison against those csrc outputs in one CUDA
graph. Every replay therefore validates every output element.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from cula.kda.chunk_intra import chunk_kda_fwd_intra as csrc_chunk_kda_fwd_intra
from cula.ops.kda.sm100.intra_fused import BT, K_DIM, chunk_kda_fwd_intra_sm100_from_gk
from cula.ops.kda.sm100.recompute_wu import recompute_w_u_fwd

OUTPUT_NAMES = ("W", "U", "KG", "Aqk", "Akk")


def _bitwise_stats(
    outputs: tuple[torch.Tensor, ...], references: tuple[torch.Tensor, ...]
) -> dict[str, dict[str, float | int | bool]]:
    stats = {}
    for name, output, reference in zip(OUTPUT_NAMES, outputs, references, strict=True):
        mismatch_count = torch.count_nonzero(output != reference).item()
        max_abs = (output.float() - reference.float()).abs().max().item()
        stats[name] = {
            "equal": mismatch_count == 0,
            "mismatches": mismatch_count,
            "max_abs": max_abs,
        }
    return stats


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
    if args.seqlen <= 0 or args.seqlen % (4 * BT) != 0:
        raise ValueError(f"--seqlen must be a positive multiple of {4 * BT}")
    if args.batch <= 0 or args.heads <= 0:
        raise ValueError("--batch and --heads must be positive")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("an SM100 CUDA device is required")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    shape = (args.batch, args.seqlen, args.heads, K_DIM)
    q = F.normalize(torch.randn(*shape, device=device).float(), dim=-1).bfloat16()
    k = F.normalize(torch.randn(*shape, device=device).float(), dim=-1).bfloat16()
    gk = torch.randn(*shape, device=device, dtype=torch.float32) * 0.02
    beta_dtype = torch.bfloat16 if args.beta_dtype == "bfloat16" else torch.float32
    beta = torch.randn(*shape[:-1], device=device).sigmoid().to(beta_dtype)
    scale = K_DIM**-0.5

    def run_cutedsl() -> tuple[torch.Tensor, ...]:
        aqk, akk = chunk_kda_fwd_intra_sm100_from_gk(
            q=q,
            k=k,
            gk=gk,
            beta=beta,
            scale=scale,
            fp32_akk_inv=True,
        )
        w, u, _, kg = recompute_w_u_fwd(k, k, beta, akk, gk)
        return w, u, kg, aqk, akk

    print(
        f"device={torch.cuda.get_device_name(0)} shape={shape} beta_dtype={args.beta_dtype} iterations={args.iterations}",
        flush=True,
    )

    w_ref, u_ref, _, kg_ref, aqk_ref, akk_ref = csrc_chunk_kda_fwd_intra(
        q=q,
        k=k,
        v=k,
        gk=gk,
        beta=beta,
        scale=scale,
        chunk_size=BT,
        safe_gate=True,
    )
    references = (w_ref, u_ref, kg_ref, aqk_ref, akk_ref)

    outputs = run_cutedsl()
    torch.cuda.synchronize()
    bitwise = _bitwise_stats(outputs, references)
    print("BITWISE_JSON=" + json.dumps(bitwise, sort_keys=True), flush=True)
    if not all(values["equal"] for values in bitwise.values()):
        raise AssertionError("CuTeDSL output is not bitwise equal to csrc")
    if not all(torch.isfinite(output).all().item() for output in outputs):
        raise AssertionError("CuTeDSL output contains NaN or Inf")

    # Warm compilation, allocator, and concatenation paths before capture.
    for _ in range(3):
        warm = run_cutedsl()
        torch.cat([tensor.reshape(-1) for tensor in warm])
    torch.cuda.synchronize()

    reference_flat = torch.cat([tensor.reshape(-1) for tensor in references])
    mismatch_count = torch.zeros((), dtype=torch.int64, device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run_cutedsl()
        captured_flat = torch.cat([tensor.reshape(-1) for tensor in captured])
        mismatch_count.add_(torch.count_nonzero(captured_flat != reference_flat))

    torch.cuda.synchronize()
    if mismatch_count.item() != 0:
        raise AssertionError(f"graph capture differed from csrc: mismatches={mismatch_count.item()}")

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
        if mismatches != 0:
            raise AssertionError(f"non-deterministic output after {completed} iterations: mismatches={mismatches}")

    elapsed = time.perf_counter() - started
    report = {
        "status": "passed",
        "device": torch.cuda.get_device_name(0),
        "device_index_visible": torch.cuda.current_device(),
        "shape": shape,
        "beta_dtype": args.beta_dtype,
        "iterations": args.iterations,
        "mismatches": mismatch_count.item(),
        "elapsed_seconds": elapsed,
        "iterations_per_second": args.iterations / elapsed,
        "bitwise_csrc": bitwise,
    }
    print("RESULT_JSON=" + json.dumps(report, sort_keys=True), flush=True)
    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")


if __name__ == "__main__":
    main()
