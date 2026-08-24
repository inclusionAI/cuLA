#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Stress SM100 CuTeDSL KDA varlen forward determinism and csrc accuracy.

The stress phase captures the complete CuTeDSL forward chain and an exact
comparison against a golden output in one CUDA graph. Every replay therefore
checks every output element; a device-side mismatch counter is inspected at
each checkpoint and after the requested number of iterations.
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
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.constant import RCP_LN2

from benchmarks.utils import exclusive_cumsum, set_seed
from cula.kda.chunk_intra import chunk_kda_fwd_intra as csrc_chunk_kda_fwd_intra
from cula.ops.kda.sm100.intra_fused import BT, K_DIM, chunk_kda_fwd_intra_sm100_varlen
from cula.ops.kda.sm100.recompute_wu import recompute_w_u_from_preprocessed

OUTPUT_NAMES = ("k_scaled", "kg", "q_scaled", "gk_last_exp", "Aqk", "Akk", "w", "u")
ACCURACY_LIMITS = {
    "k_scaled": (1e-4, 2e-3),
    "kg": (5e-4, 2e-3),
    "q_scaled": (1e-4, 2e-3),
    "gk_last_exp": (1e-3, 2e-2),
    "Aqk": (2e-3, 2e-3),
    "Akk": (1e-4, 2e-3),
    "w": (1e-2, 4e-3),
    "u": (1e-3, 2e-3),
}


def _parse_seq_lens(text: str) -> list[int]:
    seq_lens = [int(value) for value in text.replace(",", " ").split()]
    if len(seq_lens) < 2 or any(length <= 0 for length in seq_lens):
        raise ValueError("--seq-lens must contain at least two positive lengths")
    return seq_lens


def _make_lower_mask(seq_lens: list[int], heads: int, device: torch.device) -> torch.Tensor:
    total_t = sum(seq_lens)
    mask = torch.zeros((1, total_t, heads, BT), dtype=torch.bool, device=device)
    cols = torch.arange(BT, device=device)
    offset = 0
    for seq_len in seq_lens:
        rows = torch.arange(seq_len, device=device) % BT
        seq_mask = (cols[None, :] <= rows[:, None]).view(1, seq_len, 1, BT)
        mask[:, offset : offset + seq_len] = seq_mask
        offset += seq_len
    return mask


def _gk_last_exp(gk: torch.Tensor, seq_lens: list[int]) -> torch.Tensor:
    rows = []
    offset = 0
    for seq_len in seq_lens:
        for chunk_start in range(0, seq_len, BT):
            last_row = offset + min(chunk_start + BT, seq_len) - 1
            rows.append(gk[:, last_row])
        offset += seq_len
    return torch.stack(rows, dim=1).contiguous().exp2()


def _accuracy_stats(ref: torch.Tensor, out: torch.Tensor) -> dict[str, float]:
    ref_f = ref.float()
    out_f = out.float()
    diff = (out_f - ref_f).abs()
    rmse = diff.square().mean().sqrt()
    ref_rms = ref_f.square().mean().sqrt()
    return {
        "rel_rmse": (rmse / (ref_rms + 1e-8)).item(),
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
    }


def _check_accuracy(
    outputs: tuple[torch.Tensor, ...],
    references: tuple[torch.Tensor, ...],
    lower_mask: torch.Tensor,
) -> dict[str, dict[str, float]]:
    stats = {}
    for name, out, ref in zip(OUTPUT_NAMES, outputs, references):
        if name in ("Aqk", "Akk"):
            out = out[lower_mask]
            ref = ref[lower_mask]
        values = _accuracy_stats(ref, out)
        rel_limit, abs_limit = ACCURACY_LIMITS[name]
        if values["rel_rmse"] > rel_limit or values["max_abs"] > abs_limit:
            raise AssertionError(
                f"{name} accuracy failed: rel_rmse={values['rel_rmse']:.6e} "
                f"(limit {rel_limit:.1e}), max_abs={values['max_abs']:.6e} "
                f"(limit {abs_limit:.1e})"
            )
        stats[name] = values

    for name, output in (("Aqk_upper", outputs[4]), ("Akk_upper", outputs[5])):
        upper_max = output[~lower_mask].abs().max().item()
        if upper_max != 0.0:
            raise AssertionError(f"{name} is not exactly zero: max_abs={upper_max}")
        stats[name] = {"rel_rmse": 0.0, "max_abs": upper_max, "mean_abs": 0.0}
    return stats


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
    beta = torch.randn(1, total_t, args.heads, device=device).sigmoid().bfloat16()
    a_log = torch.randn(args.heads, device=device)
    dt_bias = torch.randn(args.heads * K_DIM, device=device)
    cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=device)
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    scale = K_DIM**-0.5
    lower_bound = -5.0

    def run_cutedsl() -> tuple[torch.Tensor, ...]:
        intra = chunk_kda_fwd_intra_sm100_varlen(
            q=q,
            k=k,
            g=g,
            beta=beta,
            A_log=a_log,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            scale=scale,
            dt_bias=dt_bias,
            safe_gate=True,
            lower_bound=lower_bound,
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

    gk = kda_gate_chunk_cumsum(
        g=g,
        A_log=a_log,
        dt_bias=dt_bias,
        scale=RCP_LN2,
        chunk_size=BT,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        lower_bound=lower_bound,
    )
    w_ref, u_ref, _, kg_ref, aqk_ref, akk_ref = csrc_chunk_kda_fwd_intra(
        q=q,
        k=k,
        v=k,
        gk=gk,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=BT,
        safe_gate=True,
    )
    references = (
        (k.float() * gk.exp2()).bfloat16(),
        kg_ref,
        (q.float() * gk.exp2()).bfloat16(),
        _gk_last_exp(gk, seq_lens),
        aqk_ref,
        akk_ref,
        w_ref,
        u_ref,
    )

    outputs = run_cutedsl()
    torch.cuda.synchronize()
    lower_mask = _make_lower_mask(seq_lens, args.heads, device)
    accuracy = _check_accuracy(outputs, references, lower_mask)
    for name, values in accuracy.items():
        print(
            f"accuracy {name:<12} rel_rmse={values['rel_rmse']:.6e} "
            f"max_abs={values['max_abs']:.6e} mean_abs={values['mean_abs']:.6e}",
            flush=True,
        )

    golden = tuple(output.clone() for output in outputs)
    if not all(torch.isfinite(output).all().item() for output in golden):
        raise AssertionError("golden CuTeDSL output contains NaN or Inf")

    # Warm all allocator and concatenation paths before graph capture.
    for _ in range(3):
        warm = run_cutedsl()
        torch.cat([tensor.reshape(-1) for index, tensor in enumerate(warm) if index != 3])
    torch.cuda.synchronize()

    golden_bf16 = torch.cat([tensor.reshape(-1) for index, tensor in enumerate(golden) if index != 3])
    golden_fp32 = golden[3].reshape(-1)
    mismatch_count = torch.zeros((), dtype=torch.int64, device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run_cutedsl()
        captured_bf16 = torch.cat([tensor.reshape(-1) for index, tensor in enumerate(captured) if index != 3])
        mismatch_count.add_(torch.count_nonzero(captured_bf16 != golden_bf16))
        mismatch_count.add_(torch.count_nonzero(captured[3].reshape(-1) != golden_fp32))

    torch.cuda.synchronize()
    if mismatch_count.item() != 0:
        raise AssertionError(f"graph capture already differed from golden: mismatches={mismatch_count.item()}")

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
        rate = completed / elapsed
        print(
            f"progress={completed}/{args.iterations} mismatches={mismatches} "
            f"elapsed_s={elapsed:.3f} iterations_per_s={rate:.1f}",
            flush=True,
        )
        if mismatches != 0:
            raise AssertionError(f"non-deterministic output after {completed} iterations: mismatches={mismatches}")

    elapsed = time.perf_counter() - started
    report = {
        "status": "passed",
        "device": torch.cuda.get_device_name(0),
        "device_index_visible": torch.cuda.current_device(),
        "seq_lens": seq_lens,
        "total_t": total_t,
        "heads": args.heads,
        "iterations": args.iterations,
        "mismatches": mismatch_count.item(),
        "elapsed_seconds": elapsed,
        "iterations_per_second": args.iterations / elapsed,
        "accuracy": accuracy,
    }
    print("RESULT_JSON=" + json.dumps(report, sort_keys=True), flush=True)
    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")


if __name__ == "__main__":
    main()
