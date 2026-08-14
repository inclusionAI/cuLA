#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark and determinism stress test for the persistent CUDA SM90 bwd-intra kernel.

The kernel uses warp-level ``mma.sync.m16n8k8`` and can also run on
SM100/SM103. This script calls the low-level kernel directly so the same
benchmark can validate every supported architecture.
"""

import argparse
import itertools
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import triton  # noqa: E402
from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra as fla_bwd_intra  # noqa: E402
from fla.ops.utils import prepare_chunk_indices  # noqa: E402

from cula.ops.kda.sm90.bwd_intra import kda_bwd_intra_mma  # noqa: E402

K = 128
BT = 64
DEVICE = torch.device("cuda")


def _balanced_lengths(total_tokens: int, num_seqs: int) -> list[int]:
    base, remainder = divmod(total_tokens, num_seqs)
    return [base] * (num_seqs - remainder) + [base + 1] * remainder


def _quasi_balanced_lengths(total_tokens: int, num_seqs: int, seed: int = 42) -> list[int]:
    rng = random.Random(seed)
    weights = [rng.uniform(1.0, 2.5) for _ in range(num_seqs)]
    lengths = [max(BT, int(total_tokens * weight / sum(weights))) for weight in weights]
    delta = total_tokens - sum(lengths)
    order = sorted(range(num_seqs), key=lengths.__getitem__, reverse=delta < 0)
    for idx in itertools.cycle(order):
        if delta == 0:
            break
        if delta < 0 and lengths[idx] == BT:
            continue
        lengths[idx] += 1 if delta > 0 else -1
        delta += -1 if delta > 0 else 1
    return lengths


def _make_inputs(lengths: list[int], heads: int, beta_dtype: torch.dtype = torch.bfloat16):
    torch.manual_seed(42)
    total_tokens = sum(lengths)
    offsets = list(itertools.accumulate(lengths, initial=0))
    cu_seqlens = torch.tensor(offsets, device=DEVICE, dtype=torch.int32)
    chunk_indices = prepare_chunk_indices(cu_seqlens.to(torch.long), BT).to(torch.int32).contiguous()

    q = torch.randn(1, total_tokens, heads, K, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    g = torch.randn(1, total_tokens, heads, K, device=DEVICE, dtype=torch.float32) / 10
    beta = torch.randn(1, total_tokens, heads, device=DEVICE, dtype=beta_dtype)
    d_aq = torch.randn(1, total_tokens, heads, BT, device=DEVICE, dtype=torch.float32)
    d_ak = torch.randn_like(d_aq)
    dq = torch.randn(1, total_tokens, heads, K, device=DEVICE, dtype=torch.float32)
    dk = torch.randn_like(dq)
    db = torch.randn(1, total_tokens, heads, device=DEVICE, dtype=torch.float32)
    dg = torch.randn_like(dq)
    return (q, k, g, beta, d_aq, d_ak, dq, dk, db, dg, cu_seqlens, chunk_indices)


def _prepare_cula(inputs):
    q, k, _, _, _, _, _, _, db, dg, _, _ = inputs
    outputs = (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(db),
        torch.empty_like(dg),
    )

    def run():
        return kda_bwd_intra_mma(*inputs, *outputs, BT)

    return run, outputs


def _run_fla(inputs):
    return fla_bwd_intra(*inputs, chunk_size=BT, safe_gate=True)


def _error_metrics(reference: torch.Tensor, actual: torch.Tensor) -> tuple[float, float, float]:
    reference = reference.float()
    actual = actual.float()
    diff = reference - actual
    rmse = diff.square().mean().sqrt().item()
    relative_rmse = rmse / (reference.square().mean().sqrt().item() + 1e-8)
    relative_max = diff.abs().max().item() / (reference.abs().max().item() + 1e-8)
    return rmse, relative_rmse, relative_max


def check_accuracy(lengths: list[int], heads: int) -> tuple[float, float]:
    inputs = _make_inputs(lengths, heads)
    run_cula, outputs = _prepare_cula(inputs)
    run_cula()
    reference = _run_fla(inputs)
    torch.cuda.synchronize()

    names = ("dq", "dk", "db", "dg")
    metrics = [_error_metrics(ref, out) for ref, out in zip(reference, outputs)]
    for name, (rmse, relative_rmse, relative_max) in zip(names, metrics):
        print(f"    {name}: RMSE={rmse:.6e} rRMSE={relative_rmse:.6e} rMAX={relative_max:.6e}")
    return max(value[1] for value in metrics), max(value[2] for value in metrics)


def check_determinism(iters: int, heads: int = 4, total_tokens: int = 512, num_seqs: int = 4) -> None:
    lengths = _balanced_lengths(total_tokens, num_seqs)
    inputs = _make_inputs(lengths, heads)
    run, outputs = _prepare_cula(inputs)
    run()
    torch.cuda.synchronize()
    reference = tuple(output.clone() for output in outputs)

    for iteration in range(iters):
        run()
        for name, output, expected in zip(("dq", "dk", "db", "dg"), outputs, reference):
            if not torch.equal(output, expected):
                max_diff = (output.float() - expected.float()).abs().max().item()
                raise AssertionError(f"{name} is non-deterministic at iteration {iteration}: max_diff={max_diff}")
        if (iteration + 1) % 1000 == 0 or iteration + 1 == iters:
            print(f"  determinism: {iteration + 1}/{iters}", flush=True)


def _do_bench(fn, warmup: int, rep: int) -> tuple[float, float, float]:
    result = triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.2, 0.8])
    return tuple(float(value) for value in result)


def run_benchmarks(heads_list: list[int], warmup: int, rep: int) -> None:
    configs = (
        ("uniform", [8192]),
        ("uniform", [32768]),
        ("varlen", _quasi_balanced_lengths(8192, 8)),
        ("varlen", _quasi_balanced_lengths(32768, 8)),
    )
    print(f"{'Config':<32} {'cuLA p50':>11} {'FLA p50':>11} {'speedup':>9} {'cuLA p20-p80':>22} {'FLA p20-p80':>22}")
    print("-" * 114)
    speedups = []
    for heads in heads_list:
        for kind, lengths in configs:
            inputs = _make_inputs(lengths, heads)
            run_cula, _ = _prepare_cula(inputs)
            run_cula()
            _run_fla(inputs)
            torch.cuda.synchronize()

            cula_p50, cula_p20, cula_p80 = _do_bench(run_cula, warmup, rep)
            fla_p50, fla_p20, fla_p80 = _do_bench(lambda: _run_fla(inputs), warmup, rep)
            speedup = fla_p50 / cula_p50
            speedups.append(speedup)
            label = f"H={heads} {kind} T={sum(lengths)} N={len(lengths)}"
            print(
                f"{label:<32} {cula_p50:>9.3f}ms {fla_p50:>9.3f}ms {speedup:>8.2f}x "
                f"{cula_p20:>8.3f}-{cula_p80:<8.3f} {fla_p20:>8.3f}-{fla_p80:<8.3f}"
            )
            relative_rmse, relative_max = check_accuracy(lengths, heads)
            print(f"    worst: rRMSE={relative_rmse:.6e} rMAX={relative_max:.6e}")
            torch.cuda.empty_cache()
    geometric_mean = torch.tensor(speedups, dtype=torch.float64).log().mean().exp().item()
    print(f"geomean speedup over FLA: {geometric_mean:.3f}x")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--determinism-iters", type=int, default=0)
    parser.add_argument("--determinism-only", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    capability = torch.cuda.get_device_capability()
    print(f"device: {torch.cuda.get_device_name()} SM{capability[0]}{capability[1]}")
    print(f"torch: {torch.__version__} cuda: {torch.version.cuda}")
    if capability not in ((9, 0), (10, 0), (10, 3)):
        raise RuntimeError(f"requires SM90, SM100, or SM103; got SM{capability[0]}{capability[1]}")

    if args.determinism_iters:
        check_determinism(args.determinism_iters)
        print(f"determinism PASS ({args.determinism_iters} iterations)")
    if not args.determinism_only:
        run_benchmarks(args.heads, args.warmup, args.rep)


if __name__ == "__main__":
    main()
