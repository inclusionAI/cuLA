#!/usr/bin/env python3
# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the canonical GDN prefill matrix on Hopper SM90.

The default invocation always runs the 10 fixed-length and 18 variable-length
rows. Compilation/setup latency is recorded separately and is excluded from
the steady-state CUDA-event measurement.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import pathlib
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cula.gdn import (  # noqa: E402
    chunk_gated_delta_rule,
    get_sm90_gdn_prefill_backend,
    get_sm90_gdn_prefill_backend_identity,
)

HEAD_SIZE = 128
NUM_HEADS = 64
DTYPE = torch.bfloat16
FIXED_BATCHES = (1, 2)
FIXED_LENGTHS = (512, 1024, 4096, 8192, 16384)
VARLEN_NUM_SEQS = (10, 20)
VARLEN_TOTAL_LENGTHS = (4096, 8192, 16384)
VARLEN_DISTRIBUTIONS = ("uniform", "random", "skewed")


@dataclass(frozen=True)
class BenchmarkRow:
    row_id: str
    row_index: int
    mode: str
    seq_lens: tuple[int, ...]
    distribution: str
    batch_size: int | None
    seq_len: int | None
    num_seqs: int
    total_tokens: int


def _exclusive_cumsum(values: tuple[int, ...]) -> list[int]:
    result = [0]
    for value in values:
        result.append(result[-1] + value)
    return result


def _uniform_seq_lens(num_seqs: int, total_len: int) -> tuple[int, ...]:
    base, remainder = divmod(total_len, num_seqs)
    return tuple(base + (index < remainder) for index in range(num_seqs))


def _random_seq_lens(num_seqs: int, total_len: int, seed: int) -> tuple[int, ...]:
    rng = random.Random(seed)
    cuts = sorted(rng.sample(range(1, total_len), num_seqs - 1))
    return tuple(end - start for start, end in zip([0, *cuts], [*cuts, total_len]))


def _skewed_seq_lens(num_seqs: int, total_len: int) -> tuple[int, ...]:
    remaining = total_len - num_seqs
    weights = tuple((num_seqs - index) ** 2 for index in range(num_seqs))
    weight_sum = sum(weights)
    lengths = [1 + remaining * weight // weight_sum for weight in weights]
    lengths[0] += total_len - sum(lengths)
    return tuple(lengths)


def _varlen_seq_lens(distribution: str, num_seqs: int, total_len: int, seed: int) -> tuple[int, ...]:
    if distribution == "uniform":
        return _uniform_seq_lens(num_seqs, total_len)
    if distribution == "random":
        return _random_seq_lens(num_seqs, total_len, seed)
    if distribution == "skewed":
        return _skewed_seq_lens(num_seqs, total_len)
    raise ValueError(f"unknown distribution: {distribution}")


def build_gdn_prefill_matrix(base_seed: int) -> tuple[BenchmarkRow, ...]:
    """Return the immutable canonical 28-row workload."""

    rows: list[BenchmarkRow] = []
    for batch_size in FIXED_BATCHES:
        for seq_len in FIXED_LENGTHS:
            index = len(rows)
            rows.append(
                BenchmarkRow(
                    row_id=f"fixed-b{batch_size}-t{seq_len}",
                    row_index=index,
                    mode="fixed",
                    seq_lens=(seq_len,) * batch_size,
                    distribution="-",
                    batch_size=batch_size,
                    seq_len=seq_len,
                    num_seqs=batch_size,
                    total_tokens=batch_size * seq_len,
                ),
            )
    for num_seqs in VARLEN_NUM_SEQS:
        for total_len in VARLEN_TOTAL_LENGTHS:
            for distribution in VARLEN_DISTRIBUTIONS:
                index = len(rows)
                rows.append(
                    BenchmarkRow(
                        row_id=f"varlen-n{num_seqs}-t{total_len}-{distribution}",
                        row_index=index,
                        mode="varlen",
                        seq_lens=_varlen_seq_lens(distribution, num_seqs, total_len, base_seed + index),
                        distribution=distribution,
                        batch_size=None,
                        seq_len=None,
                        num_seqs=num_seqs,
                        total_tokens=total_len,
                    ),
                )
    if len(rows) != 28 or len({row.row_id for row in rows}) != 28:
        raise AssertionError("GDN prefill benchmark must contain exactly 28 unique rows")
    return tuple(rows)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    byte_tensor = tensor.detach().contiguous().view(torch.uint8)
    if byte_tensor.is_cuda:
        byte_tensor = byte_tensor.cpu()
    return hashlib.sha256(memoryview(byte_tensor.numpy())).hexdigest()


def _make_inputs(row: BenchmarkRow, base_seed: int, device: torch.device) -> dict[str, torch.Tensor]:
    row_seed = base_seed + row.row_index
    generator = torch.Generator(device="cpu").manual_seed(row_seed)
    shape = (row.total_tokens, NUM_HEADS, HEAD_SIZE)
    gate_shape = (row.total_tokens, NUM_HEADS)
    cpu_inputs = {
        "q": (torch.randn(shape, generator=generator) * 0.03).to(DTYPE),
        "k": (torch.randn(shape, generator=generator) * 0.01).to(DTYPE),
        "v": (torch.randn(shape, generator=generator) * 0.1).to(DTYPE),
        "g": torch.rand(gate_shape, generator=generator, dtype=torch.float32) * 0.1 + 0.85,
        "beta": torch.rand(gate_shape, generator=generator, dtype=torch.float32) * 0.5,
        "cu_seqlens": torch.tensor(_exclusive_cumsum(row.seq_lens), dtype=torch.int64),
    }
    cpu_hashes = {name: _tensor_sha256(tensor) for name, tensor in cpu_inputs.items()}
    inputs = {name: tensor.to(device=device) for name, tensor in cpu_inputs.items()}
    torch.cuda.synchronize(device)
    cuda_hashes = {name: _tensor_sha256(tensor) for name, tensor in inputs.items()}
    if cpu_hashes != cuda_hashes:
        raise AssertionError(f"CPU/CUDA input bytes differ for {row.row_id}")
    return inputs


def _call_kernel(inputs: dict[str, torch.Tensor], output: torch.Tensor) -> torch.Tensor:
    result = chunk_gated_delta_rule(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        g=inputs["g"],
        beta=inputs["beta"],
        scale=1.0 / math.sqrt(HEAD_SIZE),
        cu_seqlens=inputs["cu_seqlens"],
        output=output,
    )
    if isinstance(result, tuple) or result.data_ptr() != output.data_ptr():
        raise RuntimeError("GDN benchmark requires the preallocated output path without final state")
    return result


def _measure_row(
    row: BenchmarkRow,
    *,
    seed: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, Any]:
    inputs = _make_inputs(row, seed, device)
    output = torch.empty((row.total_tokens, NUM_HEADS, HEAD_SIZE), dtype=DTYPE, device=device)

    setup_start = time.perf_counter()
    _call_kernel(inputs, output)
    torch.cuda.synchronize(device)
    setup_ms = (time.perf_counter() - setup_start) * 1000.0

    for _ in range(warmup):
        _call_kernel(inputs, output)
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        _call_kernel(inputs, output)
    end.record()
    end.synchronize()
    latency_ms = start.elapsed_time(end) / iterations

    if not bool(torch.isfinite(output).all()) or float(output.float().abs().sum().item()) == 0.0:
        raise AssertionError(f"invalid output for {row.row_id}")
    return {
        **asdict(row),
        "seq_lens": list(row.seq_lens),
        "setup_ms_excluded": setup_ms,
        "warmup": warmup,
        "iterations": iterations,
        "latency_ms": latency_ms,
        "output_sha256": _tensor_sha256(output),
        "status": "PASS",
    }


def _git_source() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = (
            subprocess.run(
                ["git", "diff", "--quiet"],
                cwd=REPO_ROOT,
                check=False,
            ).returncode
            != 0
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "repo_root": str(REPO_ROOT),
        "commit": commit,
        "tracked_worktree_dirty": dirty,
        "backend": get_sm90_gdn_prefill_backend(),
        "backend_identity": get_sm90_gdn_prefill_backend_identity(),
    }


def _environment(device: torch.device) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    return {
        "captured_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "hostname": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "device": device.index,
        "gpu_name": props.name,
        "compute_capability": [props.major, props.minor],
    }


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("gdn-sm90-benchmark.json"))
    parser.add_argument("--list-matrix", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be non-negative and --iterations must be positive")
    return args


def main() -> None:
    args = parse_args()
    matrix = build_gdn_prefill_matrix(args.seed)
    if args.list_matrix:
        print(json.dumps([asdict(row) for row in matrix], indent=2))
        return
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda", args.device)
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise SystemExit(f"GDN prefill requires SM90, got {torch.cuda.get_device_capability(device)}")

    record: dict[str, Any] = {
        "schema": "cula.gdn.sm90.benchmark.v1",
        "status": "RUNNING",
        "protocol": {
            "matrix_rows": 28,
            "fixed_rows": 10,
            "varlen_rows": 18,
            "heads": [NUM_HEADS, NUM_HEADS, NUM_HEADS],
            "head_size": HEAD_SIZE,
            "dtype": str(DTYPE),
            "seed": args.seed,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "statistic": "CUDA-event mean",
            "compile_and_setup_excluded": True,
        },
        "source": _git_source(),
        "environment": _environment(device),
        "rows": [],
    }
    for position, row in enumerate(matrix, 1):
        print(f"[{position:02d}/28] {row.row_id}", flush=True)
        result = _measure_row(
            row,
            seed=args.seed,
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
        )
        record["rows"].append(result)
        print(f"  {result['latency_ms']:.6f} ms", flush=True)

    record["status"] = "PASS"
    record["coverage"] = "28/28"
    _write_json(args.output, record)
    print(f"GDN_SM90_BENCHMARK_PASS output={args.output}", flush=True)


if __name__ == "__main__":
    main()
