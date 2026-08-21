#!/usr/bin/env python3
# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the canonical GDN2 prefill matrix on Hopper SM90.

The five rows cover MHA, GVA2, GVA4, packed variable-length input, all four
initial/final-state modes, and the shortest and longest release sentinels.
Compilation is recorded separately and excluded from CUDA-event timing.

When ``--implementation both`` is selected, the FLA callable is measured from
the same public logical-input boundary. Its required GVA head expansion is
inside the timed call.
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
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cula.gdn2 import (  # noqa: E402
    chunk_gdn2,
    get_sm90_gdn2_backend,
    get_sm90_gdn2_backend_identity,
)

HEAD_SIZE = 128
QUERY_HEADS = 16
VALUE_SIZE = 128
DTYPE = torch.bfloat16


@dataclass(frozen=True)
class BenchmarkRow:
    row_id: str
    lengths: tuple[int, ...]
    value_heads: int
    initial_state: bool
    output_final_state: bool
    seed: int

    @property
    def total_tokens(self) -> int:
        return sum(self.lengths)


CANONICAL_MATRIX = (
    BenchmarkRow(
        "S1-MHA-T64-H16-H0NONE-HTOFF",
        (64,),
        16,
        False,
        False,
        6651,
    ),
    BenchmarkRow(
        "S2-MHA-T1024-H16-H0-HTON",
        (1024,),
        16,
        True,
        True,
        6652,
    ),
    BenchmarkRow(
        "S3-MHA-PACKED-T4096-H16-H0NONE-HTON",
        (
            63,
            129,
            257,
            31,
            512,
            65,
            128,
            17,
            333,
            91,
            211,
            7,
            401,
            255,
            144,
            73,
            289,
            377,
            19,
            694,
        ),
        16,
        False,
        True,
        6653,
    ),
    BenchmarkRow(
        "S4-GVA4-T1024-H16-H64-H0-HTON",
        (1024,),
        64,
        True,
        True,
        6654,
    ),
    BenchmarkRow(
        "S5-GVA2-T1024-H16-H32-H0NONE-HTON",
        (1024,),
        32,
        False,
        True,
        6655,
    ),
)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    bytes_tensor = tensor.detach().contiguous().view(torch.uint8)
    if bytes_tensor.is_cuda:
        bytes_tensor = bytes_tensor.cpu()
    return hashlib.sha256(memoryview(bytes_tensor.numpy())).hexdigest()


def _make_inputs(
    row: BenchmarkRow,
    device: torch.device,
) -> dict[str, torch.Tensor | None]:
    generator = torch.Generator(device="cpu").manual_seed(row.seed)
    q_shape = (row.total_tokens, QUERY_HEADS, HEAD_SIZE)
    v_shape = (row.total_tokens, row.value_heads, VALUE_SIZE)

    def bf16_normal(
        shape: tuple[int, ...],
        standard_deviation: float,
    ) -> torch.Tensor:
        return (torch.randn(shape, generator=generator) * standard_deviation).to(DTYPE)

    offsets = [0]
    for length in row.lengths:
        offsets.append(offsets[-1] + length)
    cpu_inputs: dict[str, torch.Tensor | None] = {
        "q": bf16_normal(q_shape, 0.03),
        "k": bf16_normal(q_shape, 0.01),
        "v": bf16_normal(v_shape, 0.1),
        "g": -torch.rand(q_shape, generator=generator) * 0.05,
        "b": torch.rand(q_shape, generator=generator).to(DTYPE),
        "w": torch.rand(v_shape, generator=generator).to(DTYPE),
        "cu_seqlens": torch.tensor(offsets, dtype=torch.int64),
        "initial_state": None,
    }
    if row.initial_state:
        cpu_inputs["initial_state"] = (
            torch.randn(
                (
                    len(row.lengths),
                    row.value_heads,
                    VALUE_SIZE,
                    HEAD_SIZE,
                ),
                generator=generator,
            )
            * 0.005
        )
    cpu_hashes = {name: _tensor_sha256(tensor) for name, tensor in cpu_inputs.items() if tensor is not None}
    inputs = {name: None if tensor is None else tensor.to(device=device) for name, tensor in cpu_inputs.items()}
    torch.cuda.synchronize(device)
    device_hashes = {name: _tensor_sha256(tensor) for name, tensor in inputs.items() if tensor is not None}
    if device_hashes != cpu_hashes:
        raise AssertionError(f"CPU/CUDA input bytes differ for {row.row_id}")
    return inputs


def _product_call(
    inputs: dict[str, torch.Tensor | None],
    row: BenchmarkRow,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return chunk_gdn2(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["b"],
        inputs["w"],
        initial_state=inputs["initial_state"],
        output_final_state=row.output_final_state,
        cu_seqlens=inputs["cu_seqlens"],
        scale=HEAD_SIZE**-0.5,
    )


def _fla_call(
    inputs: dict[str, torch.Tensor | None],
    row: BenchmarkRow,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    from fla.ops.gdn2.chunk import chunk_gdn2 as fla_chunk_gdn2

    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]
    g = inputs["g"]
    b = inputs["b"]
    w = inputs["w"]
    cu_seqlens = inputs["cu_seqlens"]
    assert all(tensor is not None for tensor in (q, k, v, g, b, w, cu_seqlens))
    group_size = row.value_heads // QUERY_HEADS
    if group_size > 1:
        owner = torch.arange(
            row.value_heads,
            dtype=torch.int64,
            device=q.device,
        ).div(group_size, rounding_mode="floor")
        q = q.index_select(1, owner)
        k = k.index_select(1, owner)
        g = g.index_select(1, owner)
        b = b.index_select(1, owner)
    result = fla_chunk_gdn2(
        q=q.unsqueeze(0),
        k=k.unsqueeze(0),
        v=v.unsqueeze(0),
        g=g.unsqueeze(0),
        b=b.unsqueeze(0),
        w=w.unsqueeze(0),
        initial_state=inputs["initial_state"],
        scale=HEAD_SIZE**-0.5,
        output_final_state=row.output_final_state,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=None,
        safe_gate=False,
        disable_recompute=False,
        return_intermediate_states=False,
        state_v_first=True,
    )
    if not isinstance(result, tuple) or len(result) < 2:
        raise TypeError("FLA chunk_gdn2 must return (output, final_state)")
    output, final_state = result[:2]
    output = output.squeeze(0)
    if row.output_final_state:
        if final_state is None:
            raise RuntimeError("FLA did not return the requested final state")
        return output, final_state
    if final_state is not None:
        raise RuntimeError("FLA returned a disabled final state")
    return output


def _normalise_result(
    result: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    *,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if output_final_state:
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("expected (output, final_state)")
        return result
    if isinstance(result, tuple):
        raise TypeError("expected an output tensor")
    return result, None


def _measure(
    call: Callable[
        [],
        torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ],
    *,
    output_final_state: bool,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> tuple[dict[str, Any], tuple[torch.Tensor, torch.Tensor | None]]:
    started = time.perf_counter()
    first = call()
    torch.cuda.synchronize(device)
    setup_ms = (time.perf_counter() - started) * 1000.0
    first_output, first_state = _normalise_result(
        first,
        output_final_state=output_final_state,
    )

    for _ in range(warmup):
        call()
    torch.cuda.synchronize(device)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        call()
        end.record()
    torch.cuda.synchronize(device)
    samples = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
    if not all(math.isfinite(value) and value > 0 for value in samples):
        raise RuntimeError("invalid CUDA-event timing sample")
    return (
        {
            "setup_ms_excluded": setup_ms,
            "warmup": warmup,
            "iterations": iterations,
            "timer": "cuda_event",
            "raw_per_iteration_ms": samples,
            "average_ms": statistics.fmean(samples),
            "median_ms": statistics.median(samples),
            "minimum_ms": min(samples),
            "maximum_ms": max(samples),
            "output_sha256": _tensor_sha256(first_output),
            "final_state_sha256": (None if first_state is None else _tensor_sha256(first_state)),
        },
        (first_output, first_state),
    )


def _source_identity() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit, status = None, None
    return {
        "repo_root": str(REPO_ROOT),
        "commit": commit,
        "worktree_status": status,
        "backend": get_sm90_gdn2_backend(),
        "backend_identity": get_sm90_gdn2_backend_identity(),
    }


def _environment(device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    result = {
        "captured_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "hostname": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "device": device.index,
        "gpu_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
    }
    try:
        result["fla"] = importlib.metadata.version(
            "flash-linear-attention",
        )
    except importlib.metadata.PackageNotFoundError:
        result["fla"] = None
    return result


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--implementation",
        choices=("product", "both"),
        default="both",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("gdn2-sm90-benchmark.json"),
    )
    parser.add_argument("--list-matrix", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        parser.error(
            "--warmup must be non-negative and --iterations must be positive",
        )
    return args


def main() -> None:
    args = _parse_args()
    if args.list_matrix:
        print(json.dumps([asdict(row) for row in CANONICAL_MATRIX], indent=2))
        return
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda", args.device)
    capability = torch.cuda.get_device_capability(device)
    if capability != (9, 0):
        raise SystemExit(f"GDN2 prefill requires SM90, got {capability}")

    record: dict[str, Any] = {
        "schema": "cula.gdn2.sm90.benchmark.v1",
        "status": "RUNNING",
        "protocol": {
            "matrix_rows": len(CANONICAL_MATRIX),
            "query_heads": QUERY_HEADS,
            "supported_value_heads": [16, 32, 64],
            "key_size": HEAD_SIZE,
            "value_size": VALUE_SIZE,
            "dtype": str(DTYPE),
            "warmup": args.warmup,
            "iterations": args.iterations,
            "statistic": "arithmetic mean of raw CUDA-event samples",
            "compile_and_setup_excluded": True,
            "fla_gva_expansion_inside_timed_call": True,
        },
        "source": _source_identity(),
        "environment": _environment(device),
        "rows": [],
    }
    for position, row in enumerate(CANONICAL_MATRIX, 1):
        print(
            f"[{position:02d}/{len(CANONICAL_MATRIX):02d}] {row.row_id}",
            flush=True,
        )
        inputs = _make_inputs(row, device)
        immutable_before = {name: _tensor_sha256(tensor) for name, tensor in inputs.items() if tensor is not None}
        product_timing, product_result = _measure(
            lambda: _product_call(inputs, row),
            output_final_state=row.output_final_state,
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
        )
        row_record: dict[str, Any] = {
            **asdict(row),
            "total_tokens": row.total_tokens,
            "product": product_timing,
            "status": "PASS",
        }
        if args.implementation == "both":
            fla_timing, fla_result = _measure(
                lambda: _fla_call(inputs, row),
                output_final_state=row.output_final_state,
                warmup=args.warmup,
                iterations=args.iterations,
                device=device,
            )
            torch.testing.assert_close(
                product_result[0].float(),
                fla_result[0].float(),
                rtol=0.01,
                atol=0.01,
            )
            if row.output_final_state:
                assert product_result[1] is not None
                assert fla_result[1] is not None
                torch.testing.assert_close(
                    product_result[1],
                    fla_result[1],
                    rtol=0.001,
                    atol=0.005,
                )
            row_record["fla"] = fla_timing
            row_record["speedup_over_fla"] = fla_timing["average_ms"] / product_timing["average_ms"]
        immutable_after = {name: _tensor_sha256(tensor) for name, tensor in inputs.items() if tensor is not None}
        if immutable_after != immutable_before:
            raise RuntimeError(f"input mutated for {row.row_id}")
        record["rows"].append(row_record)
        print(
            f"  product={product_timing['average_ms']:.6f} ms"
            + ("" if "speedup_over_fla" not in row_record else f" speedup={row_record['speedup_over_fla']:.3f}x"),
            flush=True,
        )

    record["status"] = "PASS"
    record["coverage"] = f"{len(CANONICAL_MATRIX)}/{len(CANONICAL_MATRIX)}"
    _write_json(args.output, record)
    print(f"GDN2_SM90_BENCHMARK_PASS output={args.output}", flush=True)


if __name__ == "__main__":
    main()
