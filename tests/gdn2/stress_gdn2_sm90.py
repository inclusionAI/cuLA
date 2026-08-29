#!/usr/bin/env python3
# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Run source-bound deterministic stress for the SM90 GDN2 product path."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from cula.gdn2 import (  # noqa: E402
    chunk_gdn2,
    get_sm90_gdn2_backend_identity,
)
from cula.ops.gdn2.sm90.config import (  # noqa: E402
    HEAD_SIZE,
    SM90_BACKEND_ID,
    SUPPORTED_Q_HEADS,
    VALUE_SIZE,
)

S3_LENGTHS = (
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
)


@dataclass(frozen=True)
class StressSpec:
    case_id: str
    lengths: tuple[int, ...]
    value_heads: int
    initial_state: bool
    output_final_state: bool
    seed: int


STRESS_MATRIX = (
    StressSpec(
        "S1-MHA-T64",
        (64,),
        16,
        False,
        False,
        7801,
    ),
    StressSpec(
        "S2-MHA-T1024",
        (1024,),
        16,
        True,
        True,
        7802,
    ),
    StressSpec(
        "S3-MHA-PACKED-T4096",
        S3_LENGTHS,
        16,
        False,
        True,
        7803,
    ),
    StressSpec(
        "N32-MHA-IRREGULAR",
        (1, 63, 64, 65) * 8,
        16,
        True,
        False,
        7804,
    ),
    StressSpec(
        "GVA2-PACKED",
        (1, 63, 65, 2),
        32,
        False,
        True,
        7805,
    ),
    StressSpec(
        "GVA4-PACKED",
        (65, 1, 129, 63),
        64,
        True,
        True,
        7806,
    ),
)


@dataclass
class StressCase:
    spec: StressSpec
    inputs: dict[str, torch.Tensor | None]
    input_hashes_before: dict[str, str]
    output_storage: torch.Tensor
    output: torch.Tensor
    output_redzones_before: tuple[str, str]
    state_storage: torch.Tensor | None
    output_state: torch.Tensor | None
    state_redzones_before: tuple[str, str] | None
    baseline_output: torch.Tensor
    baseline_state: torch.Tensor | None
    baseline_output_sha256: str
    baseline_state_sha256: str | None
    output_mismatches: torch.Tensor
    state_mismatches: torch.Tensor
    nonfinite_values: torch.Tensor
    stress_launches: int = 0


def _tensor_sha256(tensor: torch.Tensor) -> str:
    byte_view = tensor.detach().contiguous().view(torch.uint8)
    if byte_view.is_cuda:
        byte_view = byte_view.cpu()
    return hashlib.sha256(
        memoryview(byte_view.numpy()),
    ).hexdigest()


def _cpu_scalar(tensor: torch.Tensor) -> object:
    """Read a scalar without PyTorch's pinned-host item() staging allocation."""
    return tensor.detach().cpu().item()


def _redzone_hashes(storage: torch.Tensor) -> tuple[str, str]:
    return (
        _tensor_sha256(storage[:1]),
        _tensor_sha256(storage[-1:]),
    )


def _make_inputs(
    spec: StressSpec,
    device: torch.device,
) -> dict[str, torch.Tensor | None]:
    generator = torch.Generator(device="cpu").manual_seed(spec.seed)
    total_tokens = sum(spec.lengths)
    q_shape = (total_tokens, SUPPORTED_Q_HEADS, HEAD_SIZE)
    v_shape = (total_tokens, spec.value_heads, VALUE_SIZE)

    def bf16_normal(
        shape: tuple[int, ...],
        standard_deviation: float,
    ) -> torch.Tensor:
        return (torch.randn(shape, generator=generator) * standard_deviation).to(torch.bfloat16)

    offsets = [0]
    for length in spec.lengths:
        offsets.append(offsets[-1] + length)
    cpu_inputs: dict[str, torch.Tensor | None] = {
        "q": bf16_normal(q_shape, 0.03),
        "k": bf16_normal(q_shape, 0.01),
        "v": bf16_normal(v_shape, 0.1),
        "g": (
            -torch.rand(
                q_shape,
                generator=generator,
                dtype=torch.float32,
            )
            * 0.05
        ),
        "b": torch.rand(
            q_shape,
            generator=generator,
        ).to(torch.bfloat16),
        "w": torch.rand(
            v_shape,
            generator=generator,
        ).to(torch.bfloat16),
        "cu_seqlens": torch.tensor(
            offsets,
            dtype=torch.int64,
        ),
        "initial_state": None,
    }
    if spec.initial_state:
        cpu_inputs["initial_state"] = (
            torch.randn(
                (
                    len(spec.lengths),
                    spec.value_heads,
                    VALUE_SIZE,
                    HEAD_SIZE,
                ),
                generator=generator,
            )
            * 0.005
        )
    return {name: (None if tensor is None else tensor.to(device=device)) for name, tensor in cpu_inputs.items()}


def _input_hashes(
    inputs: dict[str, torch.Tensor | None],
) -> dict[str, str]:
    return {name: _tensor_sha256(tensor) for name, tensor in sorted(inputs.items()) if tensor is not None}


def _launch(case: StressCase) -> None:
    result = chunk_gdn2(
        case.inputs["q"],
        case.inputs["k"],
        case.inputs["v"],
        case.inputs["g"],
        case.inputs["b"],
        case.inputs["w"],
        initial_state=case.inputs["initial_state"],
        output_final_state=case.spec.output_final_state,
        cu_seqlens=case.inputs["cu_seqlens"],
        scale=HEAD_SIZE**-0.5,
        output=case.output,
        output_state=case.output_state,
        validate_inputs=False,
    )
    if case.spec.output_final_state:
        output, state = result
        if output is not case.output or state is not case.output_state:
            raise RuntimeError(
                f"preallocated result identity drift: {case.spec.case_id}",
            )
    elif result is not case.output:
        raise RuntimeError(
            f"preallocated output identity drift: {case.spec.case_id}",
        )


def _accumulate_exactness(case: StressCase) -> None:
    case.output_mismatches.add_(
        torch.count_nonzero(
            case.output.view(torch.int16) != case.baseline_output.view(torch.int16),
        ),
    )
    case.nonfinite_values.add_(
        torch.count_nonzero(~torch.isfinite(case.output)),
    )
    if case.output_state is not None:
        if case.baseline_state is None:
            raise RuntimeError(
                f"missing state baseline: {case.spec.case_id}",
            )
        case.state_mismatches.add_(
            torch.count_nonzero(
                case.output_state.view(torch.int32) != case.baseline_state.view(torch.int32),
            ),
        )
        case.nonfinite_values.add_(
            torch.count_nonzero(
                ~torch.isfinite(case.output_state),
            ),
        )


def _build_case(
    spec: StressSpec,
    device: torch.device,
    warmup: int,
) -> StressCase:
    inputs = _make_inputs(spec, device)
    input_hashes_before = _input_hashes(inputs)
    total_tokens = sum(spec.lengths)
    output_storage = torch.full(
        (
            total_tokens + 2,
            spec.value_heads,
            VALUE_SIZE,
        ),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    output_storage[0].fill_(-123.0)
    output_storage[-1].fill_(-123.0)
    output = output_storage[1:-1]
    output_redzones_before = _redzone_hashes(output_storage)
    state_storage = None
    output_state = None
    state_redzones_before = None
    if spec.output_final_state:
        state_storage = torch.full(
            (
                len(spec.lengths) + 2,
                spec.value_heads,
                VALUE_SIZE,
                HEAD_SIZE,
            ),
            float("nan"),
            dtype=torch.float32,
            device=device,
        )
        state_storage[0].fill_(-987654.0)
        state_storage[-1].fill_(-987654.0)
        output_state = state_storage[1:-1]
        state_redzones_before = _redzone_hashes(
            state_storage,
        )

    placeholder = StressCase(
        spec=spec,
        inputs=inputs,
        input_hashes_before=input_hashes_before,
        output_storage=output_storage,
        output=output,
        output_redzones_before=output_redzones_before,
        state_storage=state_storage,
        output_state=output_state,
        state_redzones_before=state_redzones_before,
        baseline_output=output,
        baseline_state=output_state,
        baseline_output_sha256="",
        baseline_state_sha256=None,
        output_mismatches=torch.zeros(
            (),
            dtype=torch.int64,
            device=device,
        ),
        state_mismatches=torch.zeros(
            (),
            dtype=torch.int64,
            device=device,
        ),
        nonfinite_values=torch.zeros(
            (),
            dtype=torch.int64,
            device=device,
        ),
    )
    for _ in range(warmup):
        _launch(placeholder)
    torch.cuda.synchronize(device)
    if not bool(_cpu_scalar(torch.isfinite(output).all())):
        raise RuntimeError(
            f"non-finite baseline output: {spec.case_id}",
        )
    baseline_output = output.detach().clone()
    baseline_state = None if output_state is None else output_state.detach().clone()
    if baseline_state is not None and not bool(
        _cpu_scalar(torch.isfinite(baseline_state).all()),
    ):
        raise RuntimeError(
            f"non-finite baseline state: {spec.case_id}",
        )
    placeholder.baseline_output = baseline_output
    placeholder.baseline_state = baseline_state
    placeholder.baseline_output_sha256 = _tensor_sha256(
        baseline_output,
    )
    placeholder.baseline_state_sha256 = None if baseline_state is None else _tensor_sha256(baseline_state)
    return placeholder


def _normalise_gpu_uuid(value: object) -> str:
    uuid = str(value)
    return uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_source_manifest(
    source_root: pathlib.Path,
    manifest_path: pathlib.Path,
) -> dict[str, Any]:
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8"),
    )
    aggregate = hashlib.sha256()
    observed: list[tuple[str, str, int]] = []
    for entry in manifest["files"]:
        relative = pathlib.Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"unsafe source-manifest path: {relative}",
            )
        path = (source_root / relative).resolve(strict=True)
        if not path.is_relative_to(source_root) or not path.is_file():
            raise ValueError(
                f"source path escapes root: {relative}",
            )
        sha256 = _sha256(path)
        size_bytes = path.stat().st_size
        if sha256 != entry["sha256"] or size_bytes != entry["size_bytes"]:
            raise RuntimeError(
                f"source-manifest mismatch: {relative}",
            )
        observed.append(
            (relative.as_posix(), sha256, size_bytes),
        )
    for relative, sha256, size_bytes in sorted(observed):
        aggregate.update(
            f"{relative}\0{sha256}\0{size_bytes}\n".encode(),
        )
    if aggregate.hexdigest() != manifest["aggregate_sha256"]:
        raise RuntimeError(
            "source-manifest aggregate mismatch",
        )
    return {
        "mode": "manifest",
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "aggregate_sha256": aggregate.hexdigest(),
        "file_count": len(observed),
    }


def _git_source_identity(
    source_root: pathlib.Path,
) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=source_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit, status = None, None
    return {
        "mode": "git",
        "repo_root": str(source_root),
        "commit": commit,
        "worktree_status": status,
    }


def _environment(
    device: torch.device,
) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "hostname": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cutlass_dsl": importlib.metadata.version(
            "nvidia-cutlass-dsl",
        ),
        "gpu_name": properties.name,
        "gpu_uuid": _normalise_gpu_uuid(properties.uuid),
        "compute_capability": [
            properties.major,
            properties.minor,
        ],
    }


def _case_result(case: StressCase) -> dict[str, Any]:
    output_mismatches = int(_cpu_scalar(case.output_mismatches))
    state_mismatches = int(_cpu_scalar(case.state_mismatches))
    nonfinite_values = int(_cpu_scalar(case.nonfinite_values))
    input_hashes_after = _input_hashes(case.inputs)
    output_redzones_after = _redzone_hashes(
        case.output_storage,
    )
    state_redzones_after = None if case.state_storage is None else _redzone_hashes(case.state_storage)
    final_output_sha256 = _tensor_sha256(case.output)
    final_state_sha256 = None if case.output_state is None else _tensor_sha256(case.output_state)
    result = {
        "case_id": case.spec.case_id,
        "lengths": list(case.spec.lengths),
        "total_tokens": sum(case.spec.lengths),
        "num_sequences": len(case.spec.lengths),
        "value_heads": case.spec.value_heads,
        "initial_state": case.spec.initial_state,
        "output_final_state": (case.spec.output_final_state),
        "seed": case.spec.seed,
        "stress_launches": case.stress_launches,
        "output_mismatches": output_mismatches,
        "state_mismatches": state_mismatches,
        "nonfinite_values": nonfinite_values,
        "input_immutability": (input_hashes_after == case.input_hashes_before),
        "output_redzones": (output_redzones_after == case.output_redzones_before),
        "state_redzones": (state_redzones_after == case.state_redzones_before),
        "baseline_output_sha256": (case.baseline_output_sha256),
        "final_output_sha256": final_output_sha256,
        "baseline_state_sha256": (case.baseline_state_sha256),
        "final_state_sha256": final_state_sha256,
    }
    result["status"] = (
        "PASS"
        if (
            case.stress_launches > 0
            and output_mismatches == 0
            and state_mismatches == 0
            and nonfinite_values == 0
            and result["input_immutability"]
            and result["output_redzones"]
            and result["state_redzones"]
            and final_output_sha256 == case.baseline_output_sha256
            and final_state_sha256 == case.baseline_state_sha256
        )
        else "FAIL"
    )
    return result


def _write_json(
    path: pathlib.Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("gdn2-sm90-stress.json"),
    )
    parser.add_argument(
        "--source-root",
        type=pathlib.Path,
        default=REPO_ROOT,
    )
    parser.add_argument(
        "--source-manifest",
        type=pathlib.Path,
    )
    parser.add_argument("--required-gpu-uuid")
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--list-matrix", action="store_true")
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.warmup <= 0:
        parser.error("--warmup must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    return args


def _run(args: argparse.Namespace) -> dict[str, Any]:
    started_at_utc = dt.datetime.now(dt.UTC).isoformat()
    source_root = args.source_root.resolve(strict=True)
    source = (
        _git_source_identity(source_root)
        if args.source_manifest is None
        else _verify_source_manifest(
            source_root,
            args.source_manifest.resolve(strict=True),
        )
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    environment = _environment(device)
    if environment["compute_capability"] != [9, 0]:
        raise RuntimeError(
            "GDN2 deterministic stress requires SM90",
        )
    if args.required_gpu_uuid is not None and environment["gpu_uuid"] != args.required_gpu_uuid:
        raise RuntimeError(
            f"GPU UUID mismatch: {environment['gpu_uuid']} != {args.required_gpu_uuid}",
        )
    if get_sm90_gdn2_backend_identity() != SM90_BACKEND_ID:
        raise RuntimeError("product backend identity drift")

    cases = [_build_case(spec, device, args.warmup) for spec in STRESS_MATRIX]
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for iteration in range(args.iterations):
        case = cases[iteration % len(cases)]
        _launch(case)
        _accumulate_exactness(case)
        case.stress_launches += 1
        completed = iteration + 1
        if args.progress_every > 0 and completed % args.progress_every == 0:
            print(
                f"GDN2_SM90_STRESS_PROGRESS completed={completed}/{args.iterations}",
                flush=True,
            )
    torch.cuda.synchronize(device)
    duration_seconds = time.perf_counter() - started
    if get_sm90_gdn2_backend_identity() != SM90_BACKEND_ID:
        raise RuntimeError("product backend identity drift after stress")
    case_results = [_case_result(case) for case in cases]
    status = (
        "PASS"
        if (
            sum(case["stress_launches"] for case in case_results) == args.iterations
            and all(case["status"] == "PASS" for case in case_results)
        )
        else "FAIL"
    )
    return {
        "schema": "cula.gdn2.sm90.deterministic-stress.v1",
        "status": status,
        "started_at_utc": started_at_utc,
        "finished_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "source": source,
        "environment": environment,
        "backend_identity": SM90_BACKEND_ID,
        "fallback": False,
        "protocol": {
            "iterations": args.iterations,
            "warmup_per_case": args.warmup,
            "matrix_rows": len(STRESS_MATRIX),
            "order": "round_robin",
            "preallocated_output_and_state": True,
            "validate_inputs": False,
            "bitwise_output_and_state_check_each_launch": True,
            "finite_check_each_launch": True,
            "host_synchronization_inside_stress_loop": False,
        },
        "duration_seconds": duration_seconds,
        "product_launches": args.iterations,
        "cases": case_results,
        "claim_boundary": (
            "One process, one CUDA device, fixed per-case inputs and "
            "initial states, round-robin public product launches, and "
            "device-side exactness/finite accumulation for every launch."
        ),
    }


def main() -> None:
    args = _parse_args()
    if args.list_matrix:
        print(
            json.dumps(
                [
                    {
                        "case_id": spec.case_id,
                        "lengths": list(spec.lengths),
                        "total_tokens": sum(spec.lengths),
                        "value_heads": spec.value_heads,
                        "initial_state": spec.initial_state,
                        "output_final_state": (spec.output_final_state),
                        "seed": spec.seed,
                    }
                    for spec in STRESS_MATRIX
                ],
                indent=2,
            ),
        )
        return
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"fresh output required: {output}")
    payload = _run(args)
    _write_json(output, payload)
    if payload["status"] != "PASS":
        raise RuntimeError(
            f"GDN2 deterministic stress failed: {output}",
        )
    print(
        "GDN2_SM90_STRESS_PASS "
        f"launches={payload['product_launches']} "
        f"duration_seconds={payload['duration_seconds']:.3f} "
        f"output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
