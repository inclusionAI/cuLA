#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Compare FlashInfer CAKE with cuLA FlashKDA auto intracard CP."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path

import flashinfer.kda_prefill as flashinfer_kda_prefill
import torch
from flashinfer import recurrent_kda

from cula.kda.flashkda import cula_kda_prefill
from cula.ops.kda.cp_mode import CPMode
from cula.ops.kda.sm90.cp.plan import plan_prefill

D = 128


def _parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _middle_half_mean(samples: list[float]) -> float:
    ordered = sorted(samples)
    kept = ordered[len(ordered) // 4 : 3 * len(ordered) // 4]
    return statistics.fmean(kept or ordered)


def _time_round(fn, *, warmup: int, samples: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return _middle_half_mean([start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)])


def _relative_rms(actual: torch.Tensor, reference: torch.Tensor) -> float:
    delta = actual.float() - reference.float()
    return float((delta.square().mean().sqrt() / reference.float().square().mean().sqrt().clamp_min(1e-8)).item())


def _cake_route(device: torch.device, heads: int, length: int) -> str:
    return flashinfer_kda_prefill._select_flash_kda_bf16_route(
        compute_capability=torch.cuda.get_device_capability(device),
        sm_count=torch.cuda.get_device_properties(device).multi_processor_count,
        fixed_layout=True,
        num_sequences=1,
        num_heads=heads,
        uniform_sequences=True,
        max_sequence_length=length,
        use_initial_state=False,
        store_final_state=False,
    )


@torch.inference_mode()
def _benchmark_shape(*, heads: int, length: int, device: torch.device, warmup: int, samples: int, rounds: int):
    generator = torch.Generator(device=device).manual_seed(20260829 + heads * 10007 + length)
    shape = (1, length, heads, D)
    q = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
    k = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
    v = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
    g = (0.1 * torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)).contiguous()
    beta = torch.randn((1, length, heads), dtype=torch.bfloat16, device=device, generator=generator)
    a_log = 0.1 * torch.randn(heads, dtype=torch.float32, device=device, generator=generator)
    dt_bias = 0.1 * torch.randn((heads, D), dtype=torch.float32, device=device, generator=generator)
    cake_out = torch.empty_like(v)
    cula_out = torch.empty_like(v)
    common = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": a_log,
        "dt_bias": dt_bias,
        "scale": D**-0.5,
        "initial_state": None,
        "output_final_state": False,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "lower_bound": -5.0,
    }

    def run_cake():
        return recurrent_kda(**common, output=cake_out, beta_is_logit=True, backend="cake")

    def run_cula():
        return cula_kda_prefill(
            **common,
            out=cula_out,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            use_intracard_cp="auto",
        )

    run_cake()
    run_cula()
    torch.cuda.synchronize()
    plan = plan_prefill([length // 16], heads, device, CPMode.AUTO)
    accuracy = {
        "relative_rms": _relative_rms(cake_out, cula_out),
        "max_abs": float((cake_out.float() - cula_out.float()).abs().max().item()),
    }

    cake_rounds = []
    cula_rounds = []
    for round_index in range(rounds):
        paths = (("cake", run_cake), ("cula", run_cula))
        if round_index % 2:
            paths = tuple(reversed(paths))
        for name, fn in paths:
            elapsed = _time_round(fn, warmup=warmup, samples=samples)
            (cake_rounds if name == "cake" else cula_rounds).append(elapsed)

    cake_ms = statistics.median(cake_rounds)
    cula_ms = statistics.median(cula_rounds)
    return {
        "heads": heads,
        "length": length,
        "cake_route": _cake_route(device, heads, length),
        "cake_ms": cake_ms,
        "flashkda_cp_ms": cula_ms,
        "winner": "cake" if cake_ms < cula_ms else "flashkda_cp",
        "winner_speedup": max(cake_ms, cula_ms) / min(cake_ms, cula_ms),
        "flashkda_cp_active": not plan.trivial,
        "flashkda_segments": plan.n_seg_total,
        "accuracy": accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", default="2,4,8")
    parser.add_argument("--lengths", default="16384,32768,65536,131072")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    results = []
    for heads in _parse_ints(args.heads):
        for length in _parse_ints(args.lengths):
            result = _benchmark_shape(
                heads=heads,
                length=length,
                device=device,
                warmup=args.warmup,
                samples=args.samples,
                rounds=args.rounds,
            )
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
            gc.collect()
            torch.cuda.empty_cache()

    properties = torch.cuda.get_device_properties(device)
    report = {
        "environment": {
            "gpu": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "sm_count": properties.multi_processor_count,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "settings": {
            "batch": 1,
            "head_dim": D,
            "dtype": "bfloat16",
            "warmup": args.warmup,
            "samples": args.samples,
            "rounds": args.rounds,
            "output_final_state": False,
            "preallocated_output": True,
        },
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
