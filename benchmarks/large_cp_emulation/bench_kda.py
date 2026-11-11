# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Emulate a large logical Kimi CP group on fewer physical GPUs.

The benchmark executes the complete cuLA KDA path: local summary, boundary
transport and cuDNN-composed apply, local forward, and distributed backward. It maps one
representative logical rank to each physical process and runs ``CP - 1`` real
NCCL ring hops per direction. This is a critical-path emulator, not a claim
about real CP64 topology or contention.

Example:
  torchrun --standalone --nproc-per-node=4 benchmarks/large_cp_emulation/bench_kda.py
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from benchmarks.large_cp_emulation.topology import LogicalCPEmulationPlan  # noqa: E402
from benchmarks.large_cp_emulation.transport import LogicalCPRingTransport  # noqa: E402
from cula.distributed import KimiCPBoundaryTransport, cudnn_kda_cp  # noqa: E402


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _run(args) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size < 2:
        raise ValueError("logical CP emulation requires at least two physical processes")
    emulation = LogicalCPEmulationPlan.build(
        [0, args.global_tokens],
        logical_cp=args.logical_cp,
        physical_rank=rank,
        physical_world_size=world_size,
    )
    local_tokens = emulation.logical_plan.local_tokens
    logical_rank = emulation.logical_rank
    torch.manual_seed(args.seed + logical_rank)

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, dtype=dtype, device=device)

    q = torch.nn.functional.normalize(randn(local_tokens, args.heads, args.key_dim).float(), dim=-1).to(torch.bfloat16)
    k = torch.nn.functional.normalize(randn(local_tokens, args.heads, args.key_dim).float(), dim=-1).to(torch.bfloat16)
    v = randn(local_tokens, args.heads, args.value_dim)
    g = -(0.01 + 0.04 * torch.rand(local_tokens, args.heads, args.key_dim, dtype=torch.float32, device=device))
    beta = torch.sigmoid(randn(local_tokens, args.heads, dtype=torch.float32))
    grad_output = (randn(local_tokens, args.heads, args.value_dim) / float(args.global_tokens)).contiguous()

    plan = emulation.runtime_plan
    cu_seqlens = torch.tensor(plan.local_cu_seqlens, dtype=torch.int32, device=device)
    transport = LogicalCPRingTransport(
        args.logical_cp,
        rank,
        world_size,
        ring_hops=emulation.transport_hops,
    )
    assert isinstance(transport, KimiCPBoundaryTransport)

    def execute(measure: bool):
        inputs = [tensor.detach().requires_grad_(True) for tensor in (q, k, v, g, beta)]
        transport.clear_events()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if measure:
            start.record()
        output = cudnn_kda_cp(
            *inputs,
            cu_seqlens,
            plan=plan,
            group=dist.group.WORLD,
            boundary_transport=transport,
        )
        output.backward(grad_output)
        if measure:
            end.record()
            end.synchronize()
        else:
            torch.cuda.synchronize(device)
        transport_times = {}
        if measure:
            for direction, begin, after_comm, after_prepare in transport.events:
                transport_times[f"comm_{direction}_ms"] = begin.elapsed_time(after_comm)
                transport_times[f"prepare_{direction}_ms"] = after_comm.elapsed_time(after_prepare)
        return {
            "cuda_total_ms": start.elapsed_time(end) if measure else None,
            "transport_ms": transport_times,
            "proxy_loss": (output.float() * grad_output.float()).sum().item(),
            "output_norm": output.float().norm().item(),
            "grad_norms": [tensor.grad.float().norm().item() for tensor in inputs],
            "finite": bool(torch.isfinite(output).all()) and all(bool(torch.isfinite(tensor.grad).all()) for tensor in inputs),
        }

    dist.barrier(device_ids=[local_rank])
    execute(False)
    dist.barrier(device_ids=[local_rank])
    torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(args.repeats):
        dist.barrier(device_ids=[local_rank])
        wall_start = time.perf_counter()
        sample = execute(True)
        sample["wall_ms"] = (time.perf_counter() - wall_start) * 1000
        samples.append(sample)

    record = {
        "physical_rank": rank,
        "logical_rank": logical_rank,
        "samples": samples,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "transport_workspace_gib": transport.workspace_bytes / 2**30,
    }
    records = [None] * world_size
    dist.all_gather_object(records, record)
    if rank == 0:
        critical_cuda = [
            max(records[r]["samples"][i]["cuda_total_ms"] for r in range(world_size)) for i in range(args.repeats)
        ]
        critical_wall = [max(records[r]["samples"][i]["wall_ms"] for r in range(world_size)) for i in range(args.repeats)]
        print(
            json.dumps(
                {
                    "mode": "kda_logical_cp_e2e",
                    "logical_global_tokens": args.global_tokens,
                    "logical_cp": args.logical_cp,
                    "local_tokens": local_tokens,
                    "physical_gpus": world_size,
                    "shape": {
                        "heads": args.heads,
                        "key_dim": args.key_dim,
                        "value_dim": args.value_dim,
                    },
                    "critical_cuda_samples_ms": critical_cuda,
                    "critical_cuda_median_ms": statistics.median(critical_cuda),
                    "critical_wall_samples_ms": critical_wall,
                    "critical_wall_median_ms": statistics.median(critical_wall),
                    "records": records,
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-tokens", type=_positive_int, default=1 << 20)
    parser.add_argument("--logical-cp", type=_positive_int, default=64)
    parser.add_argument("--heads", type=_positive_int, default=32)
    parser.add_argument("--key-dim", type=_positive_int, default=128)
    parser.add_argument("--value-dim", type=_positive_int, default=128)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--seed", type=int, default=20261102)
    return parser.parse_args()


def main() -> None:
    _run(_parse_args())


if __name__ == "__main__":
    main()
