# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Emulate large logical CP selection and ring communication on fewer GPUs.

DSA mode executes one logical rank's real query/block geometry. Ring mode must
be launched with torchrun and sends the real per-rank payload for ``CP - 1``
hops. It measures the serial dependency path, not a claim about 64-GPU fabric
contention or topology.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
from functools import partial

import torch
import torch.distributed as dist

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from benchmarks.large_cp_emulation.transport import emulate_ring_hops  # noqa: E402
from cula.hybrid import DsaSelectionPlan, DsaSelectionWorkspace, cudnn_dsa_select  # noqa: E402


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("all values must be positive")
    return result


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _measure(callable_, repeats: int) -> list[float]:
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        callable_()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return values


def _candidate_prefix(position: int, block_size: int) -> int:
    quotient, remainder = divmod(position, block_size)
    return block_size * quotient * (quotient - 1) // 2 + quotient * (remainder + 1)


def run_dsa(args) -> None:
    if args.global_tokens % args.logical_cp:
        raise ValueError("global_tokens must be divisible by logical_cp")
    if args.global_tokens % args.block_size:
        raise ValueError("global_tokens must be divisible by block_size")
    local_tokens = args.global_tokens // args.logical_cp
    logical_rank = args.logical_rank if args.logical_rank is not None else args.logical_cp - 1
    if not 0 <= logical_rank < args.logical_cp:
        raise ValueError("logical_rank must be in [0, logical_cp)")
    blocks = args.global_tokens // args.block_size
    query_offset = logical_rank * local_tokens
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    qi = torch.randn(local_tokens, args.index_heads, args.head_dim, dtype=torch.bfloat16, device=device)
    kbar = torch.randn(blocks, 1, args.head_dim, dtype=torch.bfloat16, device=device)
    weights = torch.randn(local_tokens, args.index_heads, dtype=torch.float32, device=device)
    full_floats = _candidate_prefix(query_offset + local_tokens, args.block_size) - _candidate_prefix(
        query_offset, args.block_size
    )
    print(
        json.dumps(
            {
                "mode": "dsa",
                "global_tokens": args.global_tokens,
                "logical_cp": args.logical_cp,
                "logical_rank": logical_rank,
                "local_tokens": local_tokens,
                "index_heads": args.index_heads,
                "compressed_blocks": blocks,
                "full_candidate_gib": full_floats * 4 / 2**30,
            }
        ),
        flush=True,
    )

    for budget_mib in args.budgets_mib:
        plan = DsaSelectionPlan.from_cu_seqlens(
            [0, local_tokens],
            [0, blocks],
            top_k=args.top_k,
            block_size=args.block_size,
            q_causal_offsets=[query_offset],
            candidate_budget_bytes=budget_mib << 20,
        )
        workspace = DsaSelectionWorkspace.from_plan(plan, device)

        execute = partial(cudnn_dsa_select, qi, kbar, weights, plan=plan, workspace=workspace)

        execute()
        torch.cuda.synchronize(device)
        elapsed = _measure(execute, args.repeats)
        print(
            json.dumps(
                {
                    "index_heads": args.index_heads,
                    "budget_mib": budget_mib,
                    "windows": len(plan.windows),
                    "workspace_mib": plan.candidate_workspace_floats * 4 / 2**20,
                    "samples_ms": elapsed,
                    "median_ms": statistics.median(elapsed),
                }
            ),
            flush=True,
        )
        del execute, workspace
        torch.cuda.empty_cache()


def run_ring(args) -> None:
    if args.logical_cp < 2:
        raise ValueError("ring emulation requires logical_cp >= 2")
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    steps = args.logical_cp - 1
    elements = (args.payload_bytes + 3) // 4
    send = torch.randn(elements, dtype=torch.float32, device=device)
    receive = torch.empty_like(send)

    def execute():
        emulate_ring_hops(
            send,
            receive,
            physical_rank=rank,
            physical_world_size=world_size,
            hops=steps,
        )

    execute()
    torch.cuda.synchronize(device)
    dist.barrier(device_ids=[local_rank])
    elapsed = _measure(execute, args.repeats)
    local = torch.tensor(elapsed, dtype=torch.float64, device=device)
    gathered = torch.empty(world_size * args.repeats, dtype=torch.float64, device=device)
    all_gather_single = dist.all_gather_single if hasattr(dist, "all_gather_single") else dist.all_gather_into_tensor
    all_gather_single(gathered, local)
    if rank == 0:
        samples = gathered.cpu().reshape(world_size, args.repeats)
        rank_medians = [statistics.median(row.tolist()) for row in samples]
        critical_samples = samples.max(dim=0).values.tolist()
        critical = statistics.median(critical_samples)
        print(
            json.dumps(
                {
                    "mode": "ring",
                    "physical_gpus": world_size,
                    "logical_cp": args.logical_cp,
                    "steps": steps,
                    "payload_bytes": args.payload_bytes,
                    "rank_median_ms": rank_medians,
                    "critical_samples_ms": critical_samples,
                    "critical_ms": critical,
                    "per_hop_us": critical * 1000 / steps,
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    dsa = subparsers.add_parser("dsa")
    dsa.add_argument("--global-tokens", type=_positive_int, default=1 << 20)
    dsa.add_argument("--logical-cp", type=_positive_int, default=64)
    dsa.add_argument("--logical-rank", type=int)
    dsa.add_argument("--budgets-mib", type=_csv_ints, default=(512, 1024, 2048, 4096))
    dsa.add_argument("--block-size", type=_positive_int, default=4)
    dsa.add_argument("--top-k", type=_positive_int, default=512)
    dsa.add_argument("--index-heads", type=_positive_int, default=64)
    dsa.add_argument("--head-dim", type=_positive_int, default=128)
    dsa.add_argument("--device", type=int, default=0)
    dsa.add_argument("--repeats", type=_positive_int, default=5)
    dsa.add_argument("--seed", type=int, default=53)
    ring = subparsers.add_parser("ring")
    ring.add_argument("--logical-cp", type=_positive_int, default=64)
    ring.add_argument("--payload-bytes", type=_positive_int, default=4 << 20)
    ring.add_argument("--repeats", type=_positive_int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "dsa":
        run_dsa(args)
    else:
        run_ring(args)


if __name__ == "__main__":
    main()
