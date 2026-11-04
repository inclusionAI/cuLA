# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Measure DSA indexer cost under contiguous and sample-alternating CP layouts."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from benchmarks.large_cp_emulation.query_layout import (  # noqa: E402
    QueryLayout,
    build_alternating_sequence_query_layout,
    build_contiguous_query_layout,
)
from benchmarks.large_cp_emulation.workload import PackedTHDWorkload  # noqa: E402
from cula.hybrid import (  # noqa: E402
    DsaSelectionPlan,
    DsaSelectionWorkspace,
    IndexPoolPlan,
    cudnn_dsa_select,
    cudnn_index_pool,
)


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _global_block_prefix(boundaries: tuple[int, ...], block_size: int) -> tuple[int, ...]:
    result = [0]
    for left, right in zip(boundaries, boundaries[1:]):
        result.append(result[-1] + (right - left) // block_size)
    return tuple(result)


def _layout_for_rank(
    name: str,
    boundaries: tuple[int, ...],
    *,
    rank: int,
    world_size: int,
    block_size: int,
) -> QueryLayout:
    builder = (
        build_contiguous_query_layout
        if name == "contiguous"
        else build_alternating_sequence_query_layout
    )
    return builder(
        boundaries,
        rank=rank,
        world_size=world_size,
        block_size=block_size,
    )


def _block_ids(layout: QueryLayout, block_size: int) -> tuple[int, ...]:
    return tuple(
        block
        for span in layout.spans
        for block in range(span.token_start // block_size, span.token_end // block_size)
    )


def _selection_key_ids(
    layout: QueryLayout,
    block_prefix: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        block
        for span in layout.spans
        for block in range(block_prefix[span.sequence], block_prefix[span.sequence + 1])
    )


def _run(args: argparse.Namespace) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    workload = PackedTHDWorkload.short_long_mix(
        args.global_tokens,
        short_sequences=args.short_sequences,
        short_tokens=args.short_tokens,
        long_sequences=args.long_sequences,
    )
    boundaries = workload.global_cu_seqlens
    layouts = tuple(
        _layout_for_rank(
            args.layout,
            boundaries,
            rank=owner,
            world_size=world_size,
            block_size=args.block_size,
        )
        for owner in range(world_size)
    )
    layout = layouts[rank]
    local_tokens = layout.local_tokens
    local_blocks = local_tokens // args.block_size
    if any(candidate.local_tokens != local_tokens for candidate in layouts):
        raise ValueError("this benchmark requires equal tokens per physical rank")
    if any(len(_block_ids(candidate, args.block_size)) != local_blocks for candidate in layouts):
        raise ValueError("this benchmark requires equal pooled blocks per physical rank")

    block_prefix = _global_block_prefix(boundaries, args.block_size)
    gathered_block_ids = torch.tensor(
        [block for candidate in layouts for block in _block_ids(candidate, args.block_size)],
        dtype=torch.int64,
        device=device,
    )
    selection_key_ids = torch.tensor(
        _selection_key_ids(layout, block_prefix),
        dtype=torch.int64,
        device=device,
    )
    select_cu_k = [0]
    for span in layout.spans:
        select_cu_k.append(
            select_cu_k[-1] + block_prefix[span.sequence + 1] - block_prefix[span.sequence]
        )

    pool_plan = IndexPoolPlan.from_cu_seqlens(
        layout.local_cu_seqlens,
        head_dim=args.index_dim,
        ratio=args.block_size,
    )
    pool_cu_tokens, pool_cu_blocks = pool_plan.materialize(device)
    budget = None if args.dsa_budget_mib == 0 else args.dsa_budget_mib << 20
    select_plan = DsaSelectionPlan.from_cu_seqlens(
        layout.local_cu_seqlens,
        select_cu_k,
        top_k=args.top_k,
        block_size=args.block_size,
        q_causal_offsets=layout.causal_offsets,
        candidate_budget_bytes=budget,
    )
    workspace = DsaSelectionWorkspace.from_plan(select_plan, device)

    torch.manual_seed(args.seed + rank)
    qi = torch.randn(local_tokens, args.index_heads, args.index_dim, dtype=torch.bfloat16, device=device)
    local_kv = torch.randn(local_tokens, args.index_dim, dtype=torch.bfloat16, device=device)
    pool_score = torch.randn_like(local_kv)
    ape = torch.randn(args.block_size, args.index_dim, dtype=torch.float32, device=device)
    weights = torch.randn(local_tokens, args.index_heads, dtype=torch.float32, device=device)
    gathered_kbar = torch.empty(
        world_size * local_blocks,
        args.index_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    global_kbar = torch.empty(
        boundaries[-1] // args.block_size,
        args.index_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    selection_kbar = torch.empty(
        selection_key_ids.numel(),
        1,
        args.index_dim,
        dtype=torch.bfloat16,
        device=device,
    )

    def execute(measure: bool) -> dict[str, object]:
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        events[0].record()
        local_kbar = cudnn_index_pool(
            local_kv,
            pool_score,
            ape,
            pool_cu_tokens,
            pool_cu_blocks,
            plan=pool_plan,
            deterministic=True,
        )
        events[1].record()
        if hasattr(dist, "all_gather_single"):
            dist.all_gather_single(gathered_kbar, local_kbar)
        else:  # pragma: no cover - compatibility with older PyTorch releases
            dist.all_gather_into_tensor(gathered_kbar, local_kbar)
        global_kbar.index_copy_(0, gathered_block_ids, gathered_kbar)
        torch.index_select(global_kbar, 0, selection_key_ids, out=selection_kbar[:, 0])
        events[2].record()
        result = cudnn_dsa_select(
            qi,
            selection_kbar,
            weights,
            plan=select_plan,
            workspace=workspace,
            deterministic=True,
        )
        events[3].record()
        events[3].synchronize()
        return {
            "total_ms": events[0].elapsed_time(events[3]) if measure else None,
            "pool_ms": events[0].elapsed_time(events[1]) if measure else None,
            "remap_comm_ms": events[1].elapsed_time(events[2]) if measure else None,
            "select_ms": events[2].elapsed_time(events[3]) if measure else None,
            "id_sum": int(result.block_ids.to(torch.int64).sum().item()),
            "score_sum": float(result.scores[torch.isfinite(result.scores)].double().sum().item()),
        }

    for _ in range(args.warmups):
        execute(False)
    torch.cuda.reset_peak_memory_stats(device)
    samples = [execute(True) for _ in range(args.repeats)]
    record = {
        "rank": rank,
        "tokens": local_tokens,
        "spans": len(layout.spans),
        "candidate_floats": layout.candidate_floats,
        "windows": len(select_plan.windows),
        "candidate_workspace_gib": select_plan.candidate_workspace_floats * 4 / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "samples": samples,
    }
    records = [None] * world_size
    dist.all_gather_object(records, record)
    if rank == 0:
        critical = [
            max(records[owner]["samples"][sample]["total_ms"] for owner in range(world_size))
            for sample in range(args.repeats)
        ]
        print(
            json.dumps(
                {
                    "layout": args.layout,
                    "global_tokens": args.global_tokens,
                    "physical_gpus": world_size,
                    "sequence_lengths": workload.sequence_lengths,
                    "critical_samples_ms": critical,
                    "critical_median_ms": statistics.median(critical),
                    "ranks": [
                        {
                            key: record[key]
                            for key in (
                                "rank",
                                "tokens",
                                "spans",
                                "candidate_floats",
                                "windows",
                                "candidate_workspace_gib",
                                "peak_allocated_gib",
                            )
                        }
                        | {
                            f"{stage}_median_ms": statistics.median(
                                sample[stage] for sample in record["samples"]
                            )
                            for stage in ("pool_ms", "remap_comm_ms", "select_ms", "total_ms")
                        }
                        | {
                            "id_sum_stable": all(
                                sample["id_sum"] == record["samples"][0]["id_sum"]
                                for sample in record["samples"]
                            ),
                            "score_sum_span": max(
                                sample["score_sum"] for sample in record["samples"]
                            )
                            - min(sample["score_sum"] for sample in record["samples"]),
                        }
                        for record in records
                    ],
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=("contiguous", "alternating"), required=True)
    parser.add_argument("--global-tokens", type=_positive_int, default=1 << 20)
    parser.add_argument("--short-sequences", type=_positive_int, default=20)
    parser.add_argument("--short-tokens", type=_positive_int, default=1024)
    parser.add_argument("--long-sequences", type=_positive_int, default=2)
    parser.add_argument("--block-size", type=_positive_int, default=4)
    parser.add_argument("--top-k", type=_positive_int, default=512)
    parser.add_argument("--index-heads", type=_positive_int, default=4)
    parser.add_argument("--index-dim", type=_positive_int, default=128)
    parser.add_argument("--dsa-budget-mib", type=int, default=8192)
    parser.add_argument("--warmups", type=_positive_int, default=2)
    parser.add_argument("--repeats", type=_positive_int, default=5)
    parser.add_argument("--seed", type=int, default=20261104)
    return parser.parse_args()


if __name__ == "__main__":
    _run(_parse_args())
