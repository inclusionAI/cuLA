# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Comparable contiguous, per-document zigzag and adaptive THD layouts."""

from cula._metadata import balanced_shards
from cula.distributed import THDRankLayout, THDSpan, plan_thd_layout, plan_thd_selection, zigzag_thd_layout


def query_layouts(boundaries, world, policy, *, cost_model=None, planner_strategy="balanced"):
    if policy == "planner":
        if planner_strategy == "selection":
            if cost_model is None:
                raise ValueError("selection planning requires calibrated costs")
            source_rows = 4 * ((boundaries[-1] // 4 + world - 1) // world)
            plan = plan_thd_selection(boundaries, world_size=world, cost_model=cost_model, max_tokens_per_rank=2 * source_rows)
        elif planner_strategy == "balanced":
            plan = plan_thd_layout(boundaries, world_size=world, cost_model=cost_model)
        else:
            raise ValueError(f"unknown planner strategy: {planner_strategy}")
        return plan.ranks, plan.policy
    assignments = [[] for _ in range(world)]
    if policy == "contiguous":
        offsets = tuple(n * 4 for n in balanced_shards(boundaries[-1] // 4, world))
        for rank, (left, right) in enumerate(zip(offsets, offsets[1:])):
            for seq, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
                a, z = max(left, start), min(right, end)
                if a < z:
                    assignments[rank].append(THDSpan(seq, a, z, start, end, rank))
    elif policy == "zigzag":
        return zigzag_thd_layout(boundaries, world_size=world), policy
    else:
        raise ValueError(f"unknown layout: {policy}")
    ranks = tuple(
        THDRankLayout(rank, tuple(sorted(spans, key=lambda s: s.token_start))) for rank, spans in enumerate(assignments)
    )
    return ranks, policy
