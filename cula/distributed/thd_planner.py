# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only load planner for packed THD context parallelism.

The planner balances two independent resources: token-linear work used by KDA
and position-dependent causal candidate work used by CSA.  It deliberately
does not inspect CUDA tensors or kernel state, so one immutable plan can be
built at microbatch ingress and reused by the complete model stack.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
from operator import index

import torch

from cula._metadata import as_boundaries, balanced_shards


@dataclass(frozen=True, slots=True)
class THDSpan:
    """One contiguous interval of a global packed sequence."""

    sequence: int
    token_start: int
    token_end: int
    sequence_start: int
    sequence_end: int
    owner: int

    @property
    def tokens(self) -> int:
        return self.token_end - self.token_start

    @property
    def causal_offset(self) -> int:
        return self.token_start - self.sequence_start


@dataclass(frozen=True, slots=True)
class THDRankLayout:
    rank: int
    spans: tuple[THDSpan, ...]

    @property
    def local_tokens(self) -> int:
        return sum(span.tokens for span in self.spans)

    @property
    def local_cu_seqlens(self) -> tuple[int, ...]:
        result = [0]
        for span in self.spans:
            result.append(result[-1] + span.tokens)
        return tuple(result)

    @property
    def causal_offsets(self) -> tuple[int, ...]:
        return tuple(span.causal_offset for span in self.spans)


@dataclass(frozen=True, slots=True)
class THDLayoutMetrics:
    token_loads: tuple[int, ...]
    causal_loads: tuple[int, ...]
    boundary_loads: tuple[int, ...]
    token_imbalance: float
    causal_imbalance: float
    score: float


@dataclass(frozen=True, slots=True)
class THDLayoutPlan:
    """Complete global span assignment selected by :func:`plan_thd_layout`."""

    world_size: int
    block_size: int
    policy: str
    ranks: tuple[THDRankLayout, ...]
    metrics: THDLayoutMetrics
    evaluated_scores: tuple[tuple[str, float], ...]

    def for_rank(self, rank: int) -> THDRankLayout:
        if not isinstance(rank, int) or not 0 <= rank < self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size})")
        return self.ranks[rank]


@dataclass(frozen=True, slots=True)
class THDPlannerConfig:
    """Shape-independent planner controls.

    ``max_chunks_per_rank`` controls the finest evaluated planning grid.  The
    planner compares every grid from one through this value, so a dominant
    causal sequence can form an early/late zigzag without forcing the same
    fragmentation on batches whose long sequences already balance each other.
    """

    block_size: int = 4
    min_split_tokens: int = 16 << 10
    max_chunks_per_rank: int = 6
    token_weight: float = 1.0
    causal_weight: float = 1.0
    boundary_weight: float = 0.02


@dataclass(frozen=True, slots=True)
class _PlanningItem:
    sequence: int
    token_start: int
    token_end: int
    sequence_start: int
    sequence_end: int
    causal_work: int

    @property
    def tokens(self) -> int:
        return self.token_end - self.token_start


def _candidate_prefix(position: int, block_size: int) -> int:
    quotient, remainder = divmod(position, block_size)
    return block_size * quotient * (quotient - 1) // 2 + quotient * (remainder + 1)


def _candidate_work(start: int, tokens: int, block_size: int) -> int:
    return _candidate_prefix(start + tokens, block_size) - _candidate_prefix(start, block_size)


def _validate_inputs(
    global_cu_seqlens: torch.Tensor | Sequence[int],
    *,
    world_size: int,
    config: THDPlannerConfig,
) -> tuple[int, ...]:
    boundaries = as_boundaries(global_cu_seqlens, name="global_cu_seqlens")
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    if boundaries[-1] < world_size:
        raise ValueError("packed context parallelism requires at least one token per rank")
    for name in ("block_size", "min_split_tokens", "max_chunks_per_rank"):
        raw_value = getattr(config, name)
        if isinstance(raw_value, bool):
            raise TypeError(f"{name} must be an integer")
        value = index(raw_value)
        if value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config.min_split_tokens % config.block_size:
        raise ValueError("min_split_tokens must be block aligned")
    if any((right - left) % config.block_size for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("every sequence length must be block aligned")
    for name in ("token_weight", "causal_weight", "boundary_weight"):
        value = getattr(config, name)
        if not isinstance(value, int | float) or value < 0:
            raise ValueError(f"{name} must be non-negative")
    if config.token_weight == 0 and config.causal_weight == 0:
        raise ValueError("at least one compute weight must be positive")
    return boundaries


def _make_item(
    sequence: int,
    start: int,
    end: int,
    sequence_start: int,
    sequence_end: int,
    block_size: int,
) -> _PlanningItem:
    return _PlanningItem(
        sequence,
        start,
        end,
        sequence_start,
        sequence_end,
        _candidate_work(start - sequence_start, end - start, block_size),
    )


def _contiguous_items(boundaries: tuple[int, ...], world_size: int, block_size: int) -> list[tuple[_PlanningItem, int]]:
    total_blocks = boundaries[-1] // block_size
    block_shards = balanced_shards(total_blocks, world_size)
    shard_offsets = tuple(value * block_size for value in block_shards)
    result = []
    for rank, (shard_start, shard_end) in enumerate(zip(shard_offsets, shard_offsets[1:])):
        for sequence, (sequence_start, sequence_end) in enumerate(zip(boundaries, boundaries[1:])):
            start = max(shard_start, sequence_start)
            end = min(shard_end, sequence_end)
            if start < end:
                result.append(
                    (_make_item(sequence, start, end, sequence_start, sequence_end, block_size), rank)
                )
    return result


def _adaptive_items(
    boundaries: tuple[int, ...],
    world_size: int,
    config: THDPlannerConfig,
    *,
    chunks_per_rank: int,
) -> list[_PlanningItem]:
    block_size = config.block_size
    total_blocks = boundaries[-1] // block_size
    requested_grid = world_size * chunks_per_rank
    target_blocks = max(1, ceil(total_blocks / requested_grid))
    min_split_blocks = config.min_split_tokens // block_size
    result = []
    for sequence, (sequence_start, sequence_end) in enumerate(zip(boundaries, boundaries[1:])):
        length_blocks = (sequence_end - sequence_start) // block_size
        pieces = min(requested_grid, max(1, ceil(length_blocks / target_blocks)))
        # A sequence shorter than two useful spans stays whole.  This is what
        # prevents short THD samples from paying a max-CP fragmentation tax.
        if length_blocks < 2 * min_split_blocks:
            pieces = 1
        offsets = balanced_shards(length_blocks, pieces)
        for left, right in zip(offsets, offsets[1:]):
            start = sequence_start + left * block_size
            end = sequence_start + right * block_size
            result.append(
                _make_item(sequence, start, end, sequence_start, sequence_end, block_size)
            )

    # Keep all ranks non-empty even for small batches containing fewer than W
    # sequences.  The minimum span is a performance preference, not a semantic
    # restriction, so only this feasibility repair may split below it.
    while len(result) < world_size:
        splittable = [item for item in result if item.tokens >= 2 * block_size]
        if not splittable:
            raise ValueError("block alignment leaves fewer non-empty spans than ranks")
        item = max(splittable, key=lambda candidate: (candidate.tokens, candidate.causal_work))
        result.remove(item)
        blocks = item.tokens // block_size
        middle = item.token_start + (blocks // 2) * block_size
        result.extend(
            (
                _make_item(
                    item.sequence,
                    item.token_start,
                    middle,
                    item.sequence_start,
                    item.sequence_end,
                    block_size,
                ),
                _make_item(
                    item.sequence,
                    middle,
                    item.token_end,
                    item.sequence_start,
                    item.sequence_end,
                    block_size,
                ),
            )
        )
    return result


def _greedy_assign(items: list[_PlanningItem], world_size: int) -> list[tuple[_PlanningItem, int]]:
    total_tokens = sum(item.tokens for item in items)
    total_causal = sum(item.causal_work for item in items)
    ideal_tokens = total_tokens / world_size
    ideal_causal = total_causal / world_size
    token_loads = [0] * world_size
    causal_loads = [0] * world_size
    item_counts = [0] * world_size

    ordered = sorted(
        items,
        key=lambda item: (
            max(item.tokens / ideal_tokens, item.causal_work / ideal_causal),
            item.causal_work,
            item.tokens,
            -item.sequence,
            -item.token_start,
        ),
        reverse=True,
    )
    result = []
    for item in ordered:
        owner = min(
            range(world_size),
            key=lambda rank: (
                max(
                    (token_loads[rank] + item.tokens) / ideal_tokens,
                    (causal_loads[rank] + item.causal_work) / ideal_causal,
                ),
                (token_loads[rank] + item.tokens) / ideal_tokens
                + (causal_loads[rank] + item.causal_work) / ideal_causal,
                item_counts[rank],
                rank,
            ),
        )
        result.append((item, owner))
        token_loads[owner] += item.tokens
        causal_loads[owner] += item.causal_work
        item_counts[owner] += 1
    return result


def _split_item(
    item: _PlanningItem,
    tokens: int,
    *,
    take_prefix: bool,
    block_size: int,
) -> tuple[_PlanningItem, _PlanningItem]:
    if tokens <= 0 or tokens >= item.tokens or tokens % block_size:
        raise ValueError("split size must be a non-empty block-aligned item prefix or suffix")
    middle = item.token_start + tokens if take_prefix else item.token_end - tokens
    left = _make_item(
        item.sequence,
        item.token_start,
        middle,
        item.sequence_start,
        item.sequence_end,
        block_size,
    )
    right = _make_item(
        item.sequence,
        middle,
        item.token_end,
        item.sequence_start,
        item.sequence_end,
        block_size,
    )
    return (left, right) if take_prefix else (right, left)


def _rebalance_tokens(
    assignments: list[tuple[_PlanningItem, int]],
    *,
    world_size: int,
    block_size: int,
) -> list[tuple[_PlanningItem, int]]:
    """Repair a compute-balanced assignment to exact balanced token bins."""

    total_blocks = sum(item.tokens for item, _ in assignments) // block_size
    capacity_offsets = balanced_shards(total_blocks, world_size)
    capacities = [
        (right - left) * block_size
        for left, right in zip(capacity_offsets, capacity_offsets[1:])
    ]
    loads = [0] * world_size
    causal_loads = [0] * world_size
    for item, owner in assignments:
        loads[owner] += item.tokens
        causal_loads[owner] += item.causal_work
    ideal_causal = sum(causal_loads) / world_size
    repaired = list(assignments)

    while loads != capacities:
        receivers = [rank for rank in range(world_size) if loads[rank] < capacities[rank]]
        donors = [rank for rank in range(world_size) if loads[rank] > capacities[rank]]
        if not receivers or not donors:
            raise RuntimeError("internal error while repairing THD token capacities")
        receiver = max(receivers, key=lambda rank: (capacities[rank] - loads[rank], -rank))
        donor = max(donors, key=lambda rank: (loads[rank] - capacities[rank], -rank))
        transfer_limit = min(
            capacities[receiver] - loads[receiver],
            loads[donor] - capacities[donor],
        )

        choices = []
        for assignment_index, (item, owner) in enumerate(repaired):
            if owner != donor:
                continue
            transfer_tokens = min(item.tokens, transfer_limit)
            if transfer_tokens == item.tokens:
                variants = ((item, None),)
            else:
                variants = tuple(
                    _split_item(
                        item,
                        transfer_tokens,
                        take_prefix=take_prefix,
                        block_size=block_size,
                    )
                    for take_prefix in (True, False)
                )
            for moved, remainder in variants:
                next_donor_causal = causal_loads[donor] - moved.causal_work
                next_receiver_causal = causal_loads[receiver] + moved.causal_work
                causal_pressure = max(
                    next_donor_causal / ideal_causal,
                    next_receiver_causal / ideal_causal,
                    *(causal_loads[rank] / ideal_causal for rank in range(world_size) if rank not in (donor, receiver)),
                )
                choices.append(
                    (
                        causal_pressure,
                        remainder is not None,
                        -moved.tokens,
                        moved.sequence,
                        moved.token_start,
                        assignment_index,
                        moved,
                        remainder,
                    )
                )
        if not choices:
            raise RuntimeError("internal error: an overloaded rank has no movable THD span")
        *_, assignment_index, moved, remainder = min(choices)
        repaired.pop(assignment_index)
        if remainder is not None:
            repaired.append((remainder, donor))
        repaired.append((moved, receiver))
        loads[donor] -= moved.tokens
        loads[receiver] += moved.tokens
        causal_loads[donor] -= moved.causal_work
        causal_loads[receiver] += moved.causal_work
    return repaired


def _coalesce(assignments: list[tuple[_PlanningItem, int]]) -> list[THDSpan]:
    ordered = sorted(assignments, key=lambda pair: (pair[0].sequence, pair[0].token_start))
    result: list[THDSpan] = []
    for item, owner in ordered:
        if (
            result
            and result[-1].sequence == item.sequence
            and result[-1].owner == owner
            and result[-1].token_end == item.token_start
        ):
            previous = result.pop()
            result.append(
                THDSpan(
                    previous.sequence,
                    previous.token_start,
                    item.token_end,
                    previous.sequence_start,
                    previous.sequence_end,
                    owner,
                )
            )
        else:
            result.append(
                THDSpan(
                    item.sequence,
                    item.token_start,
                    item.token_end,
                    item.sequence_start,
                    item.sequence_end,
                    owner,
                )
            )
    return result


def _metrics(
    spans: list[THDSpan],
    *,
    world_size: int,
    block_size: int,
    config: THDPlannerConfig,
) -> THDLayoutMetrics:
    token_loads = [0] * world_size
    causal_loads = [0] * world_size
    boundary_loads = [0] * world_size
    by_sequence: dict[int, list[THDSpan]] = defaultdict(list)
    for span in spans:
        token_loads[span.owner] += span.tokens
        causal_loads[span.owner] += _candidate_work(span.causal_offset, span.tokens, block_size)
        by_sequence[span.sequence].append(span)
    for sequence_spans in by_sequence.values():
        sequence_spans.sort(key=lambda span: span.token_start)
        for left, right in zip(sequence_spans, sequence_spans[1:]):
            if left.owner != right.owner:
                boundary_loads[left.owner] += 1
                boundary_loads[right.owner] += 1

    def imbalance(loads: list[int]) -> float:
        average = sum(loads) / world_size
        return max(loads) / average if average else 1.0

    token_imbalance = imbalance(token_loads)
    causal_imbalance = imbalance(causal_loads)
    score = max(
        config.token_weight * token_imbalance,
        config.causal_weight * causal_imbalance,
    ) + config.boundary_weight * max(boundary_loads, default=0)
    return THDLayoutMetrics(
        tuple(token_loads),
        tuple(causal_loads),
        tuple(boundary_loads),
        token_imbalance,
        causal_imbalance,
        score,
    )


def _build_candidate(
    assignments: list[tuple[_PlanningItem, int]],
    *,
    policy: str,
    world_size: int,
    config: THDPlannerConfig,
) -> tuple[str, list[THDSpan], THDLayoutMetrics] | None:
    spans = _coalesce(assignments)
    metrics = _metrics(
        spans,
        world_size=world_size,
        block_size=config.block_size,
        config=config,
    )
    if any(load == 0 for load in metrics.token_loads):
        return None
    return policy, spans, metrics


def plan_thd_layout(
    global_cu_seqlens: torch.Tensor | Sequence[int],
    *,
    world_size: int,
    config: THDPlannerConfig | None = None,
) -> THDLayoutPlan:
    """Plan a block-aligned THD layout and select the best modeled candidate.

    The result is guaranteed to have a score no worse than the evaluated
    contiguous baseline under the declared cost model.  This is a model-level
    guarantee, not a claim that an analytic proxy exactly predicts every GPU.
    """

    config = THDPlannerConfig() if config is None else config
    if not isinstance(config, THDPlannerConfig):
        raise TypeError("config must be a THDPlannerConfig")
    boundaries = _validate_inputs(global_cu_seqlens, world_size=world_size, config=config)

    contiguous = _build_candidate(
        _contiguous_items(boundaries, world_size, config.block_size),
        policy="contiguous",
        world_size=world_size,
        config=config,
    )
    adaptive_candidates = []
    seen_assignments = set()
    for chunks_per_rank in range(1, config.max_chunks_per_rank + 1):
        adaptive_items = _adaptive_items(
            boundaries,
            world_size,
            config,
            chunks_per_rank=chunks_per_rank,
        )
        adaptive = _build_candidate(
            _rebalance_tokens(
                _greedy_assign(adaptive_items, world_size),
                world_size=world_size,
                block_size=config.block_size,
            ),
            policy=f"cost-balanced-{chunks_per_rank}x",
            world_size=world_size,
            config=config,
        )
        if adaptive is None:
            continue
        signature = tuple(
            (span.sequence, span.token_start, span.token_end, span.owner)
            for span in adaptive[1]
        )
        if signature not in seen_assignments:
            adaptive_candidates.append(adaptive)
            seen_assignments.add(signature)
    candidates = [candidate for candidate in (contiguous, *adaptive_candidates) if candidate is not None]
    if not candidates:
        raise RuntimeError("no feasible THD layout candidate")
    selected_policy, selected_spans, selected_metrics = min(
        candidates,
        key=lambda candidate: (
            candidate[2].score,
            candidate[0] != "contiguous",
            candidate[0],
        ),
    )
    ranks = tuple(
        THDRankLayout(
            rank,
            tuple(
                sorted(
                    (span for span in selected_spans if span.owner == rank),
                    key=lambda span: (span.sequence, span.token_start),
                )
            ),
        )
        for rank in range(world_size)
    )
    return THDLayoutPlan(
        world_size,
        config.block_size,
        selected_policy,
        ranks,
        selected_metrics,
        tuple((policy, metrics.score) for policy, _, metrics in candidates),
    )


__all__ = [
    "THDLayoutMetrics",
    "THDLayoutPlan",
    "THDPlannerConfig",
    "THDRankLayout",
    "THDSpan",
    "plan_thd_layout",
]
