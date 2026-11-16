# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only load planner for packed THD context parallelism.

The planner balances two independent resources: token-linear work used by KDA
and position-dependent causal candidate work used by CSA.  It deliberately
does not inspect CUDA tensors or kernel state, so one immutable plan can be
built at microbatch ingress and reused by the complete model stack.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from heapq import heapify, heappop, heappush
from itertools import groupby
from math import ceil, isfinite
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
    """Complete global span assignment selected by a THD planner."""

    world_size: int
    block_size: int
    policy: str
    ranks: tuple[THDRankLayout, ...]
    metrics: THDLayoutMetrics
    evaluated_scores: tuple[tuple[str, float], ...]
    score_unit: str = "normalized"

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
    min_chunk_tokens: int = 0
    """Minimum newly created selection span; zero preserves the legacy search.

    Existing source fragments and complete short documents remain legal.
    This does not change the explicitly requested full-zigzag baseline.
    """


@dataclass(frozen=True, slots=True)
class THDSelectionCostModel:
    """Calibrated eager selection-routing costs; no hardware detection or I/O.

    Coefficients must describe the caller's indexer shape and transport. Q/O
    attention and backward stay in source ownership. ``bytes_per_query`` sums
    outgoing indexer Q/weights and returning IDs/scores, without dtype changes.
    Bandwidths are effective per-rank GB/s, not advertised link speeds. Group
    rank -> node IDs is explicit; the source is a contiguous balanced shard.
    ``min_gain_fraction`` requires headroom over contiguous before migrating.
    ``min_gain_ms`` additionally requires an absolute predicted saving per use;
    callers may price one-time preparation divided by expected layout reuse.
    ``peer_gbps`` optionally prices the maximum aggregated source/destination
    edge, since sparse peer traffic may not saturate aggregate rank bandwidth.
    ``inter_node_aggregate_gbps`` optionally bounds shared node injection and
    receive bandwidth. ``peer_startup_us`` prices the busiest rank's nonempty
    peer count, summed over the ordered forward/return collectives. Calibrate
    these independently of kernel arrival waits; they are effective costs,
    not physical-link specifications. Unspecified limits preserve old profiles.
    ``local_copy_gbps`` prices the diagonal of the eager all-to-all: a globally
    nonidentity exchange still copies rows retained by the same rank. This
    shares the local transfer resource; it is absent for an identity layout.
    """

    candidate_ns: float
    query_ns: float
    span_us: float
    bytes_per_query: int
    intra_node_gbps: float
    inter_node_gbps: float
    rank_node_ids: tuple[int, ...]
    packing_ns_per_query: float = 0.0
    exchange_us: float = 0.0
    min_gain_fraction: float = 0.05
    peer_gbps: float | None = None
    min_gain_ms: float = 0.0
    peer_startup_us: float = 0.0
    inter_node_aggregate_gbps: float | None = None
    overlap_intra_inter: bool = False
    local_copy_gbps: float | None = None

    def __post_init__(self):
        if not isinstance(self.overlap_intra_inter, bool):
            raise TypeError("overlap_intra_inter must be a boolean")
        for name in (
            "candidate_ns",
            "query_ns",
            "span_us",
            "packing_ns_per_query",
            "exchange_us",
            "min_gain_fraction",
            "min_gain_ms",
            "peer_startup_us",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("intra_node_gbps", "inter_node_gbps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.min_gain_fraction >= 1:
            raise ValueError("min_gain_fraction must be below one")
        for name in ("peer_gbps", "inter_node_aggregate_gbps", "local_copy_gbps"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value) or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive when supplied")
        if isinstance(self.bytes_per_query, bool) or not isinstance(self.bytes_per_query, int) or self.bytes_per_query <= 0:
            raise ValueError("bytes_per_query must be a positive integer")
        if not isinstance(self.rank_node_ids, tuple) or not self.rank_node_ids:
            raise TypeError("rank_node_ids must be a nonempty tuple")
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in self.rank_node_ids):
            raise ValueError("node ids must be nonnegative integers")

    def _migration_threshold_ms(self, baseline_ms):
        return baseline_ms - max(baseline_ms * self.min_gain_fraction, self.min_gain_ms)

    def _outgoing_capacity_gbps(self, rank):
        kinds = {node == self.rank_node_ids[rank] for peer, node in enumerate(self.rank_node_ids) if peer != rank}
        rates = [self.intra_node_gbps if local else self.inter_node_gbps for local in kinds]
        return sum(rates) if self.overlap_intra_inter and rates else max(rates, default=self.intra_node_gbps)

    def estimate_ms(self, ranks: tuple[THDRankLayout, ...], *, block_size: int) -> float:
        world = len(ranks)
        if len(self.rank_node_ids) != world:
            raise ValueError("cost-model node map must have one entry per process-group rank")
        total = sum(r.local_tokens for r in ranks)
        offsets = tuple(x * block_size for x in balanced_shards(total // block_size, world))
        compute, incoming, outgoing = [0.0] * world, [0.0] * world, [0.0] * world
        intra_in, intra_out, inter_in, inter_out = ([0.0] * world for _ in range(4))
        local_rows = [0] * world
        peer_rows = defaultdict(int)
        source_cursor = list(offsets[:-1])
        send_ordered, receive_ordered = [True] * world, [True] * world
        identity = True
        for rank, layout in enumerate(ranks):
            cursor = offsets[rank]
            last_source = -1
            for span in layout.spans:
                start, end = span.token_start, span.token_end
                tokens = end - start
                compute[rank] += (
                    self.candidate_ns * _candidate_work(start - span.sequence_start, tokens, block_size) / 1e6
                    + self.query_ns * tokens / 1e6
                    + self.span_us / 1e3
                )
                identity &= start == cursor
                cursor += tokens
                # Pinned source spans are common in mixed packs. Their exact
                # one-source intersection needs neither a bisect nor a walk.
                if offsets[rank] <= start and end <= offsets[rank + 1]:
                    local_rows[rank] += tokens
                    send_ordered[rank] &= start == source_cursor[rank]
                    source_cursor[rank] += tokens
                    receive_ordered[rank] &= rank >= last_source
                    last_source = rank
                    continue
                for source, rows in _source_intersections(span, offsets):
                    send_ordered[source] &= max(offsets[source], span.token_start) == source_cursor[source]
                    source_cursor[source] += rows
                    receive_ordered[rank] &= source >= last_source
                    last_source = source
                    if source != rank:
                        gbps = (
                            self.intra_node_gbps
                            if self.rank_node_ids[source] == self.rank_node_ids[rank]
                            else self.inter_node_gbps
                        )
                        cost = rows * self.bytes_per_query / (gbps * 1e6)
                        outgoing[source] += cost
                        incoming[rank] += cost
                        local = self.rank_node_ids[source] == self.rank_node_ids[rank]
                        (intra_out if local else inter_out)[source] += cost
                        (intra_in if local else inter_in)[rank] += cost
                        peer_rows[source, rank] += rows
                    else:
                        local_rows[rank] += rows
            identity &= cursor == offsets[rank + 1]
        if identity:
            return max(compute)
        if self.local_copy_gbps is not None:
            for rank, rows in enumerate(local_rows):
                cost = rows * self.bytes_per_query / (self.local_copy_gbps * 1e6)
                incoming[rank] += cost
                outgoing[rank] += cost
                intra_in[rank] += cost
                intra_out[rank] += cost
        # Execution has ordered exchange/compute/return phases. Model their
        # critical durations separately rather than assuming unmeasured overlap.
        transfer = max(*incoming, *outgoing)
        if self.overlap_intra_inter:
            transfer = max(*intra_in, *intra_out, *inter_in, *inter_out)
        if self.peer_gbps is not None:
            transfer = max(transfer, max(peer_rows.values(), default=0) * self.bytes_per_query / (self.peer_gbps * 1e6))
        node_in, node_out = defaultdict(int), defaultdict(int)
        send_peers, receive_peers = [0] * world, [0] * world
        for (source, target), rows in peer_rows.items():
            send_peers[source] += 1
            receive_peers[target] += 1
            a, b = self.rank_node_ids[source], self.rank_node_ids[target]
            if a != b:
                node_out[a] += rows
                node_in[b] += rows
        if self.inter_node_aggregate_gbps is not None:
            node_rows = max((*node_in.values(), *node_out.values()), default=0)
            transfer = max(transfer, node_rows * self.bytes_per_query / (self.inter_node_aggregate_gbps * 1e6))
        transfer += max(*send_peers, *receive_peers) * self.peer_startup_us / 1e3
        # Match the redistributor's no-op permutation elimination. A local
        # forward pack and its inverse restore together copy bytes_per_query;
        # the destination-side pair does the same only if receive order differs.
        copies = max(
            (offsets[r + 1] - offsets[r]) * (not send_ordered[r]) + layout.local_tokens * (not receive_ordered[r])
            for r, layout in enumerate(ranks)
        )
        return max(compute) + transfer + copies * self.packing_ns_per_query / 1e6 + self.exchange_us / 1e3


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


def _source_intersections(span: THDSpan, offsets: tuple[int, ...]):
    for source in range(bisect_right(offsets, span.token_start) - 1, bisect_left(offsets, span.token_end)):
        yield source, min(offsets[source + 1], span.token_end) - max(offsets[source], span.token_start)


def _transfer_lower_bound(assignments, world, block_size, model, compute_mean, baseline):
    """Optimistic lower bound on critical compute plus outgoing traffic.

    To reduce source rank i's compute below C, some candidate/token work must
    leave it. A moved row can remove no more than the largest causal position
    on that source. Ignore span overhead, incoming traffic, packing, capacity
    and fixed exchange cost: all make this a lower bound, not a prediction.
    Minimizing the maximum of affine functions only needs their intersections
    and the interval endpoints. It remains valid for unequal destination rows.
    """
    compute, density = [0.0] * world, [0.0] * world
    for item, rank in assignments:
        compute[rank] += (model.candidate_ns * item.causal_work + model.query_ns * item.tokens) / 1e6
        density[rank] = max(
            density[rank],
            (model.candidate_ns * ((item.token_end - item.sequence_start) // block_size) + model.query_ns) / 1e6,
        )
    lines = [(1.0, 0.0)]
    for rank in range(world):
        if not density[rank]:
            continue
        rate = model._outgoing_capacity_gbps(rank)
        slope = model.bytes_per_query / (rate * 1e6 * density[rank])
        lines.append((1 - slope, slope * compute[rank]))
    points = [compute_mean, baseline]
    for i, (a, b) in enumerate(lines):
        for c, d in lines[:i]:
            if a != c:
                cross = (d - b) / (a - c)
                if compute_mean < cross < baseline:
                    points.append(cross)
    return min(max(a * x + b for a, b in lines) for x in points)


def _validate_inputs(
    global_cu_seqlens: torch.Tensor | Sequence[int],
    *,
    world_size: int,
    config: THDPlannerConfig,
) -> tuple[int, ...]:
    boundaries = as_boundaries(global_cu_seqlens, name="global_cu_seqlens")
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    for name in ("block_size", "min_split_tokens", "max_chunks_per_rank"):
        raw_value = getattr(config, name)
        if isinstance(raw_value, bool):
            raise TypeError(f"{name} must be an integer")
        value = index(raw_value)
        if value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config.min_split_tokens % config.block_size:
        raise ValueError("min_split_tokens must be block aligned")
    if (
        isinstance(config.min_chunk_tokens, bool)
        or not isinstance(config.min_chunk_tokens, int)
        or config.min_chunk_tokens < 0
        or config.min_chunk_tokens % config.block_size
    ):
        raise ValueError("min_chunk_tokens must be nonnegative and block aligned")
    if any((right - left) % config.block_size for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("every sequence length must be block aligned")
    if boundaries[-1] // config.block_size < world_size:
        raise ValueError("packed context parallelism requires at least one block per rank")
    for name in ("token_weight", "causal_weight", "boundary_weight"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
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
        for sequence in range(bisect_right(boundaries, shard_start) - 1, bisect_left(boundaries, shard_end)):
            sequence_start, sequence_end = boundaries[sequence : sequence + 2]
            start = max(shard_start, sequence_start)
            end = min(shard_end, sequence_end)
            if start < end:
                result.append((_make_item(sequence, start, end, sequence_start, sequence_end, block_size), rank))
    return result


def _adaptive_items(
    boundaries: tuple[int, ...],
    world_size: int,
    config: THDPlannerConfig,
    *,
    chunks_per_rank: int,
    sequences: set[int] | None = None,
) -> list[_PlanningItem]:
    block_size = config.block_size
    total_blocks = boundaries[-1] // block_size
    requested_grid = world_size * chunks_per_rank
    target_blocks = max(1, ceil(total_blocks / requested_grid))
    min_split_blocks = config.min_split_tokens // block_size
    result = []
    for sequence in range(len(boundaries) - 1) if sequences is None else sorted(sequences):
        sequence_start, sequence_end = boundaries[sequence : sequence + 2]
        length_blocks = (sequence_end - sequence_start) // block_size
        pieces = min(requested_grid, max(1, ceil(length_blocks / target_blocks)))
        # A sequence shorter than two useful spans stays whole.  This is what
        # prevents short THD samples from paying a max-CP fragmentation tax.
        if length_blocks < 2 * min_split_blocks:
            pieces = 1
        if config.min_chunk_tokens:
            pieces = min(pieces, max(1, length_blocks // (config.min_chunk_tokens // block_size)))
        offsets = balanced_shards(length_blocks, pieces)
        for left, right in zip(offsets, offsets[1:]):
            start = sequence_start + left * block_size
            end = sequence_start + right * block_size
            result.append(_make_item(sequence, start, end, sequence_start, sequence_end, block_size))

    # Keep all ranks non-empty even for small batches containing fewer than W
    # sequences.  The minimum span is a performance preference, not a semantic
    # restriction, so only this feasibility repair may split below it.
    while sequences is None and len(result) < world_size:
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
    for (tokens, work), group in groupby(ordered, key=lambda item: (item.tokens, item.causal_work)):

        def pressure(rank):
            token_pressure = (token_loads[rank] + tokens) / ideal_tokens
            causal_pressure = (causal_loads[rank] + work) / ideal_causal
            return max(token_pressure, causal_pressure), token_pressure + causal_pressure, item_counts[rank], rank

        # Consecutive equal-cost pieces change only the selected rank's key.
        # A heap retains the exact LPT ordering without scanning all CP ranks
        # for every tiny document. Unique-cost pieces keep the simple minimum.
        pieces = list(group)
        heap = [pressure(rank) for rank in range(world_size)]
        if len(pieces) > 1:
            heapify(heap)
        for item in pieces:
            owner = heappop(heap)[-1] if len(pieces) > 1 else min(heap)[-1]
            result.append((item, owner))
            token_loads[owner] += tokens
            causal_loads[owner] += work
            item_counts[owner] += 1
            if len(pieces) > 1:
                heappush(heap, pressure(owner))
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
    capacities = [(right - left) * block_size for left, right in zip(capacity_offsets, capacity_offsets[1:])]
    loads = [0] * world_size
    causal_loads = [0] * world_size
    for item, owner in assignments:
        loads[owner] += item.tokens
        causal_loads[owner] += item.causal_work
    ideal_causal = sum(causal_loads) / world_size
    # Stable serials preserve the old global-list tie order, while looking only
    # at the selected donor avoids rescanning every unrelated short document.
    bins = [dict() for _ in range(world_size)]
    shapes = [defaultdict(list) for _ in range(world_size)]

    def insert(owner, serial, item):
        bins[owner][serial] = item
        shape = (item.tokens, item.token_start - item.sequence_start)
        heappush(shapes[owner][shape], (item.sequence, item.token_start, serial))

    for serial, (item, owner) in enumerate(assignments):
        insert(owner, serial, item)
    next_serial = len(assignments)

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

        other_pressure = max(
            (causal_loads[r] / ideal_causal for r in range(world_size) if r not in (donor, receiver)), default=0.0
        )
        best = None
        # Equal (length, causal offset) pieces have identical whole/prefix/
        # suffix costs. Only the earliest semantic tie-break can win, so one
        # heap head represents all equivalent short documents exactly.
        for heap in shapes[donor].values():
            serial = heap[0][2]
            item = bins[donor][serial]
            transfer_tokens = min(item.tokens, transfer_limit)
            split = transfer_tokens != item.tokens
            for take_prefix in (True, False) if split else (True,):
                moved_start = item.token_start if take_prefix else item.token_end - transfer_tokens
                moved_work = (
                    _candidate_work(moved_start - item.sequence_start, transfer_tokens, block_size)
                    if split
                    else item.causal_work
                )
                causal_pressure = max(
                    (causal_loads[donor] - moved_work) / ideal_causal,
                    (causal_loads[receiver] + moved_work) / ideal_causal,
                    other_pressure,
                )
                choice = (causal_pressure, split, -transfer_tokens, item.sequence, moved_start, serial, take_prefix)
                if best is None or choice < best:
                    best = choice
        if best is None:
            raise RuntimeError("internal error: an overloaded rank has no movable THD span")
        _, split, negative_tokens, _, _, serial, take_prefix = best
        item = bins[donor].pop(serial)
        shape = (item.tokens, item.token_start - item.sequence_start)
        heappop(shapes[donor][shape])
        if not shapes[donor][shape]:
            del shapes[donor][shape]
        moved, remainder = (
            _split_item(item, -negative_tokens, take_prefix=take_prefix, block_size=block_size) if split else (item, None)
        )
        if remainder is not None:
            insert(donor, next_serial, remainder)
            next_serial += 1
        insert(receiver, next_serial, moved)
        next_serial += 1
        loads[donor] -= moved.tokens
        loads[receiver] += moved.tokens
        causal_loads[donor] -= moved.causal_work
        causal_loads[receiver] += moved.causal_work
    return [
        (item, owner) for _, item, owner in sorted((i, item, r) for r, items in enumerate(bins) for i, item in items.items())
    ]


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


def _minimum_assignment(costs: list[list[float]]) -> tuple[int, ...]:
    """Minimum-cost bin -> rank permutation, O(W^3), with stable ties.

    The primal-dual augmenting-path algorithm keeps rank ownership bijective.
    It needs no optional optimizer dependency and never enumerates W! maps.
    """
    size = len(costs)
    u, v = [0.0] * (size + 1), [0.0] * (size + 1)
    matched, previous = [0] * (size + 1), [0] * (size + 1)
    for row in range(1, size + 1):
        matched[0] = row
        column = 0
        best = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column] = True
            active = matched[column]
            step, next_column = float("inf"), 0
            for candidate in range(1, size + 1):
                if not used[candidate]:
                    reduced = costs[active - 1][candidate - 1] - u[active] - v[candidate]
                    if reduced < best[candidate]:
                        best[candidate], previous[candidate] = reduced, column
                    if best[candidate] < step:
                        step, next_column = best[candidate], candidate
            for candidate in range(size + 1):
                if used[candidate]:
                    u[matched[candidate]] += step
                    v[candidate] -= step
                else:
                    best[candidate] -= step
            column = next_column
            if matched[column] == 0:
                break
        while column:
            parent = previous[column]
            matched[column] = matched[parent]
            column = parent
    assignment = [0] * size
    for column in range(1, size + 1):
        assignment[matched[column] - 1] = column - 1
    return tuple(assignment)


def _localize_spans(spans: list[THDSpan], world: int, block_size: int, model: THDSelectionCostModel) -> list[THDSpan]:
    """Relabel equal-capacity bins to minimize aggregate transfer cost.

    Computation and fragmentation stay identical. This minimizes a sum, not
    the critical path; the caller must still score the result against the
    original candidate with the full max-rank model.
    """
    offsets = tuple(x * block_size for x in balanced_shards(sum(s.tokens for s in spans) // block_size, world))
    overlaps = [[0] * world for _ in range(world)]
    for span in spans:
        for source, rows in _source_intersections(span, offsets):
            overlaps[span.owner][source] += rows
    capacities = defaultdict(list)
    for rank, (a, z) in enumerate(zip(offsets, offsets[1:])):
        capacities[z - a].append(rank)
    owners = list(range(world))
    for peers in capacities.values():
        costs = [
            [
                sum(
                    rows
                    / (model.intra_node_gbps if model.rank_node_ids[s] == model.rank_node_ids[r] else model.inter_node_gbps)
                    for s, rows in enumerate(overlaps[owner])
                    if s != r
                )
                for r in peers
            ]
            for owner in peers
        ]
        assignment = _minimum_assignment(costs)
        for owner, destination in zip(peers, assignment):
            owners[owner] = peers[destination]
    return [replace(span, owner=owners[span.owner]) for span in spans]


def plan_thd_layout(
    global_cu_seqlens: torch.Tensor | Sequence[int],
    *,
    world_size: int,
    config: THDPlannerConfig | None = None,
    cost_model: THDSelectionCostModel | None = None,
) -> THDLayoutPlan:
    """Plan a block-aligned THD layout and select the best modeled candidate.

    The result is guaranteed to have a score no worse than the evaluated
    contiguous baseline under the declared cost model.  This is a model-level
    guarantee, not a claim that an analytic proxy exactly predicts every GPU.
    """

    config = THDPlannerConfig() if config is None else config
    if not isinstance(config, THDPlannerConfig):
        raise TypeError("config must be a THDPlannerConfig")
    if cost_model is not None and not isinstance(cost_model, THDSelectionCostModel):
        raise TypeError("cost_model must be a THDSelectionCostModel")
    boundaries = _validate_inputs(global_cu_seqlens, world_size=world_size, config=config)
    if cost_model is not None and len(cost_model.rank_node_ids) != world_size:
        raise ValueError("cost-model node map must have one entry per process-group rank")

    source_items = _contiguous_items(boundaries, world_size, config.block_size)
    contiguous = _build_candidate(
        source_items,
        policy="contiguous",
        world_size=world_size,
        config=config,
    )
    if cost_model is not None and contiguous is not None:
        policy, spans, metrics = contiguous
        ranks = tuple(THDRankLayout(r, tuple(s for s in spans if s.owner == r)) for r in range(world_size))
        baseline_ms = cost_model.estimate_ms(ranks, block_size=config.block_size)
        # Every exact partition has the same candidate/token sums and at least
        # one span per document. Even free transport cannot beat their mean.
        lower_bound = (
            cost_model.candidate_ns * sum(metrics.causal_loads) / 1e6
            + cost_model.query_ns * boundaries[-1] / 1e6
            + cost_model.span_us * (len(boundaries) - 1) / 1e3
        ) / world_size
        lower_bound = _transfer_lower_bound(source_items, world_size, config.block_size, cost_model, lower_bound, baseline_ms)
        if lower_bound >= cost_model._migration_threshold_ms(baseline_ms):
            return THDLayoutPlan(
                world_size,
                config.block_size,
                policy,
                ranks,
                replace(metrics, score=baseline_ms),
                ((policy, baseline_ms),),
                "ms",
            )
    adaptive_candidates = []
    seen_assignments = set()
    seen_items = set()
    for chunks_per_rank in range(1, config.max_chunks_per_rank + 1):
        adaptive_items = _adaptive_items(
            boundaries,
            world_size,
            config,
            chunks_per_rank=chunks_per_rank,
        )
        item_signature = tuple(adaptive_items)
        if item_signature in seen_items:
            continue
        seen_items.add(item_signature)
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
        signature = tuple((span.sequence, span.token_start, span.token_end, span.owner) for span in adaptive[1])
        if signature not in seen_assignments:
            adaptive_candidates.append(adaptive)
            seen_assignments.add(signature)
    candidates = [candidate for candidate in (contiguous, *adaptive_candidates) if candidate is not None]
    if not candidates:
        raise RuntimeError("no feasible THD layout candidate")
    if cost_model is not None:
        localized = []
        for policy, spans, _ in candidates:
            if policy == "contiguous":
                continue
            nearby = _localize_spans(spans, world_size, config.block_size, cost_model)
            if nearby != spans:
                localized.append(
                    (
                        policy + "-local",
                        nearby,
                        _metrics(nearby, world_size=world_size, block_size=config.block_size, config=config),
                    )
                )
        candidates += localized
        candidates = [
            (
                policy,
                spans,
                replace(
                    metrics,
                    score=cost_model.estimate_ms(
                        tuple(THDRankLayout(r, tuple(s for s in spans if s.owner == r)) for r in range(world_size)),
                        block_size=config.block_size,
                    ),
                ),
            )
            for policy, spans, metrics in candidates
        ]
    selected_policy, selected_spans, selected_metrics = min(
        candidates,
        key=lambda candidate: (
            candidate[2].score,
            candidate[0] != "contiguous",
            candidate[0],
        ),
    )
    if cost_model is not None:
        baseline = next(c for c in candidates if c[0] == "contiguous")
        if selected_metrics.score >= cost_model._migration_threshold_ms(baseline[2].score):
            selected_policy, selected_spans, selected_metrics = baseline
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
        "ms" if cost_model is not None else "normalized",
    )


def _selection_assign(items, *, pinned_loads, pinned_compute, world, capacity, offsets, model, block_size):
    """Keep cheap source spans local; place long-span work under a row cap.

    Unlike a shared KDA/CSA layout, selection ownership need not have equal
    token counts. Its invariant is bounded rows and exact coverage. Placement
    scores ordered compute/transport phases using the same calibrated units
    as final candidate selection, with stable load/locality tie breaks.
    """
    # Copy the small per-rank state; all grids share the same pinned spans.
    loads, compute = list(pinned_loads), list(pinned_compute)
    incoming, outgoing = [0.0] * world, [0.0] * world
    intra_in, intra_out, inter_in, inter_out = ([0.0] * world for _ in range(4))
    peer_rows = [[0] * world for _ in range(world)]
    peer_peak = 0
    send_peers, receive_peers = [0] * world, [0] * world
    degree_peak, node_peak = 0, 0
    node_in, node_out = defaultdict(int), defaultdict(int)
    # Pinned spans initialize loads but are materialized once by the caller.
    result = []

    def work(item):
        return model.candidate_ns * item.causal_work / 1e6 + model.query_ns * item.tokens / 1e6 + model.span_us / 1e3

    for original in sorted(items, key=lambda item: (-work(item), -item.tokens, item.sequence, item.token_start)):
        item = original
        while item is not None:
            # A cap is not an exact bin size. Split only when no remaining
            # bin can hold this item, keeping short sequences intact otherwise.
            free = max(capacity - n for n in loads)
            if free <= 0:
                raise RuntimeError("internal error: selection row capacity exhausted")
            placed, remainder = (
                _split_item(item, free, take_prefix=True, block_size=block_size) if item.tokens > free else (item, None)
            )
            intersections = tuple(_source_intersections(placed, offsets))
            cost = work(placed)
            max_compute, max_incoming, max_outgoing = max(compute), max(incoming), max(outgoing)
            max_copies = max(offsets[r + 1] - offsets[r] + n for r, n in enumerate(loads))
            if model.overlap_intra_inter:
                max_incoming, max_outgoing = max(*intra_in, *inter_in), max(*intra_out, *inter_out)
            transfer_floor = max(max_incoming, max_outgoing)
            if model.peer_gbps is not None:
                transfer_floor = max(transfer_floor, peer_peak * model.bytes_per_query / (model.peer_gbps * 1e6))
            if model.inter_node_aggregate_gbps is not None:
                transfer_floor = max(
                    transfer_floor, node_peak * model.bytes_per_query / (model.inter_node_aggregate_gbps * 1e6)
                )
            transfer_floor += degree_peak * model.peer_startup_us / 1e3
            best = None
            for rank in range(world):
                if loads[rank] + placed.tokens > capacity:
                    continue
                rank_compute = max(max_compute, compute[rank] + cost)
                copies = max(max_copies, offsets[rank + 1] - offsets[rank] + loads[rank] + placed.tokens)
                # New traffic cannot reduce an existing bottleneck. Evaluate
                # the expensive peer/node deltas only if this rank can still
                # beat the incumbent. Keep equality for the full tie break.
                floor = rank_compute + transfer_floor
                if peer_peak:
                    floor += copies * model.packing_ns_per_query / 1e6 + model.exchange_us / 1e3
                if best is not None and floor > best[0][0]:
                    continue
                additions = []
                moved = 0
                for source, rows in intersections:
                    if source == rank:
                        continue
                    rate = (
                        model.intra_node_gbps
                        if model.rank_node_ids[source] == model.rank_node_ids[rank]
                        else model.inter_node_gbps
                    )
                    additions.append((source, rows * model.bytes_per_query / (rate * 1e6)))
                    moved += rows
                received = sum(value for _, value in additions)
                transfer = max(max_incoming, incoming[rank] + received, max_outgoing)
                transfer = max(transfer, max((outgoing[s] + value for s, value in additions), default=0.0))
                if model.overlap_intra_inter:
                    local_received = sum(v for s, v in additions if model.rank_node_ids[s] == model.rank_node_ids[rank])
                    remote_received = sum(v for s, v in additions if model.rank_node_ids[s] != model.rank_node_ids[rank])
                    transfer = max(
                        max_incoming, max_outgoing, intra_in[rank] + local_received, inter_in[rank] + remote_received
                    )
                    transfer = max(
                        transfer,
                        max(
                            (
                                (intra_out if model.rank_node_ids[s] == model.rank_node_ids[rank] else inter_out)[s] + v
                                for s, v in additions
                            ),
                            default=0.0,
                        ),
                    )
                candidate_peer_peak = max(
                    peer_peak, max((peer_rows[s][rank] + n for s, n in intersections if s != rank), default=0)
                )
                if model.peer_gbps is not None:
                    transfer = max(transfer, candidate_peer_peak * model.bytes_per_query / (model.peer_gbps * 1e6))
                new_edges = [s for s, n in intersections if s != rank and not peer_rows[s][rank]]
                candidate_degree = max(
                    degree_peak,
                    receive_peers[rank] + len(new_edges),
                    max((send_peers[s] + 1 for s in new_edges), default=0),
                )
                candidate_node = node_peak
                if model.inter_node_aggregate_gbps is not None:
                    node_additions = defaultdict(int)
                    for s, n in intersections:
                        if model.rank_node_ids[s] != model.rank_node_ids[rank]:
                            node_additions[model.rank_node_ids[s]] += n
                    candidate_node = max(
                        node_peak,
                        node_in[model.rank_node_ids[rank]] + sum(node_additions.values()),
                        max((node_out[n] + rows for n, rows in node_additions.items()), default=0),
                    )
                    transfer = max(transfer, candidate_node * model.bytes_per_query / (model.inter_node_aggregate_gbps * 1e6))
                transfer += candidate_degree * model.peer_startup_us / 1e3
                # Partial placement uses an upper bound on copies. Final
                # scoring checks the actual source/destination permutations.
                critical = rank_compute + transfer
                if candidate_peer_peak:
                    critical += copies * model.packing_ns_per_query / 1e6 + model.exchange_us / 1e3
                key = (critical, compute[rank] + cost + incoming[rank] + received, moved, loads[rank], rank)
                if best is None or key < best[0]:
                    best = key, rank, additions, received, candidate_peer_peak, candidate_degree, candidate_node
            _, owner, additions, received, peer_peak, degree_peak, node_peak = best
            result.append((placed, owner))
            loads[owner] += placed.tokens
            compute[owner] += cost
            incoming[owner] += received
            for source, value in additions:
                outgoing[source] += value
                local = model.rank_node_ids[source] == model.rank_node_ids[owner]
                (intra_out if local else inter_out)[source] += value
                (intra_in if local else inter_in)[owner] += value
            for source, rows in intersections:
                if source != owner:
                    if not peer_rows[source][owner]:
                        send_peers[source] += 1
                        receive_peers[owner] += 1
                    peer_rows[source][owner] += rows
                    a, b = model.rank_node_ids[source], model.rank_node_ids[owner]
                    if a != b:
                        node_out[a] += rows
                        node_in[b] += rows
            item = remainder
    return result


def _paired_selection_items(boundaries, movable, world, block_size, offsets, model):
    """Pair early/late chunks only within long documents, keeping shorts local."""
    result = []
    for sequence in sorted(movable):
        start, end = boundaries[sequence : sequence + 2]
        # Unlike balanced_shards, empty pieces are legitimate for tiny custom
        # split thresholds. They never become empty THDSpan objects.
        base, remainder = divmod((end - start) // block_size, 2 * world)
        cuts = [start + (i * base + min(i, remainder)) * block_size for i in range(2 * world + 1)]
        pairs = []
        for rank in range(world):
            pairs.append(
                [
                    _make_item(sequence, cuts[i], cuts[i + 1], start, end, block_size)
                    for i in (rank, 2 * world - 1 - rank)
                    if cuts[i] < cuts[i + 1]
                ]
            )
        costs = []
        for pair in pairs:
            overlaps = [0] * world
            for item in pair:
                for source, rows in _source_intersections(item, offsets):
                    overlaps[source] += rows
            costs.append(
                [
                    sum(
                        rows
                        / (
                            model.intra_node_gbps
                            if model.rank_node_ids[source] == model.rank_node_ids[rank]
                            else model.inter_node_gbps
                        )
                        for source, rows in enumerate(overlaps)
                        if source != rank
                    )
                    for rank in range(world)
                ]
            )
        owners = _minimum_assignment(costs)
        result.extend((item, owners[rank]) for rank, pair in enumerate(pairs) for item in pair)
    return result


def zigzag_thd_layout(
    global_cu_seqlens: torch.Tensor | Sequence[int], *, world_size: int, block_size: int = 4
) -> tuple[THDRankLayout, ...]:
    """Conventional per-document early/late pairing, including short documents.

    Rank r owns pieces r and 2*CP-1-r. Empty pieces are omitted; adjacent
    pieces remain distinct, matching the selection backend's execution spans.
    This layout can have unequal or empty ranks and does not impose a row cap.
    """
    config = THDPlannerConfig(block_size=block_size, min_split_tokens=block_size)
    boundaries = _validate_inputs(global_cu_seqlens, world_size=world_size, config=config)
    ranks = [[] for _ in range(world_size)]
    for sequence, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        base, remainder = divmod((end - start) // block_size, 2 * world_size)
        cuts = [start + (i * base + min(i, remainder)) * block_size for i in range(2 * world_size + 1)]
        for rank in range(world_size):
            for piece in (rank, 2 * world_size - 1 - rank):
                a, z = cuts[piece : piece + 2]
                if a < z:
                    ranks[rank].append(THDSpan(sequence, a, z, start, end, rank))
    return tuple(THDRankLayout(r, tuple(spans)) for r, spans in enumerate(ranks))


def _zigzag_lower_bound(boundaries, source, world, block_size, model):
    """Average compute plus exact outgoing traffic, without building THD spans.

    Consecutive zigzag pieces assigned to the same node form interval runs.
    Intersect each source span with its rank's two pieces and its node's runs:
    this gives exact self, intra-node and inter-node row counts. The number of
    runs is small for contiguous node ranks; arbitrary explicit topologies are
    supported too. Average compute and outgoing-only transport are lower bounds
    on their respective critical times. Packing and incoming/peer bottlenecks
    are omitted, so this cannot overprice the concrete full-zigzag candidate.
    """
    piece_nodes = [model.rank_node_ids[min(i, 2 * world - 1 - i)] for i in range(2 * world)]
    node_runs = defaultdict(list)
    begin = 0
    for node, pieces in groupby(piece_nodes):
        end = begin + sum(1 for _ in pieces)
        node_runs[node].append((begin, end))
        begin = end
    geometry, total_work, spans = [], 0, 0
    for a, z in zip(boundaries, boundaries[1:]):
        blocks = (z - a) // block_size
        geometry.append(divmod(blocks, 2 * world))
        total_work += _candidate_work(0, z - a, block_size)
        spans += min(blocks, 2 * world)
    intra, inter = [0] * world, [0] * world
    for item, rank in source:
        base, remainder = geometry[item.sequence]

        def overlap(left, right):
            a = item.sequence_start + (left * base + min(left, remainder)) * block_size
            z = item.sequence_start + (right * base + min(right, remainder)) * block_size
            return max(0, min(item.token_end, z) - max(item.token_start, a))

        own_rows = overlap(rank, rank + 1) + overlap(2 * world - 1 - rank, 2 * world - rank)
        node_rows = sum(overlap(a, z) for a, z in node_runs[model.rank_node_ids[rank]])
        intra[rank] += node_rows - own_rows
        inter[rank] += item.tokens - node_rows
    transfer = 0.0
    for local, remote in zip(intra, inter):
        a, b = local / model.intra_node_gbps, remote / model.inter_node_gbps
        transfer = max(transfer, (max(a, b) if model.overlap_intra_inter else a + b) * model.bytes_per_query / 1e6)
    if model.inter_node_aggregate_gbps is not None:
        node_out = defaultdict(int)
        for rank, rows in enumerate(inter):
            node_out[model.rank_node_ids[rank]] += rows
        transfer = max(transfer, max(node_out.values()) * model.bytes_per_query / (model.inter_node_aggregate_gbps * 1e6))
    if any(intra) or any(inter):
        # Each used link kind requires at least one distinct remote peer.
        peers = max((a > 0) + (b > 0) for a, b in zip(intra, inter))
        transfer += (model.exchange_us + model.peer_startup_us * peers) / 1e3
    compute = (model.candidate_ns * total_work + model.query_ns * boundaries[-1]) / (world * 1e6)
    compute += model.span_us * spans / (world * 1e3)
    return compute + transfer


def _zigzag_meets_floor(boundaries, world, block_size, floor, source_fragments):
    """Check grain feasibility before pricing or materializing full zigzag.

    Whole documents with sufficiently large pieces need no piece walk. For
    finer pieces, retain exactly the same original-source-fragment exception
    used by the other candidates. This rejects no otherwise legal layout.
    """
    if not floor:
        return True
    for sequence, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        blocks = (end - start) // block_size
        base, remainder = divmod(blocks, 2 * world)
        if max(1, base) * block_size >= floor:
            continue
        left = start
        for piece in range(min(blocks, 2 * world)):
            right = left + (base + (piece < remainder)) * block_size
            if right - left < floor and (sequence, left, right) not in source_fragments:
                return False
            left = right
    return True


def _source_selection_candidate(source, world, model):
    """Price known identity ownership without reconstructing its transport.

    Source items already have exact causal work and global token order. Reuse
    both for baseline metrics instead of recomputing candidate prefixes and
    source intersections several times on the common no-migration path.
    """
    groups = [[] for _ in range(world)]
    tokens, causal, boundaries, compute = ([0] * world for _ in range(4))
    previous = None
    for item, rank in source:
        groups[rank].append(
            THDSpan(item.sequence, item.token_start, item.token_end, item.sequence_start, item.sequence_end, rank)
        )
        tokens[rank] += item.tokens
        causal[rank] += item.causal_work
        compute[rank] += model.candidate_ns * item.causal_work / 1e6 + model.query_ns * item.tokens / 1e6 + model.span_us / 1e3
        if previous is not None and previous[0].sequence == item.sequence and previous[1] != rank:
            boundaries[previous[1]] += 1
            boundaries[rank] += 1
        previous = item, rank
    metrics = THDLayoutMetrics(
        tuple(tokens),
        tuple(causal),
        tuple(boundaries),
        max(tokens) / (sum(tokens) / world),
        max(causal) / (sum(causal) / world) if sum(causal) else 1.0,
        max(compute),
    )
    return tuple(THDRankLayout(r, tuple(g)) for r, g in enumerate(groups)), metrics


def plan_thd_selection(
    global_cu_seqlens: torch.Tensor | Sequence[int],
    *,
    world_size: int,
    max_tokens_per_rank: int,
    cost_model: THDSelectionCostModel,
    config: THDPlannerConfig | None = None,
) -> THDLayoutPlan:
    """Plan selection-only routing with an explicit per-rank row capacity.

    Compare contiguous, complete per-document zigzag and locality-preserving
    long-document candidates in the same cost model. Only the latter pin
    short source spans and split documents at least twice ``min_split_tokens``.
    Target row counts
    may differ: attention and KDA must retain source ownership, and selection
    IDs/scores must be restored through the inverse exchange before use.

    The capacity includes pinned rows and must fit every contiguous source
    shard. It bounds query-shaped buffers, not total backend workspace; the
    caller must also budget selection scratch and gathered keys. This eager
    CPU planner does no device probing, timing or implicit caching.
    """
    config = THDPlannerConfig() if config is None else config
    if not isinstance(config, THDPlannerConfig):
        raise TypeError("config must be a THDPlannerConfig")
    if not isinstance(cost_model, THDSelectionCostModel):
        raise TypeError("cost_model must be a THDSelectionCostModel")
    boundaries = _validate_inputs(global_cu_seqlens, world_size=world_size, config=config)
    if len(cost_model.rank_node_ids) != world_size:
        raise ValueError("cost-model node map must have one entry per process-group rank")
    if isinstance(max_tokens_per_rank, bool) or not isinstance(max_tokens_per_rank, int):
        raise TypeError("max_tokens_per_rank must be an integer")
    offsets = tuple(x * config.block_size for x in balanced_shards(boundaries[-1] // config.block_size, world_size))
    if max_tokens_per_rank % config.block_size or max_tokens_per_rank < max(b - a for a, b in zip(offsets, offsets[1:])):
        raise ValueError("max_tokens_per_rank must be block aligned and fit every source shard")
    base = _contiguous_items(boundaries, world_size, config.block_size)
    source_fragments = (
        {(item.sequence, item.token_start, item.token_end) for item, _ in base} if config.min_chunk_tokens else set()
    )
    movable = {i for i, (a, z) in enumerate(zip(boundaries, boundaries[1:])) if z - a >= 2 * config.min_split_tokens}
    pinned = [(item, rank) for item, rank in base if item.sequence not in movable]
    pinned_spans = ()
    source_ranks, source_metrics = _source_selection_candidate(base, world_size, cost_model)
    candidates = [("contiguous", source_ranks, source_metrics.score)]

    def add(assignments, policy):
        # Pinned and movable documents are disjoint. Reusing pinned spans
        # avoids rebuilding hundreds of short-document objects per grid.
        spans = sorted((*pinned_spans, *_coalesce(assignments)), key=lambda s: (s.sequence, s.token_start))
        # Capacity repair can split an otherwise useful item into tiny pieces.
        # Reject the whole candidate instead of silently weakening the floor.
        # Existing short source fragments are unavoidable and remain legal.
        if policy != "contiguous" and any(
            span.tokens < config.min_chunk_tokens and (span.sequence, span.token_start, span.token_end) not in source_fragments
            for span in spans
        ):
            return
        grouped = [[] for _ in range(world_size)]
        for span in spans:
            grouped[span.owner].append(span)
        ranks = tuple(THDRankLayout(r, tuple(group)) for r, group in enumerate(grouped))
        if any(rank.local_tokens > max_tokens_per_rank for rank in ranks):
            return
        score = cost_model.estimate_ms(ranks, block_size=config.block_size)
        candidates.append((policy, ranks, score))

    baseline_ms = candidates[0][2]
    lower_bound = baseline_ms
    # This bound allows arbitrary row movement, including short documents.
    # It may prune full zigzag too, but the long-document threshold may not.
    if baseline_ms > cost_model.min_gain_ms:
        lower_bound = (
            cost_model.candidate_ns * sum(item.causal_work for item, _ in base) / 1e6
            + cost_model.query_ns * boundaries[-1] / 1e6
            + cost_model.span_us * (len(boundaries) - 1) / 1e3
        ) / world_size
        lower_bound = _transfer_lower_bound(base, world_size, config.block_size, cost_model, lower_bound, baseline_ms)
    if movable and lower_bound < cost_model._migration_threshold_ms(baseline_ms):
        pinned_loads, pinned_compute = [0] * world_size, [0.0] * world_size
        for item, owner in pinned:
            pinned_loads[owner] += item.tokens
            pinned_compute[owner] += (
                cost_model.candidate_ns * item.causal_work / 1e6
                + cost_model.query_ns * item.tokens / 1e6
                + cost_model.span_us / 1e3
            )
        # These source-local spans remain in every long-document candidate.
        # Nonnegative work and transport cannot reduce their critical compute
        # time. This bound applies only to the pinned family, not full zigzag.
        if max(pinned_compute) < cost_model._migration_threshold_ms(baseline_ms):
            pinned_spans = _coalesce(pinned)
            add(
                _paired_selection_items(boundaries, movable, world_size, config.block_size, offsets, cost_model),
                "selection-paired-local",
            )
            seen = set()
            for grid in range(1, config.max_chunks_per_rank + 1):
                items = tuple(_adaptive_items(boundaries, world_size, config, chunks_per_rank=grid, sequences=movable))
                if items in seen:
                    continue
                seen.add(items)
                add(
                    _selection_assign(
                        items,
                        pinned_loads=pinned_loads,
                        pinned_compute=pinned_compute,
                        world=world_size,
                        capacity=max_tokens_per_rank,
                        offsets=offsets,
                        model=cost_model,
                        block_size=config.block_size,
                    ),
                    f"selection-local-{grid}x",
                )
    # Score the cheap local candidates first. They tighten the incumbent,
    # allowing an admissible bound to reject a fragmented full zigzag without
    # allocating O(documents * CP) span objects. No candidate-specific heuristic
    # threshold is used: any full zigzag that can win is still materialized.
    threshold = min(cost_model._migration_threshold_ms(baseline_ms), min(c[2] for c in candidates))
    if (
        lower_bound < threshold
        and _zigzag_meets_floor(boundaries, world_size, config.block_size, config.min_chunk_tokens, source_fragments)
        and _zigzag_lower_bound(boundaries, base, world_size, config.block_size, cost_model) < threshold
    ):
        zigzag = zigzag_thd_layout(boundaries, world_size=world_size, block_size=config.block_size)
        if all(rank.local_tokens <= max_tokens_per_rank for rank in zigzag) and all(
            span.tokens >= config.min_chunk_tokens or (span.sequence, span.token_start, span.token_end) in source_fragments
            for rank in zigzag
            for span in rank.spans
        ):
            candidates.append(("zigzag", zigzag, cost_model.estimate_ms(zigzag, block_size=config.block_size)))
    selected = min(candidates, key=lambda c: (c[2], c[0] != "contiguous", c[0]))
    if selected[2] >= cost_model._migration_threshold_ms(baseline_ms):
        selected = candidates[0]
    policy, ranks, score = selected
    metrics = (
        source_metrics
        if policy == "contiguous"
        else _metrics(
            [span for rank in ranks for span in rank.spans], world_size=world_size, block_size=config.block_size, config=config
        )
    )
    metrics = replace(metrics, score=score)
    return THDLayoutPlan(
        world_size,
        config.block_size,
        policy,
        ranks,
        metrics,
        tuple((name, value) for name, _, value in candidates),
        "ms",
    )


__all__ = [
    "THDLayoutMetrics",
    "THDLayoutPlan",
    "THDPlannerConfig",
    "THDSelectionCostModel",
    "THDRankLayout",
    "THDSpan",
    "plan_thd_layout",
    "plan_thd_selection",
    "zigzag_thd_layout",
]
