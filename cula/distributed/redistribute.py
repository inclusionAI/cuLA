# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Lossless row redistribution from contiguous CP shards to THD span layouts.

Only row ownership/order changes. Floating point values are neither reduced
nor cast. The backward is the exact inverse exchange. Preparation is CPU
metadata work plus device index construction; execution is eager PyTorch/NCCL
and allocates outputs. No CUDA Graph contract is implied.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable

from cula._metadata import _range_indices
from cula.distributed.thd_planner import THDRankLayout


@dataclass(frozen=True)
class THDRedistributionPlan:
    rank: int
    world_size: int
    source_rows: int
    target_rows: int
    send_counts: tuple[int, ...]
    receive_counts: tuple[int, ...]
    send_ranges: tuple[tuple[int, int], ...]
    receive_ranges: tuple[tuple[int, int], ...]
    identity: bool

    @classmethod
    def build(cls, ranks: tuple[THDRankLayout, ...], source_offsets: tuple[int, ...], rank: int):
        world = len(ranks)
        if world < 1 or len(source_offsets) != world + 1 or source_offsets[0] != 0:
            raise ValueError("source_offsets must describe one contiguous shard per rank starting at zero")
        if any(isinstance(x, bool) or not isinstance(x, int) for x in source_offsets):
            raise TypeError("source offsets must be integers")
        if any(a > b for a, b in zip(source_offsets, source_offsets[1:])):
            raise ValueError("source offsets must be nondecreasing")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world:
            raise ValueError("invalid rank")
        intervals = []
        for owner, layout in enumerate(ranks):
            if isinstance(layout.rank, bool) or not isinstance(layout.rank, int):
                raise TypeError("layout ranks must be integers")
            if layout.rank != owner:
                raise ValueError("rank layouts must be in group-rank order")
            for span in layout.spans:
                if any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in (span.owner, span.token_start, span.token_end, span.sequence_start, span.sequence_end)
                ):
                    raise TypeError("span owners and interval bounds must be integers")
                if span.owner != owner:
                    raise ValueError("span owner does not match its rank layout")
                if not (span.sequence_start <= span.token_start < span.token_end <= span.sequence_end):
                    raise ValueError("span must be a nonempty interval within its sequence")
                intervals.append((span.token_start, span.token_end))
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor:
                raise ValueError("target layout must cover every source row exactly once, without gaps or overlap")
            cursor = end
        if cursor != source_offsets[-1]:
            raise ValueError("target layout does not cover the source extent")

        send_ranges, receive_ranges = [], []
        send_counts, receive_counts = [0] * world, [0] * world
        identity = True
        left, right = source_offsets[rank : rank + 2]
        for destination, layout in enumerate(ranks):
            target_offset = 0
            for span in layout.spans:
                # Global identity is required: one locally unchanged rank must
                # still join the collective if any peer changes ownership.
                identity &= span.token_start == source_offsets[destination] + target_offset
                a, z = max(left, span.token_start), min(right, span.token_end)
                if a < z:
                    send_counts[destination] += z - a
                    send_ranges.append((a - left, z - left))
                target_offset += span.tokens
            identity &= target_offset == source_offsets[destination + 1] - source_offsets[destination]

        # Receive buffer groups rows by source, then by destination span order.
        for source, (left, right) in enumerate(zip(source_offsets, source_offsets[1:])):
            target_offset = 0
            for span in ranks[rank].spans:
                a, z = max(left, span.token_start), min(right, span.token_end)
                if a < z:
                    receive_counts[source] += z - a
                    receive_ranges.append((target_offset + a - span.token_start, target_offset + z - span.token_start))
                target_offset += span.tokens
        return cls(
            rank,
            world,
            source_offsets[rank + 1] - source_offsets[rank],
            ranks[rank].local_tokens,
            tuple(send_counts),
            tuple(receive_counts),
            tuple(send_ranges),
            tuple(receive_ranges),
            bool(identity),
        )

    @property
    def remote_rows(self):
        return sum(self.send_counts) - self.send_counts[self.rank]


def _permutations(ranges, rows, device):
    # Avoid allocating or applying an index tensor for a naturally ordered side.
    cursor, ordered = 0, True
    for start, end in ranges:
        ordered &= start == cursor
        cursor = end
    if ordered and cursor == rows:
        return None, None
    order = _range_indices(ranges, device)
    if order.numel() != rows:
        raise ValueError("incomplete permutation")
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(rows, device=device)
    return order, inverse


class THDRedistributor:
    """Prepared eager exchange with explicit forward and inverse mappings.

    Every rank must prepare from identical global layouts and call exchanges
    in the same order. Trailing shapes and dtypes must agree across peers.
    The process group must stay alive for every backward using this object.
    """

    def __init__(self, plan: THDRedistributionPlan, device, *, group=None):
        if dist.get_world_size(group) != plan.world_size or dist.get_rank(group) != plan.rank:
            raise ValueError("plan identity does not match the process group")
        self.plan, self.group, self.device = plan, group, torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.send_order, self.send_inverse = _permutations(plan.send_ranges, plan.source_rows, self.device)
        self.receive_order, self.receive_inverse = _permutations(plan.receive_ranges, plan.target_rows, self.device)

    def _exchange(self, tensor, reverse):
        plan = self.plan
        rows = plan.target_rows if reverse else plan.source_rows
        if tensor.ndim < 1 or tensor.shape[0] != rows or tensor.device != self.device:
            raise ValueError(f"expected {rows} rows on {self.device}, got {tuple(tensor.shape)} on {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError("redistribution requires contiguous rows; explicit packing belongs to the caller")
        if plan.identity:
            return tensor
        order = self.receive_order if reverse else self.send_order
        inverse = self.send_inverse if reverse else self.receive_inverse
        send_counts, receive_counts = (
            (plan.receive_counts, plan.send_counts) if reverse else (plan.send_counts, plan.receive_counts)
        )
        packed = tensor if order is None else tensor.index_select(0, order)
        received = tensor.new_empty((sum(receive_counts), *tensor.shape[1:]))
        dist.all_to_all_single(
            received, packed, output_split_sizes=list(receive_counts), input_split_sizes=list(send_counts), group=self.group
        )
        return received if inverse is None else received.index_select(0, inverse)

    def distribute(self, tensor):
        return _Redistribute.apply(tensor, self, False)

    def restore(self, tensor):
        return _Redistribute.apply(tensor, self, True)


class _Redistribute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, exchange, reverse):
        ctx.exchange, ctx.reverse = exchange, reverse
        return exchange._exchange(tensor, reverse)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        return ctx.exchange._exchange(grad.contiguous(), not ctx.reverse), None, None
