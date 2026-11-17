# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""CPU protocol tests, including unequal counts and a nontrivial receive order."""

from dataclasses import replace

import pytest
import torch

from benchmarks.cp_layouts import query_layouts
from cula._metadata import balanced_shards
from cula.distributed import THDRedistributionPlan, THDRedistributor


@pytest.mark.parametrize("policy", ["contiguous", "zigzag", "planner"])
@pytest.mark.parametrize("boundaries", [(0, 64), (0, 4, 12, 32, 64), (0, 32768), (0, 4, 8, 12, 16)])
def test_exchange_and_its_transpose(monkeypatch, policy, boundaries):
    world = 4
    layouts, _ = query_layouts(boundaries, world, policy)
    # Reverse local span order to require a genuine receive permutation too.
    if policy != "contiguous":
        layouts = tuple(replace(x, spans=x.spans[::-1]) for x in layouts)
    offsets = balanced_shards(boundaries[-1], world)
    plans = [THDRedistributionPlan.build(layouts, offsets, rank) for rank in range(world)]
    source = [torch.arange(offsets[r], offsets[r + 1], dtype=torch.float64)[:, None].repeat(1, 3) for r in range(world)]
    target = [
        (
            torch.cat([torch.arange(s.token_start, s.token_end, dtype=torch.float64) for s in x.spans])
            if x.spans
            else torch.empty(0, dtype=torch.float64)
        )[:, None].repeat(1, 3)
        for x in layouts
    ]
    for rank, plan in enumerate(plans):
        monkeypatch.setattr("torch.distributed.get_world_size", lambda group: world)
        monkeypatch.setattr("torch.distributed.get_rank", lambda group: rank)
        exchange = THDRedistributor(plan, "cpu")
        calls = []
        reverse, factor = False, 1

        def collective(out, packed, output_split_sizes, input_split_sizes, group):
            calls.append(reverse)
            tensors = [x * factor for x in (target if reverse else source)]
            parts = []
            for sender, peer_plan in enumerate(plans):
                from cula.distributed.redistribute import _permutations

                ranges = peer_plan.receive_ranges if reverse else peer_plan.send_ranges
                rows = peer_plan.target_rows if reverse else peer_plan.source_rows
                order, _ = _permutations(ranges, rows, "cpu")
                data = tensors[sender] if order is None else tensors[sender].index_select(0, order)
                counts = peer_plan.receive_counts if reverse else peer_plan.send_counts
                begin = sum(counts[:rank])
                parts.append(data[begin : begin + counts[rank]])
            out.copy_(torch.cat(parts))
            expected_counts = plan.receive_counts if reverse else plan.send_counts
            assert tuple(input_split_sizes) == expected_counts
            assert sum(output_split_sizes) == out.shape[0]
            # Verify the sender's local packing as well, independently of the
            # mocked receive construction.
            ranges = plan.receive_ranges if reverse else plan.send_ranges
            idx = torch.tensor([i for a, z in ranges for i in range(a, z)], dtype=torch.int64)
            assert torch.equal(packed, tensors[rank].index_select(0, idx))

        monkeypatch.setattr("torch.distributed.all_to_all_single", collective)
        leaf = source[rank].clone().requires_grad_()
        routed = exchange.distribute(leaf)
        assert torch.equal(routed, target[rank])
        reverse, factor = True, 2
        # Nonuniform gradients expose a wrong inverse even when a sum-of-rows
        # test would pass. Empty destinations still participate in backward.
        routed.backward(target[rank] * factor)
        assert torch.equal(leaf.grad, source[rank] * factor)
        reverse, factor = True, 1
        leaf = target[rank].clone().requires_grad_()
        restored = exchange.restore(leaf)
        assert torch.equal(restored, source[rank])
        reverse, factor = False, 3
        restored.backward(source[rank] * factor)
        assert torch.equal(leaf.grad, target[rank] * factor)
        assert calls == ([] if plan.identity else [False, True, True, False])


def test_reject_overlap():
    layouts, _ = query_layouts((0, 64), 4, "contiguous")
    broken = (layouts[0], replace(layouts[1], spans=tuple(replace(span, owner=1) for span in layouts[0].spans)), *layouts[2:])
    with pytest.raises(ValueError, match="without gaps or overlap"):
        THDRedistributionPlan.build(broken, (0, 16, 32, 48, 64), 0)


@pytest.mark.parametrize("field,value", [("owner", False), ("token_start", 0.0), ("token_end", 16.0), ("sequence_end", 64.0)])
def test_reject_noninteger_span_metadata_before_collectives(field, value):
    layouts, _ = query_layouts((0, 64), 4, "contiguous")
    span = replace(layouts[0].spans[0], **{field: value})
    layouts = (replace(layouts[0], spans=(span,)), *layouts[1:])
    with pytest.raises(TypeError, match="must be integers"):
        THDRedistributionPlan.build(layouts, (0, 16, 32, 48, 64), 0)


def test_locally_unchanged_rank_cannot_skip_a_peer_exchange():
    layouts, _ = query_layouts((0, 64), 4, "contiguous")
    layouts = tuple(
        replace(layouts[source], rank=target, spans=tuple(replace(s, owner=target) for s in layouts[source].spans))
        for target, source in enumerate((0, 2, 1, 3))
    )
    plans = [THDRedistributionPlan.build(layouts, (0, 16, 32, 48, 64), r) for r in range(4)]
    assert plans[0].remote_rows == plans[3].remote_rows == 0
    assert all(not plan.identity for plan in plans)


def test_prepare_indices_do_not_follow_the_framework_default_device():
    from cula.distributed.redistribute import _permutations

    with torch.device("meta"):
        order, inverse = _permutations(((4, 8), (0, 4)), 8, "cpu")
    assert order.device.type == inverse.device.type == "cpu"
    assert order.tolist() == [4, 5, 6, 7, 0, 1, 2, 3]
    assert inverse.tolist() == [4, 5, 6, 7, 0, 1, 2, 3]


def test_unindexed_cuda_device_is_resolved_at_prepare(monkeypatch):
    layouts, _ = query_layouts((0, 64), 4, "contiguous")
    plan = THDRedistributionPlan.build(layouts, (0, 16, 32, 48, 64), 2)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 4)
    monkeypatch.setattr("torch.distributed.get_rank", lambda group: 2)
    monkeypatch.setattr("torch.cuda.current_device", lambda: 3)
    # Identity needs no device allocation, so this also runs without CUDA.
    exchange = THDRedistributor(plan, "cuda")
    assert exchange.device == torch.device("cuda:3")
