# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Host planning and execution-contract checks without a native dependency."""

from dataclasses import replace
from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch

from benchmarks.cp_layouts import query_layouts
from cula.distributed import THDPlannerConfig, THDSelectionCostModel, plan_thd_selection
from cula.hybrid import CsaCPSelectionPlan, CsaCPSelectionWorkspace, csa_cp, cudnn_csa_select_cp
from cula.hybrid.dsa import DsaSelectionPlan


def costs(world):
    return THDSelectionCostModel(
        candidate_ns=0.1,
        query_ns=8,
        span_us=5,
        bytes_per_query=2 * 4 * 2 + 2 * 2 + 8 * 8,
        intra_node_gbps=40,
        inter_node_gbps=20,
        rank_node_ids=(0,) * world,
    )


@pytest.mark.parametrize("tokens", [128 << 10, 256 << 10, 1 << 20])
@pytest.mark.parametrize("world", [8, 16])
@pytest.mark.parametrize("kind", ["single", "uniform", "near-uniform", "mixed", "long-tail"])
@pytest.mark.parametrize("policy", ["contiguous", "zigzag", "planner"])
def test_plans_match_existing_execution_metadata(tokens, world, kind, policy):
    if kind == "single":
        lengths = [tokens]
    elif kind == "uniform":
        lengths = [tokens // 1024] * 1024
    elif kind == "near-uniform":
        lengths = [tokens // 512 - 4, tokens // 512 + 4] * 256
    elif kind == "mixed":
        lengths = [64] * 1000 + [(tokens - 64000) // 2] * 2
    else:
        lengths = [64] * 500 + [256] * 12 + [tokens - 35072]
    cu = (0, *accumulate(lengths))
    model = costs(world)
    layouts, selected = query_layouts(cu, world, policy, cost_model=model, planner_strategy="selection")
    # First, interior and last source ranks cover both complete and split docs.
    for rank in (0, world // 2, world - 1):
        p = CsaCPSelectionPlan.from_cu_seqlens(cu, rank=rank, world_size=world, top_k=8, policy=policy, cost_model=model)
        assert p.ranks == layouts and p.policy == selected
        local = layouts[rank]
        assert p.exchange.source_rows == tokens // world
        assert p.exchange.target_rows == local.local_tokens
        ranges = tuple((s.sequence_start // 4, s.sequence_end // 4) for s in local.spans)
        assert p.key_ranges == ranges
        if not local.local_tokens:
            assert p.selection is None
            continue
        prefix = (0, *accumulate(z - a for a, z in ranges))
        expected = DsaSelectionPlan.from_cu_seqlens(
            local.local_cu_seqlens,
            prefix,
            top_k=8,
            q_causal_offsets=local.causal_offsets,
        )
        assert p.selection == expected


@pytest.mark.parametrize(
    "options",
    [
        {"world_size": 0},
        {"rank": True},
        {"rank": 8},
        {"top_k": 0},
        {"top_k": 2049},
        {"top_k": True},
        {"policy": "unknown"},
        {"max_tokens_per_rank": 15},
        {"candidate_budget_bytes": 0},
        {"candidate_budget_bytes": True},
        {"cost_model": costs(4)},
    ],
)
def test_invalid_plan_options(options):
    kwargs = dict(rank=0, world_size=8, top_k=8, policy="contiguous")
    kwargs.update(options)
    with pytest.raises((ValueError, TypeError)):
        CsaCPSelectionPlan.from_cu_seqlens((0, 128), **kwargs)


@pytest.mark.parametrize("cu", [(0, 5, 128), (0, 16, 12), (1, 128), (0, 2**31), (0, 4)])
def test_invalid_global_metadata_fails_on_every_rank(cu):
    for rank in range(8):
        with pytest.raises(ValueError):
            CsaCPSelectionPlan.from_cu_seqlens(cu, rank=rank, world_size=8, top_k=8, policy="contiguous")


def test_planner_needs_explicit_costs_and_never_reads_device_metadata():
    with pytest.raises(ValueError, match="cost_model"):
        CsaCPSelectionPlan.from_cu_seqlens((0, 128), rank=0, world_size=8, top_k=8)
    with pytest.raises(ValueError, match="CPU tensor"):
        CsaCPSelectionPlan.from_cu_seqlens(torch.empty(2, device="meta"), rank=0, world_size=8, top_k=8, policy="contiguous")


def test_empty_zigzag_destination_and_block_balanced_sources():
    plans = [CsaCPSelectionPlan.from_cu_seqlens((0, 4, 36), rank=r, world_size=8, top_k=8, policy="zigzag") for r in range(8)]
    assert all(p.exchange.source_rows % 4 == 0 for p in plans)
    assert sum(p.exchange.target_rows for p in plans) == 36
    # Unequal source counts are intentional: nine blocks across eight ranks.
    assert {p.exchange.source_rows for p in plans} == {4, 8}
    tiny = [
        CsaCPSelectionPlan.from_cu_seqlens(
            tuple(range(0, 36, 4)), rank=r, world_size=8, top_k=8, policy="zigzag", max_tokens_per_rank=32
        )
        for r in range(8)
    ]
    assert any(p.selection is None for p in tiny)


@pytest.mark.parametrize("identity", [True, False])
def test_source_order_result_lifetime_and_validation(monkeypatch, identity):
    plan = CsaCPSelectionPlan.from_cu_seqlens((0, 16), rank=0, world_size=1, top_k=8, policy="contiguous", cost_model=costs(1))
    plan = replace(plan, exchange=replace(plan.exchange, identity=identity))
    calls = []

    def route(tensor):
        calls.append("route")
        return tensor if identity else tensor.flip(0)

    def restore(tensor):
        calls.append("restore")
        return tensor if identity else tensor.flip(0)

    exchange = SimpleNamespace(device=torch.device("cpu"), group=None, distribute=route, restore=restore)
    native = SimpleNamespace(block_ids=torch.empty(16, 8, dtype=torch.int32), scores=torch.empty(16, 8))
    workspace = CsaCPSelectionWorkspace(plan, exchange, native, slice(0, 4), None)
    monkeypatch.setattr(csa_cp.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(csa_cp.dist, "get_world_size", lambda group: 1)
    seen = []

    def select(q, k, w, **kwargs):
        seen.append(kwargs)
        native.block_ids.copy_(q[:, 0, 0, None].expand(-1, 8).int())
        native.scores.copy_(q[:, 0, 0, None].expand(-1, 8).float())
        return native

    monkeypatch.setattr(csa_cp, "cudnn_dsa_select", select)
    qi = torch.arange(16, dtype=torch.bfloat16)[:, None, None].expand(-1, 2, 4).clone().requires_grad_()
    keys, weights = torch.ones(4, 1, 4, dtype=torch.bfloat16), torch.ones(16, 2, dtype=torch.bfloat16)
    result = cudnn_csa_select_cp(qi, keys, weights, workspace=workspace, sm_scale=0.5)
    assert torch.equal(result.block_ids[:, 0], torch.arange(16))
    assert torch.equal(result.scores[:, 0], torch.arange(16).float())
    native.block_ids.fill_(-1)
    native.scores.zero_()
    assert torch.equal(result.block_ids[:, 0], torch.arange(16))
    assert torch.equal(result.scores[:, 0], torch.arange(16).float())
    assert not result.scores.requires_grad
    assert calls == ["route", "route", "restore", "restore"]
    assert seen[0]["deterministic"] is False and seen[0]["sm_scale"] == 0.5
    for bad_q, bad_k, bad_w in [
        (qi.float(), keys, weights),
        (qi, keys.float(), weights),
        (qi[:, :, ::2], keys, weights),
        (qi, keys, weights[:, :1]),
    ]:
        with pytest.raises((TypeError, ValueError)):
            cudnn_csa_select_cp(bad_q, bad_k, bad_w, workspace=workspace)
    with pytest.raises(ValueError, match="bytes_per_query"):
        cudnn_csa_select_cp(qi, keys, weights.float(), workspace=workspace)
    with pytest.raises(ValueError, match="finite"):
        cudnn_csa_select_cp(qi, keys, weights, workspace=workspace, sm_scale=float("nan"))
    with pytest.raises(TypeError, match="bool"):
        cudnn_csa_select_cp(qi, keys, weights, workspace=workspace, deterministic=1)
    assert len(calls) == 4  # Invalid calls fail before any collective.


@pytest.mark.parametrize("world", [8, 16, 64])
@pytest.mark.parametrize("tokens", [128 << 10, 256 << 10])
def test_effective_chunk_floor_limits_high_cp_fragmentation(world, tokens):
    from cula.distributed.thd_planner import _adaptive_items

    config = THDPlannerConfig(min_chunk_tokens=1024)
    pieces = _adaptive_items((0, tokens), world, config, chunks_per_rank=6, sequences={0})
    assert all(item.tokens >= 1024 for item in pieces)
    assert len(pieces) <= tokens // 1024
    plan = plan_thd_selection(
        (0, tokens), world_size=world, max_tokens_per_rank=2 * tokens // world, cost_model=costs(world), config=config
    )
    assert all(span.tokens >= 1024 for rank in plan.ranks for span in rank.spans)
    assert sum(rank.local_tokens for rank in plan.ranks) == tokens


def test_chunk_floor_preserves_short_source_fragments_and_fallback():
    cu = (0, 4, 128 << 10)
    plan = plan_thd_selection(
        cu, world_size=64, max_tokens_per_rank=4096, cost_model=costs(64), config=THDPlannerConfig(min_chunk_tokens=4096)
    )
    assert sum(rank.local_tokens for rank in plan.ranks) == cu[-1]
    assert any(s.tokens == 4 for r in plan.ranks for s in r.spans)
    # An infeasible grain preference must not remove the contiguous fallback.
    assert plan.policy == "contiguous"


@pytest.mark.parametrize("floor", [-4, 1, True, 1.5])
def test_invalid_chunk_floor(floor):
    with pytest.raises((ValueError, TypeError)):
        plan_thd_selection(
            (0, 128),
            world_size=8,
            max_tokens_per_rank=32,
            cost_model=costs(8),
            config=THDPlannerConfig(min_chunk_tokens=floor),
        )


@pytest.mark.parametrize("world", [1, 8, 16, 64])
@pytest.mark.parametrize("lengths", [[4] * 1024, [4, 128, 512, 65536], [131072], [64] * 998 + [33600, 33600]])
def test_identity_metrics_match_general_cost_model(world, lengths):
    from cula.distributed.thd_planner import _contiguous_items, _metrics, _source_selection_candidate

    cu = (0, *accumulate(lengths))
    config, model = THDPlannerConfig(), costs(world)
    ranks, metrics = _source_selection_candidate(_contiguous_items(cu, world, 4), world, model)
    expected = _metrics([s for r in ranks for s in r.spans], world_size=world, block_size=4, config=config)
    score = model.estimate_ms(ranks, block_size=4)
    assert metrics == replace(expected, score=score)


@pytest.mark.parametrize("ranges", [[], [(0, 0)], [(3, 9)], [(7, 11), (0, 3), (7, 11)], [(5, 5), (12, 16)]])
def test_range_indices_preserve_duplicates_gaps_and_order(ranges):
    from cula._metadata import _range_indices

    actual = _range_indices(ranges, torch.device("cpu"))
    expected = torch.tensor([i for a, z in ranges for i in range(a, z)], dtype=torch.int64)
    assert torch.equal(actual, expected)
