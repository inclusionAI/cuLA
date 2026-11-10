# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from cula.distributed.plan import KimiCPPlan, build_kimi_cp_plan


def _global_cu(lengths):
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return torch.tensor(values, dtype=torch.int32)


def test_256k_twenty_short_two_long_boundary_plans():
    global_cu = _global_cu([1024] * 20 + [120832, 120832])
    plans = [KimiCPPlan.from_global_cu_seqlens(global_cu, rank=rank, world_size=8) for rank in range(8)]

    assert all(plan.local_tokens == 32768 for plan in plans)
    assert plans[0].local_cu_seqlens == tuple(range(0, 20 * 1024 + 1, 1024)) + (32768,)
    assert (plans[0].first_sequence, plans[0].last_sequence) == (0, 20)
    assert (plans[0].pre_num_ranks, plans[0].post_num_ranks) == (0, 4)

    # Rank 4 ends the first long sample and starts the second one.
    assert (plans[4].first_sequence, plans[4].last_sequence) == (20, 21)
    assert plans[4].local_cu_seqlens == (0, 10240, 32768)
    assert (plans[4].first_boundary_end, plans[4].last_boundary_start) == (10240, 10240)
    assert (plans[4].pre_num_ranks, plans[4].post_num_ranks) == (4, 3)

    assert (plans[7].first_sequence, plans[7].last_sequence) == (21, 21)
    assert (plans[7].pre_num_ranks, plans[7].post_num_ranks) == (3, 0)


def test_custom_uneven_shards_derive_ordered_boundary_chains():
    global_cu = _global_cu([5, 11, 7])
    offsets = (0, 4, 10, 17, 23)
    plans = [
        KimiCPPlan.from_global_cu_seqlens(
            global_cu,
            rank=rank,
            world_size=4,
            shard_offsets=offsets,
        )
        for rank in range(4)
    ]

    assert plans[0].local_cu_seqlens == (0, 4)
    assert (plans[0].pre_num_ranks, plans[0].post_num_ranks) == (0, 1)
    assert plans[1].local_cu_seqlens == (0, 1, 6)
    assert (plans[1].first_sequence, plans[1].last_sequence) == (0, 1)
    assert (plans[1].pre_num_ranks, plans[1].post_num_ranks) == (1, 1)
    assert plans[2].local_cu_seqlens == (0, 6, 7)
    assert (plans[2].pre_num_ranks, plans[2].post_num_ranks) == (1, 1)
    assert plans[3].local_cu_seqlens == (0, 6)
    assert (plans[3].pre_num_ranks, plans[3].post_num_ranks) == (1, 0)


def test_build_single_rank_plan_without_distributed_runtime():
    plan = build_kimi_cp_plan([0, 3, 8])

    assert plan.rank == 0
    assert plan.world_size == 1
    assert plan.local_cu_seqlens == (0, 3, 8)
    assert plan.pre_num_ranks == plan.post_num_ranks == 0
    plan.validate_group(None)
    plan.validate_local_cu_seqlens(torch.tensor([0, 3, 8], dtype=torch.int32), device=torch.device("cpu"))


@pytest.mark.parametrize(
    ("global_cu", "message"),
    [
        (torch.tensor([1, 4], dtype=torch.int32), "start at zero"),
        (torch.tensor([0, 4, 4], dtype=torch.int32), "strictly increasing"),
        (torch.tensor([[0, 4]], dtype=torch.int32), "one-dimensional"),
        (torch.tensor([0.0, 4.0]), "int32 or int64"),
    ],
)
def test_invalid_global_boundaries_fail_without_cuda_sync(global_cu, message):
    with pytest.raises((TypeError, ValueError), match=message):
        KimiCPPlan.from_global_cu_seqlens(global_cu, rank=0, world_size=1)


def test_invalid_or_empty_shards_are_rejected():
    with pytest.raises(TypeError, match="integer sequence"):
        KimiCPPlan.from_global_cu_seqlens([0.0, 8.0], rank=0, world_size=1)
    with pytest.raises(ValueError, match="at least one token"):
        KimiCPPlan.from_global_cu_seqlens([0, 2], rank=0, world_size=4)
    with pytest.raises(ValueError, match="strictly increasing"):
        KimiCPPlan.from_global_cu_seqlens(
            [0, 8],
            rank=0,
            world_size=2,
            shard_offsets=(0, 0, 8),
        )


def test_cpu_local_boundaries_must_match_plan():
    plan = KimiCPPlan.from_global_cu_seqlens([0, 4, 8], rank=0, world_size=2)
    with pytest.raises(ValueError, match="do not match"):
        plan.validate_local_cu_seqlens(
            torch.tensor([0, 3], dtype=torch.int32),
            device=torch.device("cpu"),
        )
