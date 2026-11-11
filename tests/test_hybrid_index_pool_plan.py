# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import random

import pytest
import torch

from cula.hybrid import IndexPoolPlan, IndexPoolShardPlan


def test_index_pool_plan_compacts_each_packed_sequence_independently():
    plan = IndexPoolPlan.from_cu_seqlens(
        torch.tensor([0, 5, 16, 24, 27], dtype=torch.int32),
        head_dim=128,
        ratio=4,
        coff=1,
    )

    assert plan.cu_seqlens == (0, 5, 16, 24, 27)
    assert plan.cu_seqlens_comp == (0, 1, 3, 5, 5)
    assert plan.total_tokens == 27
    assert plan.total_comp == 5
    assert plan.num_sequences == 4
    assert plan.input_width == 128


def test_overlapping_plan_changes_projection_width_not_block_count():
    plan = IndexPoolPlan.from_cu_seqlens([0, 8, 17], head_dim=512, ratio=4, coff=2)

    assert plan.input_width == 1024
    assert plan.cu_seqlens_comp == (0, 2, 4)


def test_plan_materializes_reusable_int32_metadata():
    plan = IndexPoolPlan.from_cu_seqlens([0, 5, 16], head_dim=128)
    cu_seqlens, cu_seqlens_comp = plan.materialize("cpu")

    assert cu_seqlens.dtype == torch.int32
    assert cu_seqlens_comp.dtype == torch.int32
    assert torch.equal(cu_seqlens, torch.tensor([0, 5, 16], dtype=torch.int32))
    assert torch.equal(cu_seqlens_comp, torch.tensor([0, 1, 3], dtype=torch.int32))


def test_aligned_cp_shards_pool_locally_and_gather_only_blocks():
    # Rank boundary 13 is sequence 1's block-2 boundary: sequence 1 starts at 5.
    plans = [
        IndexPoolShardPlan.from_global_cu_seqlens(
            [0, 5, 16, 24, 27],
            rank=rank,
            world_size=2,
            head_dim=128,
            shard_offsets=(0, 13, 27),
        )
        for rank in range(2)
    ]

    assert all(plan.all_shards_direct for plan in plans)
    assert all(plan.direct_local_pool for plan in plans)
    assert plans[0].block_offsets == (0, 3, 5)
    assert plans[0].local_plan is not None
    assert plans[0].local_plan.cu_seqlens == (0, 5, 13)
    assert plans[0].local_plan.cu_seqlens_comp == (0, 1, 3)
    assert plans[1].local_plan is not None
    assert plans[1].local_plan.cu_seqlens == (0, 3, 11, 14)
    assert plans[1].local_plan.cu_seqlens_comp == (0, 0, 2, 2)
    assert plans[0].global_blocks == 5
    assert plans[0].block_counts == (3, 2)
    assert not plans[0].equal_block_counts


def test_misaligned_cp_shard_derives_bounded_pooling_halo():
    plans = [
        IndexPoolShardPlan.from_global_cu_seqlens(
            [0, 5, 16, 24, 27],
            rank=rank,
            world_size=2,
            head_dim=128,
            shard_offsets=(0, 12, 27),
        )
        for rank in range(2)
    ]

    assert not plans[0].all_shards_direct
    assert plans[0].recv_right_tokens == 1
    assert plans[1].send_left_tokens == 1
    assert plans[0].block_offsets == (0, 3, 5)
    assert plans[0].local_plan is None
    assert plans[1].local_plan is None
    assert all(plan.halo_supported for plan in plans)


def test_tiny_shards_declare_single_neighbor_halo_unsupported():
    plan = IndexPoolShardPlan.from_global_cu_seqlens(
        [0, 8],
        rank=0,
        world_size=4,
        head_dim=128,
        shard_offsets=(0, 2, 4, 6, 8),
    )

    assert not plan.halo_supported


def test_cp_seam_inside_dropped_sequence_tail_needs_no_halo():
    plans = [
        IndexPoolShardPlan.from_global_cu_seqlens(
            [0, 6, 14],
            rank=rank,
            world_size=2,
            head_dim=128,
            shard_offsets=(0, 5, 14),
        )
        for rank in range(2)
    ]

    assert plans[0].recv_right_tokens == 0
    assert plans[1].send_left_tokens == 0
    assert all(plan.all_shards_direct for plan in plans)
    assert plans[0].local_plan is not None
    assert plans[1].local_plan is not None


def test_random_shards_assign_every_complete_block_exactly_once():
    generator = random.Random(47)
    for _ in range(200):
        lengths = [generator.randint(1, 64) for _ in range(generator.randint(1, 12))]
        cu_seqlens = [0]
        for length in lengths:
            cu_seqlens.append(cu_seqlens[-1] + length)
        world_size = generator.randint(1, min(8, cu_seqlens[-1]))
        cuts = sorted(generator.sample(range(1, cu_seqlens[-1]), world_size - 1))
        shard_offsets = (0, *cuts, cu_seqlens[-1])
        plans = [
            IndexPoolShardPlan.from_global_cu_seqlens(
                cu_seqlens,
                rank=rank,
                world_size=world_size,
                head_dim=128,
                shard_offsets=shard_offsets,
            )
            for rank in range(world_size)
        ]

        expected_blocks = sum(length // 4 for length in lengths)
        assert plans[0].block_offsets[-1] == expected_blocks
        assert sum(plan.local_blocks for plan in plans) == expected_blocks
        for rank in range(world_size - 1):
            assert plans[rank].recv_right_tokens == plans[rank + 1].send_left_tokens


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"head_dim": 0}, "head_dim"),
        ({"head_dim": 128, "ratio": 0}, "ratio"),
        ({"head_dim": 128, "coff": 3}, "coff"),
        ({"head_dim": 128, "ratio": 32}, "no complete"),
    ],
)
def test_invalid_index_pool_plans_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        IndexPoolPlan.from_cu_seqlens([0, 8], **kwargs)
