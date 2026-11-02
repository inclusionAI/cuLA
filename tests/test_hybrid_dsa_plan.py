# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from cula.hybrid import DsaSelectionPlan


def test_selection_plan_coalesces_short_packed_sequences():
    plan = DsaSelectionPlan.from_cu_seqlens(
        [0, 16, 36],
        [0, 4, 9],
        top_k=3,
        candidate_budget_bytes=1 << 20,
    )

    assert plan.total_queries == 36
    assert plan.total_blocks == 9
    assert plan.num_sequences == 2
    assert len(plan.windows) == 1
    window = plan.windows[0]
    assert window.cu_seqlens_q == (0, 16, 36)
    assert window.cu_seqlens_k == (0, 4, 9)
    assert window.q_causal_offsets == (0, 0)
    assert window.candidate_offsets == (0, 28, 73)


def test_selection_plan_splits_one_sequence_to_bound_peak_scratch():
    plan = DsaSelectionPlan.from_cu_seqlens(
        [0, 32],
        [0, 8],
        top_k=4,
        candidate_budget_bytes=80 * 4,
    )

    assert len(plan.windows) == 2
    assert max(window.candidate_floats for window in plan.windows) <= 80
    assert [window.query_start for window in plan.windows] == [0, plan.windows[0].query_end]
    assert plan.windows[-1].query_end == 32
    assert all(window.key_start == 0 and window.key_end == 8 for window in plan.windows)
    assert plan.windows[1].q_causal_offsets == (plan.windows[1].query_start,)


def test_selection_plan_keeps_chunked_prefill_position_offset():
    plan = DsaSelectionPlan.from_cu_seqlens(
        [0, 8],
        [0, 6],
        top_k=4,
        q_causal_offsets=[16],
        candidate_budget_bytes=1 << 20,
    )

    assert plan.windows[0].q_causal_offsets == (16,)
    assert plan.windows[0].candidate_floats == 38


def test_selection_plan_accepts_cpu_offset_tensor():
    plan = DsaSelectionPlan.from_cu_seqlens(
        [0, 8],
        [0, 6],
        top_k=4,
        q_causal_offsets=torch.tensor([16], dtype=torch.int32),
    )

    assert plan.q_causal_offsets == (16,)


def test_selection_plan_permits_a_sequence_without_a_complete_block():
    plan = DsaSelectionPlan.from_cu_seqlens(
        [0, 3, 11],
        [0, 0, 2],
        top_k=2,
        candidate_budget_bytes=1 << 20,
    )

    assert plan.windows[0].cu_seqlens_k == (0, 0, 2)
    assert plan.windows[0].candidate_offsets == (0, 0, 6)


def test_selection_plan_covers_heterogeneous_128k_without_exceeding_budget():
    lengths = [1024] * 20 + [55296] * 2
    cu_q = [0]
    cu_k = [0]
    for length in lengths:
        cu_q.append(cu_q[-1] + length)
        cu_k.append(cu_k[-1] + length // 4)

    budget = 512 << 20
    plan = DsaSelectionPlan.from_cu_seqlens(
        cu_q,
        cu_k,
        top_k=512,
        candidate_budget_bytes=budget,
    )

    assert plan.total_queries == 131072
    assert plan.total_blocks == 32768
    assert len(plan.windows) == 7
    assert plan.candidate_workspace_floats * 4 <= budget
    assert plan.windows[0].cu_seqlens_q == tuple(i * 1024 for i in range(21))
    assert plan.windows[0].candidate_floats * 4 < 10 << 20


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_k": 0}, "top_k"),
        ({"top_k": 3, "block_size": 0}, "block_size"),
        ({"top_k": 3, "candidate_budget_bytes": 3}, "candidate_budget_bytes"),
        ({"top_k": 3, "q_causal_offsets": [-1]}, "non-negative"),
    ],
)
def test_invalid_selection_plans_are_rejected(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        DsaSelectionPlan.from_cu_seqlens([0, 8], [0, 2], **kwargs)


def test_selection_plan_rejects_missing_visible_blocks():
    with pytest.raises(ValueError, match="requires at least 4"):
        DsaSelectionPlan.from_cu_seqlens([0, 16], [0, 3], top_k=3)


def test_selection_plan_rejects_budget_smaller_than_one_causal_row():
    with pytest.raises(ValueError, match="too small for one row"):
        DsaSelectionPlan.from_cu_seqlens(
            [0, 16],
            [0, 29],
            top_k=3,
            q_causal_offsets=[100],
            candidate_budget_bytes=4,
        )
