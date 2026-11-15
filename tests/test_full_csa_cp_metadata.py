# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the experimental full-CSA benchmark's addressing contract."""

import pytest
import torch

from benchmarks.bench_full_csa_cp import expand_blocks, query_metadata
from cula.distributed import THDRankLayout, THDSpan


@pytest.mark.parametrize("visible", [1, 2, 3, 4, 5, 6, 7, 8, 17])
@pytest.mark.parametrize("origin", [0, 128, 1024])
def test_complete_blocks_and_causal_tail(visible, origin):
    count = min(visible // 4, 4)
    ids = torch.full((1, 8), -1, dtype=torch.int32)
    # Valid entries intentionally have holes; ids/scores must stay paired.
    ids[0, 1 : 2 * count : 2] = torch.arange(count)
    scores = ids.float() + 0.25
    sorted_ids, sorted_scores, members, tokens, lengths = expand_blocks(
        ids, scores, torch.tensor([origin]), torch.tensor([visible])
    )
    actual = tokens[0, : lengths.item()].tolist()
    expected = list(range(origin, origin + 4 * count))
    expected += list(range(origin + visible // 4 * 4, origin + visible))
    assert actual == expected
    assert len(actual) == len(set(actual))
    assert torch.all(tokens[0, lengths.item() :] == -1)
    assert torch.all(members[0, 4 * count :] == -1)
    assert torch.equal(sorted_scores, sorted_ids.float() + 0.25)


def test_all_selected_blocks_leave_room_for_three_tail_tokens():
    ids = torch.arange(128, dtype=torch.int32)[None]
    _, _, _, tokens, lengths = expand_blocks(ids, ids.float(), torch.tensor([2048]), torch.tensor([515]))
    assert lengths.item() == 515
    assert tokens.shape[1] % 128 == 0
    assert torch.equal(tokens[0, :515], torch.arange(2048, 2048 + 515))
    assert torch.all(tokens[0, 515:] == -1)


@pytest.mark.parametrize("spans", [(), ((128, 160, 64, 256),), ((512, 520, 256, 1024), (64, 128, 0, 256))])
def test_bulk_metadata_preserves_addresses_including_reordered_and_empty_spans(spans):
    layout = THDRankLayout(0, tuple(THDSpan(i, a, z, start, end, 0) for i, (a, z, start, end) in enumerate(spans)))
    origin, visible, rows = query_metadata(layout, "cpu")
    expected = [(start, t - start + 1, t) for a, z, start, end in spans for t in range(a, z)]
    assert torch.stack((origin, visible, rows), dim=1).tolist() == [list(row) for row in expected]
    assert origin.dtype == visible.dtype == rows.dtype == torch.int32
    assert all(x.is_contiguous() for x in (origin, visible, rows))


def test_bulk_metadata_is_prepared_on_cpu_under_a_different_default_device():
    layout = THDRankLayout(0, (THDSpan(0, 4, 8, 0, 16, 0),))
    with torch.device("meta"):
        origin, visible, rows = query_metadata(layout, "cpu")
    assert origin.tolist() == [0] * 4
    assert visible.tolist() == [5, 6, 7, 8]
    assert rows.tolist() == [4, 5, 6, 7]
