# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from cula.distributed import kda as cp


def test_boundary_summary_pack_round_trip():
    h = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    transition = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)

    packed = cp._pack_boundary_summary(h, transition)
    h_result, transition_result = cp._unpack_boundary_summaries(packed, h.shape[-2])

    assert packed.shape == (2, 7, 4)
    assert torch.equal(h_result, h)
    assert torch.equal(transition_result, transition)


def test_boundary_chain_selects_only_matching_ranks(monkeypatch):
    h_all = torch.arange(1, 5, dtype=torch.float32).reshape(4, 1, 1, 1)
    transition_all = torch.zeros(4, 1, 1, 1)

    def fake_merge(h, transition, *, reverse):
        assert transition.shape == h.shape
        states = torch.zeros_like(h)
        if reverse:
            for index in range(len(h) - 2, -1, -1):
                states[index] = states[index + 1] + h[index + 1]
        else:
            for index in range(1, len(h)):
                states[index] = states[index - 1] + h[index - 1]
        return states

    monkeypatch.setattr(cp, "_merge_rank_summaries", fake_merge)

    incoming = cp._merge_boundary_chain(
        h_all,
        transition_all,
        rank=2,
        num_remote_ranks=2,
        reverse=False,
    )
    outgoing = cp._merge_boundary_chain(
        h_all,
        transition_all,
        rank=1,
        num_remote_ranks=2,
        reverse=True,
    )
    no_remote_state = cp._merge_boundary_chain(
        h_all,
        transition_all,
        rank=0,
        num_remote_ranks=0,
        reverse=False,
    )

    assert incoming.item() == 3.0
    assert outgoing.item() == 7.0
    assert no_remote_state.item() == 0.0


def test_multi_rank_requires_boundary_metadata(monkeypatch):
    monkeypatch.setattr(cp, "_distributed_identity", lambda group: (2, 0))
    q = torch.zeros(2, 1, 1)
    cu = torch.tensor([0, 2], dtype=torch.int32)
    with pytest.raises(ValueError, match="pre_num_ranks and post_num_ranks"):
        cp.cudnn_kda_cp_thd(q, q, q, q, q[..., 0], cu)
