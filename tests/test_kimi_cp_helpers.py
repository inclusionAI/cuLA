# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch

from cula.distributed import kda as cp


def test_boundary_summary_pack_round_trip_preserves_fp32():
    h = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    transition = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)

    packed = cp._pack_boundary_summary(h, transition)
    h_result, transition_result = cp._unpack_boundary_summaries(packed, h.shape[-2])

    assert packed.shape == (2, 7, 4)
    assert packed.dtype == torch.float32
    assert torch.equal(h_result, h)
    assert torch.equal(transition_result, transition)


def test_boundary_chain_selects_only_matching_remote_ranks(monkeypatch):
    h_all = torch.arange(1, 5, dtype=torch.float32).reshape(4, 1, 1, 1)
    transition_all = torch.zeros(4, 1, 1, 1)
    calls = []

    def fake_merge(h, transition, *, reverse):
        calls.append((h.flatten().tolist(), reverse))
        assert transition.shape == h.shape
        return h.sum(dim=0)

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
    assert calls == [([1.0, 2.0], False), ([3.0, 4.0], True)]


def test_boundary_pack_rejects_precision_downgrade():
    h = torch.zeros(2, 3, 4, dtype=torch.bfloat16)
    transition = torch.zeros(2, 4, 4, dtype=torch.float32)

    try:
        cp._pack_boundary_summary(h, transition)
    except TypeError as error:
        assert "FP32" in str(error)
    else:
        raise AssertionError("BF16 boundary state must be rejected")


def test_merge_uses_cudnn_fixed_ieee_api_without_precision_option(monkeypatch):
    observed = {}

    def fake_merge(h, transition, *, reverse):
        observed.update(h=h, transition=transition, reverse=reverse)
        return {"final_state": h[-1:]}

    monkeypatch.setattr(cp, "_load_cudnn_cp_ops", lambda: (fake_merge, None, None, None))
    h = torch.ones(2, 1, 1, 1, dtype=torch.float32)
    transition = torch.zeros_like(h)

    result = cp._merge_rank_summaries(h, transition, reverse=True)

    assert result.item() == 1.0
    assert observed["reverse"] is True
