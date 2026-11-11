# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from cula.hybrid import dsa


def test_cudnn_dsa_loader_prefers_public_namespace(monkeypatch):
    select = object()
    fake_cudnn = SimpleNamespace(DSA=SimpleNamespace(indexer_forward_top_k_wrapper=select))
    monkeypatch.setattr(dsa, "import_module", lambda name: fake_cudnn)
    dsa._load_cudnn_dsa_select.cache_clear()
    try:
        assert dsa._load_cudnn_dsa_select() is select
    finally:
        dsa._load_cudnn_dsa_select.cache_clear()


def test_cudnn_dsa_selection_defaults_to_nondeterministic(monkeypatch):
    observed = []

    def select(*args, **kwargs):
        del args
        observed.append(kwargs["deterministic"])

    window = dsa.DsaSelectionWindow(
        query_start=0,
        query_end=1,
        key_start=0,
        key_end=1,
        cu_seqlens_q=(0, 1),
        cu_seqlens_k=(0, 1),
        q_causal_offsets=(0,),
        candidate_offsets=(0, 1),
    )
    plan = dsa.DsaSelectionPlan(
        block_size=4,
        top_k=1,
        cu_seqlens_q=(0, 1),
        cu_seqlens_k=(0, 1),
        q_causal_offsets=(0,),
        candidate_budget_bytes=4,
        windows=(window,),
    )
    metadata = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 1], dtype=torch.int32),
        q_causal_offsets=None,
        candidate_offsets=torch.tensor([0, 1], dtype=torch.int64),
    )
    workspace = dsa.DsaSelectionWorkspace(
        plan=plan,
        candidates=torch.empty(1),
        block_ids=torch.empty((1, 1), dtype=torch.int32),
        scores=torch.empty((1, 1)),
        windows=(metadata,),
    )
    monkeypatch.setattr(dsa, "_validate_selection_inputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(dsa, "_load_cudnn_dsa_select", lambda: select)

    dsa.cudnn_dsa_select(
        torch.empty((1, 1, 1)),
        torch.empty((1, 1, 1)),
        torch.empty((1, 1)),
        plan=plan,
        workspace=workspace,
    )

    assert observed == [False]
