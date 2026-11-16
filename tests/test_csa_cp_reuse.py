# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Host checks for preparation accounting and multi-layer fixture ownership."""

from types import SimpleNamespace

import pytest
import torch

from benchmarks.bench_csa_cp_reuse import POLICIES, clone_holder, compare_tensors, summarize
from benchmarks.bench_full_csa_cp import FullCSA


def fixture():
    layer = FullCSA.__new__(FullCSA)
    layer.selection_api = True
    layer.redistribution_scope = "selection"
    layer.cu, layer.rank, layer.world, layer.top_k, layer.heads = (0, 16), 0, 1, 8, 64
    layer.source_q = torch.zeros(16, 1, 1)
    layer.indexer = SimpleNamespace(select_plan=object(), select_workspace=object(), qi=object())
    layer.last, layer.events, layer._layout_options = {}, [], {"layout": "contiguous"}
    for name in (
        "cp_selection",
        "selected_policy",
        "selection_layout",
        "selection_exchange",
        "layout",
        "exchange",
        "n",
        "doc_start",
        "visible",
        "query_global_rows",
        "selection_key_slice",
        "selection_key_ids",
    ):
        setattr(layer, name, object())
    return layer


def test_prepared_layout_shares_only_metadata():
    first, second = fixture(), fixture()
    source, qi = second.source_q, second.indexer.qi
    second.use_prepared_layout(first)
    assert second.cp_selection is first.cp_selection
    assert second.indexer.select_workspace is first.indexer.select_workspace
    assert second.source_q is source and second.indexer.qi is qi
    assert second._layout_options == first._layout_options
    assert second._layout_options is not first._layout_options
    holder = clone_holder(second)
    assert holder.source_q is second.source_q and holder.indexer.qi is qi
    assert holder.indexer is not second.indexer
    assert holder.events is not second.events and holder.last is not second.last


@pytest.mark.parametrize("change", ["cu", "heads", "scope", "pending"])
def test_layout_rebinding_rejects_incompatible_or_pending_layers(change):
    first, second = fixture(), fixture()
    if change == "pending":
        second.last = {"saved_forward": object()}
    elif change == "scope":
        second.redistribution_scope = "queries"
    elif change == "cu":
        second.cu = (0, 8, 16)
    else:
        second.heads = 32
    with pytest.raises(ValueError):
        second.use_prepared_layout(first)


def test_accounting_uses_critical_rank_and_charges_preparation_once():
    records = []
    for times, prep in [([4, 12, 8], [1, 5, 3]), ([10, 6, 9], [2, 3, 4])]:
        records.append(
            dict(
                samples={p: [dict(stack_ms=t, allocated_gib=2, reserved_gib=3) for t in times] for p in POLICIES},
                preparation={p: [dict(common_ms=1, layout_ms=x) for x in prep] for p in POLICIES},
            )
        )
    for value in summarize(records, 3, 3, 4).values():
        assert value["stack_ms"] == 10
        assert value["prepare_ms"] == 5
        assert value["prepare_plus_stack_ms"] == 15
        assert value["amortized_per_layer_ms"] == 3.75
        assert value["prepare_fraction"] == pytest.approx(1 / 3)


def test_tensor_error_uses_contiguous_reference_norm():
    error = compare_tensors(torch.tensor([2.0, 4.0]), torch.tensor([1.0, 2.0]))
    assert error["relative_rms"] == 1.0 and error["max_abs"] == 2.0 and error["finite"]
    assert not compare_tensors(torch.tensor([float("nan")]), torch.ones(1))["finite"]
