# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

import cula.kda as kda
from cula.kda.prefill import kda_prefill
from cula.ops.kda.cp_mode import CPMode


def _inputs():
    q = torch.empty(1, 2, 1, 128)
    return q, q.clone(), q.clone(), q.clone(), torch.empty(1, 2, 1)


def test_public_prefill_routes_directly_to_flashkda(monkeypatch):
    expected = (object(), object())
    seen = {}

    def fake_prefill(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return expected

    monkeypatch.setitem(sys.modules, "cula.kda.flashkda", SimpleNamespace(cula_kda_prefill=fake_prefill))
    q, k, v, g, beta = _inputs()

    actual = kda_prefill(
        q,
        k,
        v,
        g,
        beta,
        A_log=torch.empty(1),
        dt_bias=torch.empty(1, 128),
        use_intracard_cp="auto",
    )

    assert actual is expected
    assert all(actual is expected for actual, expected in zip(seen["args"], (q, k, v, g, beta), strict=True))
    assert seen["kwargs"]["use_intracard_cp"] is CPMode.AUTO
    assert seen["kwargs"]["safe_gate"] is True
    assert seen["kwargs"]["use_gate_in_kernel"] is True
    assert seen["kwargs"]["use_qk_l2norm_in_kernel"] is True


def test_removed_native_entry_points_are_not_exported():
    assert "kda_prefill_hopper" not in kda.__all__
    assert "kda_prefill_hopper_opt" not in kda.__all__
    assert "kda_prefill_hopper_auto" not in kda.__all__


def test_public_prefill_rejects_unknown_keyword_before_launch():
    with pytest.raises(TypeError, match="unexpected keyword"):
        kda_prefill(*_inputs(), unknown_native_option=True)
