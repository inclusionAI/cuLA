# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

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
