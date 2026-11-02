# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from cula.hybrid import index_pool


def test_cudnn_index_pool_loader_uses_csa_namespace(monkeypatch):
    forward = object()
    backward = object()
    fake_cudnn = SimpleNamespace(
        CSA=SimpleNamespace(
            csa_compressor_forward_wrapper=forward,
            csa_compressor_backward_wrapper=backward,
        )
    )
    monkeypatch.setattr(index_pool, "import_module", lambda name: fake_cudnn)
    index_pool._load_cudnn_index_pool_ops.cache_clear()
    try:
        assert index_pool._load_cudnn_index_pool_ops() == (forward, backward)
    finally:
        index_pool._load_cudnn_index_pool_ops.cache_clear()
