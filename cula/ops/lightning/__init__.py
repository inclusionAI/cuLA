# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Lightning Attention backend kernels (prefill + decode) — non-KDA."""

__all__ = [
    "get_lightning_attn_prefill_backend_identity",
    "lightning_attn_fwd",
    "lightning_attn_fwd_varlen",
]

_LAZY = {
    "get_lightning_attn_prefill_backend_identity": (
        "cula.ops.lightning.prefill",
        "get_lightning_attn_prefill_backend_identity",
    ),
    "lightning_attn_fwd": ("cula.ops.lightning.prefill", "lightning_attn_fwd"),
    "lightning_attn_fwd_varlen": (
        "cula.ops.lightning.prefill",
        "lightning_attn_fwd_varlen",
    ),
}


def __getattr__(name):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(target[0]), target[1])


def __dir__():
    return sorted(__all__)
