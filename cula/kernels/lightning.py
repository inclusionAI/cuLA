# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Standalone Lightning Attention operator exports."""

from cula.kernels._lazy import resolve

_LAZY = {
    "lightning_attn_fwd": ("cula.ops.lightning.prefill", "lightning_attn_fwd"),
    "lightning_attn_fwd_varlen": ("cula.ops.lightning.prefill", "lightning_attn_fwd_varlen"),
    "linear_attention_decode": ("cula.ops.lightning.decode", "linear_attention_decode"),
    "linear_attention_decode_mtp": ("cula.lightning.la_decode_mtp", "linear_attention_decode_mtp"),
    "linear_attention_verify_kvbuffer": (
        "cula.lightning.la_verify_kvbuffer",
        "linear_attention_verify_kvbuffer",
    ),
    "linear_attention_state_update_kvbuffer": (
        "cula.lightning.la_state_update_kvbuffer",
        "linear_attention_state_update_kvbuffer",
    ),
    "linear_attention_state_update_kvbuffer_fused": (
        "cula.lightning.la_state_update_kvbuffer",
        "linear_attention_state_update_kvbuffer_fused",
    ),
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    return resolve(_LAZY, __name__, name)


def __dir__():
    return __all__
