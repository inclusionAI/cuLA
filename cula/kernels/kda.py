# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Standalone KDA operator and atomic-kernel exports."""

from cula.kernels._lazy import resolve

_LAZY = {
    # Complete inference operators.
    "kda_prefill": ("cula.kda.prefill", "kda_prefill"),
    "kda_decode": ("cula.ops.kda.decode.cute", "kda_decode"),
    "kda_packed_decode": ("cula.ops.kda.decode.cute", "kda_packed_decode"),
    "kda_decode_mtp": ("cula.ops.kda.decode.mtp", "kda_decode_mtp"),
    "kda_decode_mtp_recurrent": ("cula.ops.kda.decode.mtp", "kda_decode_mtp_recurrent"),
    "kda_decode_mtp_recurrent_ws": ("cula.ops.kda.decode.mtp", "kda_decode_mtp_recurrent_ws"),
    "kda_conv_decode_mtp_verify": ("cula.ops.kda.decode.mtp_conv", "kda_conv_decode_mtp_verify"),
    "kda_conv_decode_mtp_kvbuffer": (
        "cula.ops.kda.decode.mtp_conv_kvbuffer",
        "kda_conv_decode_mtp_kvbuffer",
    ),
    "kda_flush_kvbuffer": ("cula.ops.kda.decode.mtp_kvbuffer", "kda_flush_kvbuffer"),
    "kda_flush_kvbuffer_all_layers": (
        "cula.ops.kda.decode.mtp_kvbuffer",
        "kda_flush_kvbuffer_all_layers",
    ),
    # SM100 modular forward/training primitives.
    "chunk_kda_fwd_intra": ("cula.kda.chunk_intra", "chunk_kda_fwd_intra"),
    "chunk_kda_fwd_intra_sm100_from_gk": (
        "cula.ops.kda.sm100.intra_fused",
        "chunk_kda_fwd_intra_sm100_from_gk",
    ),
    "chunk_kda_fwd_full_sm100_from_gk": (
        "cula.ops.kda.sm100.full_port",
        "chunk_kda_fwd_full_sm100_from_gk",
    ),
    "recompute_w_u_fwd": ("cula.ops.kda.sm100.recompute_wu", "recompute_w_u_fwd"),
    "recompute_w_u_from_preprocessed": (
        "cula.ops.kda.sm100.recompute_wu",
        "recompute_w_u_from_preprocessed",
    ),
    "chunk_gated_delta_rule_fwd_h": (
        "cula.ops.kda.sm100.delta_h",
        "chunk_gated_delta_rule_fwd_h",
    ),
    "chunk_gla_fwd_o": ("cula.ops.kda.sm100.fwd_o", "chunk_gla_fwd_o"),
    "kda_bwd_wy_dqkg_sm100": (
        "cula.ops.kda.sm100.bwd_wy_dqkg",
        "chunk_kda_bwd_wy_dqkg_fused",
    ),
    "kda_bwd_wy_dqkg_sm90": (
        "cula.ops.kda.sm90.bwd_wy_dqkg",
        "chunk_kda_bwd_wy_dqkg_fused",
    ),
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    return resolve(_LAZY, __name__, name)


def __dir__():
    return __all__
