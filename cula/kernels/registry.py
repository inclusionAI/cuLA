# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Static inventory for the supported standalone kernel surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

KernelFamily = Literal["kda", "lightning"]
KernelLevel = Literal["operator", "composite", "primitive"]
KernelStatus = Literal["stable", "specialized", "experimental"]


@dataclass(frozen=True, slots=True)
class KernelSpec:
    name: str
    family: KernelFamily
    level: KernelLevel
    architectures: tuple[str, ...]
    status: KernelStatus
    summary: str


_KERNELS = (
    KernelSpec("kda_prefill", "kda", "operator", ("sm90", "sm100", "sm103"), "stable", "Forward-only FlashKDA prefill."),
    KernelSpec("kda_decode", "kda", "operator", ("sm90+",), "stable", "Single-token and packed recurrent decode."),
    KernelSpec("kda_decode_mtp", "kda", "operator", ("sm90+",), "specialized", "Multi-token prediction verify/decode."),
    KernelSpec(
        "kda_conv_decode_mtp_kvbuffer",
        "kda",
        "operator",
        ("sm90+",),
        "specialized",
        "Fused short-convolution, KDA MTP, and KV-buffer update.",
    ),
    KernelSpec(
        "chunk_kda_fwd_intra",
        "kda",
        "composite",
        ("sm100", "sm103"),
        "specialized",
        "Chunk intra matrices plus W/U preparation from preprocessed gates.",
    ),
    KernelSpec(
        "chunk_kda_fwd_intra_sm100_from_gk",
        "kda",
        "primitive",
        ("sm100", "sm103"),
        "specialized",
        "Bitwise-oriented Aqk/Akk construction from cumulative FP32 gates.",
    ),
    KernelSpec(
        "recompute_w_u_fwd",
        "kda",
        "primitive",
        ("sm100", "sm103"),
        "specialized",
        "W/U/KG preparation for modular chunk KDA.",
    ),
    KernelSpec(
        "chunk_gated_delta_rule_fwd_h",
        "kda",
        "primitive",
        ("sm100", "sm103"),
        "specialized",
        "Inter-chunk recurrent-state propagation.",
    ),
    KernelSpec(
        "chunk_gla_fwd_o",
        "kda",
        "primitive",
        ("sm100", "sm103"),
        "specialized",
        "Chunk output projection from recurrent state and intra attention.",
    ),
    KernelSpec(
        "kda_bwd_wy_dqkg_sm100",
        "kda",
        "primitive",
        ("sm100", "sm103"),
        "experimental",
        "Fused W/Y-related KDA backward gradients.",
    ),
    KernelSpec(
        "lightning_attn_fwd",
        "lightning",
        "operator",
        ("sm90", "sm100", "sm103"),
        "stable",
        "Fixed-length Lightning Attention prefill.",
    ),
    KernelSpec(
        "lightning_attn_fwd_varlen",
        "lightning",
        "operator",
        ("sm90", "sm100", "sm103"),
        "stable",
        "Packed variable-length Lightning Attention prefill.",
    ),
    KernelSpec(
        "linear_attention_decode_mtp",
        "lightning",
        "operator",
        ("sm90+",),
        "specialized",
        "Multi-token Lightning Attention verification.",
    ),
    KernelSpec(
        "linear_attention_verify_kvbuffer",
        "lightning",
        "operator",
        ("sm90+",),
        "specialized",
        "MTP verification against a pooled state/KV buffer.",
    ),
)

_BY_NAME = {spec.name: spec for spec in _KERNELS}


def list_kernels(*, family: KernelFamily | None = None, level: KernelLevel | None = None) -> tuple[KernelSpec, ...]:
    """Return immutable kernel metadata, optionally filtered by family/level."""

    return tuple(
        spec for spec in _KERNELS if (family is None or spec.family == family) and (level is None or spec.level == level)
    )


def get_kernel(name: str) -> KernelSpec:
    """Return one kernel specification by public name."""

    try:
        return _BY_NAME[name]
    except KeyError as error:
        raise KeyError(f"unknown standalone kernel {name!r}") from error


__all__ = ["KernelSpec", "get_kernel", "list_kernels"]
