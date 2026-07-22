# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Stable host-side contract for the Hopper GDN prefill kernel."""

from __future__ import annotations

from enum import Enum

EXPECTED_CUTLASS_DSL_VERSION = "4.5.1"
SM90_BACKEND_ID = "sm90_cutedsl_gdn"

CHUNK_SIZE = 64
HEAD_SIZE = 128
THREADS_PER_CTA = 512


class HeadMode(str, Enum):  # noqa: UP042 - Python 3.10 remains supported
    MHA = "mha"
    GQA = "gqa"
    GVA = "gva"


def classify_head_mode(num_q_heads: int, num_k_heads: int, num_v_heads: int) -> HeadMode:
    """Validate the MHA, grouped-query, or grouped-value head relation."""

    if min(num_q_heads, num_k_heads, num_v_heads) <= 0:
        raise ValueError("head counts must be positive")
    if num_q_heads == num_k_heads == num_v_heads:
        return HeadMode.MHA
    if num_k_heads == num_v_heads and num_q_heads > num_k_heads and num_q_heads % num_k_heads == 0:
        return HeadMode.GQA
    if num_q_heads == num_k_heads and num_v_heads > num_q_heads and num_v_heads % num_q_heads == 0:
        return HeadMode.GVA
    raise ValueError(
        "SM90 GDN supports MHA (Hq=Hk=Hv), GQA (Hq multiple of Hk=Hv), or GVA (Hv multiple of Hq=Hk)",
    )
