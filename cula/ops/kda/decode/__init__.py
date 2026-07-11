# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""KDA single-token decode backend (CuTe DSL) and its FLA reference."""

from cula.ops.kda.decode.cute import fused_sigmoid_gating_delta_rule_update, kda_decode

__all__ = ["kda_decode", "fused_sigmoid_gating_delta_rule_update"]
