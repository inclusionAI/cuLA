# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""KDA backend kernels, organized by arch (sm90 / sm100).

sm90/   SM90 (Hopper) two-kernel (K1+K2) FlashKDA prefill, fwd-only (+ cp/)
sm100/  SM100 (Blackwell) modular-chunk recurrence/output/bwd kernels (+ cp/)
decode/         single-token decode (CuTe DSL + FLA reference)
experimental/   unwired fully-fused WIP
cp_mode.py      shared intracard-CP mode vocabulary (CPMode / NotSplittableError)

"""
