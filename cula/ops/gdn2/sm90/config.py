# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Stable host-side contract for Hopper SM90a GDN2 prefill."""

CHUNK_SIZE = 64
HEAD_SIZE = 128
VALUE_SIZE = 128
THREADS_PER_CTA = 384
MAX_SEQUENCES = 32
SUPPORTED_Q_HEADS = 16
SUPPORTED_V_HEADS = (16, 32, 64)
SM90_BACKEND_ID = "sm90a_cutedsl_gdn2_prefill_v1"

# Elementwise log-decay lower bound. The blockwise-rebased intra-chunk
# factorization keeps every stored exponent within a 15-token in-block span;
# g >= -5 leaves the largest such exponent (75 nats) a factor ~9e5 below the
# BF16/FP32 overflow boundary (~88.72 nats). See
# docs/gdn2_sm90_stable_factor.md.
SUPPORTED_G_MIN = -5.0

# The one supported nvidia-cutlass-dsl range for this backend.
# is_sm90_gdn2_available(), the runtime dispatch error, and
# docs/gdn2_sm90_api.md all derive from this string, and the installed
# version is read through cula.ops._mlir_compat so this backend and the
# shared gateway cannot disagree about which toolchain is in use.
#
# Relationship to the repository-wide contract in cula/ops/_mlir_compat.py
# (_SUPPORTED_MIN/_SUPPORTED_MAX/_EXCLUDED_VERSIONS, kept in sync with
# pyproject.toml): the upper bound is the same 4.7 and is enforced there on
# every private-dialect access, which this kernel triggers at import. GDN2
# only raises the floor, from 4.4.2 to 4.5.1, because the kernel needs
# `cutlass.cute.nvgpu.OperandMajorMode`, which 4.4.x does not provide -- the
# backend cannot be imported there at all. Raising the shared floor also
# subsumes the shared 4.5.0 exclusion. Keep the upper bound in step with the
# gateway when it moves.
CUTLASS_DSL_REQUIREMENT = "nvidia-cutlass-dsl>=4.5.1,<4.7"
