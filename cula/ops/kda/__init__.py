# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""KDA backend kernels migrated to the arch-first layout.

sm100/          SM100 (Blackwell) modular-chunk recurrence/output/bwd kernels
decode/         single-token decode (CuTe DSL + FLA reference)
experimental/   unwired fully-fused WIP
policy.py       CP dispatch policy (use_cp / use_intracard_cp)

"""

# TODO: The SM90 (Hopper) prefill is still the C++ kernel under csrc/kda/sm90 (CuTeDSL
# port pending); it is not yet part of this package.
