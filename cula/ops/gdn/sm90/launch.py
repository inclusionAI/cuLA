# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Production host adapter for the upstream-derived SM90 GDN CuTe DSL kernel."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .config import HEAD_SIZE
from .delta_rule import run_delta_rule_prefill

if TYPE_CHECKING:
    from cula.gdn.prefill import _GDNPrefillInputs


def launch_sm90_gdn_prefill(inputs: _GDNPrefillInputs) -> None:
    """Launch the 512-thread SM90 CuTe DSL port without a C++ fallback."""

    with torch.cuda.device(inputs.q.device):
        cu_seqlens = inputs.cu_seqlens
        if cu_seqlens.dtype != torch.int64:
            cu_seqlens = cu_seqlens.to(torch.int64)
        if not cu_seqlens.is_contiguous():
            cu_seqlens = cu_seqlens.contiguous()

        final_state = inputs.output_state
        if final_state is None:
            final_state = torch.empty(
                (
                    inputs.num_seqs,
                    inputs.output.shape[1],
                    HEAD_SIZE,
                    HEAD_SIZE,
                ),
                dtype=torch.float32,
                device=inputs.q.device,
            )

        run_delta_rule_prefill(
            inputs.output,
            final_state,
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.initial_state,
            inputs.alpha,
            inputs.beta,
            cu_seqlens,
            inputs.scale,
        )
