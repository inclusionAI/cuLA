# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util

import torch

from cula.backends import BaseBackend


def _is_sm90(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (9, 0)


class FlashKDABackend(BaseBackend):
    backend_type = "flashkda"
    env_var = "CULA_BACKEND_FLASHKDA"
    priority = 1

    def probe(self):
        if importlib.util.find_spec("cutlass") is None:
            return False, "nvidia-cutlass-dsl not installed"
        return True, None

    def kda_prefill_verifier(
        self,
        q,
        k,
        v,
        g,
        beta,
        scale=None,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        cu_seqlens=None,
        chunk_indices=None,
        **kwargs,
    ):
        if not _is_sm90(q.device):
            return False, "requires an SM90 (Hopper) device"
        if v.shape[2] != q.shape[2]:
            return False, f"no GVA support (HV={v.shape[2]} != H={q.shape[2]})"
        if any(not t.is_contiguous() for t in (q, k, v, g)):
            return False, "requires contiguous q/k/v/g (no implicit copies)"
        if g.dtype != torch.bfloat16:
            return False, f"requires bfloat16 g (no implicit cast), got {g.dtype}"
        if not use_gate_in_kernel:
            return False, "requires use_gate_in_kernel=True"
        if not safe_gate:
            return False, "requires safe_gate=True"
        if not use_qk_l2norm_in_kernel:
            return False, "always applies qk l2norm in kernel"
        A_log, dt_bias = kwargs.get("A_log"), kwargs.get("dt_bias")
        if A_log is None or dt_bias is None:
            return False, "requires A_log and dt_bias"
        if isinstance(A_log, torch.Tensor) and not A_log.is_contiguous():
            return False, "requires contiguous A_log"
        if isinstance(dt_bias, torch.Tensor) and not dt_bias.is_contiguous():
            return False, "requires contiguous dt_bias"
        if isinstance(cu_seqlens, torch.Tensor) and not cu_seqlens.is_contiguous():
            return False, "requires contiguous cu_seqlens"
        if isinstance(initial_state, torch.Tensor) and not initial_state.is_contiguous():
            return False, "requires contiguous initial_state"
        return True, None

    def kda_prefill(self, q, k, v, g, beta, **kwargs):
        from cula.kda.flashkda import cula_kda_prefill

        kwargs.setdefault("use_intracard_cp", "auto")
        return cula_kda_prefill(q, k, v, g, beta, **kwargs)
