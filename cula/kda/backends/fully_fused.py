# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util

import torch

from cula.backends import BaseBackend
from cula.ops.kda.cp_mode import CPMode


def _is_sm90(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (9, 0)


class FullyFusedBackend(BaseBackend):
    backend_type = "fully_fused"
    env_var = "CULA_BACKEND_FULLY_FUSED"
    priority = 2

    def probe(self):
        if importlib.util.find_spec("cula._cudac_sm90") is None:
            return False, "cula._cudac_sm90 extension not built (rebuild with CULA_DISABLE_SM90=0)"
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
        H, HV = q.shape[2], v.shape[2]
        if HV % H != 0:
            return False, f"requires HV to be a multiple of H, got HV={HV} H={H}"
        if not safe_gate:
            return False, "requires safe_gate=True"
        if kwargs.get("use_beta_sigmoid_in_kernel", False) is not False:
            return False, "does not support use_beta_sigmoid_in_kernel=True"
        if kwargs.get("use_intracard_cp") is CPMode.FORCE:
            return False, "does not support forced intracard CP"
        if kwargs.get("out") is not None or kwargs.get("final_state") is not None:
            return False, "does not support preallocated out or final_state buffers"
        return True, None

    def kda_prefill(self, q, k, v, g, beta, **kwargs):
        from cula.kda.auto_route import cula_kda_prefill_auto

        cp_mode = kwargs.pop("use_intracard_cp", None)
        for name in ("use_beta_sigmoid_in_kernel", "out", "final_state"):
            kwargs.pop(name, None)
        return cula_kda_prefill_auto(q, k, v, g, beta, auto_cp=cp_mode is not CPMode.OFF, **kwargs)
