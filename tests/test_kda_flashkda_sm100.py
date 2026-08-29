# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""GB200 validation for the SM90-derived FlashKDA compatibility path."""

import pytest
import torch
from fla.ops.kda.chunk import chunk_kda as fla_chunk_kda
from fla.utils import assert_close

from cula.kda import kda_prefill

pytestmark = [
    pytest.mark.sm100_only,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]


@pytest.mark.kda_fast
def test_flashkda_auto_cp_matches_fla(monkeypatch):
    from cula.ops.kda.sm90.cp import driver

    B, T, H, D = 1, 4096, 2, 128
    device = torch.device("cuda")
    torch.manual_seed(42)
    q = torch.rand(B, T, H, D, dtype=torch.bfloat16, device=device)
    k = torch.rand(B, T, H, D, dtype=torch.bfloat16, device=device)
    v = torch.rand(B, T, H, D, dtype=torch.bfloat16, device=device)
    g = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device)
    beta = torch.randn(B, T, H, dtype=torch.float32, device=device).sigmoid().to(torch.bfloat16)
    A_log = torch.randn(H, dtype=torch.float32, device=device)
    dt_bias = torch.randn(H * D, dtype=torch.float32, device=device)

    original_run_cp = driver._run_cp
    entered_cp = False

    def record_run_cp(*args, **kwargs):
        nonlocal entered_cp
        entered_cp = True
        return original_run_cp(*args, **kwargs)

    monkeypatch.setattr(driver, "_run_cp", record_run_cp)

    with torch.inference_mode():
        ref_o, ref_ht = fla_chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            A_log=A_log,
            dt_bias=dt_bias,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=-5.0,
        )
        actual_o, actual_ht_vk = kda_prefill(
            q,
            k,
            v,
            g,
            beta,
            A_log=A_log,
            dt_bias=dt_bias,
            output_final_state=True,
            use_intracard_cp="auto",
        )

    assert entered_cp, "expected the SM100 public backend call to enter FlashKDA intracard CP"
    assert_close("o", ref_o, actual_o, 0.005)
    assert_close("ht", ref_ht, actual_ht_vk.transpose(-2, -1), 0.005)
