# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Frozen bitwise regression vectors from the removed SM100 csrc forward."""

import hashlib

import pytest
import torch

from cula.kda.chunk_intra import chunk_kda_fwd_intra

pytestmark = pytest.mark.cula_slow

_CSRC_SHA256 = {
    (torch.float32, "mha"): "106ea353dc56f883005d039481bb0bd4c1db6e5c6816c9a9524101271779d26d",
    (torch.float32, "gva_varlen"): "22d07fdbbd4c7e2593a9a3e862dd6b73f898e384ace1d1af47c7cdc7769889c2",
    (torch.float32, "tail"): "56a93b44f09ccc26611e370ec68de243e5ebcb34eeafd193dc04c3c01ab0a32f",
    (torch.bfloat16, "mha"): "124605da8d60287ec41ed63f8e4afcdc468fdf3183da87426d7e43ce5d073790",
    (torch.bfloat16, "gva_varlen"): "7aea0a7eedc5832b5d68f9a8b5d82323dce5ffdb97b0e3078bd9c0dc4b201f42",
    (torch.bfloat16, "tail"): "e7df48c614161effdd7ab1829c5f77198ac3f79c275d3187961252cbd7ae9264",
}

_CASES = {
    "mha": (1, 256, 4, 4, (0, 256), 101),
    "gva_varlen": (1, 193, 2, 4, (0, 63, 193), 103),
    "tail": (2, 63, 2, 4, (0, 63, 126), 107),
}


def _requires_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 CUDA device required")


def _digest(outputs):
    result = hashlib.sha256()
    for output in outputs:
        if output is None:
            result.update(b"NONE")
        else:
            result.update(output.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


@pytest.mark.parametrize("beta_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("case_name", ["mha", "gva_varlen", "tail"])
def test_pure_cutedsl_matches_frozen_csrc_bits(beta_dtype, case_name):
    _requires_sm100()
    batch, length, h_qk, h_v, offsets, seed = _CASES[case_name]
    torch.manual_seed(seed)
    dim = 128
    q = torch.randn(batch, length, h_qk, dim, device="cuda", dtype=torch.bfloat16) * 0.1
    k = torch.randn(batch, length, h_qk, dim, device="cuda", dtype=torch.bfloat16) * 0.1
    v = torch.randn(batch, length, h_v, dim, device="cuda", dtype=torch.bfloat16) * 0.1
    gk = torch.randn(batch, length, h_v, dim, device="cuda", dtype=torch.float32) * 0.02
    # Generate in FP32 before casting so the frozen BF16 vectors do not
    # contain exact zeros. IEEE +0/-0 are value-equivalent, but a raw SHA256
    # intentionally treats their sign bits as different.
    beta = torch.randn(batch, length, h_v, device="cuda").sigmoid().to(beta_dtype)

    cu_seqlens = None
    if case_name == "gva_varlen":
        cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    outputs = chunk_kda_fwd_intra(
        q,
        k,
        v,
        gk,
        beta,
        scale=dim**-0.5,
        cu_seqlens=cu_seqlens,
        safe_gate=True,
        disable_recompute=True,
    )
    torch.cuda.synchronize()
    assert _digest(outputs) == _CSRC_SHA256[(beta_dtype, case_name)]
