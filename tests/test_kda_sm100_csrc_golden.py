# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Frozen bitwise regression vectors from the removed SM100 csrc forward."""

import hashlib

import pytest
import torch

from cula.kda.chunk_intra import chunk_kda_fwd_intra

pytestmark = pytest.mark.cula_slow

_CSRC_SHA256 = {
    (torch.float32, "mha"): "b4fc1c3e4f98beb7c372c050465cfea30e6deec1c4f9a39e4e7c1295cd3b65a5",
    (torch.float32, "gva_varlen"): "a0349594fa31fbc78c79f43d3c88883413197cc8acbc98a1c87597e70d3dc214",
    (torch.float32, "tail"): "3241a28b76ea1e752de642e7872a739022973a08d9be25f9736ede3163004eff",
    (torch.bfloat16, "mha"): "0668ccdd9eb7b6999ee69581a23877177fd4f01d39e172fd4d230fe11a743ddb",
    (torch.bfloat16, "gva_varlen"): "1dadb1ed9712abd9c0fbaf067db37f110afdb58b9e3eca08ad951f5ccf11597c",
    (torch.bfloat16, "tail"): "e750eb9df65f9fe15dd4ccf0ee87b8480d4eb4e75fe2c6b7f7ba45893996918f",
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
    beta = torch.rand(batch, length, h_v, device="cuda", dtype=beta_dtype)

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
