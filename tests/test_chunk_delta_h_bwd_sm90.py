#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for the SM90 CuTe DSL bwd_dhu prototype."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu as fla_bwd_dhu

from cula.ops.chunk_delta_h_bwd import chunk_gated_delta_rule_bwd_dhu_sm90


def _is_sm90() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


pytestmark = [
    pytest.mark.sm90_only,
    pytest.mark.skipif(not _is_sm90(), reason="SM90/Hopper GPU is required"),
]


def _make_inputs(B, T, H, K, V, use_gk=False, use_dht=False, use_h0=False, seed=42):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda") * 0.1
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda") * 0.1
    w = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda") * 0.1
    do = torch.randn(B, T, H, V, dtype=torch.bfloat16, device="cuda") * 0.1
    dv = torch.randn(B, T, H, V, dtype=torch.bfloat16, device="cuda") * 0.1

    gk = None
    if use_gk:
        gk = -torch.abs(torch.randn(B, T, H, K, dtype=torch.float32, device="cuda") * 0.01).cumsum(dim=1)

    dht = None
    if use_dht:
        dht = torch.randn(B, H, K, V, dtype=torch.float32, device="cuda") * 0.01

    h0 = None
    if use_h0:
        h0 = torch.empty(B, H, K, V, dtype=torch.float32, device="cuda")

    return q, k, w, do, dv, gk, dht, h0


def _run_case(B, T, H, K, V, use_gk=False, use_dht=False, use_h0=False, use_exp2=False):
    q, k, w, do, dv, gk, dht, h0 = _make_inputs(B, T, H, K, V, use_gk, use_dht, use_h0)
    scale = K**-0.5

    ref_dh, ref_dh0, ref_dv2 = fla_bwd_dhu(
        q=q,
        k=k,
        w=w,
        do=do,
        dv=dv,
        gk=gk,
        h0=h0,
        dht=dht,
        scale=scale,
        chunk_size=64,
        use_exp2=use_exp2,
    )

    got_dh, got_dh0, got_dv2 = chunk_gated_delta_rule_bwd_dhu_sm90(
        q=q,
        k=k,
        w=w,
        do=do,
        dv=dv,
        gk=gk,
        h0=h0,
        dht=dht,
        scale=scale,
        chunk_size=64,
        use_exp2=use_exp2,
    )

    torch.testing.assert_close(got_dh.float(), ref_dh.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(got_dv2.float(), ref_dv2.float(), atol=3e-2, rtol=3e-2)
    if use_h0:
        assert got_dh0 is not None
        torch.testing.assert_close(got_dh0, ref_dh0, atol=3e-2, rtol=3e-2)
    else:
        assert got_dh0 is None


@pytest.mark.parametrize("T", [64, 128])
@pytest.mark.parametrize("V", [32, 64])
def test_bwd_dhu_no_gating(T, V):
    _run_case(B=1, T=T, H=1, K=64, V=V)


def test_bwd_dhu_with_gk_exp2_and_dht():
    _run_case(B=1, T=128, H=2, K=64, V=64, use_gk=True, use_dht=True, use_exp2=True)


def test_bwd_dhu_k128_with_gk_exp2_and_dht():
    _run_case(B=1, T=128, H=1, K=128, V=64, use_gk=True, use_dht=True, use_exp2=True)


def test_bwd_dhu_returns_dh0():
    _run_case(B=2, T=128, H=1, K=64, V=64, use_h0=True)
