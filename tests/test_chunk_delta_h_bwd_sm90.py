#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for the SM90 CuTe DSL WGMMA bwd_dhu path."""

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


def _make_inputs(
    B,
    T,
    H,
    K,
    V,
    use_g=False,
    use_gk=False,
    use_dht=False,
    use_h0=False,
    seed=42,
    transpose_state_layout=False,
):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda") * 0.1
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda") * 0.1
    w = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda") * 0.1
    do = torch.randn(B, T, H, V, dtype=torch.bfloat16, device="cuda") * 0.1
    dv = torch.randn(B, T, H, V, dtype=torch.bfloat16, device="cuda") * 0.1

    g = None
    if use_g:
        g = -torch.abs(torch.randn(B, T, H, dtype=torch.float32, device="cuda") * 0.01).cumsum(dim=1)

    gk = None
    if use_gk:
        gk = -torch.abs(torch.randn(B, T, H, K, dtype=torch.float32, device="cuda") * 0.01).cumsum(dim=1)

    dht = None
    if use_dht:
        state_shape = (B, H, V, K) if transpose_state_layout else (B, H, K, V)
        dht = torch.randn(state_shape, dtype=torch.float32, device="cuda") * 0.01

    h0 = None
    if use_h0:
        state_shape = (B, H, V, K) if transpose_state_layout else (B, H, K, V)
        h0 = torch.empty(state_shape, dtype=torch.float32, device="cuda")

    return q, k, w, do, dv, g, gk, dht, h0


def _run_case(
    B,
    T,
    H,
    K,
    V,
    use_g=False,
    use_gk=False,
    use_dht=False,
    use_h0=False,
    use_exp2=False,
    transpose_state_layout=False,
    seed=42,
):
    q, k, w, do, dv, g, gk, dht, h0 = _make_inputs(
        B,
        T,
        H,
        K,
        V,
        use_g,
        use_gk,
        use_dht,
        use_h0,
        seed=seed,
        transpose_state_layout=transpose_state_layout,
    )
    scale = K**-0.5

    ref_dh, ref_dh0, ref_dv2 = fla_bwd_dhu(
        q=q,
        k=k,
        w=w,
        do=do,
        dv=dv,
        g=g,
        gk=gk,
        h0=h0,
        dht=dht,
        scale=scale,
        chunk_size=64,
        use_exp2=use_exp2,
        transpose_state_layout=transpose_state_layout,
    )

    got_dh, got_dh0, got_dv2 = chunk_gated_delta_rule_bwd_dhu_sm90(
        q=q,
        k=k,
        w=w,
        do=do,
        dv=dv,
        g=g,
        gk=gk,
        h0=h0,
        dht=dht,
        scale=scale,
        chunk_size=64,
        use_exp2=use_exp2,
        transpose_state_layout=transpose_state_layout,
    )

    torch.testing.assert_close(got_dh.float(), ref_dh.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(got_dv2.float(), ref_dv2.float(), atol=3e-2, rtol=3e-2)
    if use_h0:
        assert got_dh0 is not None
        torch.testing.assert_close(got_dh0, ref_dh0, atol=3e-2, rtol=3e-2)
    else:
        assert got_dh0 is None


def _make_varlen_inputs(
    seq_lens,
    H,
    K,
    V,
    use_g=False,
    use_gk=False,
    use_dht=False,
    use_h0=False,
    seed=42,
    transpose_state_layout=False,
):
    torch.manual_seed(seed)
    T_total = sum(seq_lens)
    N = len(seq_lens)
    cu = [0]
    for seq_len in seq_lens:
        cu.append(cu[-1] + seq_len)

    q = torch.randn(1, T_total, H, K, dtype=torch.bfloat16, device="cuda") * 0.1
    k = torch.randn(1, T_total, H, K, dtype=torch.bfloat16, device="cuda") * 0.1
    w = torch.randn(1, T_total, H, K, dtype=torch.bfloat16, device="cuda") * 0.1
    do = torch.randn(1, T_total, H, V, dtype=torch.bfloat16, device="cuda") * 0.1
    dv = torch.randn(1, T_total, H, V, dtype=torch.bfloat16, device="cuda") * 0.1

    g = None
    if use_g:
        g = torch.empty(1, T_total, H, dtype=torch.float32, device="cuda")
        for i in range(N):
            bos, eos = cu[i], cu[i + 1]
            seg = torch.randn(1, eos - bos, H, dtype=torch.float32, device="cuda") * 0.01
            g[:, bos:eos] = -torch.abs(seg).cumsum(dim=1)

    gk = None
    if use_gk:
        gk = torch.empty(1, T_total, H, K, dtype=torch.float32, device="cuda")
        for i in range(N):
            bos, eos = cu[i], cu[i + 1]
            seg = torch.randn(1, eos - bos, H, K, dtype=torch.float32, device="cuda") * 0.01
            gk[:, bos:eos] = -torch.abs(seg).cumsum(dim=1)

    state_shape = (N, H, V, K) if transpose_state_layout else (N, H, K, V)
    dht = torch.randn(state_shape, dtype=torch.float32, device="cuda") * 0.01 if use_dht else None
    h0 = torch.empty(state_shape, dtype=torch.float32, device="cuda") if use_h0 else None
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device="cuda")
    return q, k, w, do, dv, g, gk, dht, h0, cu_seqlens


def _run_varlen_case(
    seq_lens,
    H,
    K,
    V,
    use_g=False,
    use_gk=False,
    use_dht=False,
    use_h0=False,
    use_exp2=False,
    transpose_state_layout=False,
    seed=42,
):
    q, k, w, do, dv, g, gk, dht, h0, cu_seqlens = _make_varlen_inputs(
        seq_lens,
        H,
        K,
        V,
        use_g=use_g,
        use_gk=use_gk,
        use_dht=use_dht,
        use_h0=use_h0,
        seed=seed,
        transpose_state_layout=transpose_state_layout,
    )
    scale = K**-0.5
    ref_dh, ref_dh0, ref_dv2 = fla_bwd_dhu(
        q=q,
        k=k,
        w=w,
        do=do,
        dv=dv,
        g=g,
        gk=gk,
        h0=h0,
        dht=dht,
        scale=scale,
        cu_seqlens=cu_seqlens.long(),
        chunk_size=64,
        use_exp2=use_exp2,
        transpose_state_layout=transpose_state_layout,
    )
    got_dh, got_dh0, got_dv2 = chunk_gated_delta_rule_bwd_dhu_sm90(
        q=q,
        k=k,
        w=w,
        do=do,
        dv=dv,
        g=g,
        gk=gk,
        h0=h0,
        dht=dht,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=64,
        use_exp2=use_exp2,
        transpose_state_layout=transpose_state_layout,
    )
    torch.testing.assert_close(got_dh.float(), ref_dh.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(got_dv2.float(), ref_dv2.float(), atol=3e-2, rtol=3e-2)
    if use_h0:
        assert got_dh0 is not None
        torch.testing.assert_close(got_dh0, ref_dh0, atol=3e-2, rtol=3e-2)
    else:
        assert got_dh0 is None


@pytest.mark.parametrize("T", [64, 128])
@pytest.mark.parametrize("V", [64, 128])
def test_bwd_dhu_no_gating(T, V):
    _run_case(B=1, T=T, H=1, K=64, V=V)


def test_bwd_dhu_with_gk_exp2_and_dht():
    _run_case(B=1, T=128, H=2, K=64, V=64, use_gk=True, use_dht=True, use_exp2=True)


def test_bwd_dhu_with_scalar_g_exp2_and_dht():
    _run_case(B=1, T=128, H=2, K=64, V=64, use_g=True, use_dht=True, use_exp2=True)


def test_bwd_dhu_with_scalar_g_and_gk():
    _run_case(B=1, T=128, H=1, K=128, V=64, use_g=True, use_gk=True, use_dht=True, use_exp2=True)


def test_bwd_dhu_k128_with_gk_exp2_and_dht():
    _run_case(B=1, T=128, H=1, K=128, V=64, use_gk=True, use_dht=True, use_exp2=True)


def test_bwd_dhu_returns_dh0():
    _run_case(B=2, T=128, H=1, K=64, V=64, use_h0=True)


@pytest.mark.parametrize(
    "case",
    [
        dict(B=1, T=256, H=4, K=64, V=64, use_gk=True, use_dht=True, use_exp2=True),
        dict(B=1, T=256, H=4, K=64, V=64, use_g=True, use_dht=True, use_exp2=True),
        dict(B=1, T=256, H=2, K=128, V=64, use_g=True, use_gk=True, use_dht=True, use_exp2=True),
        dict(B=2, T=256, H=2, K=128, V=64, use_gk=True, use_dht=True, use_h0=True, use_exp2=True),
        dict(B=1, T=512, H=4, K=128, V=128, use_gk=True, use_dht=True, use_exp2=True),
        dict(B=1, T=128, H=2, K=256, V=64, use_gk=True, use_dht=True, use_exp2=True),
    ],
    ids=[
        "k64-v64-multihead-gk-dht",
        "k64-v64-multihead-g-dht",
        "k128-v64-g-and-gk",
        "k128-v64-batch-h0",
        "k128-v128-long",
        "k256-v64",
    ],
)
def test_bwd_dhu_forward_aligned_cases(case):
    _run_case(**case, seed=123)


def test_bwd_dhu_transpose_state_layout():
    _run_case(
        B=1,
        T=128,
        H=2,
        K=128,
        V=64,
        use_gk=True,
        use_dht=True,
        use_h0=True,
        use_exp2=True,
        transpose_state_layout=True,
    )


@pytest.mark.parametrize(
    "case",
    [
        dict(seq_lens=[64, 128], H=1, K=64, V=64),
        dict(seq_lens=[50, 192, 100], H=2, K=64, V=64, use_gk=True, use_dht=True, use_exp2=True),
        dict(seq_lens=[33, 128, 200], H=1, K=128, V=64, use_g=True, use_dht=True, use_h0=True, use_exp2=True),
        dict(
            seq_lens=[96, 129],
            H=1,
            K=128,
            V=64,
            use_gk=True,
            use_dht=True,
            use_h0=True,
            use_exp2=True,
            transpose_state_layout=True,
        ),
    ],
    ids=["basic", "gk-dht", "g-h0", "transpose-gk-h0"],
)
def test_bwd_dhu_varlen_cases(case):
    _run_varlen_case(**case, seed=321)
