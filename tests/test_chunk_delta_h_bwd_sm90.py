#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for the SM90 CuTe DSL WGMMA bwd_dhu path.

These cases follow tests/test_chunk_delta_h.py where the backward API permits.
For bwd_dhu, fwd's initial_state/output_final_state pair maps to dht/dh0.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu as fla_bwd_dhu

from cula.ops.chunk_delta_h_bwd import chunk_gated_delta_rule_bwd_dhu_sm90

BT = 64
ATOL = 3e-2
RTOL = 3e-2
device = "cuda"


def _is_sm90() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


pytestmark = [
    pytest.mark.sm90_only,
    pytest.mark.skipif(not _is_sm90(), reason="SM90/Hopper GPU is required"),
]


def run_fla_ref(
    q,
    k,
    w,
    do,
    dv,
    g=None,
    gk=None,
    dht=None,
    dh0=None,
    cu_seqlens=None,
    use_exp2=True,
    transpose_state_layout=False,
):
    return fla_bwd_dhu(
        q=q,
        k=k,
        w=w,
        do=do,
        dv=dv,
        g=g,
        gk=gk,
        h0=dh0,
        dht=dht,
        scale=q.shape[-1] ** -0.5,
        cu_seqlens=cu_seqlens.long() if cu_seqlens is not None else None,
        chunk_size=BT,
        use_exp2=use_exp2,
        transpose_state_layout=transpose_state_layout,
    )


def run_cute_dsl(
    q,
    k,
    w,
    do,
    dv,
    g=None,
    gk=None,
    dht=None,
    dh0=None,
    cu_seqlens=None,
    use_exp2=True,
    transpose_state_layout=False,
):
    return chunk_gated_delta_rule_bwd_dhu_sm90(
        q=q,
        k=k,
        w=w,
        do=do,
        dv=dv,
        g=g,
        gk=gk,
        h0=dh0,
        dht=dht,
        scale=q.shape[-1] ** -0.5,
        cu_seqlens=cu_seqlens,
        chunk_size=BT,
        use_exp2=use_exp2,
        transpose_state_layout=transpose_state_layout,
    )


def _assert_bwd_close(got, ref, expect_dh0, msg):
    got_dh, got_dh0, got_dv2 = got
    ref_dh, ref_dh0, ref_dv2 = ref
    torch.testing.assert_close(got_dh.float(), ref_dh.float(), atol=ATOL, rtol=RTOL, msg=f"{msg}: dh")
    torch.testing.assert_close(got_dv2.float(), ref_dv2.float(), atol=ATOL, rtol=RTOL, msg=f"{msg}: dv2")
    if expect_dh0:
        assert got_dh0 is not None
        torch.testing.assert_close(got_dh0.float(), ref_dh0.float(), atol=ATOL, rtol=RTOL, msg=f"{msg}: dh0")
    else:
        assert got_dh0 is None


def _make_inputs(
    B,
    T,
    H,
    K,
    V,
    use_g=False,
    use_gk=False,
    use_state=False,
    seed=42,
    transpose_state_layout=False,
):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device) * 0.1
    w = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device) * 0.1
    do = torch.randn(B, T, H, V, dtype=torch.bfloat16, device=device) * 0.1
    dv = torch.randn(B, T, H, V, dtype=torch.bfloat16, device=device) * 0.1

    g = None
    if use_g:
        g = -torch.abs(torch.randn(B, T, H, dtype=torch.float32, device=device) * 0.01).cumsum(dim=1)

    gk = None
    if use_gk:
        gk = -torch.abs(torch.randn(B, T, H, K, dtype=torch.float32, device=device) * 0.01).cumsum(dim=1)

    state_shape = (B, H, V, K) if transpose_state_layout else (B, H, K, V)
    dht = torch.randn(state_shape, dtype=torch.float32, device=device) * 0.01 if use_state else None
    dh0 = torch.empty(state_shape, dtype=torch.float32, device=device) if use_state else None
    return q, k, w, do, dv, g, gk, dht, dh0


def _make_varlen_inputs(
    seq_lens,
    H,
    K,
    V,
    use_g=False,
    use_gk=False,
    use_state=False,
    seed=42,
    transpose_state_layout=False,
):
    T_total = sum(seq_lens)
    num_seqs = len(seq_lens)
    cu = [0]
    for seq_len in seq_lens:
        cu.append(cu[-1] + seq_len)

    torch.manual_seed(seed)
    q = torch.randn(1, T_total, H, K, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.randn(1, T_total, H, K, dtype=torch.bfloat16, device=device) * 0.1
    w = torch.randn(1, T_total, H, K, dtype=torch.bfloat16, device=device) * 0.1
    do = torch.randn(1, T_total, H, V, dtype=torch.bfloat16, device=device) * 0.1
    dv = torch.randn(1, T_total, H, V, dtype=torch.bfloat16, device=device) * 0.1

    g = None
    if use_g:
        g = torch.empty(1, T_total, H, dtype=torch.float32, device=device)
        for i in range(num_seqs):
            bos, eos = cu[i], cu[i + 1]
            seg = torch.randn(1, eos - bos, H, dtype=torch.float32, device=device) * 0.01
            g[:, bos:eos] = -torch.abs(seg).cumsum(dim=1)

    gk = None
    if use_gk:
        gk = torch.empty(1, T_total, H, K, dtype=torch.float32, device=device)
        for i in range(num_seqs):
            bos, eos = cu[i], cu[i + 1]
            seg = torch.randn(1, eos - bos, H, K, dtype=torch.float32, device=device) * 0.01
            gk[:, bos:eos] = -torch.abs(seg).cumsum(dim=1)

    state_shape = (num_seqs, H, V, K) if transpose_state_layout else (num_seqs, H, K, V)
    dht = torch.randn(state_shape, dtype=torch.float32, device=device) * 0.01 if use_state else None
    dh0 = torch.empty(state_shape, dtype=torch.float32, device=device) if use_state else None
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)
    return q, k, w, do, dv, g, gk, dht, dh0, cu_seqlens


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("H", [1, 4])
@pytest.mark.parametrize("T", [64, 128, 256])
@pytest.mark.parametrize("K", [128])
@pytest.mark.parametrize("V", [128])
@pytest.mark.parametrize("use_gk", [False, True])
@pytest.mark.parametrize("use_state", [False, True])
def test_dhu_against_fla(B, H, T, K, V, use_gk, use_state):
    q, k, w, do, dv, g, gk, dht, dh0 = _make_inputs(B, T, H, K, V, use_gk=use_gk, use_state=use_state)
    ref = run_fla_ref(q, k, w, do, dv, g=g, gk=gk, dht=dht, dh0=dh0)
    got = run_cute_dsl(q, k, w, do, dv, g=g, gk=gk, dht=dht, dh0=dh0)
    _assert_bwd_close(got, ref, use_state, f"B={B} H={H} T={T} gk={use_gk} state={use_state}")


@pytest.mark.parametrize(
    "B,T,H,K,V",
    [
        (1, 64, 1, 128, 128),
        (2, 128, 4, 128, 128),
        (4, 512, 4, 128, 128),
    ],
)
def test_dv2_no_gating(B, T, H, K, V):
    q, k, w, do, dv, g, gk, dht, dh0 = _make_inputs(B, T, H, K, V)
    ref = run_fla_ref(q, k, w, do, dv, g=g, gk=gk, dht=dht, dh0=dh0)
    got = run_cute_dsl(q, k, w, do, dv, g=g, gk=gk, dht=dht, dh0=dh0)
    _assert_bwd_close(got, ref, False, f"dv2 no-gating B={B} T={T} H={H}")


@pytest.mark.parametrize(
    "seq_lens",
    [
        [128, 128],
        [50, 192, 100],
        [33, 128, 200, 95],
    ],
)
@pytest.mark.parametrize("H", [1, 4])
@pytest.mark.parametrize("use_gk", [False, True])
@pytest.mark.parametrize("use_state", [False, True])
def test_varlen_against_fla(seq_lens, H, use_gk, use_state):
    K, V = 128, 128
    q, k, w, do, dv, g, gk, dht, dh0, cu_seqlens = _make_varlen_inputs(seq_lens, H, K, V, use_gk=use_gk, use_state=use_state)
    ref = run_fla_ref(q, k, w, do, dv, g=g, gk=gk, dht=dht, dh0=dh0, cu_seqlens=cu_seqlens)
    got = run_cute_dsl(q, k, w, do, dv, g=g, gk=gk, dht=dht, dh0=dh0, cu_seqlens=cu_seqlens)
    _assert_bwd_close(got, ref, use_state, f"varlen seqs={seq_lens} H={H} gk={use_gk} state={use_state}")


def test_varlen_vs_nonvarlen():
    H, K, V = 2, 128, 128
    T = 256
    q, k, w, do, dv, g, gk, dht, dh0 = _make_inputs(1, T, H, K, V, use_gk=True, use_state=True)
    dh_nv, dh0_nv, dv2_nv = run_cute_dsl(q, k, w, do, dv, g=g, gk=gk, dht=dht, dh0=dh0)

    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)
    dh_vl, dh0_vl, dv2_vl = run_cute_dsl(q, k, w, do, dv, g=g, gk=gk, dht=dht, dh0=dh0, cu_seqlens=cu_seqlens)

    torch.testing.assert_close(dh_nv.float(), dh_vl.float(), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(dv2_nv.float(), dv2_vl.float(), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(dh0_nv.float(), dh0_vl.float(), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "use_g,use_gk",
    [
        (True, False),
        (True, True),
    ],
)
def test_scalar_g_features(use_g, use_gk):
    q, k, w, do, dv, g, gk, dht, dh0 = _make_inputs(
        B=1,
        T=128,
        H=2,
        K=128,
        V=128,
        use_g=use_g,
        use_gk=use_gk,
        use_state=True,
        seed=123,
    )
    ref = run_fla_ref(q, k, w, do, dv, g=g, gk=gk, dht=dht, dh0=dh0)
    got = run_cute_dsl(q, k, w, do, dv, g=g, gk=gk, dht=dht, dh0=dh0)
    _assert_bwd_close(got, ref, True, f"scalar-g g={use_g} gk={use_gk}")


@pytest.mark.parametrize(
    "T,use_g,use_gk,transpose_state_layout",
    [
        (65, False, False, False),
        (127, True, False, False),
        (129, False, True, True),
        (191, True, True, True),
    ],
    ids=["t65-plain", "t127-g", "t129-gk-trans", "t191-g-gk-trans"],
)
def test_tail_chunk_sizes(T, use_g, use_gk, transpose_state_layout):
    q, k, w, do, dv, g, gk, dht, dh0 = _make_inputs(
        B=1,
        T=T,
        H=2,
        K=128,
        V=128,
        use_g=use_g,
        use_gk=use_gk,
        use_state=True,
        seed=1000 + T,
        transpose_state_layout=transpose_state_layout,
    )
    ref = run_fla_ref(
        q,
        k,
        w,
        do,
        dv,
        g=g,
        gk=gk,
        dht=dht,
        dh0=dh0,
        transpose_state_layout=transpose_state_layout,
    )
    got = run_cute_dsl(
        q,
        k,
        w,
        do,
        dv,
        g=g,
        gk=gk,
        dht=dht,
        dh0=dh0,
        transpose_state_layout=transpose_state_layout,
    )
    _assert_bwd_close(got, ref, True, f"T={T} g={use_g} gk={use_gk} trans={transpose_state_layout}")


@pytest.mark.parametrize(
    "use_g,use_gk,transpose_state_layout",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (True, True, False),
        (False, False, True),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ],
)
def test_varlen_tail_chunk_sizes(use_g, use_gk, transpose_state_layout):
    seq_lens = [1, 63, 64, 65, 127, 128, 129]
    q, k, w, do, dv, g, gk, dht, dh0, cu_seqlens = _make_varlen_inputs(
        seq_lens,
        H=1,
        K=128,
        V=128,
        use_g=use_g,
        use_gk=use_gk,
        use_state=True,
        seed=2000 + int(use_g) * 10 + int(use_gk) * 20 + int(transpose_state_layout) * 40,
        transpose_state_layout=transpose_state_layout,
    )
    ref = run_fla_ref(
        q,
        k,
        w,
        do,
        dv,
        g=g,
        gk=gk,
        dht=dht,
        dh0=dh0,
        cu_seqlens=cu_seqlens,
        transpose_state_layout=transpose_state_layout,
    )
    got = run_cute_dsl(
        q,
        k,
        w,
        do,
        dv,
        g=g,
        gk=gk,
        dht=dht,
        dh0=dh0,
        cu_seqlens=cu_seqlens,
        transpose_state_layout=transpose_state_layout,
    )
    _assert_bwd_close(
        got,
        ref,
        True,
        f"varlen tails g={use_g} gk={use_gk} trans={transpose_state_layout}",
    )


def test_transpose_state_layout():
    q, k, w, do, dv, g, gk, dht, dh0 = _make_inputs(
        B=1,
        T=128,
        H=2,
        K=128,
        V=128,
        use_gk=True,
        use_state=True,
        seed=456,
        transpose_state_layout=True,
    )
    ref = run_fla_ref(q, k, w, do, dv, g=g, gk=gk, dht=dht, dh0=dh0, transpose_state_layout=True)
    got = run_cute_dsl(q, k, w, do, dv, g=g, gk=gk, dht=dht, dh0=dh0, transpose_state_layout=True)
    _assert_bwd_close(got, ref, True, "transpose state layout")
