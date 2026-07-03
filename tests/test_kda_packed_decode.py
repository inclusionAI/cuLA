#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests for kda_packed_decode (packed-QKV CuTe DSL KDA decode kernel).

Numerical ground truth = the existing non-packed ``kda_decode``: the same base
inputs (q, k, v, a, b, A_log, dt_bias, state) are repacked into a mixed_qkv of
shape ``[N, qkv_dim] = [Q(H·K) | K(H·K) | V(HV·V)]`` (head-major, last dim
contiguous) and ``kda_packed_decode`` is checked against ``kda_decode``.
"""

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cula.kda import kda_decode, kda_packed_decode
from tests.test_kda_decode import _assert_close, make_inputs  # noqa: F401

K = 128


def _pack_mixed_qkv(q, k, v, N):
    """q,k: (N,H,K) bf16 ; v: (N,HV,V) bf16 -> mixed_qkv (N, qkv_dim) bf16."""
    return torch.cat([q.view(N, -1), k.view(N, -1), v.view(N, -1)], dim=-1)


# ---------------------------------------------------------------------------
# Dense: packed vs non-packed kda_decode
# ---------------------------------------------------------------------------
def _run_nonpacked_dense(q, k, v, a, b, A_log, dt_bias, state, scale):
    N, H, Kd = q.shape
    state_ref = state.clone()
    o = kda_decode(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q.unsqueeze(1).contiguous(),
        k=k.unsqueeze(1).contiguous(),
        v=v.unsqueeze(1).contiguous(),
        a=a.unsqueeze(1).contiguous(),
        b=b.unsqueeze(1).contiguous(),
        initial_state_source=state_ref,
        initial_state_indices=torch.arange(N, device=q.device, dtype=torch.int32),
        scale=scale,
        use_qk_l2norm_in_kernel=True,
    )
    return o.squeeze(1), state_ref  # (N,HV,V), (N,HV,V,K)


def _run_packed_dense(q, k, v, a, b, A_log, dt_bias, state, scale):
    N, H, Kd = q.shape
    mixed = _pack_mixed_qkv(q, k, v, N)
    state_p = state.clone()
    o = kda_packed_decode(
        mixed,
        a.unsqueeze(1).contiguous(),
        b.unsqueeze(1).contiguous(),
        A_log=A_log,
        dt_bias=dt_bias,
        state=state_p,
        state_indices=torch.arange(N, device=q.device, dtype=torch.int32),
        scale=scale,
        use_qk_l2norm_in_kernel=True,
    )
    return o.squeeze(1), state_p  # (N,HV,V), (N,HV,V,K)


@pytest.mark.parametrize("N", [1, 2, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("H,HV", [(8, 16), (16, 32)])
@pytest.mark.parametrize("V", [128, 256])
def test_packed_dense(N, H, HV, V):
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs(N, H, HV, K, V)

    o_ref, state_ref = _run_nonpacked_dense(q, k, v, a, b, A_log, dt_bias, state, scale)
    o_p, state_p = _run_packed_dense(q, k, v, a, b, A_log, dt_bias, state, scale)

    # Sanity: the packed should be ~identical to non-packed (same kernel, strided view).
    _assert_close("output", o_ref.float(), o_p.float())
    _assert_close("state", state_ref, state_p)


# ---------------------------------------------------------------------------
# Varlen: packed vs non-packed kda_decode
# ---------------------------------------------------------------------------
def _run_nonpacked_varlen(q, k, v, a, b, A_log, dt_bias, state, scale):
    N, H, Kd = q.shape
    state_ref = state.clone()
    o = kda_decode(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q.unsqueeze(0).contiguous(),
        k=k.unsqueeze(0).contiguous(),
        v=v.unsqueeze(0).contiguous(),
        a=a.contiguous(),
        b=b.contiguous(),
        initial_state_source=state_ref,
        initial_state_indices=torch.arange(N, device=q.device, dtype=torch.int32),
        cu_seqlens=torch.arange(N + 1, device=q.device, dtype=torch.int32),
        scale=scale,
        use_qk_l2norm_in_kernel=True,
    )
    return o.squeeze(0), state_ref  # (N,HV,V), (N,HV,V,K)


def _run_packed_varlen(q, k, v, a, b, A_log, dt_bias, state, scale):
    N, H, Kd = q.shape
    mixed = _pack_mixed_qkv(q, k, v, N)
    state_p = state.clone()
    o = kda_packed_decode(
        mixed,
        a.contiguous(),
        b.contiguous(),
        A_log=A_log,
        dt_bias=dt_bias,
        state=state_p,
        state_indices=torch.arange(N, device=q.device, dtype=torch.int32),
        cu_seqlens=torch.arange(N + 1, device=q.device, dtype=torch.int32),
        scale=scale,
        use_qk_l2norm_in_kernel=True,
    )
    return o.squeeze(0), state_p  # (N,HV,V), (N,HV,V,K)


@pytest.mark.parametrize("N", [2, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("H,HV", [(8, 16), (16, 32)])
@pytest.mark.parametrize("V", [128, 256])
def test_packed_varlen(N, H, HV, V):
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs(N, H, HV, K, V)

    o_ref, state_ref = _run_nonpacked_varlen(q, k, v, a, b, A_log, dt_bias, state, scale)
    o_p, state_p = _run_packed_varlen(q, k, v, a, b, A_log, dt_bias, state, scale)

    _assert_close("output", o_ref.float(), o_p.float())
    _assert_close("state", state_ref, state_p)


# ---------------------------------------------------------------------------
# Equal-head coverage: H == HV (group ratio = 1)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("N", [1, 8, 64])
@pytest.mark.parametrize("H", [8, 16, 32, 64])
def test_packed_dense_equal_heads(N, H):
    HV, V = H, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs(N, H, HV, K, V)

    o_ref, state_ref = _run_nonpacked_dense(q, k, v, a, b, A_log, dt_bias, state, scale)
    o_p, state_p = _run_packed_dense(q, k, v, a, b, A_log, dt_bias, state, scale)

    _assert_close("output", o_ref.float(), o_p.float())
    _assert_close("state", state_ref, state_p)


@pytest.mark.parametrize("N", [2, 8, 64])
@pytest.mark.parametrize("H", [8, 16, 32, 64])
def test_packed_varlen_equal_heads(N, H):
    HV, V = H, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs(N, H, HV, K, V)

    o_ref, state_ref = _run_nonpacked_varlen(q, k, v, a, b, A_log, dt_bias, state, scale)
    o_p, state_p = _run_packed_varlen(q, k, v, a, b, A_log, dt_bias, state, scale)

    _assert_close("output", o_ref.float(), o_p.float())
    _assert_close("state", state_ref, state_p)


# ---------------------------------------------------------------------------
# -1 dummy slots: output 0, corresponding state untouched
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("N", [1, 2, 8])
def test_packed_minus1_dummy(N):
    H, HV, V = 8, 16, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs(N, H, HV, K, V)

    # pool larger than N; mark the first token as dummy (-1).
    pool = N + 2
    state_pool = torch.zeros(pool, HV, V, K, device="cuda", dtype=torch.float32)
    real_idx = torch.arange(N, device="cuda", dtype=torch.int32) + 1  # use slots 1..N
    state_pool[real_idx] = state
    indices = real_idx.clone()
    indices[0] = -1  # first batch token is a dummy

    mixed = _pack_mixed_qkv(q, k, v, N)
    state_before = state_pool.clone()
    o = kda_packed_decode(
        mixed,
        a.unsqueeze(1).contiguous(),
        b.unsqueeze(1).contiguous(),
        A_log=A_log,
        dt_bias=dt_bias,
        state=state_pool,
        state_indices=indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
    ).squeeze(1)  # (N,HV,V)

    # Dummy tokens (pool_idx == -1) skip ALL kernel work — the kernel neither
    # reads/writes their state slot nor writes their output row. So:
    #   - the dummy's pool slot is untouched, and
    #   - the dummy's output row is UNINITIALIZED (leave it out of checks).
    dummy_slot = int(real_idx[0]) if indices[0] == -1 else None
    assert dummy_slot is not None
    assert torch.equal(state_pool[dummy_slot], state_before[dummy_slot]), "dummy state slot was modified"

    # Compare against non-packed using the SAME pool + SAME indices (no remap),
    # so packed and non-packed write the same physical state slots.
    state_ref_pool = state_before.clone()
    o_ref = kda_decode(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q.unsqueeze(1).contiguous(),
        k=k.unsqueeze(1).contiguous(),
        v=v.unsqueeze(1).contiguous(),
        a=a.unsqueeze(1).contiguous(),
        b=b.unsqueeze(1).contiguous(),
        initial_state_source=state_ref_pool,
        initial_state_indices=indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
    ).squeeze(1)

    if N > 1:
        _assert_close("real output", o_ref[1:].float(), o[1:].float())
        _assert_close("real state", state_ref_pool[real_idx[1:]], state_pool[real_idx[1:]])
    else:
        # N == 1: the only token is the dummy, so there is no real row to check.
        # Just confirm the dummy slot's state survived untouched (already asserted).
        pass


# ---------------------------------------------------------------------------
# KV state layout
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("is_varlen", [False, True])
def test_packed_kv_state_layout(is_varlen):
    N, H, HV, V = 8, 8, 16, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state_vk = make_inputs(N, H, HV, K, V)
    state_kv = state_vk.permute(0, 1, 3, 2).contiguous()  # (N,HV,K,V)

    mixed = _pack_mixed_qkv(q, k, v, N)
    indices = torch.arange(N, device="cuda", dtype=torch.int32)
    cu_seqlens = torch.arange(N + 1, device="cuda", dtype=torch.int32) if is_varlen else None

    if is_varlen:
        state_p = state_kv.clone()
        o_kv = kda_packed_decode(
            mixed,
            a.contiguous(),
            b.contiguous(),
            A_log=A_log,
            dt_bias=dt_bias,
            state=state_p,
            state_indices=indices,
            cu_seqlens=cu_seqlens,
            scale=scale,
            state_layout="kv",
        ).squeeze(0)
    else:
        state_p = state_kv.clone()
        o_kv = kda_packed_decode(
            mixed,
            a.unsqueeze(1).contiguous(),
            b.unsqueeze(1).contiguous(),
            A_log=A_log,
            dt_bias=dt_bias,
            state=state_p,
            state_indices=indices,
            scale=scale,
            state_layout="kv",
        ).squeeze(1)

    # Compare against vk-layout non-packed reference (same numerics).
    state_vk_ref = state_vk.clone()
    if is_varlen:
        o_vk = kda_decode(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q.unsqueeze(0).contiguous(),
            k=k.unsqueeze(0).contiguous(),
            v=v.unsqueeze(0).contiguous(),
            a=a.contiguous(),
            b=b.contiguous(),
            initial_state_source=state_vk_ref,
            initial_state_indices=indices,
            cu_seqlens=cu_seqlens,
            scale=scale,
        ).squeeze(0)
    else:
        o_vk = kda_decode(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q.unsqueeze(1).contiguous(),
            k=k.unsqueeze(1).contiguous(),
            v=v.unsqueeze(1).contiguous(),
            a=a.unsqueeze(1).contiguous(),
            b=b.unsqueeze(1).contiguous(),
            initial_state_source=state_vk_ref,
            initial_state_indices=indices,
            scale=scale,
        ).squeeze(1)

    _assert_close("kv output", o_kv.float(), o_vk.float())
    # state: kv layout transposed back to vk for comparison
    _assert_close("kv state", state_vk_ref, state_p.permute(0, 1, 3, 2).contiguous())


# ---------------------------------------------------------------------------
# No L2 norm path
# ---------------------------------------------------------------------------
def test_packed_no_l2norm():
    N, H, HV, V = 8, 8, 16, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs(N, H, HV, K, V)

    mixed = _pack_mixed_qkv(q, k, v, N)
    indices = torch.arange(N, device="cuda", dtype=torch.int32)

    state_p = state.clone()
    o_p = kda_packed_decode(
        mixed,
        a.unsqueeze(1).contiguous(),
        b.unsqueeze(1).contiguous(),
        A_log=A_log,
        dt_bias=dt_bias,
        state=state_p,
        state_indices=indices,
        scale=scale,
        use_qk_l2norm_in_kernel=False,
    ).squeeze(1)

    state_ref = state.clone()
    o_ref = kda_decode(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q.unsqueeze(1).contiguous(),
        k=k.unsqueeze(1).contiguous(),
        v=v.unsqueeze(1).contiguous(),
        a=a.unsqueeze(1).contiguous(),
        b=b.unsqueeze(1).contiguous(),
        initial_state_source=state_ref,
        initial_state_indices=indices,
        scale=scale,
        use_qk_l2norm_in_kernel=False,
    ).squeeze(1)

    _assert_close("output", o_ref.float(), o_p.float())
    _assert_close("state", state_ref, state_p)


# ---------------------------------------------------------------------------
# Focused API compatibility and validation coverage
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("is_varlen", [False, True])
def test_packed_explicit_out_scale_and_softplus(is_varlen):
    N, H, HV, V = 8, 8, 16, 128
    scale = 0.25
    softplus_beta = 0.75
    softplus_threshold = 10.0
    q, k, v, a, b, A_log, dt_bias, state = make_inputs(N, H, HV, K, V)

    mixed = _pack_mixed_qkv(q, k, v, N)
    indices = torch.arange(N, device="cuda", dtype=torch.int32)
    cu_seqlens = torch.arange(N + 1, device="cuda", dtype=torch.int32) if is_varlen else None

    state_p = state.clone()
    if is_varlen:
        out = torch.empty(1, N, HV, V, device="cuda", dtype=torch.bfloat16)
        o_p = kda_packed_decode(
            mixed,
            a.contiguous(),
            b.contiguous(),
            A_log=A_log,
            dt_bias=dt_bias,
            state=state_p,
            state_indices=indices,
            out=out,
            cu_seqlens=cu_seqlens,
            scale=scale,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
        )
        assert o_p is out
        o_p_cmp = o_p.squeeze(0)
        state_ref = state.clone()
        o_ref = kda_decode(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q.unsqueeze(0).contiguous(),
            k=k.unsqueeze(0).contiguous(),
            v=v.unsqueeze(0).contiguous(),
            a=a.contiguous(),
            b=b.contiguous(),
            initial_state_source=state_ref,
            initial_state_indices=indices,
            cu_seqlens=cu_seqlens,
            scale=scale,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
        ).squeeze(0)
    else:
        out = torch.empty(N, 1, HV, V, device="cuda", dtype=torch.bfloat16)
        o_p = kda_packed_decode(
            mixed,
            a.unsqueeze(1).contiguous(),
            b.unsqueeze(1).contiguous(),
            A_log=A_log,
            dt_bias=dt_bias,
            state=state_p,
            state_indices=indices,
            out=out,
            scale=scale,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
        )
        assert o_p is out
        o_p_cmp = o_p.squeeze(1)
        state_ref = state.clone()
        o_ref = kda_decode(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q.unsqueeze(1).contiguous(),
            k=k.unsqueeze(1).contiguous(),
            v=v.unsqueeze(1).contiguous(),
            a=a.unsqueeze(1).contiguous(),
            b=b.unsqueeze(1).contiguous(),
            initial_state_source=state_ref,
            initial_state_indices=indices,
            scale=scale,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
        ).squeeze(1)

    _assert_close("output", o_ref.float(), o_p_cmp.float())
    _assert_close("state", state_ref, state_p)


@pytest.mark.parametrize("is_varlen", [False, True])
def test_packed_slow_path_compatible_a_b_shapes(is_varlen):
    N, H, HV, V = 8, 8, 16, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs(N, H, HV, K, V)

    mixed = _pack_mixed_qkv(q, k, v, N)
    indices = torch.arange(N, device="cuda", dtype=torch.int32)
    cu_seqlens = torch.arange(N + 1, device="cuda", dtype=torch.int32) if is_varlen else None

    state_p = state.clone()
    if is_varlen:
        # Fast path expects a=(N,HV,K), b=(N,HV); these compatible public
        # shapes force the general normalization path.
        o_p = kda_packed_decode(
            mixed,
            a.reshape(1, N, HV * K).contiguous(),
            b.unsqueeze(0).contiguous(),
            A_log=A_log,
            dt_bias=dt_bias,
            state=state_p,
            state_indices=indices,
            cu_seqlens=cu_seqlens,
            scale=scale,
        ).squeeze(0)
        state_ref = state.clone()
        o_ref = kda_decode(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q.unsqueeze(0).contiguous(),
            k=k.unsqueeze(0).contiguous(),
            v=v.unsqueeze(0).contiguous(),
            a=a.contiguous(),
            b=b.contiguous(),
            initial_state_source=state_ref,
            initial_state_indices=indices,
            cu_seqlens=cu_seqlens,
            scale=scale,
        ).squeeze(0)
    else:
        # Fast path does not accept dense a=(N,HV*K), so this covers the
        # packed general path while preserving the same numerics.
        o_p = kda_packed_decode(
            mixed,
            a.reshape(N, HV * K).contiguous(),
            b.contiguous(),
            A_log=A_log,
            dt_bias=dt_bias,
            state=state_p,
            state_indices=indices,
            scale=scale,
        ).squeeze(1)
        state_ref = state.clone()
        o_ref = kda_decode(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q.unsqueeze(1).contiguous(),
            k=k.unsqueeze(1).contiguous(),
            v=v.unsqueeze(1).contiguous(),
            a=a.unsqueeze(1).contiguous(),
            b=b.unsqueeze(1).contiguous(),
            initial_state_source=state_ref,
            initial_state_indices=indices,
            scale=scale,
        ).squeeze(1)

    _assert_close("output", o_ref.float(), o_p.float())
    _assert_close("state", state_ref, state_p)


def test_packed_mixed_qkv_allows_padded_row_stride():
    N, H, HV, V = 8, 8, 16, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs(N, H, HV, K, V)

    mixed = _pack_mixed_qkv(q, k, v, N)
    padded = torch.empty(N, mixed.shape[1] + 8, device="cuda", dtype=torch.bfloat16)
    mixed_padded_view = padded[:, : mixed.shape[1]]
    mixed_padded_view.copy_(mixed)
    assert mixed_padded_view.stride(-1) == 1
    assert mixed_padded_view.stride(0) != mixed.shape[1]

    indices = torch.arange(N, device="cuda", dtype=torch.int32)
    state_p = state.clone()
    o_p = kda_packed_decode(
        mixed_padded_view,
        a.unsqueeze(1).contiguous(),
        b.unsqueeze(1).contiguous(),
        A_log=A_log,
        dt_bias=dt_bias,
        state=state_p,
        state_indices=indices,
        scale=scale,
    ).squeeze(1)

    state_ref = state.clone()
    o_ref = kda_decode(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q.unsqueeze(1).contiguous(),
        k=k.unsqueeze(1).contiguous(),
        v=v.unsqueeze(1).contiguous(),
        a=a.unsqueeze(1).contiguous(),
        b=b.unsqueeze(1).contiguous(),
        initial_state_source=state_ref,
        initial_state_indices=indices,
        scale=scale,
    ).squeeze(1)

    _assert_close("output", o_ref.float(), o_p.float())
    _assert_close("state", state_ref, state_p)


def test_packed_mixed_qkv_allows_padded_row_stride_n1():
    N, H, HV, V = 1, 8, 16, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs(N, H, HV, K, V)

    mixed = _pack_mixed_qkv(q, k, v, N)
    padded = torch.empty(N, mixed.shape[1] + 8, device="cuda", dtype=torch.bfloat16)
    mixed_padded_view = padded[:, : mixed.shape[1]]
    mixed_padded_view.copy_(mixed)
    assert mixed_padded_view.stride(-1) == 1
    assert mixed_padded_view.stride(0) != mixed.shape[1]

    indices = torch.arange(N, device="cuda", dtype=torch.int32)
    state_p = state.clone()
    o_p = kda_packed_decode(
        mixed_padded_view,
        a.unsqueeze(1).contiguous(),
        b.unsqueeze(1).contiguous(),
        A_log=A_log,
        dt_bias=dt_bias,
        state=state_p,
        state_indices=indices,
        scale=scale,
    ).squeeze(1)

    state_ref = state.clone()
    o_ref = kda_decode(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q.unsqueeze(1).contiguous(),
        k=k.unsqueeze(1).contiguous(),
        v=v.unsqueeze(1).contiguous(),
        a=a.unsqueeze(1).contiguous(),
        b=b.unsqueeze(1).contiguous(),
        initial_state_source=state_ref,
        initial_state_indices=indices,
        scale=scale,
    ).squeeze(1)

    _assert_close("output", o_ref.float(), o_p.float())
    _assert_close("state", state_ref, state_p)


@pytest.mark.parametrize("is_varlen", [False, True])
def test_packed_general_path_accepts_noncontiguous_a_b(is_varlen):
    N, H, HV, V = 4, 8, 16, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state = make_inputs(N, H, HV, K, V)
    mixed = _pack_mixed_qkv(q, k, v, N)

    a_flat = a.reshape(N, HV * K)
    a_padded = torch.empty(N, HV * K + 1, device="cuda", dtype=torch.bfloat16)
    a_arg = a_padded[:, : HV * K]
    a_arg.copy_(a_flat)
    b_padded = torch.empty(N, HV + 1, device="cuda", dtype=torch.bfloat16)
    b_arg = b_padded[:, :HV]
    b_arg.copy_(b)
    assert not a_arg.is_contiguous()
    assert not b_arg.is_contiguous()

    state_p = state.clone()
    if is_varlen:
        o_p = kda_packed_decode(
            mixed,
            a_arg,
            b_arg,
            A_log=A_log,
            dt_bias=dt_bias,
            state=state_p,
            state_indices=torch.arange(N, device="cuda", dtype=torch.int32),
            cu_seqlens=torch.arange(N + 1, device="cuda", dtype=torch.int32),
            scale=scale,
        ).squeeze(0)
        state_ref = state.clone()
        o_ref = kda_decode(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q.unsqueeze(0).contiguous(),
            k=k.unsqueeze(0).contiguous(),
            v=v.unsqueeze(0).contiguous(),
            a=a.contiguous(),
            b=b.contiguous(),
            initial_state_source=state_ref,
            initial_state_indices=torch.arange(N, device="cuda", dtype=torch.int32),
            cu_seqlens=torch.arange(N + 1, device="cuda", dtype=torch.int32),
            scale=scale,
        ).squeeze(0)
    else:
        o_p = kda_packed_decode(
            mixed,
            a_arg,
            b_arg,
            A_log=A_log,
            dt_bias=dt_bias,
            state=state_p,
            state_indices=torch.arange(N, device="cuda", dtype=torch.int32),
            scale=scale,
        ).squeeze(1)
        state_ref = state.clone()
        o_ref = kda_decode(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q.unsqueeze(1).contiguous(),
            k=k.unsqueeze(1).contiguous(),
            v=v.unsqueeze(1).contiguous(),
            a=a.unsqueeze(1).contiguous(),
            b=b.unsqueeze(1).contiguous(),
            initial_state_source=state_ref,
            initial_state_indices=torch.arange(N, device="cuda", dtype=torch.int32),
            scale=scale,
        ).squeeze(1)

    _assert_close("output", o_ref.float(), o_p.float())
    _assert_close("state", state_ref, state_p)


# ---------------------------------------------------------------------------
# Padded batch + CUDA graph capture/replay
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("is_varlen", [False, True])
def test_packed_padded_batch(is_varlen):
    N_real, N_pad, H, HV, V = 6, 8, 8, 16, 128
    scale = K**-0.5
    q, k, v, a, b, A_log, dt_bias, state_real = make_inputs(N_real, H, HV, K, V)

    # Pad to N_pad with extra rows; mark them as dummy (-1).
    q_pad = torch.zeros(N_pad, H, K, device="cuda", dtype=torch.bfloat16)
    k_pad = torch.zeros(N_pad, H, K, device="cuda", dtype=torch.bfloat16)
    v_pad = torch.zeros(N_pad, HV, V, device="cuda", dtype=torch.bfloat16)
    a_pad = torch.zeros(N_pad, HV, K, device="cuda", dtype=torch.bfloat16)
    b_pad = torch.zeros(N_pad, HV, device="cuda", dtype=torch.bfloat16)
    q_pad[:N_real], k_pad[:N_real], v_pad[:N_real] = q, k, v
    a_pad[:N_real] = a
    b_pad[:N_real] = b

    pool = N_pad
    state_pool = torch.zeros(pool, HV, V, K, device="cuda", dtype=torch.float32)
    indices = torch.arange(N_pad, device="cuda", dtype=torch.int32)
    indices[N_real:] = -1  # padded slots are dummies
    state_pool[:N_real] = state_real

    mixed = _pack_mixed_qkv(q_pad, k_pad, v_pad, N_pad)
    cu_seqlens = torch.arange(N_pad + 1, device="cuda", dtype=torch.int32) if is_varlen else None

    # Forward args shared by eager + graph path
    if is_varlen:
        a_arg, b_arg = a_pad.contiguous(), b_pad.contiguous()
    else:
        a_arg, b_arg = a_pad.unsqueeze(1).contiguous(), b_pad.unsqueeze(1).contiguous()

    state_p = state_pool.clone()
    o_p = kda_packed_decode(
        mixed,
        a_arg,
        b_arg,
        A_log=A_log,
        dt_bias=dt_bias,
        state=state_p,
        state_indices=indices,
        cu_seqlens=cu_seqlens,
        scale=scale,
    )
    if is_varlen:
        o_real = o_p[:, :N_real]  # (1,N_real,HV,V)
    else:
        o_real = o_p[:N_real]  # (N_real,1,HV,V)

    # Compare real rows against non-packed run on the real-only inputs.
    state_ref = state_real.clone()
    if is_varlen:
        o_ref = kda_decode(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q.unsqueeze(0).contiguous(),
            k=k.unsqueeze(0).contiguous(),
            v=v.unsqueeze(0).contiguous(),
            a=a.contiguous(),
            b=b.contiguous(),
            initial_state_source=state_ref,
            initial_state_indices=torch.arange(N_real, device="cuda", dtype=torch.int32),
            cu_seqlens=torch.arange(N_real + 1, device="cuda", dtype=torch.int32),
            scale=scale,
        )  # (1,N_real,HV,V)
    else:
        o_ref = kda_decode(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q.unsqueeze(1).contiguous(),
            k=k.unsqueeze(1).contiguous(),
            v=v.unsqueeze(1).contiguous(),
            a=a.unsqueeze(1).contiguous(),
            b=b.unsqueeze(1).contiguous(),
            initial_state_source=state_ref,
            initial_state_indices=torch.arange(N_real, device="cuda", dtype=torch.int32),
            scale=scale,
        )  # (N_real,1,HV,V)

    _assert_close("real output", o_ref.float(), o_real.float())
    # dummy (padded) state slots must be untouched; their output rows are
    # UNINITIALIZED by the kernel (no write), so we don't assert on them.
    assert torch.equal(state_p[N_real:], state_pool[N_real:]), "dummy state slots were modified"

    # CUDA graph capture + replay smoke (capture must not raise; replay must not raise).
    # Run once eagerly first so the kernel is compiled and any warming cu_seqlens
    # cache is populated before capture.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        state_g = state_pool.clone()
        _ = kda_packed_decode(
            mixed,
            a_arg,
            b_arg,
            A_log=A_log,
            dt_bias=dt_bias,
            state=state_g,
            state_indices=indices,
            cu_seqlens=cu_seqlens,
            scale=scale,
        )
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_g2 = kda_packed_decode(
            mixed,
            a_arg,
            b_arg,
            A_log=A_log,
            dt_bias=dt_bias,
            state=state_g,
            state_indices=indices,
            cu_seqlens=cu_seqlens,
            scale=scale,
        )
    g.replay()
    assert out_g2 is not None
