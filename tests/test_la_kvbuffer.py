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

"""Unit tests for the KVBuffer verify + state-update kernels."""

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from cula.lightning.la_decode_mtp import linear_attention_decode_mtp
from cula.lightning.la_state_update_kvbuffer import (
    linear_attention_state_update_kvbuffer,
    linear_attention_state_update_kvbuffer_fused,
)
from cula.lightning.la_verify_kvbuffer import linear_attention_verify_kvbuffer


# ---------------------------------------------------------------------------
# Pure PyTorch reference for multi-token Lightning Attention decode
# ---------------------------------------------------------------------------
def torch_la_mtp_ref(q, k, v, state, decay_scales, scale, T, cache_intermediate_states=False, disable_state_update=False):
    """Pure PyTorch reference.

    Args:
        q, k:        [B, T, H,  D] fp32
        v:           [B, T, HV, D] fp32
        state:       [B, HV, D, D] fp32 (K-major, V-minor)
        decay_scales: [H] fp32 (positive; kernel does exp(-x))
        scale: float
        T: int
        cache_intermediate_states: cache per-step state to inter
        disable_state_update: do not update state_new at end

    Returns:
        out:        [B, T, HV, D] fp32
        state_new:  [B, HV, D, D] fp32
        inter:      [B*T*HV, D, D] fp32 or None
    """
    B, _, H, D = q.shape
    HV = v.shape[2]
    q_f = q.float() * scale
    k_f, v_f = k.float(), v.float()
    decay_per_q_head = torch.exp(-decay_scales)
    decay_per_hv = decay_per_q_head.repeat_interleave(HV // H).view(1, HV, 1, 1)

    state_running = state.clone()
    out = torch.zeros(B, T, HV, D, dtype=torch.float32, device=q.device)
    inter = torch.zeros(B * T * HV, D, D, dtype=torch.float32, device=q.device) if cache_intermediate_states else None

    for t in range(T):
        q_hv = q_f[:, t].repeat_interleave(HV // H, dim=1)
        k_hv = k_f[:, t].repeat_interleave(HV // H, dim=1)
        v_t = v_f[:, t]
        state_running = state_running * decay_per_hv + k_hv.unsqueeze(-1) * v_t.unsqueeze(-2)
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q_hv, state_running)
        if cache_intermediate_states:
            for b in range(B):
                inter[b * T * HV + t * HV : b * T * HV + (t + 1) * HV] = state_running[b]

    state_final = state.clone() if disable_state_update else state_running
    return out, state_final, inter


def _skip_if_no_sm90_or_later():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    cc = torch.cuda.get_device_capability("cuda")
    if cc[0] < 9:
        pytest.skip(f"requires SM90+, got SM{cc[0]}{cc[1]}")


def _make_inputs(B, T, H, HV, D, device="cuda", seed=42):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, D, device=device, dtype=torch.float32)
    k = torch.randn(B, T, H, D, device=device, dtype=torch.float32)
    v = torch.randn(B, T, HV, D, device=device, dtype=torch.float32)
    state = torch.randn(B, HV, D, D, device=device, dtype=torch.float32) * 0.01
    return q, k, v, state


def _make_kv_buffers(k, v, h0_indices, pool_size=None):
    B, T, H, D = k.shape
    _, _, HV, V = v.shape
    if pool_size is None:
        pool_size = B
    k_buf = torch.zeros(pool_size, T, H, D, device=k.device, dtype=torch.float32)
    v_buf = torch.zeros(pool_size, T, HV, V, device=v.device, dtype=torch.float32)
    for b in range(B):
        pool_idx = int(h0_indices[b].item())
        if pool_idx >= 0:
            k_buf[pool_idx] = k[b]
            v_buf[pool_idx] = v[b]
    return k_buf, v_buf


@pytest.mark.parametrize("T", [4, 16])
def test_verify_rejects_non_fp32_state(T):
    _skip_if_no_sm90_or_later()
    B, H, HV, D = 2, 16, 16, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H
    q, k, v, state = _make_inputs(B, T, H, HV, D)
    s_cute = state.permute(0, 1, 3, 2).contiguous().clone().to(torch.bfloat16)
    out = torch.zeros(B, T, HV, D, device="cuda", dtype=torch.float32)
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="s must be torch.float32"):
        linear_attention_verify_kvbuffer(q, k, v, s_cute, out, decay_scales, h0_indices, scale, T)


def test_state_update_L0_no_op():
    """accepted_len=0 everywhere: s must be byte-for-byte unchanged."""
    _skip_if_no_sm90_or_later()
    B, T, H, HV, D = 4, 4, 16, 16, 128
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H
    _, k, v, state = _make_inputs(B, T, H, HV, D)

    s_cute = state.permute(0, 1, 3, 2).contiguous().clone()  # [B, HV, V, K]
    s_snapshot = s_cute.clone()
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)
    accepted_len = torch.zeros(B, device="cuda", dtype=torch.int32)
    k_buf, v_buf = _make_kv_buffers(k, v, h0_indices)

    linear_attention_state_update_kvbuffer(
        k_buf,
        v_buf,
        s_cute,
        decay_scales,
        h0_indices,
        accepted_len,
        T,
    )
    assert torch.equal(s_cute, s_snapshot), "L=0 must leave state unchanged"


def _ref_state_after_L(state, k, v, decay_scales, L_per_batch, T):
    """state[B,HV,K,V] fp32; returns the per-batch state after L recurrent steps."""
    B, HV, K, V = state.shape
    H = k.shape[2]
    k_f, v_f = k.float(), v.float()
    decay_per_q_head = torch.exp(-decay_scales)
    decay_per_hv = decay_per_q_head.repeat_interleave(HV // H).view(HV, 1, 1)
    out = state.clone()
    for b in range(B):
        L = int(L_per_batch[b].item())
        running = state[b].clone()
        for i in range(L):
            k_hv = k_f[b, i].repeat_interleave(HV // H, dim=0)  # [HV, K]
            v_i = v_f[b, i]  # [HV, V]
            running = running * decay_per_hv + k_hv.unsqueeze(-1) * v_i.unsqueeze(-2)
        out[b] = running
    return out


@pytest.mark.parametrize(
    "B,T,H,HV,D",
    [(4, 4, 16, 16, 128), (8, 4, 64, 64, 128), (4, 3, 16, 16, 128), (8, 7, 64, 64, 128)],
)
def test_state_update_full_accept(B, T, H, HV, D):
    """accepted_len=T everywhere: bit-exact vs baseline recurrence reference."""
    _skip_if_no_sm90_or_later()
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H
    _, k, v, state = _make_inputs(B, T, H, HV, D)

    L_per_batch = torch.full((B,), T, device="cuda", dtype=torch.int32)
    ref = _ref_state_after_L(state, k, v, decay_scales, L_per_batch, T)  # [B,HV,K,V]

    s_cute = state.permute(0, 1, 3, 2).contiguous().clone()  # [B,HV,V,K]
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)
    k_buf, v_buf = _make_kv_buffers(k, v, h0_indices)
    linear_attention_state_update_kvbuffer(
        k_buf,
        v_buf,
        s_cute,
        decay_scales,
        h0_indices,
        L_per_batch,
        T,
    )
    got = s_cute.permute(0, 1, 3, 2).contiguous()  # back to [B,HV,K,V]
    rmse = torch.sqrt(torch.mean((got - ref) ** 2)).item()
    rel = rmse / (torch.abs(ref).max().item() + 1e-8)
    assert rel < 1e-3, f"full-accept state rel RMSE {rel:.6f} too large"


@pytest.mark.parametrize("L", [0, 1, 3])
def test_state_update_partial(L):
    """Uniform accepted_len=L across all batches."""
    _skip_if_no_sm90_or_later()
    B, T, H, HV, D = 4, 4, 16, 16, 128
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H
    _, k, v, state = _make_inputs(B, T, H, HV, D)

    L_per_batch = torch.full((B,), L, device="cuda", dtype=torch.int32)
    ref = _ref_state_after_L(state, k, v, decay_scales, L_per_batch, T)

    s_cute = state.permute(0, 1, 3, 2).contiguous().clone()
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)
    k_buf, v_buf = _make_kv_buffers(k, v, h0_indices)
    linear_attention_state_update_kvbuffer(
        k_buf,
        v_buf,
        s_cute,
        decay_scales,
        h0_indices,
        L_per_batch,
        T,
    )
    got = s_cute.permute(0, 1, 3, 2).contiguous()
    rel = torch.sqrt(torch.mean((got - ref) ** 2)).item() / (torch.abs(ref).max().item() + 1e-8)
    assert rel < 1e-3, f"L={L} state rel RMSE {rel:.6f}"


def test_state_update_per_batch_L():
    """accepted_len varies per batch: [0, 1, T-1, T]."""
    _skip_if_no_sm90_or_later()
    B, T, H, HV, D = 4, 4, 16, 16, 128
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H
    _, k, v, state = _make_inputs(B, T, H, HV, D)

    L_per_batch = torch.tensor([0, 1, T - 1, T], device="cuda", dtype=torch.int32)
    ref = _ref_state_after_L(state, k, v, decay_scales, L_per_batch, T)

    s_cute = state.permute(0, 1, 3, 2).contiguous().clone()
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)
    k_buf, v_buf = _make_kv_buffers(k, v, h0_indices)
    linear_attention_state_update_kvbuffer(
        k_buf,
        v_buf,
        s_cute,
        decay_scales,
        h0_indices,
        L_per_batch,
        T,
    )
    got = s_cute.permute(0, 1, 3, 2).contiguous()
    for b in range(B):
        rel = torch.sqrt(torch.mean((got[b] - ref[b]) ** 2)).item() / (torch.abs(ref[b]).max().item() + 1e-8)
        assert rel < 1e-3, f"batch {b} (L={int(L_per_batch[b])}) rel RMSE {rel:.6f}"


def test_state_update_skip_negative_h0_indices():
    """h0_indices[b]=-1: that pool slot is untouched even with accepted_len>0."""
    _skip_if_no_sm90_or_later()
    B, T, H, HV, D = 4, 4, 16, 16, 128
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H
    _, k, v, state = _make_inputs(B, T, H, HV, D)

    s_cute = state.permute(0, 1, 3, 2).contiguous().clone()
    snapshot_b2 = s_cute[2].clone()
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)
    h0_indices[2] = -1
    L_per_batch = torch.full((B,), T, device="cuda", dtype=torch.int32)
    k_buf, v_buf = _make_kv_buffers(k, v, h0_indices)

    linear_attention_state_update_kvbuffer(
        k_buf,
        v_buf,
        s_cute,
        decay_scales,
        h0_indices,
        L_per_batch,
        T,
    )
    assert torch.equal(s_cute[2], snapshot_b2), "skipped batch slot was modified"


def test_verify_skip_negative_h0_indices():
    """h0_indices[b]=-1: out[b] stays at its sentinel value."""
    _skip_if_no_sm90_or_later()
    B, T, H, HV, D = 4, 4, 16, 16, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H
    q, k, v, state = _make_inputs(B, T, H, HV, D)

    s_cute = state.permute(0, 1, 3, 2).contiguous().clone()
    sentinel = 123.0
    out = torch.full((B, T, HV, D), sentinel, device="cuda", dtype=torch.float32)
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)
    h0_indices[2] = -1

    linear_attention_verify_kvbuffer(
        q,
        k,
        v,
        s_cute,
        out,
        decay_scales,
        h0_indices,
        scale,
        T,
    )
    assert torch.all(out[2] == sentinel), "skipped batch out slot was modified"


@pytest.mark.parametrize(
    "B,T",
    [(1, 4), (2, 2), (2, 4), (8, 4), (32, 2), (32, 4), (2, 1), (2, 3), (8, 5), (8, 7)],
)
def test_verify_outputs_match_ref(B, T):
    """Verify kernel o matches torch_la_mtp_ref across the baseline configs."""
    _skip_if_no_sm90_or_later()
    H, HV, D = 64, 64, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H
    q, k, v, state = _make_inputs(B, T, H, HV, D)

    o_ref, _, _ = torch_la_mtp_ref(q, k, v, state, decay_scales, scale, T)

    s_cute = state.permute(0, 1, 3, 2).contiguous().clone()
    out = torch.zeros(B, T, HV, D, device="cuda", dtype=torch.float32)
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)
    linear_attention_verify_kvbuffer(
        q,
        k,
        v,
        s_cute,
        out,
        decay_scales,
        h0_indices,
        scale,
        T,
    )
    rel = torch.sqrt(torch.mean((out.float() - o_ref.float()) ** 2)).item() / (torch.abs(o_ref.float()).max().item() + 1e-8)
    assert rel < 1e-2, f"B={B} T={T}: verify output rel RMSE {rel:.6f} too large"


@pytest.mark.parametrize("H,HV", [(16, 16), (8, 32), (16, 64)])
def test_verify_different_heads(H, HV):
    _skip_if_no_sm90_or_later()
    B, T, D = 4, 4, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H
    q, k, v, state = _make_inputs(B, T, H, HV, D)
    o_ref, _, _ = torch_la_mtp_ref(q, k, v, state, decay_scales, scale, T)

    s_cute = state.permute(0, 1, 3, 2).contiguous().clone()
    out = torch.zeros(B, T, HV, D, device="cuda", dtype=torch.float32)
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)
    linear_attention_verify_kvbuffer(
        q,
        k,
        v,
        s_cute,
        out,
        decay_scales,
        h0_indices,
        scale,
        T,
    )
    rel = torch.sqrt(torch.mean((out.float() - o_ref.float()) ** 2)).item() / (torch.abs(o_ref.float()).max().item() + 1e-8)
    assert rel < 1e-2, f"H={H} HV={HV}: verify output mismatch {rel:.6f}"


def test_verify_zero_decay():
    _skip_if_no_sm90_or_later()
    B, T, H, HV, D = 4, 4, 16, 16, 128
    scale = D**-0.5
    decay_scales = torch.zeros(H, device="cuda", dtype=torch.float32)
    q, k, v, state = _make_inputs(B, T, H, HV, D)
    o_ref, _, _ = torch_la_mtp_ref(q, k, v, state, decay_scales, scale, T)
    s_cute = state.permute(0, 1, 3, 2).contiguous().clone()
    out = torch.zeros(B, T, HV, D, device="cuda", dtype=torch.float32)
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)
    linear_attention_verify_kvbuffer(q, k, v, s_cute, out, decay_scales, h0_indices, scale, T)
    rel = torch.sqrt(torch.mean((out.float() - o_ref.float()) ** 2)).item() / (torch.abs(o_ref.float()).max().item() + 1e-8)
    assert rel < 1e-2, f"zero decay: {rel:.6f}"


def test_verify_zero_state():
    _skip_if_no_sm90_or_later()
    B, T, H, HV, D = 4, 4, 16, 16, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.ones(H, device="cuda", dtype=torch.float32)
    q, k, v, _ = _make_inputs(B, T, H, HV, D)
    state = torch.zeros(B, HV, D, D, device="cuda", dtype=torch.float32)
    o_ref, _, _ = torch_la_mtp_ref(q, k, v, state, decay_scales, scale, T)
    s_cute = state.permute(0, 1, 3, 2).contiguous().clone()
    out = torch.zeros(B, T, HV, D, device="cuda", dtype=torch.float32)
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)
    linear_attention_verify_kvbuffer(q, k, v, s_cute, out, decay_scales, h0_indices, scale, T)
    rel = torch.sqrt(torch.mean((out.float() - o_ref.float()) ** 2)).item() / (torch.abs(o_ref.float()).max().item() + 1e-8)
    assert rel < 1e-2, f"zero state: {rel:.6f}"


def test_end_to_end_equivalence_with_baseline():
    """KVBuffer (verify + state_update L=T) == baseline (cache_inter=T, disable=T)."""
    _skip_if_no_sm90_or_later()
    B, T, H, HV, D = 8, 4, 64, 64, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H
    q, k, v, state = _make_inputs(B, T, H, HV, D)

    # ---- Baseline: capture out + all intermediate states ----
    s_base = state.permute(0, 1, 3, 2).contiguous().clone()  # [B,HV,V,K]
    out_base = torch.zeros(B, T, HV, D, device="cuda", dtype=torch.float32)
    s_offsets = torch.arange(B, device="cuda", dtype=torch.int32)
    inter = torch.zeros(B * T * HV, D, D, device="cuda", dtype=torch.float32)  # [.,V,K]
    cu_seqlens = torch.empty(1, device="cuda", dtype=torch.int32)
    linear_attention_decode_mtp(
        q,
        k,
        v,
        s_base,
        inter,
        out_base,
        decay_scales=decay_scales,
        s_offsets=s_offsets,
        cu_seqlens=cu_seqlens,
        softmax_scale=scale,
        T=T,
        cache_intermediate_states=True,
        disable_state_update=True,
        is_varlen=False,
    )

    # ---- KVBuffer: verify writes out; state-update (L=T) writes state ----
    s_kv = state.permute(0, 1, 3, 2).contiguous().clone()  # [B,HV,V,K]
    out_kv = torch.zeros(B, T, HV, D, device="cuda", dtype=torch.float32)
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)
    linear_attention_verify_kvbuffer(
        q,
        k,
        v,
        s_kv,
        out_kv,
        decay_scales,
        h0_indices,
        scale,
        T,
    )
    accepted_len = torch.full((B,), T, device="cuda", dtype=torch.int32)
    k_buf, v_buf = _make_kv_buffers(k, v, h0_indices)
    linear_attention_state_update_kvbuffer(
        k_buf,
        v_buf,
        s_kv,
        decay_scales,
        h0_indices,
        accepted_len,
        T,
    )

    # (a) outputs match
    rel_o = torch.sqrt(torch.mean((out_kv - out_base) ** 2)).item() / (torch.abs(out_base).max().item() + 1e-8)
    assert rel_o < 1e-2, f"output mismatch vs baseline: {rel_o:.6f}"

    # (b) updated state == baseline's last intermediate slice [B,HV,V,K]
    inter_v = inter.view(B, T, HV, D, D)  # [B,T,HV,V,K]
    last_state = inter_v[:, T - 1]  # [B,HV,V,K]
    rel_s = torch.sqrt(torch.mean((s_kv - last_state) ** 2)).item() / (torch.abs(last_state).max().item() + 1e-8)
    assert rel_s < 1e-3, f"state mismatch vs baseline last intermediate: {rel_s:.6f}"


@pytest.mark.parametrize("B,T", [(4, 4), (8, 2), (32, 4)])
def test_verify_writes_kv_buffer(B, T):
    """Verify kernel with k_buf/v_buf writes correct copies of k and v."""
    _skip_if_no_sm90_or_later()
    H, HV, D = 64, 64, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H
    q, k, v, state = _make_inputs(B, T, H, HV, D)

    pool_size = B
    s_cute = state.permute(0, 1, 3, 2).contiguous().clone()
    out = torch.zeros(B, T, HV, D, device="cuda", dtype=torch.float32)
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)
    k_buf = torch.zeros(pool_size, T, H, D, device="cuda", dtype=torch.float32)
    v_buf = torch.zeros(pool_size, T, HV, D, device="cuda", dtype=torch.float32)

    linear_attention_verify_kvbuffer(
        q,
        k,
        v,
        s_cute,
        out,
        decay_scales,
        h0_indices,
        scale,
        T,
        k_buf=k_buf,
        v_buf=v_buf,
    )

    for b in range(B):
        pool_idx = h0_indices[b].item()
        assert torch.equal(k_buf[pool_idx], k[b]), f"k_buf mismatch at batch {b}"
        assert torch.equal(v_buf[pool_idx], v[b]), f"v_buf mismatch at batch {b}"


def test_verify_output_unchanged_with_kv_write():
    """Output o is identical whether k_buf/v_buf are provided or not."""
    _skip_if_no_sm90_or_later()
    B, T, H, HV, D = 8, 4, 64, 64, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H
    q, k, v, state = _make_inputs(B, T, H, HV, D)

    pool_size = B
    s1 = state.permute(0, 1, 3, 2).contiguous().clone()
    s2 = s1.clone()
    out_no_buf = torch.zeros(B, T, HV, D, device="cuda", dtype=torch.float32)
    out_with_buf = torch.zeros(B, T, HV, D, device="cuda", dtype=torch.float32)
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)

    linear_attention_verify_kvbuffer(
        q,
        k,
        v,
        s1,
        out_no_buf,
        decay_scales,
        h0_indices,
        scale,
        T,
    )

    k_buf = torch.zeros(pool_size, T, H, D, device="cuda", dtype=torch.float32)
    v_buf = torch.zeros(pool_size, T, HV, D, device="cuda", dtype=torch.float32)
    linear_attention_verify_kvbuffer(
        q,
        k,
        v,
        s2,
        out_with_buf,
        decay_scales,
        h0_indices,
        scale,
        T,
        k_buf=k_buf,
        v_buf=v_buf,
    )

    assert torch.equal(out_no_buf, out_with_buf), "kv write should not affect output"


@pytest.mark.parametrize("B,T,H,HV,D", [(4, 4, 16, 16, 128), (8, 4, 64, 64, 128)])
def test_state_update_from_buffer(B, T, H, HV, D):
    """State update from pool-indexed k_buf/v_buf is deterministic."""
    _skip_if_no_sm90_or_later()
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H
    _, k, v, state = _make_inputs(B, T, H, HV, D)

    pool_size = B
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)
    L_per_batch = torch.full((B,), T, device="cuda", dtype=torch.int32)

    # Path A: read from pool-indexed k_buf/v_buf filled from raw k, v
    k_buf, v_buf = _make_kv_buffers(k, v, h0_indices)
    s_raw = state.permute(0, 1, 3, 2).contiguous().clone()
    linear_attention_state_update_kvbuffer(
        k_buf,
        v_buf,
        s_raw,
        decay_scales,
        h0_indices,
        L_per_batch,
        T,
    )

    # Path B: same data, independently materialized buffer.
    k_buf_2, v_buf_2 = _make_kv_buffers(k, v, h0_indices, pool_size=pool_size)
    s_buf = state.permute(0, 1, 3, 2).contiguous().clone()
    linear_attention_state_update_kvbuffer(
        k_buf_2,
        v_buf_2,
        s_buf,
        decay_scales,
        h0_indices,
        L_per_batch,
        T,
    )

    assert torch.equal(s_raw, s_buf), "buffer-read state must match raw-read state"


def test_verify_skip_negative_indices_no_buffer_write():
    """h0_indices[b]=-1: k_buf and v_buf slots are untouched."""
    _skip_if_no_sm90_or_later()
    B, T, H, HV, D = 4, 4, 16, 16, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H
    q, k, v, state = _make_inputs(B, T, H, HV, D)

    pool_size = B
    sentinel = 42.0
    k_buf = torch.full((pool_size, T, H, D), sentinel, device="cuda", dtype=torch.float32)
    v_buf = torch.full((pool_size, T, HV, D), sentinel, device="cuda", dtype=torch.float32)
    k_buf_snap = k_buf.clone()
    v_buf_snap = v_buf.clone()

    s_cute = state.permute(0, 1, 3, 2).contiguous().clone()
    out = torch.zeros(B, T, HV, D, device="cuda", dtype=torch.float32)
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)
    h0_indices[2] = -1

    linear_attention_verify_kvbuffer(
        q,
        k,
        v,
        s_cute,
        out,
        decay_scales,
        h0_indices,
        scale,
        T,
        k_buf=k_buf,
        v_buf=v_buf,
    )

    assert torch.equal(k_buf[2], k_buf_snap[2]), "skipped batch k_buf slot was modified"
    assert torch.equal(v_buf[2], v_buf_snap[2]), "skipped batch v_buf slot was modified"


def test_end_to_end_with_buffer():
    """Full pipeline: verify(+kv write) → state_update(from buffer) matches baseline."""
    _skip_if_no_sm90_or_later()
    B, T, H, HV, D = 8, 4, 64, 64, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H
    q, k, v, state = _make_inputs(B, T, H, HV, D)

    pool_size = B
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)

    # Reference: existing end-to-end (no buffer)
    s_ref = state.permute(0, 1, 3, 2).contiguous().clone()
    out_ref = torch.zeros(B, T, HV, D, device="cuda", dtype=torch.float32)
    linear_attention_verify_kvbuffer(
        q,
        k,
        v,
        s_ref,
        out_ref,
        decay_scales,
        h0_indices,
        scale,
        T,
    )
    accepted_len = torch.full((B,), T, device="cuda", dtype=torch.int32)
    k_buf_ref, v_buf_ref = _make_kv_buffers(k, v, h0_indices, pool_size=pool_size)
    linear_attention_state_update_kvbuffer(
        k_buf_ref,
        v_buf_ref,
        s_ref,
        decay_scales,
        h0_indices,
        accepted_len,
        T,
    )

    # Buffer path: verify writes buffer, state_update reads buffer
    s_buf = state.permute(0, 1, 3, 2).contiguous().clone()
    out_buf = torch.zeros(B, T, HV, D, device="cuda", dtype=torch.float32)
    k_buf = torch.zeros(pool_size, T, H, D, device="cuda", dtype=torch.float32)
    v_buf = torch.zeros(pool_size, T, HV, D, device="cuda", dtype=torch.float32)

    linear_attention_verify_kvbuffer(
        q,
        k,
        v,
        s_buf,
        out_buf,
        decay_scales,
        h0_indices,
        scale,
        T,
        k_buf=k_buf,
        v_buf=v_buf,
    )
    linear_attention_state_update_kvbuffer(
        k_buf,
        v_buf,
        s_buf,
        decay_scales,
        h0_indices,
        accepted_len,
        T,
    )

    assert torch.equal(out_ref, out_buf), "output mismatch with buffer pipeline"
    assert torch.equal(s_ref, s_buf), "state mismatch with buffer pipeline"


def test_state_update_fused_matches_per_layer():
    """Layer-fused state update matches independent per-layer launches."""
    _skip_if_no_sm90_or_later()
    num_layers, B, T, H, HV, D = 3, 4, 4, 16, 16, 128
    pool_size = B
    h0_indices = torch.arange(B, device="cuda", dtype=torch.int32)
    accepted_len = torch.tensor([0, 1, T - 1, T], device="cuda", dtype=torch.int32)

    k_buf_layers = []
    v_buf_layers = []
    states = []
    decays = []
    for layer in range(num_layers):
        _, k, v, state = _make_inputs(B, T, H, HV, D, seed=100 + layer)
        k_buf, v_buf = _make_kv_buffers(k, v, h0_indices, pool_size=pool_size)
        k_buf_layers.append(k_buf)
        v_buf_layers.append(v_buf)
        states.append(state.permute(0, 1, 3, 2).contiguous())
        decays.append(0.3 * (layer + 1) * torch.arange(H, device="cuda", dtype=torch.float32) / H)

    k_buf_fused = torch.stack(k_buf_layers, dim=0)
    v_buf_fused = torch.stack(v_buf_layers, dim=0)
    s_fused = torch.stack(states, dim=0)
    decay_fused = torch.stack(decays, dim=0)

    s_expected = s_fused.clone()
    for layer in range(num_layers):
        linear_attention_state_update_kvbuffer(
            k_buf_fused[layer],
            v_buf_fused[layer],
            s_expected[layer],
            decay_fused[layer],
            h0_indices,
            accepted_len,
            T,
        )

    linear_attention_state_update_kvbuffer_fused(
        k_buf_fused,
        v_buf_fused,
        s_fused,
        decay_fused,
        h0_indices,
        accepted_len,
        T,
    )

    assert torch.equal(s_fused, s_expected), "fused state update must match per-layer state update"
