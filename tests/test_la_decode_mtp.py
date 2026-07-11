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
Unit tests for la_decode_mtp (CuTe DSL Lightning Attention MTP decode kernel).

Compares against a PyTorch reference implementation of multi-token
Lightning Attention decode (T > 1).

Layouts:
  q, k:                [B, T, H,  K]   fp32
  v:                   [B, T, HV, V]   fp32
  s:                   [pool_size, HV, V, K]  fp32  (V-major, K-last)
  intermediate_states: [pool_size * T * HV, V, K] fp32, or 1-elem dummy
  out:                 [B, T, HV, V]   fp32
  decay_scales:        [H]             fp32  (positive; kernel does exp(-x))
  s_offsets:           [B]             int32 (pool index per batch; -1 to skip)
"""

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from cula.lightning.la_decode_mtp import linear_attention_decode_mtp


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_inputs(B, T, H, HV, D, device="cuda", seed=42):
    """Returns q[B,T,H,D] fp32, k[B,T,H,D] fp32, v[B,T,HV,D] fp32, state[B,HV,D,D] fp32."""
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, D, device=device, dtype=torch.float32)
    k = torch.randn(B, T, H, D, device=device, dtype=torch.float32)
    v = torch.randn(B, T, HV, D, device=device, dtype=torch.float32)
    state = torch.randn(B, HV, D, D, device=device, dtype=torch.float32) * 0.01
    return q, k, v, state


def run_la_mtp(
    q,
    k,
    v,
    state_4d,
    decay_scales,
    scale,
    T,
    cache_intermediate_states=False,
    disable_state_update=False,
):
    """
    Wraps linear_attention_decode_mtp with proper state-layout conversion.

    state_4d: [B, HV, K, V] fp32 (K-major)
    Kernel expects s: [pool_size=B, HV, V, K]; we transpose K and V.
    """
    B, HV, K, V = state_4d.shape
    H = q.shape[2]
    assert HV % H == 0, "HV must be a multiple of H"

    # pretranspose: [B, HV, V, K]
    s_cute = state_4d.permute(0, 1, 3, 2).contiguous().clone()
    out = torch.zeros(B, T, HV, V, device=q.device, dtype=torch.float32)
    s_offsets = torch.arange(B, device=q.device, dtype=torch.int32)

    if cache_intermediate_states:
        inter = torch.zeros(B * T * HV, V, K, device=q.device, dtype=torch.float32)
    else:
        inter = torch.empty(1, 1, 1, device=q.device, dtype=torch.float32)  # dummy

    cu_seqlens = torch.empty(1, device=q.device, dtype=torch.int32)  # dummy when is_varlen=False

    linear_attention_decode_mtp(
        q,
        k,
        v,
        s_cute,
        inter,
        out,
        decay_scales=decay_scales,
        s_offsets=s_offsets,
        cu_seqlens=cu_seqlens,
        softmax_scale=scale,
        T=T,
        cache_intermediate_states=cache_intermediate_states,
        disable_state_update=disable_state_update,
        is_varlen=False,
    )

    # convert state back: [B, HV, V, K] -> [B, HV, K, V]
    state_out = s_cute.permute(0, 1, 3, 2).contiguous()

    if cache_intermediate_states:
        # inter (kernel): [B*T*HV, V, K] -> ref layout [B*T*HV, K, V]
        inter_out = inter.permute(0, 2, 1).contiguous()
    else:
        inter_out = None

    return out, state_out, inter_out


# ---------------------------------------------------------------------------
# Tests vs PyTorch reference
# ---------------------------------------------------------------------------
# Each (B, T) below targets a distinct heuristic config (with H=HV=64):
#   B=1,  T=4: work_units=64   → tile_v=8,  ilp=2, smem_v=False
#   B=2,  T=2: work_units=128  → tile_v=16, ilp=4, smem_v=False
#   B=2,  T=4: work_units=128  → tile_v=16, ilp=4, smem_v=False
#   B=8,  T=4: work_units=512  → tile_v=32, ilp=4, smem_v=False
#   B=32, T=2: work_units=2048 → tile_v=64, ilp=8, smem_v=False  (state_update ON)
#   B=32, T=4: work_units=2048 → tile_v=64, ilp=4, smem_v=True
@pytest.mark.parametrize(
    "B,T,expected_config",
    [
        (1, 4, "tile_v=8_ilp=2"),
        (2, 2, "tile_v=16_ilp=4"),
        (2, 4, "tile_v=16_ilp=4"),
        (8, 4, "tile_v=32_ilp=4"),
        (32, 2, "tile_v=64_ilp=8"),
        (32, 4, "tile_v=64_ilp=4_smem_v"),
    ],
)
def test_output_vs_torch_ref(B, T, expected_config):
    _skip_if_no_sm90_or_later()
    H, HV, D = 64, 64, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H

    q, k, v, state = make_inputs(B, T, H, HV, D)
    o_ref, state_ref, _ = torch_la_mtp_ref(q, k, v, state, decay_scales, scale, T)
    o_cute, state_cute, _ = run_la_mtp(q, k, v, state, decay_scales, scale, T)

    # Output check
    rmse = torch.sqrt(torch.mean((o_cute.float() - o_ref.float()) ** 2)).item()
    max_ref = torch.abs(o_ref.float()).max().item()
    rel = rmse / (max_ref + 1e-8)
    assert rel < 0.01, f"B={B} T={T} [{expected_config}]: output rel RMSE {rel:.6f} too large"

    # State check
    state_rmse = torch.sqrt(torch.mean((state_cute - state_ref) ** 2)).item()
    state_max = torch.abs(state_ref).max().item()
    state_rel = state_rmse / (state_max + 1e-8)
    assert state_rel < 0.001, f"B={B} T={T} [{expected_config}]: state rel RMSE {state_rel:.6f} too large"


@pytest.mark.parametrize("H,HV", [(16, 16), (8, 32), (16, 64)])  # MHA + GQA
def test_different_heads(H, HV):
    """GQA support: HV is multiple of H; q/k indexed by i_h = i_hv // (HV//H)."""
    _skip_if_no_sm90_or_later()
    B, T, D = 4, 4, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H

    q, k, v, state = make_inputs(B, T, H, HV, D)
    o_ref, state_ref, _ = torch_la_mtp_ref(q, k, v, state, decay_scales, scale, T)
    o_cute, state_cute, _ = run_la_mtp(q, k, v, state, decay_scales, scale, T)

    rmse = torch.sqrt(torch.mean((o_cute.float() - o_ref.float()) ** 2)).item()
    max_ref = torch.abs(o_ref.float()).max().item()
    assert rmse / (max_ref + 1e-8) < 0.01, f"H={H} HV={HV}: output mismatch"

    state_rmse = torch.sqrt(torch.mean((state_cute - state_ref) ** 2)).item()
    state_max = torch.abs(state_ref).max().item()
    assert state_rmse / (state_max + 1e-8) < 0.001, f"H={H} HV={HV}: state mismatch"


def test_disable_state_update():
    """h0_source remains bitwise-equal to the input snapshot."""
    _skip_if_no_sm90_or_later()
    B, T, H, HV, D = 4, 4, 16, 16, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H

    q, k, v, state = make_inputs(B, T, H, HV, D)
    state_snapshot = state.clone()

    _, state_out, _ = run_la_mtp(
        q,
        k,
        v,
        state,
        decay_scales,
        scale,
        T,
        disable_state_update=True,
    )
    assert torch.equal(state_out, state_snapshot), "state was mutated despite disable_state_update=True"


def test_cache_intermediate_states():
    """Each per-t slice of inter matches the reference state_running at that step."""
    _skip_if_no_sm90_or_later()
    B, T, H, HV, D = 4, 4, 16, 16, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H

    q, k, v, state = make_inputs(B, T, H, HV, D)
    _, _, inter_ref = torch_la_mtp_ref(
        q,
        k,
        v,
        state,
        decay_scales,
        scale,
        T,
        cache_intermediate_states=True,
    )
    _, _, inter_cute = run_la_mtp(
        q,
        k,
        v,
        state,
        decay_scales,
        scale,
        T,
        cache_intermediate_states=True,
    )

    rmse = torch.sqrt(torch.mean((inter_cute - inter_ref) ** 2)).item()
    max_ref = torch.abs(inter_ref).max().item()
    assert rmse / (max_ref + 1e-8) < 0.001, f"intermediate states mismatch, rel_rmse={rmse / (max_ref + 1e-8):.6f}"

    inter_cute_v = inter_cute.view(B, T, HV, D, D)
    inter_ref_v = inter_ref.view(B, T, HV, D, D)
    for b in range(B):
        for t in range(T):
            slot_c = inter_cute_v[b, t]
            slot_r = inter_ref_v[b, t]
            slot_rmse = torch.sqrt(torch.mean((slot_c - slot_r) ** 2)).item()
            slot_max = torch.abs(slot_r).max().item()
            assert slot_rmse / (slot_max + 1e-8) < 0.001, (
                f"(b={b}, t={t}) intermediate mismatch, rel_rmse={slot_rmse / (slot_max + 1e-8):.6f}"
            )

    assert not torch.allclose(inter_cute_v[0, 0], inter_cute_v[0, 1])


def test_skip_with_negative_offset():
    """s_offsets[i]=-1: that batch's `out` slot stays at initial value."""
    _skip_if_no_sm90_or_later()
    B, T, H, HV, D = 4, 4, 16, 16, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H

    q, k, v, state = make_inputs(B, T, H, HV, D)
    s_cute = state.permute(0, 1, 3, 2).contiguous().clone()
    sentinel = 123.0
    out = torch.full((B, T, HV, D), sentinel, device=q.device, dtype=torch.float32)
    s_offsets = torch.arange(B, device=q.device, dtype=torch.int32)
    s_offsets[2] = -1  # skip batch index 2

    inter = torch.empty(1, 1, 1, device=q.device, dtype=torch.float32)
    cu_seqlens = torch.empty(1, device=q.device, dtype=torch.int32)
    linear_attention_decode_mtp(
        q,
        k,
        v,
        s_cute,
        inter,
        out,
        decay_scales=decay_scales,
        s_offsets=s_offsets,
        cu_seqlens=cu_seqlens,
        softmax_scale=scale,
        T=T,
        cache_intermediate_states=False,
        disable_state_update=False,
        is_varlen=False,
    )
    # batch 2 should be untouched (sentinel value)
    assert torch.all(out[2] == torch.full_like(out[2], sentinel)), "skipped batch was modified"
    # other batches should differ from sentinel
    assert not torch.all(out[0] == torch.full_like(out[0], sentinel)), "non-skipped batch unchanged"


def test_skip_with_negative_offset_cache_intermediate():
    _skip_if_no_sm90_or_later()
    B, T, H, HV, D = 4, 4, 16, 16, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H

    q, k, v, state = make_inputs(B, T, H, HV, D)
    s_cute = state.permute(0, 1, 3, 2).contiguous().clone()
    out = torch.zeros(B, T, HV, D, device=q.device, dtype=torch.float32)
    s_offsets = torch.arange(B, device=q.device, dtype=torch.int32)
    s_offsets[2] = -1

    inter_sentinel = 7.5
    inter = torch.full((B * T * HV, D, D), inter_sentinel, device=q.device, dtype=torch.float32)
    cu_seqlens = torch.empty(1, device=q.device, dtype=torch.int32)

    linear_attention_decode_mtp(
        q,
        k,
        v,
        s_cute,
        inter,
        out,
        decay_scales=decay_scales,
        s_offsets=s_offsets,
        cu_seqlens=cu_seqlens,
        softmax_scale=scale,
        T=T,
        cache_intermediate_states=True,
        disable_state_update=False,
        is_varlen=False,
    )

    skipped = inter[2 * T * HV : 3 * T * HV]
    assert torch.all(skipped == inter_sentinel), (
        f"intermediate_states for skipped batch was written (min={skipped.min().item()}, max={skipped.max().item()})"
    )

    others = torch.cat([inter[: 2 * T * HV], inter[3 * T * HV :]], dim=0)
    assert not torch.all(others == inter_sentinel), "non-skipped intermediate slots were not written"


def test_zero_decay():
    """With decay=0: state_new = state_old + k⊗v (no decay applied)."""
    _skip_if_no_sm90_or_later()
    B, T, H, HV, D = 4, 4, 16, 16, 128
    scale = D**-0.5
    decay_scales = torch.zeros(H, device="cuda", dtype=torch.float32)

    q, k, v, state = make_inputs(B, T, H, HV, D)
    o_ref, _, _ = torch_la_mtp_ref(q, k, v, state, decay_scales, scale, T)
    o_cute, _, _ = run_la_mtp(q, k, v, state, decay_scales, scale, T)

    rmse = torch.sqrt(torch.mean((o_cute.float() - o_ref.float()) ** 2)).item()
    max_ref = torch.abs(o_ref.float()).max().item()
    assert rmse / (max_ref + 1e-8) < 0.01, "zero decay: output mismatch"


def test_zero_state():
    """With zero initial state."""
    _skip_if_no_sm90_or_later()
    B, T, H, HV, D = 4, 4, 16, 16, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.ones(H, device="cuda", dtype=torch.float32)

    q, k, v, _ = make_inputs(B, T, H, HV, D)
    state = torch.zeros(B, HV, D, D, device="cuda", dtype=torch.float32)
    o_ref, _, _ = torch_la_mtp_ref(q, k, v, state, decay_scales, scale, T)
    o_cute, _, _ = run_la_mtp(q, k, v, state, decay_scales, scale, T)

    rmse = torch.sqrt(torch.mean((o_cute.float() - o_ref.float()) ** 2)).item()
    max_ref = torch.abs(o_ref.float()).max().item()
    assert rmse / (max_ref + 1e-8) < 0.01, "zero state: output mismatch"


def test_rejects_non_fp32_inputs():
    _skip_if_no_sm90_or_later()
    B, T, H, HV, D = 2, 4, 16, 16, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H

    q, k, v, state = make_inputs(B, T, H, HV, D)
    s_cute = state.permute(0, 1, 3, 2).contiguous().clone()
    out = torch.zeros(B, T, HV, D, device="cuda", dtype=torch.float32)
    inter = torch.empty(1, 1, 1, device="cuda", dtype=torch.float32)
    s_offsets = torch.arange(B, device="cuda", dtype=torch.int32)
    cu_seqlens = torch.empty(1, device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="q/k/v must be torch.float32"):
        linear_attention_decode_mtp(
            q.to(torch.bfloat16),
            k,
            v,
            s_cute,
            inter,
            out,
            decay_scales=decay_scales,
            s_offsets=s_offsets,
            cu_seqlens=cu_seqlens,
            softmax_scale=scale,
            T=T,
            cache_intermediate_states=False,
            disable_state_update=False,
            is_varlen=False,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
