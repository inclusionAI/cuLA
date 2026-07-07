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
Test for Lightning Attention decode state-pool indirect indexing.

Exposes the bug where la_decode ignores s_offsets and indexes state
by flattened batch_idx directly. With identity offsets the bug is invisible.
With non-identity offsets, the kernel reads/writes wrong state slots.
"""

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cula.ops.lightning.decode import linear_attention_decode


def torch_la_decode_ref(q, k, v, state, decay_scales, scale):
    """Pure PyTorch reference — state is [B, H, K, V] (BHKV)."""
    B, H, D = q.shape
    q_f = q.float() * scale
    k_f = k.float()
    v_f = v.float()
    decay = torch.exp(-decay_scales).view(1, H, 1, 1)
    state_new = state * decay + k_f.unsqueeze(-1) * v_f.unsqueeze(-2)
    o = torch.einsum("bhk,bhkv->bhv", q_f, state_new)
    return o.to(torch.bfloat16), state_new


def run_la_decode_with_pool(q, k, v, state_pool_4d, s_offsets, decay_scales, scale):
    """
    Run la_decode with a state pool and arbitrary offsets.

    state_pool_4d: [pool_size, H, K, V] — the full pool (BHKV layout)
    s_offsets: [B] — which pool slot each batch element uses
    """
    B, H, D = q.shape
    pool_size = state_pool_4d.shape[0]

    # la_decode expects BHVK layout: [pool_size*H, V, K]
    state_cute = state_pool_4d.clone().transpose(-1, -2).contiguous().reshape(pool_size * H, D, D)
    out = torch.zeros(B, H, D, device=q.device, dtype=torch.bfloat16)

    linear_attention_decode(
        q,
        k,
        v,
        state_cute,
        out,
        softmax_scale=scale,
        stride_q=0,
        stride_k=0,
        stride_v=0,
        stride_s=0,
        stride_o=0,
        s_offsets=s_offsets,
        decay_scales=decay_scales,
        HEAD_DIM=D,
        K_SPLIT_DIM=D,
        V_SPLIT_DIM=D,
    )

    state_out = state_cute.reshape(pool_size, H, D, D).transpose(-1, -2).contiguous()
    return out, state_out


# ---------------------------------------------------------------------------
# Test 1: Identity offsets (baseline — should always pass)
# ---------------------------------------------------------------------------
def test_identity_offsets():
    """Identity offsets: s_offsets=[0,1,2,3]. Bug is invisible."""
    B, H, D = 4, 8, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H

    torch.manual_seed(42)
    q = torch.randn(B, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, D, device="cuda", dtype=torch.bfloat16)
    state_4d = torch.randn(B, H, D, D, device="cuda", dtype=torch.float32) * 0.1

    s_offsets = torch.arange(B, device="cuda", dtype=torch.int32)
    out, _ = run_la_decode_with_pool(q, k, v, state_4d, s_offsets, decay_scales, scale)

    o_ref, _ = torch_la_decode_ref(q, k, v, state_4d, decay_scales, scale)
    rmse = torch.sqrt(torch.mean((out.float() - o_ref.float()) ** 2)).item()
    max_ref = torch.abs(o_ref.float()).max().item()
    rel_err = rmse / (max_ref + 1e-8)

    assert rel_err < 0.01, f"Identity offsets: rel_err={rel_err:.6f}"


# ---------------------------------------------------------------------------
# Test 2: Non-identity offsets (exposes the bug)
# ---------------------------------------------------------------------------
def test_non_identity_offsets():
    """
    pool_size=6, batch=4, offsets=[2, 0, 5, 1].
    Each batch reads a different, non-sequential pool slot.
    Bug: kernel reads slots [0,1,2,3] instead of [2,0,5,1].
    """
    B = 4
    POOL_SIZE = 6
    H, D = 8, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H

    torch.manual_seed(42)
    q = torch.randn(B, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, D, device="cuda", dtype=torch.bfloat16)

    # Large state magnitude so wrong-slot reads produce clearly different outputs
    state_pool = torch.randn(POOL_SIZE, H, D, D, device="cuda", dtype=torch.float32) * 0.1

    offsets = [2, 0, 5, 1]
    s_offsets = torch.tensor(offsets, device="cuda", dtype=torch.int32)

    out, _ = run_la_decode_with_pool(q, k, v, state_pool, s_offsets, decay_scales, scale)

    # Reference: manually select the correct state for each batch element
    state_selected = state_pool[s_offsets.long()]  # [B, H, D, D]
    o_ref, _ = torch_la_decode_ref(q, k, v, state_selected, decay_scales, scale)

    rmse = torch.sqrt(torch.mean((out.float() - o_ref.float()) ** 2)).item()
    max_ref = torch.abs(o_ref.float()).max().item()
    rel_err = rmse / (max_ref + 1e-8)

    assert rel_err < 0.01, f"Non-identity offsets {offsets}: rel_err={rel_err:.6f}"


# ---------------------------------------------------------------------------
# Test 3: Reversed offsets (another non-identity pattern)
# ---------------------------------------------------------------------------
def test_reversed_offsets():
    """
    pool_size=B, offsets=[3,2,1,0] (reversed).
    Batch 0 reads slot 3, batch 3 reads slot 0.
    """
    B, H, D = 4, 8, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H

    torch.manual_seed(42)
    q = torch.randn(B, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, D, device="cuda", dtype=torch.bfloat16)
    state_pool = torch.randn(B, H, D, D, device="cuda", dtype=torch.float32) * 0.1

    offsets = list(reversed(range(B)))
    s_offsets = torch.tensor(offsets, device="cuda", dtype=torch.int32)

    out, _ = run_la_decode_with_pool(q, k, v, state_pool, s_offsets, decay_scales, scale)

    state_selected = state_pool[s_offsets.long()]
    o_ref, _ = torch_la_decode_ref(q, k, v, state_selected, decay_scales, scale)

    rmse = torch.sqrt(torch.mean((out.float() - o_ref.float()) ** 2)).item()
    max_ref = torch.abs(o_ref.float()).max().item()
    rel_err = rmse / (max_ref + 1e-8)

    assert rel_err < 0.01, f"Reversed offsets {offsets}: rel_err={rel_err:.6f}"


# ---------------------------------------------------------------------------
# Test 4: State writeback with non-identity offsets
# ---------------------------------------------------------------------------
def test_state_writeback_non_identity():
    """
    Verify that state updates go to the correct pool slots.
    After decode, pool slot offsets[i] should have the updated state for batch i.
    Other pool slots should be unchanged.
    """
    B = 4
    POOL_SIZE = 6
    H, D = 8, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H

    torch.manual_seed(42)
    q = torch.randn(B, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, D, device="cuda", dtype=torch.bfloat16)
    state_pool = torch.randn(POOL_SIZE, H, D, D, device="cuda", dtype=torch.float32) * 0.1
    state_pool_orig = state_pool.clone()

    offsets = [2, 0, 5, 1]
    s_offsets = torch.tensor(offsets, device="cuda", dtype=torch.int32)

    _, state_out = run_la_decode_with_pool(q, k, v, state_pool, s_offsets, decay_scales, scale)

    # Reference: compute expected new state for each active batch element
    state_selected = state_pool_orig[s_offsets.long()]
    _, state_ref = torch_la_decode_ref(q, k, v, state_selected, decay_scales, scale)

    # Check that active slots were updated correctly
    for b_idx, pool_slot in enumerate(offsets):
        slot_rmse = torch.sqrt(torch.mean((state_out[pool_slot].float() - state_ref[b_idx].float()) ** 2)).item()
        slot_max = torch.abs(state_ref[b_idx].float()).max().item()
        slot_rel = slot_rmse / (slot_max + 1e-8)
        assert slot_rel < 0.001, f"State writeback: pool slot {pool_slot} (batch {b_idx}) rel_err={slot_rel:.6f}"

    # Check that inactive slots (3, 4) were NOT touched
    inactive = set(range(POOL_SIZE)) - set(offsets)
    for slot in inactive:
        diff = torch.abs(state_out[slot] - state_pool_orig[slot]).max().item()
        assert diff < 1e-8, f"Inactive pool slot {slot} was modified! max_diff={diff}"


# ---------------------------------------------------------------------------
# Test 5: Big batch (B > 32) with non-identity offsets
# ---------------------------------------------------------------------------
def test_big_batch_non_identity_offsets():
    """
    B=33 triggers the big-batch kernel path (B > 32).
    pool_size=40, shifted offsets so batch i reads slot (i + 7) % 40.
    """
    B = 33
    POOL_SIZE = 40
    H, D = 8, 128
    scale = D**-0.5
    decay_scales = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H

    torch.manual_seed(42)
    q = torch.randn(B, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, D, device="cuda", dtype=torch.bfloat16)

    state_pool = torch.randn(POOL_SIZE, H, D, D, device="cuda", dtype=torch.float32) * 0.1

    offsets = [(i + 7) % POOL_SIZE for i in range(B)]
    s_offsets = torch.tensor(offsets, device="cuda", dtype=torch.int32)

    out, _ = run_la_decode_with_pool(q, k, v, state_pool, s_offsets, decay_scales, scale)

    state_selected = state_pool[s_offsets.long()]
    o_ref, _ = torch_la_decode_ref(q, k, v, state_selected, decay_scales, scale)

    rmse = torch.sqrt(torch.mean((out.float() - o_ref.float()) ** 2)).item()
    max_ref = torch.abs(o_ref.float()).max().item()
    rel_err = rmse / (max_ref + 1e-8)

    assert rel_err < 0.01, f"Big batch non-identity offsets: rel_err={rel_err:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
