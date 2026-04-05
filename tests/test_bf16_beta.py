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

# Tests for bf16 beta support on SM90 (Hopper) and SM100 (Blackwell).
#
# Verifies that bf16 beta produces results close to fp32 beta across:
#   - SM100 modular forward (chunk_kda): fixed-length and varlen, fwd + bwd
#   - SM90 fused forward (kda_prefill_hopper): fixed-length and varlen, fwd only

import pytest
import torch
import torch.nn.functional as F
from fla.utils import assert_close, device


# ============================================================
# SM100 (Blackwell) — chunk_kda, fixed-length, fwd + bwd
# ============================================================

@pytest.mark.sm100_only
@pytest.mark.parametrize(
    ("B", "T", "H", "D", "use_gate_in_kernel"),
    [
        pytest.param(*t, id="B{}-T{}-H{}-D{}-gate{}".format(*t))
        for t in [
            (1, 63, 1, 128, False),
            (2, 500, 3, 128, False),
            (4, 1024, 4, 128, False),
            (4, 1024, 4, 128, True),
            (4, 2048, 8, 128, True),
        ]
    ],
)
def test_bf16_beta_sm100_fixed(B, T, H, D, use_gate_in_kernel):
    """SM100 chunk_kda: bf16 beta vs fp32 beta (fixed-length, fwd+bwd)."""
    from cula.kda import chunk_kda

    torch.manual_seed(42)
    q = torch.rand(B, T, H, D, dtype=torch.bfloat16)
    k = torch.rand(B, T, H, D, dtype=torch.bfloat16)
    v = torch.rand(B, T, H, D, dtype=torch.bfloat16)

    if use_gate_in_kernel:
        g = torch.randn(B, T, H, D, dtype=torch.bfloat16)
        A_log = torch.randn(H, dtype=torch.float32)
        dt_bias = torch.randn(H * D, dtype=torch.float32)
    else:
        g = F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float32)).clamp(-5, 0)
        A_log, dt_bias = None, None

    beta_fp32 = torch.randn(B, T, H, dtype=torch.float32).sigmoid()
    beta_bf16 = beta_fp32.bfloat16()
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)

    if use_gate_in_kernel:
        A_log, dt_bias = map(lambda x: x.to(device).requires_grad_(True), (A_log, dt_bias))
    q, k, v, g, beta_fp32, beta_bf16, h0 = map(
        lambda x: x.to(device).requires_grad_(True), (q, k, v, g, beta_fp32, beta_bf16, h0)
    )

    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    # --- fp32 beta forward + backward ---
    o_fp32, ht_fp32 = chunk_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(), g=g.clone(), beta=beta_fp32.clone(),
        A_log=(A_log.clone() if use_gate_in_kernel else None),
        dt_bias=(dt_bias.clone() if use_gate_in_kernel else None),
        initial_state=h0.clone(), output_final_state=True,
        use_gate_in_kernel=use_gate_in_kernel, safe_gate=True, lower_bound=-5.0,
    )
    ((o_fp32 * do).sum() + (ht_fp32 * dht).sum()).backward()
    if use_gate_in_kernel:
        fp32_dA, A_log.grad = A_log.grad.clone(), None
        fp32_dbias, dt_bias.grad = dt_bias.grad.clone(), None
    fp32_dq, q.grad = q.grad.clone(), None
    fp32_dk, k.grad = k.grad.clone(), None
    fp32_dv, v.grad = v.grad.clone(), None
    fp32_dg, g.grad = g.grad.clone(), None
    fp32_db = beta_fp32.grad.clone()
    beta_fp32.grad = None
    fp32_dh0, h0.grad = h0.grad.clone(), None

    # --- bf16 beta forward + backward ---
    o_bf16, ht_bf16 = chunk_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(), g=g.clone(), beta=beta_bf16.clone(),
        A_log=(A_log.clone() if use_gate_in_kernel else None),
        dt_bias=(dt_bias.clone() if use_gate_in_kernel else None),
        initial_state=h0.clone(), output_final_state=True,
        use_gate_in_kernel=use_gate_in_kernel, safe_gate=True, lower_bound=-5.0,
    )
    ((o_bf16 * do).sum() + (ht_bf16 * dht).sum()).backward()
    bf16_dq, bf16_dk, bf16_dv = q.grad, k.grad, v.grad
    bf16_dg, bf16_dh0 = g.grad, h0.grad
    bf16_db = beta_bf16.grad

    # --- Forward ---
    assert_close("o", o_fp32, o_bf16, 0.005)
    assert_close("ht", ht_fp32, ht_bf16, 0.005)
    # --- Backward ---
    assert_close("dq", fp32_dq, bf16_dq, 0.008)
    assert_close("dk", fp32_dk, bf16_dk, 0.008)
    assert_close("dv", fp32_dv, bf16_dv, 0.008)
    assert_close("dg", fp32_dg, bf16_dg, 0.02)
    assert_close("db", fp32_db, bf16_db, 0.02)
    assert_close("dh0", fp32_dh0, bf16_dh0, 0.008)
    if use_gate_in_kernel:
        assert_close("dA", fp32_dA, A_log.grad, 0.008)
        assert_close("dbias", fp32_dbias, dt_bias.grad, 0.008)


# ============================================================
# SM100 (Blackwell) — chunk_kda, varlen, fwd + bwd
# ============================================================

@pytest.mark.sm100_only
@pytest.mark.parametrize(
    ("H", "D", "cu_seqlens"),
    [
        pytest.param(*t, id="H{}-D{}-cu_seqlens{}".format(*t))
        for t in [
            (4, 128, [0, 15]),
            (4, 128, [0, 256, 500, 1000]),
            (4, 128, [0, 15, 100, 300, 1200, 2000]),
            (4, 128, [0, 100, 300, 1200, 3000, 4096]),
            (32, 128, [0, 247, 699, 982, 1688, 1985, 2383, 3081, 3526, 3973, 4096, 4824, 5101, 5919, 6426, 7137, 7392, 7800, 8192]),
        ]
    ],
)
def test_bf16_beta_sm100_varlen(H, D, cu_seqlens):
    """SM100 chunk_kda: bf16 beta vs fp32 beta (varlen, fwd+bwd)."""
    from cula.kda import chunk_kda

    torch.manual_seed(42)
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    cu_seqlens_cpu = cu_seqlens.cpu()
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    q = torch.randn((1, T, H, D), dtype=torch.bfloat16)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn((1, T, H, D), dtype=torch.bfloat16)
    g = F.logsigmoid(torch.randn(1, T, H, D, dtype=torch.float32)).clamp(-5, 0)

    beta_fp32 = torch.randn(1, T, H, dtype=torch.float32).sigmoid()
    beta_bf16 = beta_fp32.bfloat16()
    h0 = torch.randn((N, H, D, D), dtype=torch.float32)

    q, k, v, g, beta_fp32, beta_bf16, h0 = map(
        lambda x: x.to(device).requires_grad_(), (q, k, v, g, beta_fp32, beta_bf16, h0)
    )
    do = torch.randn_like(v)
    dht = torch.rand_like(h0)

    # --- fp32 beta ---
    o_fp32, ht_fp32 = chunk_kda(
        q=F.normalize(q.clone(), p=2, dim=-1), k=k.clone(), v=v.clone(), g=g.clone(),
        beta=beta_fp32.clone(), initial_state=h0.clone(), output_final_state=True,
        cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu,
        safe_gate=True, lower_bound=-5.0,
    )
    ((o_fp32 * do).sum() + (ht_fp32 * dht).sum()).backward()
    fp32_dq, q.grad = q.grad.clone(), None
    fp32_dk, k.grad = k.grad.clone(), None
    fp32_dv, v.grad = v.grad.clone(), None
    fp32_dg, g.grad = g.grad.clone(), None
    fp32_db = beta_fp32.grad.clone()
    beta_fp32.grad = None
    fp32_dh0, h0.grad = h0.grad.clone(), None

    # --- bf16 beta ---
    o_bf16, ht_bf16 = chunk_kda(
        q=F.normalize(q.clone(), p=2, dim=-1), k=k.clone(), v=v.clone(), g=g.clone(),
        beta=beta_bf16.clone(), initial_state=h0.clone(), output_final_state=True,
        cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu,
        safe_gate=True, lower_bound=-5.0,
    )
    ((o_bf16 * do).sum() + (ht_bf16 * dht).sum()).backward()
    bf16_dq, bf16_dk, bf16_dv = q.grad, k.grad, v.grad
    bf16_dg, bf16_dh0 = g.grad, h0.grad
    bf16_db = beta_bf16.grad

    # --- Forward ---
    assert_close("o", o_fp32, o_bf16, 0.005)
    assert_close("ht", ht_fp32, ht_bf16, 0.005)
    # --- Backward ---
    assert_close("dq", fp32_dq, bf16_dq, 0.007)
    assert_close("dk", fp32_dk, bf16_dk, 0.008)
    assert_close("dv", fp32_dv, bf16_dv, 0.007)
    assert_close("dg", fp32_dg, bf16_dg, 0.015)
    assert_close("db", fp32_db, bf16_db, 0.015)
    assert_close("dh0", fp32_dh0, bf16_dh0, 0.007)


# ============================================================
# SM90 (Hopper) — kda_prefill_hopper, fixed-length, fwd only
# ============================================================

@pytest.mark.sm90_only
@pytest.mark.parametrize(
    ("B", "T", "H", "D", "use_gate_in_kernel"),
    [
        pytest.param(*t, id="B{}-T{}-H{}-D{}-gate{}".format(*t))
        for t in [
            (1, 63, 1, 128, False),
            (2, 500, 3, 128, False),
            (4, 1024, 4, 128, False),
            (2, 1500, 4, 128, True),
            (4, 2048, 8, 128, True),
        ]
    ],
)
def test_bf16_beta_sm90_fixed(B, T, H, D, use_gate_in_kernel):
    """SM90 kda_prefill_hopper: bf16 beta vs fp32 beta (fixed-length, fwd only)."""
    from cula.utils import get_kda_fused_fwd

    cula_kda_fused_fwd = get_kda_fused_fwd(device)

    torch.manual_seed(42)
    q = torch.rand(B, T, H, D, dtype=torch.bfloat16)
    k = torch.rand(B, T, H, D, dtype=torch.bfloat16)
    v = torch.rand(B, T, H, D, dtype=torch.bfloat16)

    if use_gate_in_kernel:
        g = torch.randn(B, T, H, D, dtype=torch.bfloat16)
        A_log = torch.randn(H, dtype=torch.float32).to(device)
        dt_bias = torch.randn(H * D, dtype=torch.float32).to(device)
    else:
        g = F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float32)).clamp(-5, 0)
        A_log, dt_bias = None, None

    beta_fp32 = torch.randn(B, T, H, dtype=torch.float32).sigmoid()
    beta_bf16 = beta_fp32.bfloat16()
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)

    q, k, v, g, beta_fp32, beta_bf16, h0 = map(lambda x: x.to(device), (q, k, v, g, beta_fp32, beta_bf16, h0))

    kwargs = {}
    if use_gate_in_kernel:
        kwargs = dict(A_log=A_log, dt_bias=dt_bias)

    o_fp32, ht_fp32 = cula_kda_fused_fwd(
        q=F.normalize(q, p=2, dim=-1), k=F.normalize(k, p=2, dim=-1),
        v=v, g=g, beta=beta_fp32,
        initial_state=h0, output_final_state=True,
        use_gate_in_kernel=use_gate_in_kernel, safe_gate=True, lower_bound=-5.0,
        **kwargs,
    )

    o_bf16, ht_bf16 = cula_kda_fused_fwd(
        q=F.normalize(q, p=2, dim=-1), k=F.normalize(k, p=2, dim=-1),
        v=v, g=g, beta=beta_bf16,
        initial_state=h0, output_final_state=True,
        use_gate_in_kernel=use_gate_in_kernel, safe_gate=True, lower_bound=-5.0,
        **kwargs,
    )

    assert_close("o", o_fp32, o_bf16, 0.005)
    assert_close("ht", ht_fp32, ht_bf16, 0.005)


# ============================================================
# SM90 (Hopper) — kda_prefill_hopper, varlen, fwd only
# ============================================================

@pytest.mark.sm90_only
@pytest.mark.parametrize(
    ("H", "D", "cu_seqlens"),
    [
        pytest.param(*t, id="H{}-D{}-cu_seqlens{}".format(*t))
        for t in [
            (4, 128, [0, 15]),
            (4, 128, [0, 256, 500, 1000]),
            (4, 128, [0, 15, 100, 300, 1200, 2000]),
            (4, 128, [0, 100, 300, 1200, 3000, 4096]),
            (32, 128, [0, 247, 699, 982, 1688, 1985, 2383, 3081, 3526, 3973, 4096, 4824, 5101, 5919, 6426, 7137, 7392, 7800, 8192]),
            (32, 128, [0, 494, 1004, 1561, 1908, 2240, 2849, 3116, 4096, 4986, 5626, 6090, 6718, 7244, 7870, 8192]),
        ]
    ],
)
def test_bf16_beta_sm90_varlen(H, D, cu_seqlens):
    """SM90 kda_prefill_hopper: bf16 beta vs fp32 beta (varlen, fwd only)."""
    from cula.utils import get_kda_fused_fwd

    cula_kda_fused_fwd = get_kda_fused_fwd(device)

    torch.manual_seed(42)
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    cu_seqlens_cpu = cu_seqlens.cpu()
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    q = torch.randn((1, T, H, D), dtype=torch.bfloat16)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn((1, T, H, D), dtype=torch.bfloat16)
    g = F.logsigmoid(torch.randn(1, T, H, D, dtype=torch.float32)).clamp(-5, 0)

    beta_fp32 = torch.randn(1, T, H, dtype=torch.float32).sigmoid()
    beta_bf16 = beta_fp32.bfloat16()
    h0 = torch.randn((N, H, D, D), dtype=torch.float32)

    q, k, v, g, beta_fp32, beta_bf16, h0 = map(lambda x: x.to(device), (q, k, v, g, beta_fp32, beta_bf16, h0))

    o_fp32, ht_fp32 = cula_kda_fused_fwd(
        q=F.normalize(q, p=2, dim=-1), k=k, v=v, g=g, beta=beta_fp32,
        initial_state=h0, output_final_state=True,
        safe_gate=True, lower_bound=-5.0,
        cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu,
    )

    o_bf16, ht_bf16 = cula_kda_fused_fwd(
        q=F.normalize(q, p=2, dim=-1), k=k, v=v, g=g, beta=beta_bf16,
        initial_state=h0, output_final_state=True,
        safe_gate=True, lower_bound=-5.0,
        cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu,
    )

    assert_close("o", o_fp32, o_bf16, 0.005)
    assert_close("ht", ht_fp32, ht_bf16, 0.005)
