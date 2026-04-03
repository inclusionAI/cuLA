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

# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# Tests for the chunk-decomposed KDA implementation (python/kda/chunk.py)

import pytest
import torch
import torch.nn.functional as F
from fla.ops.kda.gate import naive_kda_gate, naive_kda_lowerbound_gate
from fla.ops.kda.naive import naive_recurrent_kda
from fla.utils import assert_close, device

from cula.kda import chunk_kda

pytestmark = pytest.mark.sm100_only


@pytest.mark.parametrize("disable_recompute", [True, False], ids=["no_recomp", "recomp"])
@pytest.mark.parametrize(
    (
        "B",
        "T",
        "H",
        "D",
        "gate_logit_normalizer",
        "mask_p",
        "use_qk_l2norm_in_kernel",
        "use_gate_in_kernel",
        "safe_gate",
        "dtype",
    ),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-gln{}-mask_p{}-l2norm{}-gate{}-safe_gate{}-{}".format(*test),
        )
        for test in [
            (1, 63, 1, 128, 1, 0, False, False, True, torch.bfloat16),
            (2, 500, 3, 128, 1, 0, False, False, True, torch.bfloat16),
            (2, 1000, 3, 128, 1, 0.5, False, False, True, torch.bfloat16),
            (3, 1024, 4, 128, 0.1, 0, False, False, True, torch.bfloat16),
            (4, 1024, 4, 128, 1, 0, False, False, True, torch.bfloat16),
            (4, 1024, 4, 128, 1, 0, True, False, True, torch.bfloat16),
            (2, 1500, 4, 128, 10, 0, False, True, True, torch.bfloat16),
            (4, 2048, 8, 128, 1, 0, False, True, True, torch.bfloat16),
        ]
    ],
)
def test_safe_gate_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    gate_logit_normalizer: float,
    mask_p: float,
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    safe_gate: bool,
    dtype: torch.dtype,
    disable_recompute: bool,
):
    torch.manual_seed(42)
    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    g = torch.randn(B, T, H, D, dtype=torch.float if not use_gate_in_kernel else dtype)
    if use_gate_in_kernel:
        A_log = torch.randn(H, dtype=torch.float)
        dt_bias = torch.randn(H * D, dtype=torch.float)
    else:
        g = F.logsigmoid(g) / gate_logit_normalizer
        g = g * (torch.rand_like(g) > mask_p)
    if safe_gate:
        lower_bound = -5.0
        if not use_gate_in_kernel:
            g = g.clamp(-5, 0)
        naive_kda_gate_fn = naive_kda_lowerbound_gate
    else:
        lower_bound = None
        naive_kda_gate_fn = naive_kda_gate

    beta = torch.randn(B, T, H, dtype=torch.float32).sigmoid()
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    if use_gate_in_kernel:
        A_log, dt_bias = map(lambda x: x.to(device).requires_grad_(True), (A_log, dt_bias))
    q, k, v, g, beta, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, g, beta, h0))

    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    ref, ref_ht = naive_recurrent_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=(naive_kda_gate_fn(g, A_log, dt_bias) if use_gate_in_kernel else g.clone()),
        beta=beta.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    if use_gate_in_kernel:
        ref_dA, A_log.grad = A_log.grad, None
        ref_dbias, dt_bias.grad = dt_bias.grad, None
    ref_dq, ref_dk, ref_dv, ref_dg, ref_db, ref_dh0 = q.grad, k.grad, v.grad, g.grad, beta.grad, h0.grad
    q.grad = k.grad = v.grad = g.grad = beta.grad = h0.grad = None

    tri, tri_ht = chunk_kda(
        q=F.normalize(q.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q.clone(),
        k=F.normalize(k.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        A_log=(A_log.clone() if use_gate_in_kernel else None),
        dt_bias=(dt_bias.clone() if use_gate_in_kernel else None),
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        disable_recompute=disable_recompute,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    if use_gate_in_kernel:
        tri_dA, A_log.grad = A_log.grad, None
        tri_dbias, dt_bias.grad = dt_bias.grad, None
    tri_dq, tri_dk, tri_dv, tri_dg, tri_db, tri_dh0 = q.grad, k.grad, v.grad, g.grad, beta.grad, h0.grad
    q.grad = k.grad = v.grad = g.grad = beta.grad = h0.grad = None

    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.008)
    assert_close("dk", ref_dk, tri_dk, 0.008)
    assert_close("dv", ref_dv, tri_dv, 0.008)
    assert_close("dg", ref_dg, tri_dg, 0.02)
    assert_close("db", ref_db, tri_db, 0.02)
    if use_gate_in_kernel:
        assert_close("dA", ref_dA, tri_dA, 0.003, warning=True)
        assert_close("dbias", ref_dbias, tri_dbias, 0.008)
    assert_close("dh0", ref_dh0, tri_dh0, 0.008)


@pytest.mark.parametrize("disable_recompute", [True, False], ids=["no_recomp", "recomp"])
@pytest.mark.parametrize(
    ("H", "D", "mask_p", "cu_seqlens", "dtype", "safe_gate"),
    [
        pytest.param(*test, id="H{}-D{}-mask_p{}-cu_seqlens{}-{}-safe_gate{}".format(*test))
        for test in [
            (4, 128, 0.1, [0, 15], torch.bfloat16, True),
            (4, 128, 0.9, [0, 256, 500, 1000], torch.bfloat16, True),
            (4, 128, 0.5, [0, 256, 500, 1000], torch.bfloat16, True),
            (4, 128, 0, [0, 15, 100, 300, 1200, 2000], torch.bfloat16, True),
            (4, 128, 0, [0, 100, 300, 1200, 3000, 4096], torch.bfloat16, True),
            # ======Varlen test with simulated trace=======
            (
                32,
                128,
                0,
                [0, 247, 699, 982, 1688, 1985, 2383, 3081, 3526, 3973, 4096, 4824, 5101, 5919, 6426, 7137, 7392, 7800, 8192],
                torch.bfloat16,
                True,
            ),
            (
                32,
                128,
                0,
                [0, 652, 1255, 1600, 2083, 2345, 2756, 3172, 3767, 4096, 4891, 5236, 5543, 6255, 6480, 6947, 7616, 8192],
                torch.bfloat16,
                True,
            ),
            (
                32,
                128,
                0,
                [0, 315, 973, 1283, 2162, 2459, 2678, 2998, 3781, 4096, 4503, 5459, 6318, 6669, 6979, 7583, 8192],
                torch.bfloat16,
                True,
            ),
            (
                32,
                128,
                0,
                [0, 494, 1004, 1561, 1908, 2240, 2849, 3116, 4096, 4986, 5626, 6090, 6718, 7244, 7870, 8192],
                torch.bfloat16,
                True,
            ),
        ]
    ],
)
def test_safe_gate_chunk_varlen(
    H: int,
    D: int,
    mask_p: float,
    cu_seqlens: list[int],
    dtype: torch.dtype,
    safe_gate: bool,
    disable_recompute: bool,
):
    torch.manual_seed(42)
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    cu_seqlens_cpu = cu_seqlens.cpu()
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    q = torch.randn((1, T, H, D), dtype=dtype)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn((1, T, H, D), dtype=dtype)
    g = F.logsigmoid(torch.randn(1, T, H, D, dtype=torch.float))
    mask = torch.rand_like(g) > mask_p
    g = g * mask + (~mask) * (-1000)
    if safe_gate:
        g = g.clamp(-5, 0)

    beta = torch.randn(1, T, H, dtype=torch.float32).sigmoid()
    h0 = torch.randn((N, H, D, D), dtype=torch.float32)

    q, k, v, g, beta, h0 = map(lambda x: x.to(device).requires_grad_(), (q, k, v, g, beta, h0))
    do = torch.randn_like(v)
    dht = torch.rand_like(h0)

    tri, tri_ht = chunk_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        safe_gate=safe_gate,
        lower_bound=-5.0 if safe_gate else None,
        disable_recompute=disable_recompute,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dg, tri_db, tri_dh0 = q.grad, k.grad, v.grad, g.grad, beta.grad, h0.grad
    q.grad = k.grad = v.grad = g.grad = beta.grad = h0.grad = None

    ref = []
    ref_ht = []
    for i in range(N):
        ref_i, ref_ht_i = naive_recurrent_kda(
            q=F.normalize(q[:, cu_seqlens[i] : cu_seqlens[i + 1]], p=2, dim=-1),
            k=k[:, cu_seqlens[i] : cu_seqlens[i + 1]],
            v=v[:, cu_seqlens[i] : cu_seqlens[i + 1]],
            beta=beta[:, cu_seqlens[i] : cu_seqlens[i + 1]],
            g=g[:, cu_seqlens[i] : cu_seqlens[i + 1]],
            initial_state=h0[i],
            output_final_state=True,
        )
        ref.append(ref_i)
        ref_ht.append(ref_ht_i)
    ref = torch.cat(ref, 1)
    ref_ht = torch.cat(ref_ht, 0)

    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dg, ref_db, ref_dh0 = q.grad, k.grad, v.grad, g.grad, beta.grad, h0.grad

    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.007)
    assert_close("dk", ref_dk, tri_dk, 0.008)
    assert_close("dv", ref_dv, tri_dv, 0.007)
    assert_close("dg", ref_dg, tri_dg, 0.015)
    assert_close("db", ref_db, tri_db, 0.015)
    assert_close("dh0", ref_dh0, tri_dh0, 0.007)
