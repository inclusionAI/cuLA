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

# Adapted from flash-linear-attention: https://github.com/fla-org/flash-linear-attention/blob/main/tests/ops/test_kda.py


import pytest
import torch
import torch.nn.functional as F
from fla.ops import chunk_kda as fla_chunk_kda
from fla.ops.kda.gate import naive_kda_gate
from fla.ops.kda.naive import naive_recurrent_kda
from fla.utils import assert_close, device

from cula.utils import get_kda_fused_fwd

pytestmark = pytest.mark.sm90_only


@pytest.mark.parametrize("beta_dtype", [torch.float32, torch.bfloat16], ids=["beta_fp32", "beta_bf16"])
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
    beta_dtype: torch.dtype,
):
    from fla.ops.kda.gate import naive_kda_lowerbound_gate

    cula_kda_fused_fwd = get_kda_fused_fwd(device)

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

    beta = torch.randn(B, T, H, dtype=torch.float32).sigmoid().to(beta_dtype)
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    # NOTE: for inference scenarios, we only use transposed state layout for better decoding performance
    h0_vk = h0.transpose(-1, -2).contiguous()
    if use_gate_in_kernel:
        A_log, dt_bias = map(lambda x: x.to(device).requires_grad_(False), (A_log, dt_bias))
    q, k, v, g, beta, h0, h0_vk = map(lambda x: x.to(device).requires_grad_(False), (q, k, v, g, beta, h0, h0_vk))

    ref, ref_ht = naive_recurrent_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=(naive_kda_gate_fn(g, A_log, dt_bias) if use_gate_in_kernel else g.clone()),
        beta=beta.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
    )

    ref_fla, ref_ht_fla = fla_chunk_kda(
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
    )

    ref_fla_trans, ref_ht_fla_trans = fla_chunk_kda(
        q=F.normalize(q.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q.clone(),
        k=F.normalize(k.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        A_log=(A_log.clone() if use_gate_in_kernel else None),
        dt_bias=(dt_bias.clone() if use_gate_in_kernel else None),
        initial_state=h0_vk.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        transpose_state_layout=True,
    )

    tri, tri_ht = cula_kda_fused_fwd(
        q=F.normalize(q.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q.clone(),
        k=F.normalize(k.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        A_log=(A_log.clone() if use_gate_in_kernel else None),
        dt_bias=(dt_bias.clone() if use_gate_in_kernel else None),
        initial_state=h0_vk.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
    )

    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht.transpose(-1, -2), 0.005)
    assert_close("o", ref_fla, tri, 0.005)
    assert_close("ht", ref_ht_fla, tri_ht.transpose(-1, -2), 0.005)
    assert_close("o", ref_fla_trans, tri, 0.005)
    assert_close("ht", ref_ht_fla_trans, tri_ht, 0.005)


# ─── Multi-Value GQA tests ─────────────────────────────────────────────
#
# KDA's "multi-value" flavor: Q and K share a single physical head count
# (num_k_heads) while V (and state, alpha/g, beta, O) use the larger
# num_heads. `num_heads / num_k_heads` consecutive state heads share one
# physical Q/K head. This matches e.g. Qwen3.5-A3B Gated DeltaNet
# (linear_num_key_heads=16, linear_num_value_heads=32).
#
# Equivalence check: running the GQA kernel on (Q=[T,num_k_heads,D],
# K=[T,num_k_heads,D], V=[T,num_heads,D]) must produce the same output as
# running the existing MHA kernel on Q and K expanded via repeat_interleave
# to [T,num_heads,D]. The rest of the setup (g, beta, state) is identical
# in both paths — they are already per-state-head.


@pytest.mark.parametrize(
    ("B", "T", "num_heads", "num_k_heads", "D", "use_qk_l2norm_in_kernel", "dtype"),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-Hk{}-D{}-l2norm{}-{}".format(*test),
        )
        for test in [
            # Qwen3.5-A3B DeltaNet shape
            (1, 128, 32, 16, 128, False, torch.bfloat16),
            (2, 256, 32, 16, 128, False, torch.bfloat16),
            (2, 1024, 32, 16, 128, True, torch.bfloat16),
            (1, 2048, 32, 16, 128, False, torch.bfloat16),
            # Other plausible GQA ratios
            (2, 512, 16, 8, 128, False, torch.bfloat16),
            (2, 512, 32, 8, 128, False, torch.bfloat16),  # k_group_size=4
        ]
    ],
)
def test_multi_value_gqa(
    B: int,
    T: int,
    num_heads: int,
    num_k_heads: int,
    D: int,
    use_qk_l2norm_in_kernel: bool,
    dtype: torch.dtype,
):
    """GQA output must match MHA output on repeat_interleave'd Q/K."""
    assert num_heads % num_k_heads == 0, "num_heads must be a multiple of num_k_heads"
    k_group_size = num_heads // num_k_heads

    cula_kda_fused_fwd = get_kda_fused_fwd(device)

    torch.manual_seed(42)
    # GQA-shaped inputs
    q_gqa = torch.rand(B, T, num_k_heads, D, dtype=dtype)
    k_gqa = torch.rand(B, T, num_k_heads, D, dtype=dtype)
    v = torch.rand(B, T, num_heads, D, dtype=dtype)
    # g and beta are per state/V head in both paths
    g = F.logsigmoid(torch.randn(B, T, num_heads, D, dtype=torch.float)).clamp(-5, 0)
    beta = torch.randn(B, T, num_heads, dtype=torch.float32).sigmoid().to(torch.float32)
    h0 = torch.randn(B, num_heads, D, D, dtype=torch.float32)
    h0_vk = h0.transpose(-1, -2).contiguous()

    q_gqa, k_gqa, v, g, beta, h0_vk = map(
        lambda x: x.to(device).requires_grad_(False),
        (q_gqa, k_gqa, v, g, beta, h0_vk),
    )

    # MHA reference: repeat Q and K heads to num_heads so they match V. The
    # MHA kernel then runs the same math on a strictly-larger tensor and the
    # result per state head should be bit-equivalent to the GQA path.
    q_mha_ref = q_gqa.repeat_interleave(k_group_size, dim=2)
    k_mha_ref = k_gqa.repeat_interleave(k_group_size, dim=2)

    def call_fused(q, k):
        q_in = F.normalize(q.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q.clone()
        k_in = F.normalize(k.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k.clone()
        return cula_kda_fused_fwd(
            q=q_in,
            k=k_in,
            v=v.clone(),
            g=g.clone(),
            beta=beta.clone(),
            initial_state=h0_vk.clone(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=False,
            safe_gate=True,
            lower_bound=-5.0,
        )

    o_gqa, ht_gqa = call_fused(q_gqa, k_gqa)
    o_mha, ht_mha = call_fused(q_mha_ref, k_mha_ref)

    assert_close("o (gqa vs mha-repeat)", o_mha, o_gqa, 0.005)
    assert_close("ht (gqa vs mha-repeat)", ht_mha, ht_gqa, 0.005)


@pytest.mark.parametrize("beta_dtype", [torch.float32, torch.bfloat16], ids=["beta_fp32", "beta_bf16"])
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
    beta_dtype: torch.dtype,
):
    cula_kda_fused_fwd = get_kda_fused_fwd(device)

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

    beta = torch.randn(1, T, H, dtype=torch.float32).sigmoid().to(beta_dtype)
    h0 = torch.randn((N, H, D, D), dtype=torch.float32)
    # NOTE: for inference scenarios, we only use transposed state layout for better decoding performance
    h0_vk = h0.transpose(-1, -2).contiguous()

    q, k, v, g, beta, h0, h0_vk = map(lambda x: x.to(device).requires_grad_(False), (q, k, v, g, beta, h0, h0_vk))

    tri, tri_ht = cula_kda_fused_fwd(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        initial_state=h0_vk.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        safe_gate=safe_gate,
        lower_bound=-5.0 if safe_gate else None,
    )

    ref_fla, ref_ht_fla = fla_chunk_kda(
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
    )

    ref_fla_trans, ref_ht_fla_trans = fla_chunk_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        initial_state=h0_vk.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        safe_gate=safe_gate,
        lower_bound=-5.0 if safe_gate else None,
        transpose_state_layout=True,
    )

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

    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht.transpose(-1, -2), 0.005)
    assert_close("o", ref_fla, tri, 0.005)
    assert_close("ht", ref_ht_fla, tri_ht.transpose(-1, -2), 0.005)
    assert_close("o", ref_fla_trans, tri, 0.005)
    assert_close("ht", ref_ht_fla_trans, tri_ht, 0.005)
