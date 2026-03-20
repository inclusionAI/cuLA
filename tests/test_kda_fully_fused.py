# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# Adapted from flash-linear-attention: https://github.com/fla-org/flash-linear-attention/blob/main/tests/ops/test_kda.py

import os

import pytest
import torch
import torch.nn.functional as F

from fla.ops.kda import chunk_kda, fused_recurrent_kda
from fla.ops.kda.gate import fused_kda_gate, naive_kda_gate
from fla.ops.kda.naive import naive_chunk_kda, naive_recurrent_kda
from fla.utils import IS_INTEL_ALCHEMIST, assert_close, device

from cula.kda.fully_fused import flash_kda_prefill

# ---------------------------------- Tests for FlashKDA Impl ----------------------------------

# FIXME: only test in dtype=torch.bfloat16, in float16, exp(-alpha) * K will overflow to nan
# NOTE: only test in head_size=128, since our kernel is only implemented for this size
# FIXME: smaller gate_logit_normalizer (e.g. 0.1) will cause exp(-alpha) * K overflow to nan
# FIXME: use_gate_in_kernel=True will cause nan sometimes ???

@pytest.mark.parametrize(
    (
        "B",
        "T",
        "H",
        "D",
        "scale",
        "gate_logit_normalizer",
        "mask_p",
        "use_qk_l2norm_in_kernel",
        "use_gate_in_kernel",
        "dtype",
        "safe_gate",
    ),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-mask_p{}-qk_l2norm{}-gate{}-dtype{}-safe_gate{}".format(
            *test),
        )
        for test in [
            # (1, 63, 1, 128, 1, 1, 0, False, False, torch.bfloat16, False),
            # (2, 500, 3, 128, 1, 1, 0, False, False, torch.bfloat16, False),
            # (2, 1000, 3, 128, 0.1, 1, 0.5, False, False, torch.bfloat16, False),
            # (3, 1024, 4, 128, 1, 0.1, 0, False, False, torch.bfloat16, False),
            # (4, 1024, 4, 128, 0.1, 1, 0, False, False, torch.bfloat16, False),
            # (4, 1024, 4, 128, 0.1, 1, 0, True, False, torch.bfloat16, False),
            # (2, 1500, 4, 128, 0.1, 10, 0, False, True, torch.bfloat16, False),
            # (4, 2048, 8, 128, 0.1, 1, 0, False, True, torch.bfloat16, False),
            # ======Safe gate, all passed=======
            # (1, 63, 1, 128, 1, 1, 0, False, False, torch.bfloat16, True),
            # (2, 500, 3, 128, 1, 1, 0, False, False, torch.bfloat16, True),
            # (2, 1000, 3, 128, 0.1, 1, 0.5, False, False, torch.bfloat16, True),
            (3, 1024, 4, 128, 1, 0.1, 0, False, False, torch.bfloat16, True),
            (4, 1024, 4, 128, 0.1, 1, 0, False, False, torch.bfloat16, True),
            (4, 1024, 4, 128, 0.1, 1, 0, True, False, torch.bfloat16, True),
            # (2, 1500, 4, 128, 0.1, 10, 0, False, True, torch.bfloat16, True),
            (4, 2048, 8, 128, 0.1, 1, 0, False, True, torch.bfloat16, True),
        ]
    ],
)
def test_safe_gate_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    mask_p: float,
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    dtype: torch.dtype,
    safe_gate: bool,
):
    try:
        from fla.ops.kda.gate import naive_kda_lowerbound_gate
    except Exception:
        raise ImportError("Please install flash-linear-attention after this commit "
            "https://github.com/fla-org/flash-linear-attention/tree/d1097c609b23b5f478f490da0fbd00060b0e9dc3")

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

    # NOTE: in our Megatron-LM's kda.py, beta is converted to float
    beta = torch.randn(B, T, H, dtype=torch.float32).sigmoid()
    # beta = torch.randn(B, T, H, dtype=dtype).sigmoid()
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
        scale=scale,
        initial_state=None,
        output_final_state=True,
    )
    # ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    # if use_gate_in_kernel:
    #     ref_dA, A_log.grad = A_log.grad, None
    #     ref_dbias, dt_bias.grad = dt_bias.grad, None
    # ref_dq, ref_dk, ref_dv, ref_dg, ref_db, ref_dh0 = q.grad, k.grad, v.grad, g.grad, beta.grad, h0.grad
    # q.grad = k.grad = v.grad = g.grad = beta.grad = h0.grad = None

    tri, tri_ht = flash_kda_prefill(
        q=F.normalize(q.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q.clone(),
        k=F.normalize(k.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        A_log=(A_log.clone() if use_gate_in_kernel else None),
        dt_bias=(dt_bias.clone() if use_gate_in_kernel else None),
        scale=scale,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
    )
    # ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    # if use_gate_in_kernel:
    #     tri_dA, A_log.grad = A_log.grad, None
    #     tri_dbias, dt_bias.grad = dt_bias.grad, None
    # tri_dq, tri_dk, tri_dv, tri_dg, tri_db, tri_dh0 = q.grad, k.grad, v.grad, g.grad, beta.grad, h0.grad
    # q.grad = k.grad = v.grad = g.grad = beta.grad = h0.grad = None

    assert_close("o", ref, tri, 0.005)
    # FIXME: not support return state currently
    # assert_close("ht", ref_ht, tri_ht, 0.005)
    # assert_close("dq", ref_dq, tri_dq, 0.008)
    # assert_close("dk", ref_dk, tri_dk, 0.008)
    # assert_close("dv", ref_dv, tri_dv, 0.008)
    # assert_close("dg", ref_dg, tri_dg, 0.02)
    # assert_close("db", ref_db, tri_db, 0.02)
    # if use_gate_in_kernel:
    #     assert_close("dA", ref_dA, tri_dA, 0.003, warning=True)
    #     assert_close("dbias", ref_dbias, tri_dbias, 0.008)
    # assert_close("dh0", ref_dh0, tri_dh0, 0.008)

@pytest.mark.parametrize(
    ("H", "D", "mask_p", "cu_seqlens", "dtype", "use_gate_in_kernel", "safe_gate"),
    [
        pytest.param(*test, id="H{}-D{}-mask_p{}-cu_seqlens{}-{}-gate{}-safe_gate{}".format(*test))
        for test in [
            # (4, 128, 0.1, [0, 15], torch.bfloat16, True, False),
            # (4, 128, 0.9, [0, 256, 500, 1000], torch.bfloat16, True, False),
            # (4, 128, 0.5, [0, 256, 500, 1000], torch.bfloat16, False, False),
            # (4, 128, 0, [0, 15, 100, 300, 1200, 2000], torch.bfloat16, True, False),
            # (4, 128, 0, [0, 100, 300, 1200, 3000, 4096], torch.bfloat16, False, False),
            # ======Safe gate, all passed=======
            (4, 128, 0.1, [0, 15], torch.bfloat16, False, True),
            (4, 128, 0.9, [0, 256, 500, 1000], torch.bfloat16, False, True),
            (4, 128, 0.5, [0, 256, 500, 1000], torch.bfloat16, False, True),
            (4, 128, 0, [0, 15, 100, 300, 1200, 2000], torch.bfloat16, False, True),
            (4, 128, 0, [0, 100, 300, 1200, 3000, 4096], torch.bfloat16, False, True),
        ]
    ],
)
def test_safe_gate_chunk_varlen(
    H: int,
    D: int,
    mask_p: float,
    cu_seqlens: list[int],
    dtype: torch.dtype,
    use_gate_in_kernel: bool,
    safe_gate: bool,
):
    try:
        from fla.ops.kda.gate import naive_kda_lowerbound_gate
    except Exception:
      raise ImportError("Please install flash-linear-attention after this commit " \
      "https://github.com/fla-org/flash-linear-attention/tree/d1097c609b23b5f478f490da0fbd00060b0e9dc3")

    torch.manual_seed(42)
    # randomly split the sequence into N segments
    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    cu_seqlens_cpu = cu_seqlens.cpu()
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, H, D), dtype=dtype)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn((1, T, H, D), dtype=dtype)
    g = torch.randn(1, T, H, D, dtype=torch.float if not use_gate_in_kernel else dtype)
    if use_gate_in_kernel:
        A_log = torch.log(torch.randn(1, 1, H, 1, dtype=torch.float32, device=device).uniform_(1, 16))
        dt_bias = torch.randn(H * D, dtype=torch.float32, device=device)
    else:
        g = F.logsigmoid(g)
        g = g * (torch.rand_like(g) > mask_p)
    mask = torch.rand_like(g) > mask_p
    g = g * mask + (~mask) * (-1000)
    if safe_gate:
        assert use_gate_in_kernel is False
        g = g.clamp(-5, 0)
    # NOTE: in our Megatron-LM's kda.py, beta is converted to float
    beta = torch.randn(1, T, H, dtype=torch.float32).sigmoid()
    h0 = torch.randn((N, H, D, D), dtype=torch.float32)

    q, k, v, g, beta, h0 = map(lambda x: x.to(device).requires_grad_(), (q, k, v, g, beta, h0))
    if use_gate_in_kernel:
        A_log, dt_bias = map(lambda x: x.to(device).requires_grad_(), (A_log, dt_bias))
    do = torch.randn_like(v)
    dht = torch.rand_like(h0)

    tri, tri_ht = flash_kda_prefill(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=k.clone(),  # k is already normalized
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        A_log=(A_log.clone() if use_gate_in_kernel else None),
        dt_bias=(dt_bias.clone() if use_gate_in_kernel else None),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
    )
    # ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    # tri_dq, tri_dk, tri_dv, tri_dg, tri_db, tri_dh0 = q.grad, k.grad, v.grad, g.grad, beta.grad, h0.grad
    # q.grad = k.grad = v.grad = g.grad = beta.grad = h0.grad = None
    # if use_gate_in_kernel:
    #     tri_dA, A_log.grad = A_log.grad, None
    #     tri_dbias, dt_bias.grad = dt_bias.grad, None

    ref = []
    ref_ht = []
    for i in range(N):
        ref_i, ref_ht_i = naive_recurrent_kda(
            q=F.normalize(q[:, cu_seqlens[i]: cu_seqlens[i + 1]], p=2, dim=-1),
            k=k[:, cu_seqlens[i]: cu_seqlens[i + 1]],  # k is already normalized
            v=v[:, cu_seqlens[i]: cu_seqlens[i + 1]],
            beta=beta[:, cu_seqlens[i]: cu_seqlens[i + 1]],
            g=(naive_kda_gate(g[:, cu_seqlens[i]: cu_seqlens[i + 1]].to(torch.float), A_log.to(torch.float),
               dt_bias.to(torch.float)) if use_gate_in_kernel else g[:, cu_seqlens[i]: cu_seqlens[i + 1]]),
            initial_state=h0[i],
            output_final_state=True,
        )
        ref.append(ref_i)
        ref_ht.append(ref_ht_i)
    ref = torch.cat(ref, 1)
    ref_ht = torch.cat(ref_ht, 0)

    # ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    # ref_dq, ref_dk, ref_dv, ref_dg, ref_db, ref_dh0 = q.grad, k.grad, v.grad, g.grad, beta.grad, h0.grad
    # if use_gate_in_kernel:
    #     ref_dA, A_log.grad = A_log.grad, None
    #     ref_dbias, dt_bias.grad = dt_bias.grad, None
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)
    # assert_close("dq", ref_dq, tri_dq, 0.007)
    # assert_close("dk", ref_dk, tri_dk, 0.008)
    # assert_close("dv", ref_dv, tri_dv, 0.007)
    # assert_close("dg", ref_dg, tri_dg, 0.015)
    # assert_close("db", ref_db, tri_db, 0.015)
    # assert_close("dh0", ref_dh0, tri_dh0, 0.007)
    # if use_gate_in_kernel:
    #     assert_close("dA", ref_dA, tri_dA, 0.008, warning=True)
    #     assert_close("dbias", ref_dbias, tri_dbias, 0.005)


# ---------------------------------- Tests for Initial State & Output Final State ----------------------------------

@pytest.mark.parametrize(
    ("B", "T", "H", "D", "scale", "dtype", "safe_gate"),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-dtype{}-safe_gate{}".format(*test),
        )
        for test in [
            (1, 128, 1, 128, 1.0, torch.bfloat16, True),
            (2, 256, 2, 128, 0.1, torch.bfloat16, True),
            (3, 1024, 4, 128, 1.0, torch.bfloat16, True),
            (4, 1024, 4, 128, 0.1, torch.bfloat16, True),
        ]
    ],
)
def test_safe_gate_chunk_with_initial_state(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    dtype: torch.dtype,
    safe_gate: bool,
):
    """Test KDA kernel with initial_state provided and output_final_state=True."""
    try:
        from fla.ops.kda.naive import naive_recurrent_kda as _check
    except Exception:
        raise ImportError("Please install flash-linear-attention after this commit "
            "https://github.com/fla-org/flash-linear-attention/tree/d1097c609b23b5f478f490da0fbd00060b0e9dc3")

    torch.manual_seed(42)
    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    g = torch.randn(B, T, H, D, dtype=torch.float)
    g = F.logsigmoid(g)
    if safe_gate:
        g = g.clamp(-5, 0)

    beta = torch.randn(B, T, H, dtype=torch.float32).sigmoid()
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)

    q, k, v, g, beta, h0 = map(lambda x: x.to(device), (q, k, v, g, beta, h0))

    # Reference: naive recurrent with initial state
    ref, ref_ht = naive_recurrent_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )

    # Test: flash_kda_prefill with initial state
    tri, tri_ht = flash_kda_prefill(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        safe_gate=safe_gate,
    )

    assert_close("o", ref, tri, 0.005)
    assert tri_ht is not None, "output_final_state=True but got None"
    assert_close("ht", ref_ht, tri_ht, 0.005)


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "scale", "dtype", "safe_gate"),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-dtype{}-safe_gate{}".format(*test),
        )
        for test in [
            (2, 256, 2, 128, 0.1, torch.bfloat16, True),
            (4, 1024, 4, 128, 0.1, torch.bfloat16, True),
        ]
    ],
)
def test_safe_gate_chunk_output_final_state_no_initial(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    dtype: torch.dtype,
    safe_gate: bool,
):
    """Test KDA kernel with output_final_state=True but no initial_state (initial_state=None)."""
    try:
        from fla.ops.kda.naive import naive_recurrent_kda as _check
    except Exception:
        raise ImportError("Please install flash-linear-attention after this commit "
            "https://github.com/fla-org/flash-linear-attention/tree/d1097c609b23b5f478f490da0fbd00060b0e9dc3")

    torch.manual_seed(42)
    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    g = torch.randn(B, T, H, D, dtype=torch.float)
    g = F.logsigmoid(g)
    if safe_gate:
        g = g.clamp(-5, 0)

    beta = torch.randn(B, T, H, dtype=torch.float32).sigmoid()

    q, k, v, g, beta = map(lambda x: x.to(device), (q, k, v, g, beta))

    # Reference: naive recurrent without initial state
    ref, ref_ht = naive_recurrent_kda(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=None,
        output_final_state=True,
    )

    # Test: flash_kda_prefill with output_final_state but no initial_state
    tri, tri_ht = flash_kda_prefill(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=None,
        output_final_state=True,
        safe_gate=safe_gate,
    )

    assert_close("o", ref, tri, 0.005)
    assert tri_ht is not None, "output_final_state=True but got None"
    assert_close("ht", ref_ht, tri_ht, 0.005)


# ---------------------------------- Tests for Varlen ----------------------------------

@pytest.mark.parametrize(
    ("seq_lens", "H", "D", "scale", "dtype", "safe_gate", "has_initial_state", "output_final_state"),
    [
        pytest.param(
            *test,
            id="seqs{}-H{}-D{}-scale{}-dtype{}-safe_gate{}-h0{}-ht{}".format(*test),
        )
        for test in [
            # equal lengths
            ([128, 128], 2, 128, 0.1, torch.bfloat16, True, False, False),
            ([256, 256], 4, 128, 0.1, torch.bfloat16, True, True, True),
            # 2 seqs, different lengths
            ([256, 128], 2, 128, 0.1, torch.bfloat16, True, False, True),
            ([64, 512], 2, 128, 0.1, torch.bfloat16, True, True, True),
            ([512, 64], 2, 128, 0.1, torch.bfloat16, True, True, False),
            # 3 seqs, unbalanced
            ([64, 192, 128], 2, 128, 0.1, torch.bfloat16, True, True, True),
            ([64, 64, 256], 2, 128, 0.1, torch.bfloat16, True, True, True),
            ([256, 64, 64], 2, 128, 0.1, torch.bfloat16, True, False, True),
            # 4+ seqs, highly unbalanced
            ([64, 128, 256, 512], 4, 128, 0.1, torch.bfloat16, True, True, True),
            ([512, 64, 64, 64], 2, 128, 0.1, torch.bfloat16, True, True, True),
            ([64, 64, 64, 64, 64], 2, 128, 0.1, torch.bfloat16, True, False, False),
            # single sequence (edge case)
            ([256], 2, 128, 0.1, torch.bfloat16, True, True, True),
            ([64], 2, 128, 0.1, torch.bfloat16, True, False, False),
            # longer sequences
            ([1024, 128], 2, 128, 0.1, torch.bfloat16, True, True, True),
            ([128, 1024, 64], 4, 128, 0.1, torch.bfloat16, True, True, True),
            # ---- non-aligned (seqlen not multiple of chunk_size=64) ----
            # tile_size +1 / -1
            ([65, 128], 2, 128, 0.1, torch.bfloat16, True, True, True),
            ([63, 128], 2, 128, 0.1, torch.bfloat16, True, True, True),
            ([128, 65], 2, 128, 0.1, torch.bfloat16, True, True, True),
            ([128, 63], 2, 128, 0.1, torch.bfloat16, True, False, True),
            # single token remainder
            ([129, 127], 2, 128, 0.1, torch.bfloat16, True, True, True),
            # multiple non-aligned
            ([65, 63, 130], 2, 128, 0.1, torch.bfloat16, True, True, True),
            ([33, 95, 200], 2, 128, 0.1, torch.bfloat16, True, True, True),
            # single non-aligned sequence
            ([65], 2, 128, 0.1, torch.bfloat16, True, True, True),
            ([127], 2, 128, 0.1, torch.bfloat16, True, False, False),
            # mixed aligned and non-aligned
            ([64, 65, 128], 2, 128, 0.1, torch.bfloat16, True, True, True),
            ([128, 100, 256], 4, 128, 0.1, torch.bfloat16, True, True, True),
            # highly unbalanced with non-aligned
            ([513, 63], 2, 128, 0.1, torch.bfloat16, True, True, True),
            ([1, 128], 2, 128, 0.1, torch.bfloat16, True, True, True),
        ]
    ],
)
def test_varlen(
    seq_lens: list,
    H: int,
    D: int,
    scale: float,
    dtype: torch.dtype,
    safe_gate: bool,
    has_initial_state: bool,
    output_final_state: bool,
):
    """Test KDA kernel with variable-length sequences via cu_seqlens."""
    try:
        from fla.ops.kda.naive import naive_recurrent_kda as _check
    except Exception:
        raise ImportError("Please install flash-linear-attention")

    torch.manual_seed(42)
    num_seqs = len(seq_lens)
    total_tokens = sum(seq_lens)
    cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lens), 0)), dtype=torch.long, device=device)

    # Generate concatenated data: [1, total_tokens, H, D]
    q = torch.rand(1, total_tokens, H, D, dtype=dtype, device=device)
    k = torch.rand(1, total_tokens, H, D, dtype=dtype, device=device)
    v = torch.rand(1, total_tokens, H, D, dtype=dtype, device=device)
    g = torch.randn(1, total_tokens, H, D, dtype=torch.float, device=device)
    g = F.logsigmoid(g)
    if safe_gate:
        g = g.clamp(-5, 0)
    beta = torch.randn(1, total_tokens, H, dtype=torch.float32, device=device).sigmoid()

    if has_initial_state:
        h0 = torch.randn(num_seqs, H, D, D, dtype=torch.float32, device=device)
    else:
        h0 = None

    # Reference: run per-sequence using naive_recurrent_kda
    ref_o_list = []
    ref_ht_list = []
    for i in range(num_seqs):
        s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        seq_h0 = h0[i:i+1] if h0 is not None else None
        ref_o_i, ref_ht_i = naive_recurrent_kda(
            q=F.normalize(q[:, s:e].clone(), p=2, dim=-1),
            k=F.normalize(k[:, s:e].clone(), p=2, dim=-1),
            v=v[:, s:e].clone(),
            g=g[:, s:e].clone(),
            beta=beta[:, s:e].clone(),
            scale=scale,
            initial_state=seq_h0,
            output_final_state=output_final_state,
        )
        ref_o_list.append(ref_o_i)
        if output_final_state:
            ref_ht_list.append(ref_ht_i)

    ref_o = torch.cat(ref_o_list, dim=1)  # [1, total_tokens, H, D]
    if output_final_state:
        ref_ht = torch.cat(ref_ht_list, dim=0)  # [num_seqs, H, D, D]

    # Test: flash_kda_prefill with cu_seqlens
    tri_o, tri_ht = flash_kda_prefill(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone() if h0 is not None else None,
        output_final_state=output_final_state,
        safe_gate=safe_gate,
        cu_seqlens=cu_seqlens,
    )

    assert_close("varlen o", ref_o, tri_o, 0.005)
    if output_final_state:
        assert tri_ht is not None, "output_final_state=True but got None"
        assert_close("varlen ht", ref_ht, tri_ht, 0.005)