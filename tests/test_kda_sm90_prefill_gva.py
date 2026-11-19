# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from cula.kda import kda_prefill
from cula.ops.kda.cp_mode import CPMode

pytestmark = [pytest.mark.sm90_only, pytest.mark.kda_fast]

D = 128
LOWER_BOUND = -5.0


def _reference_sequence(q, k, v, g, beta, A_log, dt_bias, initial_state):
    batch, length, qk_heads, dim = q.shape
    value_heads = v.shape[2]
    group_size = value_heads // qk_heads
    scale = dim**-0.5

    q = torch.nn.functional.normalize(q.float(), dim=-1) * scale
    k = torch.nn.functional.normalize(k.float(), dim=-1)
    state = initial_state.transpose(-1, -2).float().clone()
    output = torch.empty(batch, length, value_heads, dim, dtype=torch.float32, device=q.device)

    for token in range(length):
        for value_head in range(value_heads):
            qk_head = value_head // group_size
            gate = LOWER_BOUND * torch.sigmoid(
                torch.exp(A_log[value_head]) * (g[:, token, value_head].float() + dt_bias[value_head])
            )
            state[:, value_head] *= torch.exp(gate).unsqueeze(-1)
            value_new = beta[:, token, value_head].float().unsqueeze(-1) * (
                v[:, token, value_head].float() - k[:, token, qk_head] @ state[:, value_head]
            )
            state[:, value_head] = state[:, value_head] + k[:, token, qk_head].unsqueeze(-1) * value_new.unsqueeze(-2)
            output[:, token, value_head] = q[:, token, qk_head] @ state[:, value_head]

    return output, state.transpose(-1, -2).contiguous()


def _make_inputs(batch, length, num_sequences, *, beta_dtype, seed, qk_heads=2, value_heads=4):
    torch.manual_seed(seed)
    q = torch.randn(batch, length, qk_heads, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn(batch, length, value_heads, D, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(batch, length, value_heads, D, dtype=torch.bfloat16, device="cuda") * 0.2
    beta = torch.randn(batch, length, value_heads, dtype=beta_dtype, device="cuda").sigmoid()
    A_log = torch.randn(value_heads, dtype=torch.float32, device="cuda") * 0.1
    dt_bias = torch.randn(value_heads, D, dtype=torch.float32, device="cuda") * 0.1
    initial_state = torch.randn(num_sequences, value_heads, D, D, dtype=torch.float32, device="cuda") * 0.01
    return q, k, v, g, beta, A_log, dt_bias, initial_state


def _assert_close(name, actual, expected):
    actual = actual.float()
    error = (actual - expected).norm() / expected.norm().clamp_min(1e-30)
    assert error < 0.007, f"{name} relative L2: {error.item():.6f}"


@pytest.mark.parametrize(("qk_heads", "value_heads"), [(2, 2), (2, 4), (2, 8)], ids=["mha", "gva2", "gva4"])
def test_dense_gva_matches_tokenwise_reference(qk_heads, value_heads):
    batch, length = 1, 32
    q, k, v, g, beta, A_log, dt_bias, initial_state = _make_inputs(
        batch,
        length,
        batch,
        beta_dtype=torch.bfloat16,
        seed=11,
        qk_heads=qk_heads,
        value_heads=value_heads,
    )
    out = torch.empty_like(v)
    final_state = torch.empty_like(initial_state)

    with torch.no_grad():
        actual_out, actual_state = kda_prefill(
            q,
            k,
            v,
            g,
            beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=initial_state,
            output_final_state=True,
            safe_gate=True,
            lower_bound=LOWER_BOUND,
            use_intracard_cp=False,
            out=out,
            final_state=final_state,
        )

    expected_out, expected_state = _reference_sequence(q, k, v, g, beta, A_log, dt_bias, initial_state)
    assert actual_out is out
    assert actual_state is final_state
    _assert_close("dense output", actual_out, expected_out)
    _assert_close("dense final state", actual_state, expected_state)


def test_varlen_gva_matches_independent_sequences():
    lengths = (3, 5, 20)
    total_length = sum(lengths)
    cu_seqlens = torch.tensor((0, *torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32, device="cuda")
    q, k, v, g, beta, A_log, dt_bias, initial_state = _make_inputs(
        1,
        total_length,
        len(lengths),
        beta_dtype=torch.float32,
        seed=17,
    )

    with torch.no_grad():
        actual_out, actual_state = kda_prefill(
            q,
            k,
            v,
            g,
            beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens.cpu(),
            safe_gate=True,
            lower_bound=LOWER_BOUND,
            use_intracard_cp="auto",
        )

    expected_out = []
    expected_state = []
    for index, (start, end) in enumerate(zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=True)):
        sequence_out, sequence_state = _reference_sequence(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            A_log,
            dt_bias,
            initial_state[index : index + 1],
        )
        expected_out.append(sequence_out)
        expected_state.append(sequence_state)

    _assert_close("varlen output", actual_out, torch.cat(expected_out, dim=1))
    _assert_close("varlen final state", actual_state, torch.cat(expected_state, dim=0))


def test_gva_without_recurrent_state():
    q, k, v, g, beta, A_log, dt_bias, _ = _make_inputs(
        1,
        17,
        1,
        beta_dtype=torch.bfloat16,
        seed=19,
    )
    zero_state = torch.zeros(1, v.shape[2], D, D, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        actual_out, actual_state = kda_prefill(
            q,
            k,
            v,
            g,
            beta,
            A_log=A_log,
            dt_bias=dt_bias,
            output_final_state=False,
            safe_gate=True,
            lower_bound=LOWER_BOUND,
            use_intracard_cp="auto",
        )

    expected_out, _ = _reference_sequence(q, k, v, g, beta, A_log, dt_bias, zero_state)
    assert actual_state is None
    _assert_close("stateless output", actual_out, expected_out)


def test_gva_rejects_forced_intracard_cp_before_launch():
    q, k, v, g, beta, A_log, dt_bias, initial_state = _make_inputs(
        1,
        16,
        1,
        beta_dtype=torch.bfloat16,
        seed=23,
    )

    with pytest.raises(NotImplementedError, match="does not support forced intracard"):
        kda_prefill(
            q,
            k,
            v,
            g,
            beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=initial_state,
            safe_gate=True,
            lower_bound=LOWER_BOUND,
            use_intracard_cp=CPMode.FORCE,
        )


def test_gva_rejects_non_multiple_value_heads():
    q, k, v, g, beta, A_log, dt_bias, _ = _make_inputs(
        1,
        16,
        1,
        beta_dtype=torch.bfloat16,
        seed=29,
        qk_heads=2,
        value_heads=3,
    )

    with pytest.raises(ValueError, match="positive multiple"):
        kda_prefill(
            q,
            k,
            v,
            g,
            beta,
            A_log=A_log,
            dt_bias=dt_bias,
            safe_gate=True,
            lower_bound=LOWER_BOUND,
        )
