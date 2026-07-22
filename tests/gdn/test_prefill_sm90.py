# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from cula.gdn import (
    chunk_gated_delta_rule,
    get_sm90_gdn_prefill_backend,
    get_sm90_gdn_prefill_backend_identity,
    is_sm90_gdn_prefill_available,
)
from cula.ops.gdn.sm90.config import (
    HEAD_SIZE,
    SM90_BACKEND_ID,
    THREADS_PER_CTA,
)

from .published_aux_reference import published_aux_reference

OUTPUT_SENTINEL = -123.0
STATE_SENTINEL = -987654.0
OUTPUT_GUARD_TOKENS = 64
STATE_GUARD_SEQS = 1
OUTPUT_RTOL = 1e-2
OUTPUT_ATOL = 1e-2
STATE_RTOL = 1e-3
STATE_ATOL = 5e-3


@dataclass(frozen=True)
class GDNCase:
    case_id: str
    seq_lens: tuple[int, ...]
    heads: tuple[int, int, int]
    alpha: str
    beta: str
    initial_state: bool
    final_state: bool


CORRECTNESS_CASES = (
    GDNCase("C01", (5, 3), (2, 2, 2), "random", "random", False, False),
    GDNCase("C02", (5, 3), (4, 1, 1), "random", "random", True, True),
    GDNCase("C03", (5, 3), (1, 1, 2), "random", "random", False, True),
    GDNCase("C04", (1, 63, 64, 65), (2, 2, 2), "random", "random", False, False),
    GDNCase("C05", (70,), (2, 2, 2), "random", "random", False, False),
    GDNCase("C06", (70,), (2, 2, 2), "random", "random", True, True),
    GDNCase("C07", (64,), (2, 2, 2), "constant-0.1", "random", False, False),
    GDNCase("C08", (65,), (2, 2, 2), "absent", "random", False, False),
    GDNCase("C09", (65,), (2, 2, 2), "random", "absent", True, False),
    GDNCase("C10", (65,), (2, 2, 2), "absent", "absent", False, True),
    GDNCase("C11", (4,), (16, 16, 32), "random", "random", True, True),
    GDNCase("C12", (512,), (64, 64, 64), "random", "random", False, False),
    GDNCase("C13", (127, 385), (64, 64, 64), "random", "random", False, False),
    GDNCase("C14", (127, 128, 129), (1, 1, 2), "random", "random", True, True),
)
CASE_BY_ID = {case.case_id: case for case in CORRECTNESS_CASES}


def _is_sm90() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _head_indices(heads: tuple[int, int, int], device: torch.device) -> tuple[torch.Tensor, ...]:
    num_q_heads, _, num_v_heads = heads
    num_o_heads = max(num_q_heads, num_v_heads)
    output_head = torch.arange(num_o_heads, device=device)
    if num_q_heads >= num_v_heads:
        q_index = output_head
        kv_index = output_head // (num_q_heads // num_v_heads)
        return q_index, kv_index, kv_index
    qk_index = output_head // (num_v_heads // num_q_heads)
    return qk_index, qk_index, output_head


def _matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.dtype in (torch.float16, torch.bfloat16) or b.dtype in (torch.float16, torch.bfloat16):
        return a.float() @ b.float()
    return a @ b


@torch.inference_mode()
def _blockwise_reference(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    seq_lens: tuple[int, ...],
    initial_state: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent PyTorch form of the kernel's 64-token block recurrence."""

    device = q.device
    q_index, k_index, v_index = _head_indices(
        (q.shape[1], k.shape[1], v.shape[1]),
        device,
    )
    q_mapped = q[:, q_index].float()
    k_mapped = k[:, k_index].float()
    v_mapped = v[:, v_index].float()
    num_o_heads = q_mapped.shape[1]
    output = torch.empty_like(v_mapped)
    final_state = torch.empty(
        (len(seq_lens), num_o_heads, HEAD_SIZE, HEAD_SIZE),
        dtype=torch.float32,
        device=device,
    )
    offset = 0
    for seq_idx, seq_len in enumerate(seq_lens):
        state = (
            initial_state[seq_idx].float().clone()
            if initial_state is not None
            else torch.zeros(
                (num_o_heads, HEAD_SIZE, HEAD_SIZE),
                dtype=torch.float32,
                device=device,
            )
        )
        chunk_start = offset
        seq_end = offset + seq_len
        while chunk_start < seq_end:
            valid = min(64, seq_end - chunk_start)
            chunk_end = chunk_start + valid
            q_chunk = torch.zeros(
                (64, num_o_heads, HEAD_SIZE),
                dtype=torch.bfloat16,
                device=device,
            )
            k_chunk = torch.zeros_like(q_chunk)
            v_chunk = torch.zeros_like(q_chunk)
            q_chunk[:valid] = q_mapped[chunk_start:chunk_end].to(torch.bfloat16)
            k_chunk[:valid] = k_mapped[chunk_start:chunk_end].to(torch.bfloat16)
            v_chunk[:valid] = v_mapped[chunk_start:chunk_end].to(torch.bfloat16)
            alpha_chunk = torch.ones((64, num_o_heads), dtype=torch.float32, device=device)
            beta_chunk = torch.zeros_like(alpha_chunk)
            alpha_chunk[:valid] = alpha[chunk_start:chunk_end]
            beta_chunk[:valid] = beta[chunk_start:chunk_end]

            alpha_hs = alpha_chunk.T
            log_alpha = torch.log(alpha_hs + 1e-10)
            prefix_log = torch.cumsum(log_alpha, dim=-1)
            gamma_hss = prefix_log[:, :, None] - prefix_log[:, None, :]
            gamma_hs1 = prefix_log[:, :, None]
            q_hsq = q_chunk.permute(1, 0, 2)
            k_hsk = k_chunk.permute(1, 0, 2)
            v_hsv = v_chunk.permute(1, 0, 2)
            beta_hs1 = beta_chunk.T[:, :, None]

            transfer = torch.exp(gamma_hss)
            causal = torch.ones((64, 64), dtype=torch.bool, device=device).tril()
            active = causal.clone()
            active[valid:, :] = False
            active[:, valid:] = False
            qk_epilogue = torch.where(
                active[None, :, :],
                _matmul(q_hsq, k_hsk.transpose(-2, -1)) * transfer * scale,
                0.0,
            )
            kk_physical = torch.where(
                active[None, :, :],
                _matmul(k_hsk, k_hsk.transpose(-2, -1)) * transfer * beta_hs1,
                0.0,
            )
            qk_published: list[torch.Tensor] = []
            kk_published: list[torch.Tensor] = []
            beta_host = beta_chunk.T.cpu()
            for head_idx in range(num_o_heads):
                published = published_aux_reference(
                    qk_epilogue[head_idx].cpu(),
                    kk_physical[head_idx].cpu(),
                    beta_host[head_idx],
                    valid,
                )
                qk_published.append(published.qk_bf16)
                kk_published.append(published.inverse_kk_beta_bf16)
            qk_bf16 = torch.stack(qk_published).to(device)
            kk_bf16 = torch.stack(kk_published).to(device)

            has_prior_state = initial_state is not None or chunk_start != offset
            v_hvt = v_hsv.transpose(-1, -2).float()
            if has_prior_state:
                rounded_state = state.to(torch.bfloat16).float()
                output_hvt = _matmul(rounded_state, q_hsq.transpose(-1, -2))
                output_hvt *= torch.exp(gamma_hs1).transpose(-1, -2) * scale
                sk_hvt = _matmul(rounded_state, k_hsk.transpose(-1, -2))
                residual_hvt = v_hvt - sk_hvt * torch.exp(gamma_hs1).transpose(-1, -2)
            else:
                output_hvt = torch.zeros_like(v_hvt)
                residual_hvt = v_hvt
            new_v_hvt = _matmul(
                residual_hvt.to(torch.bfloat16),
                kk_bf16.transpose(-1, -2),
            )
            output_hvt += _matmul(
                new_v_hvt.to(torch.bfloat16),
                qk_bf16.transpose(-1, -2),
            )
            chunk_output = output_hvt.transpose(-1, -2).permute(1, 0, 2)
            output[chunk_start:chunk_end] = chunk_output[:valid]

            alpha_prefix = torch.exp(gamma_hs1).transpose(-1, -2)
            alpha_last = alpha_prefix[:, :, -1:]
            decayed_new_v = new_v_hvt * (alpha_last / alpha_prefix)
            state = state * alpha_last
            state += _matmul(decayed_new_v.to(torch.bfloat16), k_hsk)
            chunk_start += 64
        final_state[seq_idx] = state
        offset += seq_len
    return output.to(torch.bfloat16), final_state


def _make_case_inputs(case: GDNCase) -> dict[str, Any]:
    torch.manual_seed(0)
    device = torch.device("cuda")
    total_tokens = sum(case.seq_lens)
    num_q_heads, num_k_heads, num_v_heads = case.heads
    num_o_heads = max(num_q_heads, num_v_heads)
    q = (torch.randn(total_tokens, num_q_heads, HEAD_SIZE, device=device) * 0.03).to(torch.bfloat16)
    k = (torch.randn(total_tokens, num_k_heads, HEAD_SIZE, device=device) * 0.01).to(torch.bfloat16)
    v = (torch.randn(total_tokens, num_v_heads, HEAD_SIZE, device=device) * 0.1).to(torch.bfloat16)
    alpha_value = torch.rand(total_tokens, num_o_heads, dtype=torch.float32, device=device) * 0.1 + 0.85
    if case.alpha == "constant-0.1":
        alpha_value.fill_(0.1)
    beta_value = torch.rand(total_tokens, num_o_heads, dtype=torch.float32, device=device) * 0.5
    initial_state = None
    if case.initial_state:
        initial_state = (
            torch.randn(
                len(case.seq_lens),
                num_o_heads,
                HEAD_SIZE,
                HEAD_SIZE,
                dtype=torch.float32,
                device=device,
            )
            * 0.005
        )

    output_storage = torch.full(
        (total_tokens + 2 * OUTPUT_GUARD_TOKENS, num_o_heads, HEAD_SIZE),
        OUTPUT_SENTINEL,
        dtype=torch.bfloat16,
        device=device,
    )
    output = output_storage[OUTPUT_GUARD_TOKENS : OUTPUT_GUARD_TOKENS + total_tokens]
    state_storage = None
    output_state = None
    if case.final_state:
        state_storage = torch.full(
            (len(case.seq_lens) + 2 * STATE_GUARD_SEQS, num_o_heads, HEAD_SIZE, HEAD_SIZE),
            STATE_SENTINEL,
            dtype=torch.float32,
            device=device,
        )
        output_state = state_storage[STATE_GUARD_SEQS : STATE_GUARD_SEQS + len(case.seq_lens)]
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(case.seq_lens).cumsum(0).tolist()],
        dtype=torch.int64,
        device=device,
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "alpha": None if case.alpha == "absent" else alpha_value,
        "alpha_value": torch.ones_like(alpha_value) if case.alpha == "absent" else alpha_value,
        "beta": None if case.beta == "absent" else beta_value,
        "beta_value": torch.ones_like(beta_value) if case.beta == "absent" else beta_value,
        "initial_state": initial_state,
        "output_storage": output_storage,
        "output": output,
        "state_storage": state_storage,
        "output_state": output_state,
        "cu_seqlens": cu_seqlens,
    }


def run_correctness_case(case: GDNCase) -> dict[str, float | str | bool | list[int]]:
    inputs = _make_case_inputs(case)
    scale = 1.0 / math.sqrt(HEAD_SIZE)
    expected_output, expected_state = _blockwise_reference(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        alpha=inputs["alpha_value"],
        beta=inputs["beta_value"],
        seq_lens=case.seq_lens,
        initial_state=inputs["initial_state"],
        scale=scale,
    )
    actual = chunk_gated_delta_rule(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        g=inputs["alpha"],
        beta=inputs["beta"],
        scale=scale,
        initial_state=inputs["initial_state"],
        output_final_state=case.final_state,
        cu_seqlens=inputs["cu_seqlens"],
        output=inputs["output"],
        output_state=inputs["output_state"],
    )
    if case.final_state:
        actual_output, actual_state = actual
    else:
        actual_output = actual
        actual_state = None
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual_output,
        expected_output,
        rtol=OUTPUT_RTOL,
        atol=OUTPUT_ATOL,
    )
    output_error = (actual_output.float() - expected_output.float()).abs()
    state_max = 0.0
    state_mean = 0.0
    if actual_state is not None:
        torch.testing.assert_close(
            actual_state,
            expected_state,
            rtol=STATE_RTOL,
            atol=STATE_ATOL,
        )
        state_error = (actual_state - expected_state).abs()
        state_max = float(state_error.max().item())
        state_mean = float(state_error.mean().item())

    output_storage = inputs["output_storage"]
    assert bool((output_storage[:OUTPUT_GUARD_TOKENS] == OUTPUT_SENTINEL).all())
    assert bool((output_storage[-OUTPUT_GUARD_TOKENS:] == OUTPUT_SENTINEL).all())
    state_storage = inputs["state_storage"]
    if state_storage is not None:
        assert bool((state_storage[:STATE_GUARD_SEQS] == STATE_SENTINEL).all())
        assert bool((state_storage[-STATE_GUARD_SEQS:] == STATE_SENTINEL).all())

    return {
        "case_id": case.case_id,
        "status": "PASS",
        "seq_lens": list(case.seq_lens),
        "input_profile": "seed0-q0.03-k0.01-v0.1",
        "output_max_abs_error": float(output_error.max().item()),
        "output_mean_abs_error": float(output_error.mean().item()),
        "state_max_abs_error": state_max,
        "state_mean_abs_error": state_mean,
        "output_redzone_validated": True,
        "state_redzone_validated": state_storage is not None,
        "backend_id": get_sm90_gdn_prefill_backend_identity(),
        "fallback_used": False,
    }


REPO_ROOT = Path(__file__).parents[2]


def _source(path: str) -> str:
    return (REPO_ROOT / path).read_text()


def test_public_dispatch_uses_only_the_sm90_cutedsl_backend() -> None:
    public_source = _source("cula/gdn/prefill.py")
    launch_source = _source("cula/ops/gdn/sm90/launch.py")
    assert get_sm90_gdn_prefill_backend() == "dsl"
    assert get_sm90_gdn_prefill_backend_identity() == SM90_BACKEND_ID
    assert "cudac" not in public_source
    assert "cudac" not in launch_source
    assert "gdn_fwd_prefill_sm90" not in public_source
    assert "gdn_fwd_prefill_sm90" not in launch_source
    assert "block=(512, 1, 1)" in _source("cula/ops/gdn/sm90/delta_rule.py")
    assert THREADS_PER_CTA == 512


def test_gdn_backend_uses_ops_layout() -> None:
    public_root = REPO_ROOT / "cula" / "gdn"
    backend_root = REPO_ROOT / "cula" / "ops" / "gdn" / "sm90"
    assert not (public_root / "sm90").exists()
    assert (backend_root / "delta_rule.py").is_file()
    assert (backend_root / "launch.py").is_file()
    for path in public_root.rglob("*.py"):
        assert "import cutlass" not in path.read_text(), path


def test_product_source_has_no_cpp_or_flashinfer_dispatch() -> None:
    source_roots = (REPO_ROOT / "cula" / "gdn", REPO_ROOT / "cula" / "ops" / "gdn")
    sources = {
        path.relative_to(REPO_ROOT): path.read_text() for source_root in source_roots for path in source_root.rglob("*.py")
    }
    assert sources
    for path, source in sources.items():
        assert "gdn_fwd_prefill_sm90" not in source, path
        assert "import flashinfer" not in source, path
        assert "from flashinfer" not in source, path
        assert "cula.cudac" not in source, path


def test_gdn_prefill_benchmark_matrix_is_exact() -> None:
    from benchmarks.bench_gdn_prefill import build_gdn_prefill_matrix

    rows = build_gdn_prefill_matrix(0)
    fixed = {(row.batch_size, row.seq_len) for row in rows if row.mode == "fixed"}
    varlen = {(row.num_seqs, row.total_tokens, row.distribution) for row in rows if row.mode == "varlen"}
    assert len(rows) == 28
    assert fixed == {(batch_size, seq_len) for batch_size in (1, 2) for seq_len in (512, 1024, 4096, 8192, 16384)}
    assert varlen == {
        (num_seqs, total_tokens, distribution)
        for num_seqs in (10, 20)
        for total_tokens in (4096, 8192, 16384)
        for distribution in ("uniform", "random", "skewed")
    }
    assert all(sum(row.seq_lens) == row.total_tokens for row in rows)
    assert all(all(length > 0 for length in row.seq_lens) for row in rows)


@pytest.mark.skipif(not _is_sm90(), reason="GDN prefill requires Hopper SM90")
@pytest.mark.parametrize("case", CORRECTNESS_CASES, ids=lambda case: case.case_id)
def test_correctness_matrix(case: GDNCase) -> None:
    assert not any(name == "cula.cudac" or name.startswith("cula._cudac") for name in sys.modules)
    result = run_correctness_case(case)
    assert result["status"] == "PASS"
    assert result["backend_id"] == SM90_BACKEND_ID
    assert result["fallback_used"] is False


@pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
def test_availability_rejects_non_cuda_device_without_query(
    monkeypatch: pytest.MonkeyPatch,
    device: str | torch.device,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def unexpected_query(_device: torch.device | int | str) -> None:
        raise AssertionError("non-CUDA device reached get_device_properties")

    monkeypatch.setattr(torch.cuda, "get_device_properties", unexpected_query)
    assert is_sm90_gdn_prefill_available(device) is False


@pytest.mark.skipif(not _is_sm90(), reason="GDN prefill requires Hopper SM90")
def test_public_api_opt_in_validation_rejects_zero_length_sequences() -> None:
    inputs = _make_case_inputs(CASE_BY_ID["C01"])
    zero_length = torch.tensor([0, 0, 8], dtype=torch.int64, device="cuda")
    with pytest.raises(ValueError, match="zero-length"):
        chunk_gated_delta_rule(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            cu_seqlens=zero_length,
            validate_inputs=True,
        )


@pytest.mark.skipif(not _is_sm90(), reason="content validation requires H20/SM90 inputs")
def test_public_api_opt_in_content_validation_observes_inplace_mutations() -> None:
    inputs = _make_case_inputs(CASE_BY_ID["C01"])
    chunk_gated_delta_rule(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        g=inputs["alpha"],
        beta=inputs["beta"],
        cu_seqlens=inputs["cu_seqlens"],
        output=inputs["output"],
        validate_inputs=True,
    )

    inputs["cu_seqlens"][1] = 0
    with pytest.raises(ValueError, match="zero-length"):
        chunk_gated_delta_rule(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            g=inputs["alpha"],
            beta=inputs["beta"],
            cu_seqlens=inputs["cu_seqlens"],
            output=inputs["output"],
            validate_inputs=True,
        )
    inputs["cu_seqlens"].copy_(torch.tensor([0, 5, 8], device="cuda"))

    inputs["alpha"][0, 0] = torch.nan
    with pytest.raises(ValueError, match="strictly positive"):
        chunk_gated_delta_rule(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            g=inputs["alpha"],
            beta=inputs["beta"],
            cu_seqlens=inputs["cu_seqlens"],
            output=inputs["output"],
            validate_inputs=True,
        )
    inputs["alpha"][0, 0] = 1.0

    inputs["beta"][0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite update"):
        chunk_gated_delta_rule(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            g=inputs["alpha"],
            beta=inputs["beta"],
            cu_seqlens=inputs["cu_seqlens"],
            output=inputs["output"],
            validate_inputs=True,
        )


@pytest.mark.skipif(not _is_sm90(), reason="host-sync trace requires H20/SM90")
def test_default_fast_path_avoids_host_sync_for_fresh_input_tensors() -> None:
    inputs = _make_case_inputs(CASE_BY_ID["C01"])
    chunk_gated_delta_rule(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        g=inputs["alpha"],
        beta=inputs["beta"],
        cu_seqlens=inputs["cu_seqlens"],
        output=inputs["output"],
    )
    torch.cuda.synchronize()

    fresh = {name: tensor.clone() for name, tensor in inputs.items() if isinstance(tensor, torch.Tensor)}
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        chunk_gated_delta_rule(
            fresh["q"],
            fresh["k"],
            fresh["v"],
            g=fresh["alpha"],
            beta=fresh["beta"],
            cu_seqlens=fresh["cu_seqlens"],
            output=fresh["output"],
        )
    torch.cuda.synchronize()

    cpu_ops = {event.key for event in trace.events()}
    forbidden = {"aten::_local_scalar_dense", "aten::_to_copy", "aten::isfinite", "aten::gt"}
    assert cpu_ops.isdisjoint(forbidden), sorted(cpu_ops & forbidden)
    assert bool(torch.isfinite(fresh["output"]).all())


@pytest.mark.skipif(not _is_sm90(), reason="TVM-FFI environment stream requires H20/SM90")
def test_public_api_uses_current_nondefault_stream() -> None:
    case = CASE_BY_ID["C01"]
    inputs = _make_case_inputs(case)
    expected, _ = _blockwise_reference(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        alpha=inputs["alpha_value"],
        beta=inputs["beta_value"],
        seq_lens=case.seq_lens,
        initial_state=None,
        scale=1.0 / math.sqrt(HEAD_SIZE),
    )
    torch.cuda.synchronize()

    output = torch.full_like(inputs["output"], torch.nan)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = chunk_gated_delta_rule(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            g=inputs["alpha"],
            beta=inputs["beta"],
            cu_seqlens=inputs["cu_seqlens"],
            output=output,
        )
        finished = torch.cuda.Event()
        finished.record()
    finished.synchronize()

    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual, expected, rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL)


@pytest.mark.skipif(not _is_sm90(), reason="GDN prefill requires Hopper SM90")
def test_repeated_launch_is_deterministic_and_keeps_redzones() -> None:
    case = CASE_BY_ID["C05"]
    inputs = _make_case_inputs(case)
    hashes: list[int] = []
    for _ in range(10):
        inputs["output_storage"].fill_(OUTPUT_SENTINEL)
        actual = chunk_gated_delta_rule(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            g=inputs["alpha"],
            beta=inputs["beta"],
            cu_seqlens=inputs["cu_seqlens"],
            output=inputs["output"],
        )
        torch.cuda.synchronize()
        hashes.append(hash(actual.detach().cpu().view(torch.uint16).numpy().tobytes()))
        assert bool((inputs["output_storage"][:OUTPUT_GUARD_TOKENS] == OUTPUT_SENTINEL).all())
        assert bool((inputs["output_storage"][-OUTPUT_GUARD_TOKENS:] == OUTPUT_SENTINEL).all())
    assert len(set(hashes)) == 1


@pytest.mark.skipif(not _is_sm90(), reason="GDN prefill requires Hopper SM90")
def test_two_call_continuation_matches_one_call() -> None:
    case = CASE_BY_ID["C06"]
    inputs = _make_case_inputs(case)
    kwargs = {
        "g": inputs["alpha"],
        "beta": inputs["beta"],
        "initial_state": inputs["initial_state"],
        "output_final_state": True,
    }
    full_output, full_state = chunk_gated_delta_rule(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        cu_seqlens=inputs["cu_seqlens"],
        **kwargs,
    )
    split = 64
    first_output, first_state = chunk_gated_delta_rule(
        inputs["q"][:split].contiguous(),
        inputs["k"][:split].contiguous(),
        inputs["v"][:split].contiguous(),
        g=inputs["alpha"][:split].contiguous(),
        beta=inputs["beta"][:split].contiguous(),
        initial_state=inputs["initial_state"],
        output_final_state=True,
        cu_seqlens=torch.tensor([0, split], dtype=torch.int64, device="cuda"),
    )
    second_output, second_state = chunk_gated_delta_rule(
        inputs["q"][split:].contiguous(),
        inputs["k"][split:].contiguous(),
        inputs["v"][split:].contiguous(),
        g=inputs["alpha"][split:].contiguous(),
        beta=inputs["beta"][split:].contiguous(),
        initial_state=first_state,
        output_final_state=True,
        cu_seqlens=torch.tensor([0, 70 - split], dtype=torch.int64, device="cuda"),
    )
    torch.cuda.synchronize()
    continued_output = torch.cat((first_output, second_output), dim=0)
    torch.testing.assert_close(
        continued_output,
        full_output,
        rtol=OUTPUT_RTOL,
        atol=OUTPUT_ATOL,
    )
    torch.testing.assert_close(
        second_state,
        full_state,
        rtol=STATE_RTOL,
        atol=STATE_ATOL,
    )
