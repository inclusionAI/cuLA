# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Launch with: torchrun --standalone --nproc-per-node=2 -m pytest -q <this file>."""

import bisect
import os

import pytest
import torch
import torch.distributed as dist


def _local_cp_metadata(lengths, rank, world_size):
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    local_tokens = boundaries[-1] // world_size
    rank_start = rank * local_tokens
    rank_end = rank_start + local_tokens
    start_seq = bisect.bisect_right(boundaries[1:], rank_start)
    end_seq = bisect.bisect_left(boundaries[:-1], rank_end)
    local_cu = []
    for boundary in boundaries[start_seq : end_seq + 1]:
        clipped = min(max(boundary, rank_start), rank_end) - rank_start
        if not local_cu or clipped != local_cu[-1]:
            local_cu.append(clipped)
    first_rank = boundaries[start_seq] // local_tokens
    last_rank = (boundaries[end_seq] - 1) // local_tokens
    return local_cu, rank - first_rank, last_rank - rank


def _relative_rms(actual, expected):
    delta = actual.float() - expected.float()
    return delta.norm() / expected.float().norm().clamp_min(1e-8)


def _assert_kda_close(actual, expected, *, gradient=False):
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    if gradient:
        delta = (actual.float() - expected.float()).abs()
        assert delta.max().item() <= 1.25e-1
        assert _relative_rms(actual, expected).item() < 5e-2
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=1e-2)


@pytest.mark.distributed
@pytest.mark.sm100_only
def test_cudnn_kda_cp_thd_matches_single_rank_and_fla_forward_backward():
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("launch this deterministic fixture with exactly two processes")
    cudnn = pytest.importorskip("cudnn")
    if not hasattr(cudnn, "kimi_cp_merge"):
        pytest.skip("cuDNN Frontend build does not include KimiCPMerge")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        from cula.distributed import cudnn_kda_cp_thd
        from cula.kda import cudnn_kda_thd

        device = torch.device("cuda", local_rank)
        generator = torch.Generator(device=device).manual_seed(20261029)
        H, K, V = 8, 128, 128
        lengths = (48, 112)
        total = sum(lengths)
        local_tokens = total // world_size
        start, end = rank * local_tokens, (rank + 1) * local_tokens
        global_cu = torch.tensor([0, lengths[0], total], dtype=torch.int32, device=device)
        local_cu_values, pre_num_ranks, post_num_ranks = _local_cp_metadata(lengths, rank, world_size)
        local_cu = torch.tensor(local_cu_values, dtype=torch.int32, device=device)

        def randn(*shape, dtype=torch.bfloat16):
            return torch.randn(*shape, dtype=dtype, device=device, generator=generator)

        q = torch.nn.functional.normalize(randn(total, H, K).float(), dim=-1).to(torch.bfloat16)
        k = torch.nn.functional.normalize(randn(total, H, K).float(), dim=-1).to(torch.bfloat16)
        v = randn(total, H, V)
        g = -(0.01 + 0.04 * torch.rand(total, H, K, dtype=torch.float32, device=device, generator=generator))
        beta = torch.sigmoid(randn(total, H, dtype=torch.float32))
        grad_output = randn(total, H, V)

        full_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v, g, beta)]
        local_inputs = [tensor[start:end].detach().clone().requires_grad_(True) for tensor in (q, k, v, g, beta)]

        full_output, _ = cudnn_kda_thd(
            *full_inputs,
            global_cu,
            output_final_state=True,
            batch_invariant=True,
            plan_name="kda_frost",
        )
        full_output.backward(grad_output)
        local_output = cudnn_kda_cp_thd(
            *local_inputs,
            local_cu,
            group=dist.group.WORLD,
            pre_num_ranks=pre_num_ranks,
            post_num_ranks=post_num_ranks,
            batch_invariant=True,
        )
        local_output.backward(grad_output[start:end])

        _assert_kda_close(local_output, full_output[start:end])
        for actual, expected in zip(local_inputs, full_inputs):
            _assert_kda_close(actual.grad, expected.grad[start:end], gradient=True)

        fla_cp = pytest.importorskip("fla.ops.cp")
        fla_kda = pytest.importorskip("fla.ops.kda").chunk_kda
        cp_context = fla_cp.build_cp_context(global_cu, group=dist.group.WORLD)
        fla_inputs = [tensor[start:end].detach().clone().requires_grad_(True) for tensor in (q, k, v, g, beta)]
        fla_output, _ = fla_kda(
            *(tensor.unsqueeze(0) for tensor in fla_inputs),
            cu_seqlens=cp_context.cu_seqlens,
            cp_context=cp_context,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            use_gate_in_kernel=False,
        )
        fla_output = fla_output.squeeze(0)
        fla_output.backward(grad_output[start:end])

        _assert_kda_close(local_output, fla_output)
        for actual, expected in zip(local_inputs, fla_inputs):
            _assert_kda_close(actual.grad, expected.grad, gradient=True)

        dist.barrier()
    finally:
        dist.destroy_process_group()
