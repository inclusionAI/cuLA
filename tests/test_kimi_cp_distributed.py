# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Run with two or more ranks under ``torchrun``."""

import inspect
import os

import pytest
import torch
import torch.distributed as dist


def _relative_rms(actual, expected):
    delta = actual.float() - expected.float()
    return delta.norm() / expected.float().norm().clamp_min(1e-8)


@pytest.mark.distributed
def test_cudnn_kda_cp_matches_monolithic_forward_backward():
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("launch this deterministic fixture with at least two processes")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        cudnn_la = pytest.importorskip("cudnn.linear_attention")
        try:
            supports_initial_state = "initial_state" in inspect.signature(cudnn_la.kimi_delta_attention).parameters
        except (TypeError, ValueError):
            supports_initial_state = False
        if not supports_initial_state:
            pytest.skip("cuDNN Frontend build does not include KDA state inputs")
        capability_major = torch.cuda.get_device_capability(local_rank)[0]
        if capability_major < 9:
            pytest.skip("cuDNN Kimi CP requires SM90 or newer")
        if (
            capability_major >= 10
            and "preceding_state_update" not in inspect.signature(cudnn_la.kimi_delta_attention).parameters
        ):
            pytest.skip("the Blackwell build does not include concatenated KDA summary apply")

        def stage(name):
            print(f"rank={local_rank} stage={name}", flush=True)

        from cula.distributed import AllGatherKimiCPTransport, build_kimi_cp_plan, cudnn_kda_cp

        device = torch.device("cuda", local_rank)
        generator = torch.Generator(device=device).manual_seed(20261102)
        heads, key_dim, value_dim = 8, 128, 128
        lengths = (48, 112)
        total = sum(lengths)
        global_cu_cpu = torch.tensor([0, lengths[0], total], dtype=torch.int32)
        global_cu = global_cu_cpu.to(device)
        plan = build_kimi_cp_plan(global_cu_cpu, group=dist.group.WORLD)
        boundary_transport = AllGatherKimiCPTransport.prepare(
            plan,
            heads=heads,
            key_dim=key_dim,
            value_dim=value_dim,
            device=device,
        )
        local_cu = torch.tensor(plan.local_cu_seqlens, dtype=torch.int32, device=device)
        start, end = plan.token_start, plan.token_end

        def randn(*shape, dtype=torch.bfloat16):
            return torch.randn(*shape, dtype=dtype, device=device, generator=generator)

        q = torch.nn.functional.normalize(randn(total, heads, key_dim).float(), dim=-1).to(torch.bfloat16)
        k = torch.nn.functional.normalize(randn(total, heads, key_dim).float(), dim=-1).to(torch.bfloat16)
        v = randn(total, heads, value_dim)
        g = -(0.01 + 0.04 * torch.rand(total, heads, key_dim, dtype=torch.float32, device=device, generator=generator))
        beta = torch.sigmoid(randn(total, heads, dtype=torch.float32))
        grad_output = randn(total, heads, value_dim)

        full_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v, g, beta)]
        local_inputs = [tensor[start:end].detach().clone().requires_grad_(True) for tensor in (q, k, v, g, beta)]

        stage("monolithic-forward-start")
        full_output, _ = cudnn_la.kimi_delta_attention(
            *full_inputs,
            global_cu,
            output_final_state=True,
            batch_invariant=True,
            plan_name=None,
        )
        stage("monolithic-forward-done")
        full_output.backward(grad_output)
        stage("monolithic-backward-done")
        stage("cp-forward-start")
        local_output = cudnn_kda_cp(
            *local_inputs,
            local_cu,
            plan=plan,
            group=dist.group.WORLD,
            boundary_transport=boundary_transport,
        )
        stage("cp-forward-done")
        local_output.backward(grad_output[start:end])
        stage("cp-backward-done")

        torch.testing.assert_close(local_output, full_output[start:end], rtol=2e-2, atol=1e-2)
        for actual, expected in zip(local_inputs, full_inputs):
            assert torch.isfinite(actual.grad).all()
            delta = (actual.grad.float() - expected.grad[start:end].float()).abs()
            assert delta.max().item() <= 1.25e-1
            assert _relative_rms(actual.grad, expected.grad[start:end]).item() < 5e-2
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    test_cudnn_kda_cp_matches_monolithic_forward_backward()
