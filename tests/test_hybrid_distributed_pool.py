# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from cula.hybrid import distributed_pool
from cula.hybrid.plan import IndexPoolShardPlan


def _fake_collectives(monkeypatch, remote_rows, *, rank=0):
    calls = []

    def all_gather(output, local, *, group):
        calls.append(("gather", tuple(local.shape), group))
        gathered = output.view(2, *local.shape)
        gathered[rank].copy_(local)
        gathered[1 - rank].copy_(remote_rows)

    def reduce_scatter(output, tensor, *, group):
        calls.append(("reduce_scatter", tuple(tensor.shape), group))
        chunks = tensor.view(2, *output.shape)
        output.copy_(chunks[rank])

    monkeypatch.setattr(distributed_pool.dist, "all_gather_into_tensor", all_gather)
    monkeypatch.setattr(distributed_pool.dist, "reduce_scatter_tensor", reduce_scatter)
    monkeypatch.delattr(distributed_pool.dist, "all_gather_single", raising=False)
    monkeypatch.delattr(distributed_pool.dist, "reduce_scatter_single", raising=False)
    return calls


def test_unequal_block_gather_pads_only_compressed_axis(monkeypatch):
    local = torch.tensor([[1.0], [2.0], [3.0]])
    remote = torch.tensor([[10.0], [20.0], [0.0]])
    calls = _fake_collectives(monkeypatch, remote)

    gathered = distributed_pool._all_gather_block_rows(
        local,
        group="cp",
        block_offsets=(0, 3, 5),
        rank=0,
    )

    assert torch.equal(gathered, torch.tensor([[1.0], [2.0], [3.0], [10.0], [20.0]]))
    assert calls == [("gather", (3, 1), "cp")]


def test_cp_pool_rejects_misaligned_seam_before_launch(monkeypatch):
    plan = IndexPoolShardPlan.from_global_cu_seqlens(
        [0, 16],
        rank=0,
        world_size=2,
        head_dim=4,
        shard_offsets=(0, 6, 16),
    )
    monkeypatch.setattr(plan.__class__, "validate_group", lambda self, group: None)
    monkeypatch.setattr(
        distributed_pool,
        "_load_cudnn_index_pool_ops",
        lambda: pytest.fail("pooling must not launch"),
    )

    with pytest.raises(NotImplementedError, match="crossing a rank seam"):
        distributed_pool.cudnn_index_pool_cp(
            torch.empty(6, 4),
            torch.empty(6, 4),
            torch.empty(4, 4),
            torch.tensor([0, 6], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            plan=plan,
            group="cp",
        )


def test_cp_pool_runs_local_pool_before_block_gather(monkeypatch):
    plan = IndexPoolShardPlan.from_global_cu_seqlens(
        [0, 16],
        rank=0,
        world_size=2,
        head_dim=4,
        shard_offsets=(0, 8, 16),
    )
    monkeypatch.setattr(plan.__class__, "validate_group", lambda self, group: None)
    observed = []

    def forward(kv, score, ape, cu, cb, **kwargs):
        observed.append((kv.shape[0], kwargs["total_comp"], kwargs["coff"]))
        return {"out": torch.arange(8, dtype=kv.dtype).reshape(2, 4)}

    monkeypatch.setattr(
        distributed_pool,
        "_load_cudnn_index_pool_ops",
        lambda: (forward, pytest.fail),
    )
    monkeypatch.setattr(distributed_pool, "_validate_inputs", lambda *args, **kwargs: None)
    _fake_collectives(
        monkeypatch,
        torch.full((2, 4), 9.0),
    )
    result = distributed_pool.cudnn_index_pool_cp(
        torch.empty(8, 4),
        torch.empty(8, 4),
        torch.empty(4, 4),
        torch.tensor([0, 8], dtype=torch.int32),
        torch.tensor([0, 2], dtype=torch.int32),
        plan=plan,
        group="cp",
        deterministic=True,
    )

    assert observed == [(8, 2, 1)]
    assert torch.equal(
        result,
        torch.cat((torch.arange(8).reshape(2, 4), torch.full((2, 4), 9.0))),
    )


def test_cp_pool_backward_preserves_loss_rank_gradients(monkeypatch):
    plan = IndexPoolShardPlan.from_global_cu_seqlens(
        [0, 16],
        rank=0,
        world_size=2,
        head_dim=4,
        shard_offsets=(0, 8, 16),
    )
    monkeypatch.setattr(plan.__class__, "validate_group", lambda self, group: None)
    monkeypatch.setattr(distributed_pool, "_validate_inputs", lambda *args, **kwargs: None)

    backward_inputs = []

    def forward(kv, score, ape, cu, cb, **kwargs):
        return {"out": torch.zeros(2, 4, dtype=kv.dtype)}

    def backward(kv, score, ape, cu, cb, grad_out, **kwargs):
        backward_inputs.append(grad_out.clone())
        value = grad_out.float().sum()
        return {
            "grad_kv": torch.full_like(kv, value),
            "grad_score": torch.full_like(score, 2 * value),
            "grad_ape": torch.full_like(ape, value),
        }

    monkeypatch.setattr(
        distributed_pool,
        "_load_cudnn_index_pool_ops",
        lambda: (forward, backward),
    )
    _fake_collectives(monkeypatch, torch.full((2, 4), 9.0))

    def all_to_all(output, tensor, **kwargs):
        # Owner rank 0 receives its block gradients from loss ranks 0 and 1.
        output.view(2, 2, 4)[0].copy_(tensor[:2])
        output.view(2, 2, 4)[1].fill_(3.0)

    monkeypatch.setattr(distributed_pool.dist, "all_to_all_single", all_to_all)

    kv = torch.ones(8, 4, requires_grad=True)
    score = torch.ones(8, 4, requires_grad=True)
    ape = torch.ones(4, 4, requires_grad=True)
    output = distributed_pool.cudnn_index_pool_cp(
        kv,
        score,
        ape,
        torch.tensor([0, 8], dtype=torch.int32),
        torch.tensor([0, 2], dtype=torch.int32),
        plan=plan,
        group="cp",
    )
    output.backward(torch.cat((torch.ones(2, 4), torch.full((2, 4), 5.0))))

    assert len(backward_inputs) == 2
    assert torch.equal(backward_inputs[0], torch.ones(2, 4))
    assert torch.equal(backward_inputs[1], torch.full((2, 4), 3.0))
    assert torch.equal(kv.grad, torch.full_like(kv, 32.0))
    assert torch.equal(score.grad, torch.full_like(score, 64.0))
    # The fake reduce-scatter selects this loss rank's owner-local APE chunk.
    assert torch.equal(ape.grad, torch.full_like(ape, 8.0))
