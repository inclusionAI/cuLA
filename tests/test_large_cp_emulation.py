# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from benchmarks.large_cp_emulation import (
    LogicalCPEmulationPlan,
    PackedTHDWorkload,
    bench_hybrid,
    sample_chain_hops,
)
from benchmarks.large_cp_emulation.bench_dsa_layout import _validate_block_aligned_workload


def test_sample_chain_hops_for_one_megatoken_thd_mix():
    workload = PackedTHDWorkload.short_long_mix(1 << 20)

    assert workload.sequence_lengths == (1024,) * 20 + (514048, 514048)
    assert sample_chain_hops(workload.global_cu_seqlens, local_tokens=16384) == 31


def test_sample_chain_hops_for_rank_local_samples():
    assert sample_chain_hops((0, 1024, 2048, 4096), local_tokens=4096) == 0


def test_emulation_plan_separates_logical_and_physical_identity():
    workload = PackedTHDWorkload.short_long_mix(1 << 20)

    plan = LogicalCPEmulationPlan.build(
        workload.global_cu_seqlens,
        logical_cp=64,
        physical_rank=2,
        physical_world_size=4,
        transport_mode="sample-chain",
    )

    assert plan.logical_rank == 42
    assert (plan.logical_plan.rank, plan.logical_plan.world_size) == (42, 64)
    assert (plan.runtime_plan.rank, plan.runtime_plan.world_size) == (2, 4)
    assert plan.runtime_plan.local_cu_seqlens == plan.logical_plan.local_cu_seqlens
    assert plan.transport_hops == 31


def test_dsa_layout_rejects_non_block_aligned_sequences():
    workload = PackedTHDWorkload((0, 1023, 2048))

    with pytest.raises(ValueError, match="every packed sequence length"):
        _validate_block_aligned_workload(workload, 4)


def test_logical_dsa_transport_preserves_local_blocks_before_ring_reuses_send(monkeypatch):
    branch = object.__new__(bench_hybrid._DsaIndexerBranch)
    branch.mode = "logical"
    branch.physical_rank = 0
    branch.physical_world_size = 2
    branch.transport_hops = 2
    branch.global_block_start = 2
    branch.local_blocks = 2
    branch.global_kbar = torch.full((6, 3), -1.0)
    branch.ring_receive = torch.empty((2, 3))
    local_kbar = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    expected = local_kbar.clone()

    def overwrite_send(source, receive, **kwargs):
        del receive, kwargs
        source.fill_(99)

    monkeypatch.setattr(bench_hybrid, "emulate_ring_hops", overwrite_send)
    branch._communicate(local_kbar)

    assert torch.equal(branch.global_kbar[2:4], expected)
    assert torch.equal(local_kbar, torch.full_like(local_kbar, 99))
