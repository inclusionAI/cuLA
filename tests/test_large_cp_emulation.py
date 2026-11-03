# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from benchmarks.large_cp_emulation import (
    LogicalCPEmulationPlan,
    PackedTHDWorkload,
    sample_chain_hops,
)


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
