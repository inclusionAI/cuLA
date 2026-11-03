# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from benchmarks.bench_hybrid_kda_dsa_e2e import _sample_chain_hops


def test_sample_chain_hops_for_one_megatoken_thd_mix():
    boundaries = tuple([index * 1024 for index in range(21)] + [534528, 1048576])

    assert _sample_chain_hops(boundaries, local_tokens=16384) == 31


def test_sample_chain_hops_for_rank_local_samples():
    assert _sample_chain_hops((0, 1024, 2048, 4096), local_tokens=4096) == 0
