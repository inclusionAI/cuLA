# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest

from benchmarks.bench_kda_independent_shards import paired_samples


@pytest.mark.parametrize("warmups,repeats", [(0, 1), (1, 4), (3, 3)])
def test_paired_samples_warm_both_routes_and_record_actual_order(warmups, repeats):
    calls = []

    def run(label):
        calls.append(label)
        return {"total_ms": len(calls), "fwd_ms": 1.0}

    samples = paired_samples(run, warmups=warmups, repeats=repeats)
    assert len(calls) == 2 * (warmups + repeats)
    assert calls[: 2 * warmups].count("baseline") == warmups
    assert calls[: 2 * warmups].count("candidate") == warmups
    for pair in range(repeats):
        expected = ["baseline", "candidate"] if pair % 2 == 0 else ["candidate", "baseline"]
        offset = 2 * (warmups + pair)
        assert calls[offset : offset + 2] == expected
        for position, label in enumerate(expected):
            sample = samples[label][pair]
            assert sample["pair_index"] == pair
            assert sample["execution_order"] == expected
            assert sample["position"] == position
            assert sample["total_ms"] == offset + position + 1
            assert sample["fwd_ms"] == 1.0
    assert all(len(rows) == repeats for rows in samples.values())


@pytest.mark.parametrize("warmups,repeats", [(-1, 1), (0, 0)])
def test_paired_samples_reject_invalid_counts_before_running(warmups, repeats):
    def unexpected(label):
        pytest.fail("invalid counts executed a benchmark route")

    with pytest.raises(ValueError):
        paired_samples(unexpected, warmups=warmups, repeats=repeats)
