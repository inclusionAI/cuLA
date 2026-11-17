# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter

import pytest

from benchmarks.bench_csa_layout_matrix import performance_cases
from benchmarks.thd_1m_cases import TOTAL_TOKENS, counted_cases, heldout_cases, manifest, statistics_cases, summarize


@pytest.mark.parametrize("case", statistics_cases(), ids=lambda case: case.name)
def test_exact_packing_and_provenance(case):
    assert sum(case.lengths) == TOTAL_TOKENS
    assert all(isinstance(n, int) and n > 0 and n % 4 == 0 for n in case.lengths)
    for length, count in Counter(case.long_documents).items():
        assert case.lengths.count(length) == count
    assert 0 <= case.final_document_truncation < 16384
    stats = summarize(case)
    assert sum(b["documents"] for b in stats["token_length_histogram"]) == len(case.lengths)
    assert sum(b["tokens"] for b in stats["token_length_histogram"]) == TOTAL_TOKENS


def test_reproducible_cases_and_order_only_controls():
    cases = statistics_cases()
    assert cases == statistics_cases()
    assert len({case.name for case in cases}) == len(cases) == 18
    lookup = {case.name: case for case in cases}
    for base in ("1m-stat-mix80", "1m-stat-mix80-2x256k"):
        for order in ("ascending", "descending", "alternating"):
            candidate = lookup[f"{base}-{order}"].lengths
            assert Counter(candidate) == Counter(lookup[base].lengths)
            assert candidate != lookup[base].lengths
        assert lookup[base + "-ascending"].lengths == tuple(sorted(lookup[base].lengths))
        assert lookup[base + "-descending"].lengths == tuple(sorted(lookup[base].lengths, reverse=True))
    assert lookup["1m-stat-mix80"].lengths != statistics_cases(seed=123)[2].lengths


def test_matrix_keeps_baseline_and_manifest_matches_execution():
    baseline = performance_cases()
    assert len(baseline) == 5
    assert [c[0] for c in baseline] == ["1m-single", "1m-uniform", "1m-near-uniform", "1m-20s2l", "1m-power-law"]
    cases = performance_cases("statistics", seed=123)
    saved = manifest(seed=123)["cases"]
    assert [(name, lengths) for name, lengths, _ in cases] == [(c["name"], list(c["lengths"])) for c in saved]
    assert len(performance_cases("all")) == 39
    with pytest.raises(ValueError, match="unknown suite"):
        performance_cases("typo")


@pytest.mark.parametrize("case", counted_cases(), ids=lambda case: case.name)
def test_fixed_document_counts(case):
    assert sum(case.lengths) == TOTAL_TOKENS
    assert 256 <= len(case.lengths) <= 1024
    assert all(n > 0 and n % 4 == 0 for n in case.lengths)
    assert len(case.lengths) == int(next(s for s in case.name.split("-") if s in {"256", "512", "1000", "1024"}))


def test_counted_manifest_matches_the_actual_matrix():
    saved = manifest(seed=123, suite="counted")["cases"]
    cases = performance_cases("counted", seed=123)
    assert [(name, lengths) for name, lengths, _ in cases] == [(c["name"], list(c["lengths"])) for c in saved]


def test_heldout_lengths_vary_without_changing_budget_or_document_counts():
    first, second = heldout_cases(20261114), heldout_cases(20261115)
    assert first == heldout_cases(20261114)
    assert len(first) == 5
    assert {len(c.lengths) for c in first} == {257, 1000, 768, 513}
    for a, b in zip(first, second):
        assert a.name == b.name
        assert len(a.lengths) == len(b.lengths)
        assert sorted(a.lengths) != sorted(b.lengths)
        for case in (a, b):
            assert sum(case.lengths) == TOTAL_TOKENS
            assert all(n > 0 and n % 4 == 0 for n in case.lengths)
    saved = manifest(20261114, "heldout")["cases"]
    actual = performance_cases("heldout", 20261114)
    assert [(name, lengths) for name, lengths, _ in actual] == [(c["name"], list(c["lengths"])) for c in saved]


def test_validation_suite_has_31_distinct_exact_1m_shapes():
    from benchmarks.bench_csa_layout_matrix import performance_cases

    cases = performance_cases("validation")
    assert len(cases) == len({name for name, _, _ in cases}) == 31
    assert all(sum(lengths) == 1 << 20 and all(n > 0 and n % 4 == 0 for n in lengths) for _, lengths, _ in cases)
    for seed in (20261114, 20261115, 20261116):
        observed = {name: lengths for name, lengths, _ in cases}
        for case in heldout_cases(seed):
            assert observed[f"{case.name}-seed{seed}"] == list(case.lengths)
