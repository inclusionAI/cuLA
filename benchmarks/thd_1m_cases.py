# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Reproducible 1M THD cases inspired by a character-length count histogram.

The screenshot supplies no exact percentages or tokenizer. Weights below are
explicit synthetic priors, not recovered measurements. Long-document spikes
are additional stress cases, not claims about the screenshot's hidden tail.
"""

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from itertools import accumulate
from pathlib import Path

TOTAL_TOKENS = 1 << 20
DEFAULT_SEED = 20261112
# (lower exclusive, upper inclusive); k=1024 is a synthetic convention.
CHARACTER_BINS = ((10, 100), (100, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 16384))
COUNT_PRIORS = {
    "tiny90": (9000, 500, 200, 200, 80, 20),
    "mix80": (8000, 700, 500, 500, 250, 50),
    "mix70": (7000, 1200, 800, 600, 300, 100),
}


@dataclass(frozen=True)
class THDCase:
    name: str
    lengths: tuple[int, ...]
    prior: str
    character_divisor: int
    long_documents: tuple[int, ...]
    order: str
    seed: int
    final_document_truncation: int = 0


def _fill(budget, prior, divisor, seed):
    """Draw aligned lengths and truncate at most the last draw to fill exactly."""
    rng = random.Random(seed)
    cumulative = tuple(accumulate(COUNT_PRIORS[prior]))
    lengths, remaining, truncation = [], budget, 0
    while remaining:
        draw = rng.randrange(cumulative[-1])
        bucket = next(i for i, bound in enumerate(cumulative) if draw < bound)
        lower, upper = CHARACTER_BINS[bucket]
        # Uniform aligned token proxies within each scaled character bin.
        minimum = max(1, lower // (4 * divisor) + 1)
        maximum = max(minimum, upper // (4 * divisor))
        proposed = 4 * rng.randint(minimum, maximum)
        length = min(proposed, remaining)
        lengths.append(length)
        remaining -= length
        truncation += proposed - length
    return lengths, truncation


def _ordered(lengths, order, seed):
    values = sorted(lengths)
    if order == "ascending":
        return tuple(values)
    if order == "descending":
        return tuple(reversed(values))
    if order == "alternating":
        # Smallest, largest, second-smallest, second-largest, ...
        return tuple(
            v
            for i in range((len(values) + 1) // 2)
            for v in ((values[i], values[-1 - i]) if i != len(values) - 1 - i else (values[i],))
        )
    if order != "shuffled":
        raise ValueError(f"unknown order: {order}")
    random.Random(seed + 1).shuffle(values)
    return tuple(values)


def statistics_cases(seed=DEFAULT_SEED):
    """Return 18 block-aligned cases; same-histogram order variants share data."""
    cases = [THDCase("1m-stat-tiny64", (64,) * (TOTAL_TOKENS // 64), "fixed64", 1, (), "constant", seed)]
    specifications = [
        ("tiny90", "tiny90", 1, ()),
        ("mix80", "mix80", 1, ()),
        ("mix70", "mix70", 1, ()),
        ("mix80-div2", "mix80", 2, ()),
        ("mix80-div4", "mix80", 4, ()),
        ("mix80-1x128k", "mix80", 1, (128 << 10,)),
        ("mix80-1x256k", "mix80", 1, (256 << 10,)),
        ("mix80-1x512k", "mix80", 1, (512 << 10,)),
        ("mix80-2x256k", "mix80", 1, (256 << 10,) * 2),
        ("mix80-4x128k", "mix80", 1, (128 << 10,) * 4),
        ("mix80-8x64k", "mix80", 1, (64 << 10,) * 8),
    ]
    for name, prior, divisor, long_docs in specifications:
        small, truncated = _fill(TOTAL_TOKENS - sum(long_docs), prior, divisor, seed)
        lengths = _ordered([*small, *long_docs], "shuffled", seed)
        cases.append(THDCase(f"1m-stat-{name}", lengths, prior, divisor, long_docs, "shuffled", seed, truncated))
    for name in ("1m-stat-mix80", "1m-stat-mix80-2x256k"):
        original = next(case for case in cases if case.name == name)
        for order in ("ascending", "descending", "alternating"):
            cases.append(
                replace(original, name=f"{name}-{order}", lengths=_ordered(original.lengths, order, seed), order=order)
            )
    return tuple(cases)


def counted_cases(seed=DEFAULT_SEED):
    """1M with 256--1024 documents, controlling count independently of length."""

    def balanced(tokens, count):
        blocks, extra = divmod(tokens // 4, count)
        if tokens % 4 or blocks < 1:
            raise ValueError("invalid aligned token budget")
        return [4 * (blocks + (i < extra)) for i in range(count)]

    tiny512 = [64] * 410
    tiny1000 = [64] * 800
    specs = [
        ("uniform-256", [4096] * 256, ()),
        ("uniform-1024", [1024] * 1024, ()),
        ("near-uniform-512", [1536, 2560] * 256, ()),
        ("tiny-mix-512", tiny512 + [8192] * 90 + balanced(TOTAL_TOKENS - sum(tiny512) - 8192 * 90, 12), ()),
        ("tiny-mix-1000", tiny1000 + [2048] * 190 + balanced(TOTAL_TOKENS - sum(tiny1000) - 2048 * 190, 10), ()),
        ("mixed-256-1long", [64] * 205 + [8192] * 50, (TOTAL_TOKENS - 64 * 205 - 8192 * 50,)),
        ("mixed-512-2long", tiny512 + balanced(TOTAL_TOKENS // 2 - sum(tiny512), 100), (256 << 10,) * 2),
        ("mixed-1000-1long", tiny1000 + balanced(TOTAL_TOKENS // 2 - sum(tiny1000), 199), (512 << 10,)),
        ("mixed-512-4long", tiny512 + balanced(TOTAL_TOKENS // 2 - sum(tiny512), 98), (128 << 10,) * 4),
    ]
    cases = [
        THDCase(f"1m-count-{name}", _ordered([*short, *long], "shuffled", seed), "fixed-count", 1, long, "shuffled", seed)
        for name, short, long in specs
    ]
    base = next(c for c in cases if c.name == "1m-count-mixed-512-2long")
    for order in ("ascending", "descending"):
        cases.append(replace(base, name=f"{base.name}-{order}", lengths=_ordered(base.lengths, order, seed), order=order))
    return tuple(cases)


def heldout_cases(seed=20261114):
    """Uncalibrated, variable-length cases with 257--1000 documents.

    Both lengths and order change with the seed. Budget apportionment uses
    largest remainders after reserving one block per sample, so it cannot
    introduce empty documents or hide padding tokens in a final correction.
    """
    rng = random.Random(seed)

    def allocate(tokens, weights):
        blocks = tokens // 4 - len(weights)
        total_weight = sum(weights)
        quotas = [blocks * weight / total_weight for weight in weights]
        values = [math.floor(quota) for quota in quotas]
        for i in sorted(range(len(weights)), key=lambda i: (-(quotas[i] - values[i]), i))[: blocks - sum(values)]:
            values[i] += 1
        return [4 * (value + 1) for value in values]

    specs = []
    for count, sigma in ((257, 1.2), (1000, 1.8)):
        specs.append((f"lognormal-{count}", allocate(TOTAL_TOKENS, [rng.lognormvariate(0, sigma) for _ in range(count)]), ()))
    specs.append(("jittered-768", allocate(TOTAL_TOKENS, [rng.uniform(0.7, 1.3) for _ in range(768)]), ()))
    tiny = [4 * rng.randint(3, 25) for _ in range(614)]
    specs.append(
        ("tiny-tail-768", tiny + allocate(TOTAL_TOKENS - sum(tiny), [rng.lognormvariate(0, 2) for _ in range(154)]), ())
    )
    short = [4 * rng.randint(3, 25) for _ in range(410)] + [4 * rng.randint(128, 1024) for _ in range(101)]
    fraction = rng.uniform(0.2, 0.8)
    long = tuple(allocate(TOTAL_TOKENS - sum(short), [fraction, 1 - fraction]))
    specs.append(("trimodal-513", short + list(long), long))
    return tuple(
        THDCase(f"1m-heldout-{name}", _ordered(lengths, "shuffled", seed), "heldout-weighted", 1, long, "shuffled", seed)
        for name, lengths, long in specs
    )


def summarize(case):
    lengths = case.lengths
    boundaries = (0, 10, 100, 1024, 2048, 4096, 8192, 16384, 65536, 131072, 262144, 524288, TOTAL_TOKENS)
    histogram = []
    for lower, upper in zip(boundaries, boundaries[1:]):
        values = [n for n in lengths if lower < n <= upper]
        histogram.append(
            dict(
                lower_exclusive=lower,
                upper_inclusive=upper,
                documents=len(values),
                tokens=sum(values),
                document_fraction=len(values) / len(lengths),
                token_fraction=sum(values) / TOTAL_TOKENS,
            )
        )
    return dict(
        documents=len(lengths),
        total_tokens=sum(lengths),
        min_tokens=min(lengths),
        max_tokens=max(lengths),
        tiny_document_fraction=sum(n <= 100 for n in lengths) / len(lengths),
        tiny_token_fraction=sum(n for n in lengths if n <= 100) / TOTAL_TOKENS,
        explicit_long_token_fraction=sum(case.long_documents) / TOTAL_TOKENS,
        lengths_sha256=hashlib.sha256(json.dumps(lengths, separators=(",", ":")).encode()).hexdigest(),
        token_length_histogram=histogram,
    )


def manifest(seed=DEFAULT_SEED, suite="statistics"):
    if suite not in ("statistics", "counted", "heldout", "all"):
        raise ValueError(f"unknown THD suite: {suite}")
    cases = ()
    if suite in ("statistics", "all"):
        cases += statistics_cases(seed)
    if suite in ("counted", "all"):
        cases += counted_cases(seed)
    if suite in ("heldout", "all"):
        cases += heldout_cases(seed)
    return dict(
        schema_version=1,
        provenance="Synthetic priors inspired by a character-length document-count screenshot; not tokenized corpus data.",
        assumptions={
            "total_tokens": TOTAL_TOKENS,
            "alignment": 4,
            "k": 1024,
            "character_bins_lower_exclusive_upper_inclusive": CHARACTER_BINS,
            "document_count_prior_weights": COUNT_PRIORS,
            "character_divisor": "1/2/4 are sensitivity assumptions, not a measured tokenizer conversion.",
            "within_bin": "Uniform over aligned token lengths after scaling.",
            "packing": "Fill short-mixture budget, truncate only the last draw, add explicit long docs, then reorder.",
            "tail": "64K-512K spikes are extra stress cases; screenshot tail probabilities are unknown.",
            "fixed_count": "256/512/1000/1024 documents; tiny mixtures use about 80% 64-token documents by count, with the remaining budget assigned explicitly. These are token stress shapes, not a measured character-to-token conversion.",
        },
        cases=[dict(**asdict(case), statistics=summarize(case)) for case in cases],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--suite", choices=("statistics", "counted", "heldout", "all"), default="statistics")
    parser.add_argument("--output", type=Path, help="Write exact lengths, assumptions and count/token histograms as JSON")
    args = parser.parse_args()
    data = manifest(args.seed, args.suite)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2) + "\n")
    print("| Case | Documents | <=100 count % | <=100 token % | Max tokens | Explicit long token % |")
    print("|---|---:|---:|---:|---:|---:|")
    for case in data["cases"]:
        stats = case["statistics"]
        print(
            f"| {case['name']} | {stats['documents']} | {100 * stats['tiny_document_fraction']:.2f} | "
            f"{100 * stats['tiny_token_fraction']:.2f} | {stats['max_tokens']} | "
            f"{100 * stats['explicit_long_token_fraction']:.1f} |"
        )


if __name__ == "__main__":
    main()
