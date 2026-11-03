# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Packed-THD workload descriptions shared by large-CP benchmarks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PackedTHDWorkload:
    global_cu_seqlens: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.global_cu_seqlens) < 2 or self.global_cu_seqlens[0] != 0:
            raise ValueError("global_cu_seqlens must contain at least two entries and start at zero")
        if any(left >= right for left, right in zip(self.global_cu_seqlens, self.global_cu_seqlens[1:])):
            raise ValueError("global_cu_seqlens must be strictly increasing")

    @property
    def global_tokens(self) -> int:
        return self.global_cu_seqlens[-1]

    @property
    def sequence_lengths(self) -> tuple[int, ...]:
        return tuple(
            right - left for left, right in zip(self.global_cu_seqlens, self.global_cu_seqlens[1:])
        )

    @classmethod
    def single_sequence(cls, global_tokens: int) -> PackedTHDWorkload:
        if global_tokens < 1:
            raise ValueError("global_tokens must be positive")
        return cls((0, global_tokens))

    @classmethod
    def short_long_mix(
        cls,
        global_tokens: int,
        *,
        short_sequences: int = 20,
        short_tokens: int = 1024,
        long_sequences: int = 2,
    ) -> PackedTHDWorkload:
        for name, value in {
            "global_tokens": global_tokens,
            "short_sequences": short_sequences,
            "short_tokens": short_tokens,
            "long_sequences": long_sequences,
        }.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        short_total = short_sequences * short_tokens
        remaining = global_tokens - short_total
        if remaining <= 0 or remaining % long_sequences:
            raise ValueError("global_tokens minus the short samples must split evenly across long samples")
        lengths = [short_tokens] * short_sequences + [remaining // long_sequences] * long_sequences
        boundaries = [0]
        for length in lengths:
            boundaries.append(boundaries[-1] + length)
        return cls(tuple(boundaries))
