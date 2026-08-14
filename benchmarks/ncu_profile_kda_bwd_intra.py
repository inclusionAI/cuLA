#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Profile one warmed-up portable KDA bwd-intra kernel launch with NCU."""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from benchmarks.bench_kda_bwd_intra_sm90 import (  # noqa: E402
    _make_inputs,
    _prepare_cula,
    _quasi_balanced_lengths,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--total-tokens", type=int, default=32768)
    parser.add_argument("--num-seqs", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    lengths = _quasi_balanced_lengths(args.total_tokens, args.num_seqs)
    inputs = _make_inputs(lengths, args.heads)
    run, outputs = _prepare_cula(inputs)

    # Compile and warm up outside the profiler range. The range below then
    # contains exactly one launch of the CUDA/CuTe bwd-intra kernel.
    run()
    torch.cuda.synchronize()

    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    try:
        run()
        torch.cuda.synchronize()
    finally:
        cudart.cudaProfilerStop()

    checksum = sum(output.float().sum().item() for output in outputs)
    capability = torch.cuda.get_device_capability()
    print(
        f"profiled H={args.heads} T={sum(lengths)} N={len(lengths)} SM{capability[0]}{capability[1]} checksum={checksum:.6e}"
    )


if __name__ == "__main__":
    main()
