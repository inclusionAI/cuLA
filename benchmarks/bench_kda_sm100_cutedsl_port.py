# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the bitwise-aligned SM100 CuTeDSL KDA port against csrc."""

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from fla.ops.utils import prepare_chunk_indices

import cula.cudac as cula_cuda
from benchmarks.utils import triton_bench_fn
from cula.kda.chunk_intra import chunk_kda_fwd_intra
from cula.ops.kda.sm100.intra_fused import chunk_kda_fwd_intra_sm100_from_gk


def _make_inputs(batch, length, h_qk, h_v, dim, beta_bf16):
    generator = torch.Generator(device="cuda").manual_seed(42)
    q = torch.randn(batch, length, h_qk, dim, dtype=torch.bfloat16, device="cuda", generator=generator) * 0.1
    k = torch.randn(batch, length, h_qk, dim, dtype=torch.bfloat16, device="cuda", generator=generator) * 0.1
    v = torch.randn(batch, length, h_v, dim, dtype=torch.bfloat16, device="cuda", generator=generator) * 0.1
    gk = torch.randn(batch, length, h_v, dim, dtype=torch.float32, device="cuda", generator=generator) * 0.02
    beta = torch.rand(batch, length, h_v, dtype=torch.float32, device="cuda", generator=generator)
    if beta_bf16:
        beta = beta.bfloat16()
    cu_seqlens = torch.arange(batch + 1, dtype=torch.int32, device="cuda") * length
    chunk_indices = prepare_chunk_indices(cu_seqlens, 64)
    return q, k, v, gk, beta, cu_seqlens, chunk_indices


def _run_csrc(q, k, v, gk, beta, cu_seqlens, chunk_indices, scale):
    Aqk, Akk = _run_csrc_intra(q, k, v, gk, beta, cu_seqlens, chunk_indices, scale)
    batch, length, _, dim = q.shape
    h_v = v.shape[2]
    w = torch.empty(batch, length, h_v, dim, dtype=torch.bfloat16, device=q.device)
    u = torch.empty_like(v)
    kg = torch.empty_like(w)
    cula_cuda.recompute_w_u_cuda(
        k,
        v,
        beta,
        Akk,
        gk,
        cu_seqlens,
        chunk_indices,
        w,
        u,
        kg,
        64,
        None,
        None,
    )
    return w, u, None, kg, Aqk, Akk


def _run_csrc_intra(q, k, v, gk, beta, cu_seqlens, chunk_indices, scale):
    batch, length = q.shape[:2]
    h_v = v.shape[2]
    Aqk = torch.empty(batch, length, h_v, 64, dtype=torch.bfloat16, device=q.device)
    Akk = torch.empty_like(Aqk)
    tile_counter = torch.zeros(1, dtype=torch.int32, device=q.device)
    cula_cuda.chunk_kda_fwd_intra_cuda(
        q,
        k,
        gk,
        beta,
        cu_seqlens,
        chunk_indices,
        Aqk,
        Akk,
        tile_counter,
        scale,
        64,
        True,
        False,
    )
    return Aqk, Akk


def _run_cutedsl_intra(q, k, _v, gk, beta, _cu_seqlens, _chunk_indices, scale):
    return chunk_kda_fwd_intra_sm100_from_gk(q, k, gk, beta, scale, fp32_akk_inv=False)


def _run_cutedsl(q, k, v, gk, beta, cu_seqlens, chunk_indices, scale):
    del cu_seqlens, chunk_indices
    return chunk_kda_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=gk,
        beta=beta,
        scale=scale,
        chunk_size=64,
        safe_gate=True,
    )


def _assert_bitwise(reference, actual):
    names = ("w", "u", "qg", "kg", "Aqk", "Akk")
    for name, expected, observed in zip(names, reference, actual):
        if expected is None:
            if observed is not None:
                raise AssertionError(f"{name}: expected None")
        elif not torch.equal(expected, observed):
            mismatches = torch.count_nonzero(expected != observed).item()
            raise AssertionError(f"{name}: {mismatches} bitwise mismatches")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads-qk", type=int, default=64)
    parser.add_argument("--heads-v", type=int, default=64)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--beta-bf16", action="store_true")
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 4096, 8192])
    args = parser.parse_args()
    if args.heads_v % args.heads_qk:
        parser.error("--heads-v must be a multiple of --heads-qk")

    scale = args.dim**-0.5
    print(
        f"{'T':>8} {'csrc intra':>11} {'DSL intra':>11} {'intra x':>9} "
        f"{'csrc full':>11} {'DSL full':>11} {'full x':>9} {'bitwise':>10}"
    )
    for length in args.lengths:
        inputs = _make_inputs(
            args.batch,
            length,
            args.heads_qk,
            args.heads_v,
            args.dim,
            args.beta_bf16,
        )
        csrc = _run_csrc(*inputs, scale)
        cutedsl = _run_cutedsl(*inputs, scale)
        _assert_bitwise(csrc, cutedsl)
        csrc_intra = _run_csrc_intra(*inputs, scale)
        cutedsl_intra = _run_cutedsl_intra(*inputs, scale)
        for name, expected, observed in zip(("Aqk", "Akk"), csrc_intra, cutedsl_intra):
            if not torch.equal(expected, observed):
                mismatches = torch.count_nonzero(expected != observed).item()
                raise AssertionError(f"fused {name}: {mismatches} bitwise mismatches")

        csrc_intra_ms = triton_bench_fn(lambda: _run_csrc_intra(*inputs, scale))
        cutedsl_intra_ms = triton_bench_fn(lambda: _run_cutedsl_intra(*inputs, scale))
        csrc_ms = triton_bench_fn(lambda: _run_csrc(*inputs, scale))
        cutedsl_ms = triton_bench_fn(lambda: _run_cutedsl(*inputs, scale))
        print(
            f"{length:8d} {csrc_intra_ms:11.4f} {cutedsl_intra_ms:11.4f} "
            f"{csrc_intra_ms / cutedsl_intra_ms:9.3f} {csrc_ms:11.4f} {cutedsl_ms:11.4f} "
            f"{csrc_ms / cutedsl_ms:9.3f} {'yes':>10}"
        )


if __name__ == "__main__":
    main()
