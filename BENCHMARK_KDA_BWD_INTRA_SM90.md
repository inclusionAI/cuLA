# Benchmark Results — KDA Backward Intra (SM90)

> Measured on 2026-08-14.

> **GPU:** NVIDIA L20X, compute capability 9.0  |  **CUDA:** 12.9  |
> **PyTorch:** 2.9.1+cu129  |  **Triton:** 3.5.1

This report compares cuLA's persistent CUDA C++ KDA intra-chunk backward
kernel with the Triton implementation in flash-linear-attention (FLA). The
kernel uses warp-level `mma.sync.m16n8k8`, BF16 Q/K/beta inputs, FP32 gate and
gradient inputs, head dimension 128, and chunk size 64.

Each configuration uses `triton.testing.do_bench` with 25 ms of warmup and a
100 ms measurement window. The table reports the median latency. The Triton
cache was cleared before each FLA version was measured. Variable-length cases
contain eight quasi-balanced sequences with the indicated total token count.

## FLA v0.5.0

| Configuration | cuLA CUDA (ms) | FLA Triton (ms) | Speedup |
|---|---:|---:|---:|
| H=32, uniform, T=8192, N=1 | 0.502 | 0.808 | **1.61x** |
| H=32, uniform, T=32768, N=1 | 1.850 | 3.198 | **1.73x** |
| H=32, varlen, T=8192, N=8 | 0.505 | 0.813 | **1.61x** |
| H=32, varlen, T=32768, N=8 | 1.842 | 3.190 | **1.73x** |
| H=64, uniform, T=8192, N=1 | 0.952 | 1.612 | **1.69x** |
| H=64, uniform, T=32768, N=1 | 3.631 | 6.398 | **1.76x** |
| H=64, varlen, T=8192, N=8 | 0.949 | 1.610 | **1.70x** |
| H=64, varlen, T=32768, N=8 | 3.630 | 6.413 | **1.77x** |

The geometric-mean speedup over FLA v0.5.0 is **1.699x**.

## FLA v0.4.2

FLA v0.4.2 is about 4.5–5.5% faster than v0.5.0 for these shapes. The cuLA
kernel latency remains effectively unchanged, so its geometric-mean speedup is
lower against this baseline.

| Configuration | cuLA CUDA (ms) | FLA Triton (ms) | Speedup |
|---|---:|---:|---:|
| H=32, uniform, T=8192, N=1 | 0.503 | 0.768 | **1.53x** |
| H=32, uniform, T=32768, N=1 | 1.851 | 3.035 | **1.64x** |
| H=32, varlen, T=8192, N=8 | 0.506 | 0.776 | **1.53x** |
| H=32, varlen, T=32768, N=8 | 1.852 | 3.046 | **1.64x** |
| H=64, uniform, T=8192, N=1 | 0.952 | 1.524 | **1.60x** |
| H=64, uniform, T=32768, N=1 | 3.636 | 6.071 | **1.67x** |
| H=64, varlen, T=8192, N=8 | 0.951 | 1.534 | **1.61x** |
| H=64, varlen, T=32768, N=8 | 3.627 | 6.086 | **1.68x** |

The geometric-mean speedup over FLA v0.4.2 is **1.612x**.

## Correctness

The SM90 correctness suite compares `dq`, `dk`, `db`, and `dg` against FLA for
fixed-length, ragged variable-length, and dense-batch inputs. It also checks
deterministic output, dispatcher behavior, device validation, and the
unsupported-beta fallback. All nine tests pass on SM90.

## Reproduction

```bash
python benchmarks/bench_kda_bwd_intra_sm90.py \
  --heads 32 64 \
  --warmup 25 \
  --rep 100

python -m pytest tests/test_kda_sm90_bwd_intra.py -v
```

The repository pins FLA v0.5.0. To reproduce the v0.4.2 comparison, check out
the `v0.4.2` tag in `third_party/flash-linear-attention` and reinstall that
submodule in editable mode before running the same command.
