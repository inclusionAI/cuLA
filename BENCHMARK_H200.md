# Benchmark Results — Hopper (SM90)

> Measured on 2026-08-09.

> **GPU:** NVIDIA H200 141GB, 132 SMs  |  **CUDA:** 12.9  |  **PyTorch:** 2.9.1+cu129  |  **Triton:** 3.5.1

> FLA baseline: [flash-linear-attention v0.5.0](https://github.com/fla-org/flash-linear-attention/releases/tag/v0.5.0)

These results use cuLA's CuTe DSL FlashKDA backend. They supersede the earlier
numbers for the legacy CUTLASS C++ fused backend.

## FlashKDA Prefill vs FLA

CuTe DSL FlashKDA runs a two-kernel K1+K2 prefill pipeline. The benchmark uses
BF16 inputs, `H=64`, `D=128`, `safe_gate=True`, no initial state, 25 warmup
iterations, and 100 measured iterations aggregated with the IQR mean.

### Fixed-Length

| B | T | FLA Triton (ms) | cuLA FlashKDA (ms) | Speedup |
|---|---:|---:|---:|---:|
| 1 | 512 | 1.0013 | 0.1284 | **7.80x** |
| 1 | 1024 | 0.9747 | 0.1832 | **5.32x** |
| 1 | 4096 | 0.9994 | 0.6537 | **1.53x** |
| 1 | 8192 | 1.7774 | 1.2831 | **1.39x** |
| 1 | 16384 | 3.5038 | 2.5421 | **1.38x** |
| 2 | 512 | 0.9855 | 0.1386 | **7.11x** |
| 2 | 1024 | 1.0283 | 0.2447 | **4.20x** |
| 2 | 4096 | 1.7690 | 0.9010 | **1.96x** |
| 2 | 8192 | 3.4866 | 1.7810 | **1.96x** |
| 2 | 16384 | 6.9413 | 3.5363 | **1.96x** |

### Variable-Length

| Config | FLA Triton (ms) | cuLA FlashKDA (ms) | Speedup |
|---|---:|---:|---:|
| uniform 10 seqs, T=4096, 409–415 tokens | 1.0295 | 0.4748 | **2.17x** |
| random 10 seqs, T=4096, 24–1201 tokens | 1.0378 | 0.5544 | **1.87x** |
| skewed 10 seqs, T=4096, 227–2053 tokens | 1.0126 | 0.6700 | **1.51x** |
| uniform 20 seqs, T=4096, 204–220 tokens | 1.0510 | 0.4639 | **2.27x** |
| random 20 seqs, T=4096, 5–787 tokens | 1.0250 | 0.5513 | **1.86x** |
| skewed 20 seqs, T=4096, 107–2063 tokens | 1.0048 | 0.6767 | **1.48x** |
| uniform 10 seqs, T=8192, 819–821 tokens | 1.7923 | 0.9051 | **1.98x** |
| random 10 seqs, T=8192, 48–2401 tokens | 1.8277 | 1.0525 | **1.74x** |
| skewed 10 seqs, T=8192, 455–4097 tokens | 1.8457 | 1.2728 | **1.45x** |
| uniform 20 seqs, T=8192, 409–421 tokens | 1.8855 | 0.8628 | **2.19x** |
| random 20 seqs, T=8192, 9–1574 tokens | 1.8785 | 1.0189 | **1.84x** |
| skewed 20 seqs, T=8192, 215–4107 tokens | 1.8875 | 1.2913 | **1.46x** |
| uniform 10 seqs, T=16384, 1638–1642 tokens | 3.5091 | 1.7505 | **2.00x** |
| random 10 seqs, T=16384, 95–4802 tokens | 3.5175 | 2.0473 | **1.72x** |
| skewed 10 seqs, T=16384, 910–8194 tokens | 3.5350 | 2.4892 | **1.42x** |
| uniform 20 seqs, T=16384, 819–823 tokens | 3.5335 | 1.6558 | **2.13x** |
| random 20 seqs, T=16384, 19–3147 tokens | 3.5694 | 1.9728 | **1.81x** |
| skewed 20 seqs, T=16384, 431–8195 tokens | 3.5403 | 2.4949 | **1.42x** |

Across all 28 fixed-length and variable-length configs, FlashKDA averages
**2.39x** over FLA (minimum **1.38x**, maximum **7.80x**).

To reproduce:

```bash
python benchmarks/bench_kda_sm90_prefill.py --mode both
```

## Intracard Context Parallelism

The following comparison uses the same FlashKDA binary with
`use_intracard_cp="auto"` and `use_intracard_cp=False`. It covers the long
single-sequence and ragged packed-sequence shapes where intracard CP fills the
H200's SM array more effectively. Each row uses BF16, `D=128`,
`safe_gate=True`, 10 warmup iterations, and 100 measured iterations.

| Config | H | CP engaged | CP off (ms) | CP auto (ms) | Speedup |
|---|---:|:---:|---:|---:|---:|
| T=1023 | 4 | no | 0.1322 | 0.1447 | 0.91x |
| T=1025 | 4 | no | 0.1364 | 0.1388 | 0.98x |
| T=4K | 4 | yes | 0.4374 | 0.2163 | **2.02x** |
| T=8K | 4 | yes | 0.8636 | 0.2820 | **3.06x** |
| T=16K | 4 | yes | 1.7099 | 0.4078 | **4.19x** |
| T=32K | 4 | yes | 3.3888 | 0.5973 | **5.67x** |
| T=64K | 4 | yes | 6.7518 | 0.9658 | **6.99x** |
| T=64K+1 | 4 | yes | 7.3033 | 1.0126 | **7.21x** |
| 2x16K | 4 | yes | 1.7631 | 0.5436 | **3.24x** |
| 32K+4K | 4 | yes | 3.4000 | 0.6682 | **5.09x** |
| 32K+1K | 4 | yes | 3.3923 | 0.6046 | **5.61x** |
| 32K+1023+1025 | 4 | yes | 3.6605 | 0.6821 | **5.37x** |
| 64K+1K | 4 | yes | 6.7713 | 1.0644 | **6.36x** |
| 64K+2x1K | 4 | yes | 6.7412 | 1.0803 | **6.24x** |
| 64K+5x1K | 4 | yes | 6.7648 | 1.1064 | **6.11x** |
| 64K+1+1023+1025 | 4 | yes | 7.3010 | 1.1200 | **6.52x** |
| T=1023 | 8 | no | 0.1365 | 0.1395 | 0.98x |
| T=1025 | 8 | no | 0.1376 | 0.1391 | 0.99x |
| T=4K | 8 | yes | 0.4557 | 0.2356 | **1.93x** |
| T=8K | 8 | yes | 0.8941 | 0.3177 | **2.81x** |
| T=16K | 8 | yes | 1.7600 | 0.5132 | **3.43x** |
| T=32K | 8 | yes | 3.4958 | 0.8774 | **3.98x** |
| T=64K | 8 | yes | 6.9656 | 1.6239 | **4.29x** |
| T=64K+1 | 8 | yes | 7.5212 | 1.7221 | **4.37x** |
| 2x16K | 8 | yes | 1.8764 | 0.8328 | **2.25x** |
| 32K+4K | 8 | yes | 3.5256 | 1.0458 | **3.37x** |
| 32K+1K | 8 | yes | 3.5080 | 0.9871 | **3.55x** |
| 32K+1023+1025 | 8 | yes | 3.7879 | 1.0731 | **3.53x** |
| 64K+1K | 8 | yes | 6.9812 | 1.8345 | **3.81x** |
| 64K+2x1K | 8 | yes | 6.9297 | 1.8368 | **3.77x** |
| 64K+5x1K | 8 | yes | 6.9738 | 1.9318 | **3.61x** |
| 64K+1+1023+1025 | 8 | yes | 7.4816 | 1.9172 | **3.90x** |

Intracard CP engages for 28 of the 32 configs. On the engaged subset it
delivers a **4.11x geometric-mean** speedup, with a range of **1.93–7.21x**.
The non-CHUNK-aligned `T=64K+1` cases retain essentially the same benefit as
`T=64K`; the four 1023/1025-token rows stay on the serial path and expose only
the auto-planner overhead.

To reproduce:

```bash
python benchmarks/bench_kda_sm90_cp.py
```
