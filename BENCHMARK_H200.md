# Benchmark Results — Hopper (SM90)

> Measured on 2026-08-12.

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
| 1 | 512 | 0.9405 | 0.1244 | **7.56x** |
| 1 | 1024 | 0.9372 | 0.1579 | **5.94x** |
| 1 | 4096 | 0.9344 | 0.5701 | **1.64x** |
| 1 | 8192 | 1.7701 | 1.1150 | **1.59x** |
| 1 | 16384 | 3.4846 | 2.2266 | **1.56x** |
| 2 | 512 | 0.9274 | 0.1262 | **7.35x** |
| 2 | 1024 | 0.9300 | 0.1971 | **4.72x** |
| 2 | 4096 | 1.7628 | 0.7223 | **2.44x** |
| 2 | 8192 | 3.4622 | 1.4306 | **2.42x** |
| 2 | 16384 | 6.8872 | 2.8817 | **2.39x** |

### Variable-Length

| Config | FLA Triton (ms) | cuLA FlashKDA (ms) | Speedup |
|---|---:|---:|---:|
| uniform 10 seqs, T=4096, 409–415 tokens | 0.9819 | 0.3802 | **2.58x** |
| random 10 seqs, T=4096, 24–1201 tokens | 0.9749 | 0.4574 | **2.13x** |
| skewed 10 seqs, T=4096, 227–2053 tokens | 0.9730 | 0.5753 | **1.69x** |
| uniform 20 seqs, T=4096, 204–220 tokens | 1.0478 | 0.3768 | **2.78x** |
| random 20 seqs, T=4096, 5–787 tokens | 1.0242 | 0.4426 | **2.31x** |
| skewed 20 seqs, T=4096, 107–2063 tokens | 0.9984 | 0.5940 | **1.68x** |
| uniform 10 seqs, T=8192, 819–821 tokens | 1.7868 | 0.7329 | **2.44x** |
| random 10 seqs, T=8192, 48–2401 tokens | 1.8201 | 0.8595 | **2.12x** |
| skewed 10 seqs, T=8192, 455–4097 tokens | 1.8388 | 1.0881 | **1.69x** |
| uniform 20 seqs, T=8192, 409–421 tokens | 1.8781 | 0.6879 | **2.73x** |
| random 20 seqs, T=8192, 9–1574 tokens | 1.8714 | 0.8146 | **2.30x** |
| skewed 20 seqs, T=8192, 215–4107 tokens | 1.8815 | 1.1248 | **1.67x** |
| uniform 10 seqs, T=16384, 1638–1642 tokens | 3.5032 | 1.4460 | **2.42x** |
| random 10 seqs, T=16384, 95–4802 tokens | 3.5105 | 1.6967 | **2.07x** |
| skewed 10 seqs, T=16384, 910–8194 tokens | 3.5259 | 2.1618 | **1.63x** |
| uniform 20 seqs, T=16384, 819–823 tokens | 3.5272 | 1.3552 | **2.60x** |
| random 20 seqs, T=16384, 19–3147 tokens | 3.5600 | 1.5926 | **2.24x** |
| skewed 20 seqs, T=16384, 431–8195 tokens | 3.5296 | 2.2084 | **1.60x** |

Across all 28 fixed-length and variable-length configs, the arithmetic mean of
the per-row speedups is **2.72x** over FLA (geometric mean **2.43x**;
minimum **1.56x**, maximum **7.56x**).

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
| T=1023 | 4 | no | 0.1308 | 0.1456 | 0.90x |
| T=1025 | 4 | no | 0.1312 | 0.1362 | 0.96x |
| T=4K | 4 | yes | 0.4070 | 0.2105 | **1.93x** |
| T=8K | 4 | yes | 0.8141 | 0.2659 | **3.06x** |
| T=16K | 4 | yes | 1.6149 | 0.3792 | **4.26x** |
| T=32K | 4 | yes | 3.2087 | 0.5455 | **5.88x** |
| T=64K | 4 | yes | 6.3907 | 0.8634 | **7.40x** |
| T=64K+1 | 4 | yes | 7.3199 | 0.9349 | **7.83x** |
| 2x16K | 4 | yes | 1.6487 | 0.4960 | **3.32x** |
| 32K+4K | 4 | yes | 3.2272 | 0.6156 | **5.24x** |
| 32K+1K | 4 | yes | 3.2135 | 0.5434 | **5.91x** |
| 32K+1023+1025 | 4 | yes | 3.6700 | 0.6341 | **5.79x** |
| 64K+1K | 4 | yes | 6.4081 | 0.9721 | **6.59x** |
| 64K+2x1K | 4 | yes | 6.4054 | 0.9865 | **6.49x** |
| 64K+5x1K | 4 | yes | 6.4184 | 1.0059 | **6.38x** |
| 64K+1+1023+1025 | 4 | yes | 7.3167 | 1.0531 | **6.95x** |
| T=1023 | 8 | no | 0.1326 | 0.1361 | 0.97x |
| T=1025 | 8 | no | 0.1335 | 0.1362 | 0.98x |
| T=4K | 8 | yes | 0.4246 | 0.2174 | **1.95x** |
| T=8K | 8 | yes | 0.8376 | 0.2854 | **2.94x** |
| T=16K | 8 | yes | 1.6496 | 0.4520 | **3.65x** |
| T=32K | 8 | yes | 3.2839 | 0.7789 | **4.22x** |
| T=64K | 8 | yes | 6.5424 | 1.4323 | **4.57x** |
| T=64K+1 | 8 | yes | 7.4745 | 1.5786 | **4.73x** |
| 2x16K | 8 | yes | 1.7236 | 0.7318 | **2.36x** |
| 32K+4K | 8 | yes | 3.3146 | 0.9499 | **3.49x** |
| 32K+1K | 8 | yes | 3.2958 | 0.8917 | **3.70x** |
| 32K+1023+1025 | 8 | yes | 3.7542 | 0.9980 | **3.76x** |
| 64K+1K | 8 | yes | 6.5675 | 1.6512 | **3.98x** |
| 64K+2x1K | 8 | yes | 6.5733 | 1.6640 | **3.95x** |
| 64K+5x1K | 8 | yes | 6.5782 | 1.7464 | **3.77x** |
| 64K+1+1023+1025 | 8 | yes | 7.4786 | 1.7998 | **4.16x** |

Intracard CP engages for 28 of the 32 configs. On the engaged subset it
delivers a **4.29x geometric-mean** speedup, with a range of **1.93–7.83x**.
The non-CHUNK-aligned `T=64K+1` cases retain essentially the same benefit as
`T=64K`; the four 1023/1025-token rows stay on the serial path and expose only
the auto-planner overhead.

To reproduce:

```bash
python benchmarks/bench_kda_sm90_cp.py
```
