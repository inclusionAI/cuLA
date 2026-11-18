# KDA prefill (CUDA)

Fused-style **KDA (Kimi Delta Attention) prefill** for float32: two kernels (intra-chunk KKT / W–U, then inter-chunk recurrence). Targets **sm_89+** (CMake default: `89`). Layout matches [cuLA](https://github.com/inclusionAI/cuLA) KDA tensor conventions: `q/k/g` `[B,H,T,K]`, `v/o` `[B,H,T,V]`, `beta` `[B,H,T]`, chunk buffers `w/u` `[B,H,num_chunks,C,K|V]`.

Standalone **CMake** build plus **ctypes** benchmark (`src/main.py`). Tensor layout matches [cuLA](https://github.com/inclusionAI/cuLA) KDA conventions for a future port into `csrc/kda/sm90/`.

## Build

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build
```

Artifacts: `libkda_prefill.a`, `libkda_prefill_runtime.so` (C ABI: `kda_prefill_f32`).

## Python benchmark (optional FLA)

Requires PyTorch and, for FLA comparison, [flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention) (`fla`).

```bash
python src/main.py --suite fla-benchmark --B 2 --H 8 --T 4096 --K 64 --V 64 --C 32
```

- `local-benchmark` — CUDA only  
- `fla-only-benchmark` — FLA only  
- `fla-benchmark` — both + quick accuracy check vs `chunk_kda`

## Tuning env (optional)

| Variable | Role |
|----------|------|
| `KDA_INTRA_THREADS` | Block size for intra-chunk kernel |
| `KDA_INTER_THREADS` | Block size for inter-chunk kernel |
| `KDA_INTER_SHARDS` | V-way sharding for inter kernel (must divide `V`) |

## Benchmark vs FLA (flash-linear-attention)

**Hardware:** NVIDIA **RTX 4070**, **sm_89** (Ada). CMake built with `CMAKE_CUDA_ARCHITECTURES=89`.

**Setup:** `K=V=64`, chunk `C=32`. Local path is **float32**; FLA `chunk_kda` in `src/main.py` runs **bf16** — timing numbers are not apples-to-apples on dtype, but useful as a rough baseline.

Latest sweep (machine‑logged) is under [`analysis/ncu/20260409-193656-large-shape-compare/`](analysis/ncu/20260409-193656-large-shape-compare/README.md): summary table in that folder’s README, raw CSV [`benchmark_results.csv`](analysis/ncu/20260409-193656-large-shape-compare/benchmark_results.csv), console captures in [`benchmarks/`](analysis/ncu/20260409-193656-large-shape-compare/benchmarks/).

| Shape (B,H,T) | iters | Local ms | FLA ms | Local / FLA |
|---------------|------:|---------:|-------:|------------:|
| (2,8,4096) | 10 | 1.8884 | 1.1918 | 1.58× |
| (2,8,8192) | 10 | 3.6517 | 3.1311 | 1.17× |
| (4,8,4096) | 10 | 3.8848 | 3.0597 | 1.27× |
| (2,16,4096) | 10 | 3.8868 | 3.1079 | 1.25× |
| (2,8,16384) | 5 | 7.2716 | 6.5675 | 1.11× |

Quick accuracy line from the same harness (local fp32 vs FLA bf16): `max_abs ≈ 4.09`, `mean_abs ≈ 0.80` on the small validation shape inside `src/main.py` (see folder README).

## Profiling artifacts

`analysis/ncu/20260409-193656-large-shape-compare/` holds the FLA comparison CSV, console logs under `benchmarks/`, and a short README. Optional for builds.

## Supported shapes

Instantiated in `src/kda.cu`: `(K,V,C)` in `(64,64,64)`, `(64,64,32)`, `(128,128,64)`, `(128,128,32)`.
