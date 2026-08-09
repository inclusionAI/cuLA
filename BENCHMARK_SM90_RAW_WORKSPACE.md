# SM90 Raw Workspace Transport A/B

This benchmark isolates the FlashKDA workspace-transport change from
intracard context parallelism.  Both runs use the same process configuration,
inputs, and GPU; only the cuLA checkout changes.

| Item | Value |
|---|---|
| Baseline | `d78f0075c69135fe191358fe8daf10e7383009d4` |
| Optimized | `4e67e4f55d241af600235898b55cde09bd90c165` |
| GPU | NVIDIA H200, SM90, 143,771 MiB device memory |
| Driver / CUDA | 570.148.08 / CUDA 13.0.48 |
| PyTorch | `2.8.0a0+34c6371d24.nv25.8` |
| CuTeDSL | `nvidia-cutlass-dsl==4.6.1` |
| FLA | `flash-linear-attention==0.5.0` |
| Inputs | BF16 Q/K/V/gate, FP32 state, H=64, D=128, `safe_gate=True` |
| CP mode | Serial K1+K2 (`use_intracard_cp=None`, planner resolves to a trivial plan) |
| Timing | 25 warmup + 100 measured iterations, IQR-mean CUDA events |
| Command | `python benchmarks/bench_kda_sm90_prefill.py --mode both` |

The test machine is an **H200**, and these results belong to the H200 SM90
benchmark set.

## Fixed length

| B | T | Baseline cuLA (ms) | Optimized cuLA (ms) | A/B |
|---:|---:|---:|---:|---:|
| 1 | 512 | 0.1241 | 0.1276 | 0.973x |
| 1 | 1024 | 0.1831 | 0.1583 | **1.157x** |
| 1 | 4096 | 0.6480 | 0.5704 | **1.136x** |
| 1 | 8192 | 1.2716 | 1.1108 | **1.145x** |
| 1 | 16384 | 2.5193 | 2.2082 | **1.141x** |
| 2 | 512 | 0.1365 | 0.1294 | **1.055x** |
| 2 | 1024 | 0.2431 | 0.1972 | **1.233x** |
| 2 | 4096 | 0.8974 | 0.7174 | **1.251x** |
| 2 | 8192 | 1.7648 | 1.4348 | **1.230x** |
| 2 | 16384 | 3.5085 | 2.8760 | **1.220x** |

## Variable length

| Distribution | Sequences / total T | Baseline cuLA (ms) | Optimized cuLA (ms) | A/B |
|---|---:|---:|---:|---:|
| uniform | 10 / 4096 | 0.4746 | 0.3842 | **1.235x** |
| random | 10 / 4096 | 0.5537 | 0.4550 | **1.217x** |
| skewed | 10 / 4096 | 0.6640 | 0.5754 | **1.154x** |
| uniform | 20 / 4096 | 0.4621 | 0.3752 | **1.232x** |
| random | 20 / 4096 | 0.5487 | 0.4406 | **1.245x** |
| skewed | 20 / 4096 | 0.6717 | 0.5904 | **1.138x** |
| uniform | 10 / 8192 | 0.9028 | 0.7349 | **1.229x** |
| random | 10 / 8192 | 1.0406 | 0.8582 | **1.213x** |
| skewed | 10 / 8192 | 1.2627 | 1.0924 | **1.156x** |
| uniform | 20 / 8192 | 0.8594 | 0.6886 | **1.248x** |
| random | 20 / 8192 | 1.0135 | 0.8260 | **1.227x** |
| skewed | 20 / 8192 | 1.2605 | 1.1314 | **1.114x** |
| uniform | 10 / 16384 | 1.7411 | 1.4488 | **1.202x** |
| random | 10 / 16384 | 2.0374 | 1.6995 | **1.199x** |
| skewed | 10 / 16384 | 2.4653 | 2.1679 | **1.137x** |
| uniform | 20 / 16384 | 1.6536 | 1.3662 | **1.210x** |
| random | 20 / 16384 | 1.9532 | 1.5986 | **1.222x** |
| skewed | 20 / 16384 | 2.4597 | 2.2102 | **1.113x** |

## Summary and numerical accuracy

- 27/28 configurations are faster; the only regression is B=1, T=512,
  where the difference is within small-kernel launch noise.
- Geometric-mean A/B speedup is **1.1778x**; summing all measured cuLA
  latencies gives a **15.45%** reduction.
- The optimized run reports `relative_rms_error` 0.004573–0.004924,
  `rel_max` 0.007772–0.015000, and `mean_diff` 1.1e-5–1.3e-5 against
  FLA.  The baseline run reports the same accuracy values to the printed
  precision, so the raw byte transport does not change numerical behavior.
- Flash-Flash-KDA reports isolated H100 workspace-transport reductions of
  23%, 34%, and 37% for fixed, uneven packed, and uniform packed CHUNK=16
  inputs.  This cuLA result is a complete CP-off prefill A/B on a different
  GPU and workload mix; it is therefore a comparable direction-of-gain
  check, not an exact reproduction of those percentages.
