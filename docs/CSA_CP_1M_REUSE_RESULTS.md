# B300 CP8: 31 one-million-token layouts with four-layer plan reuse

All **31 historical THD distributions** were rerun on eight B300 GPUs using
cuLA `b65230095daaf2f3232f932fd6e6f085997bcab2`. Each pack contains exactly 1,048,576 tokens.
Contiguous, complete per-document zigzag and planner use the same four sets
of synthetic layer inputs. Each policy prepares one host plan and one device
workspace, then shares them across **four independent complete CSA instances**.
These are operator instances, not a connected training model.

The largest planner gap to the faster measured contiguous/zigzag policy is
**+1.798% including one preparation**;
31/31 cases are within 3%. Across one four-layer pass
of every distribution, total time including preparation is
11.40% lower than fixed contiguous
and 6.67% lower than fixed zigzag.
These are measured-suite results, not a guarantee for arbitrary THD layouts.
Subpercent differences can reflect timing variation.

## Execution and accounting

- All four forwards execute before reverse-order backwards. Each layer retains
  its own selection/output state while later layers overwrite shared scratch.
- Host planning, routing indices, native scratch and common pooling/source
  metadata are prepared once. Per-layer activations, gather buffers and results
  remain separate. Both public preparation factories are guarded against calls
  during native stack execution.
- The complete path includes ratio-4 pooling, pooled-key/attention-KV gathers,
  selection routing/restoration, sparse attention, detached teacher/KL,
  all seven gradients, FP32 reductions and runtime layout conversions.
- KDA, projections, residuals, MoE, optimizer and checkpoint recomputation are
  excluded. No activations or selected blocks are reused across layers.
- CSA H64/D576/V512; indexer H64/D128; Top-K512. BF16 operands and FP32 scores
  and reductions; nondeterministic native kernels. TF32 and BF16 reduced-
  precision GEMM reductions are disabled. No numerical kernel was changed.
- Two warmups and seven measured four-layer passes per policy. Policy order
  rotates and reverses; timing is the median of the maximum rank per sample.
  CUDA events include host launch gaps; wall samples are retained too.
- Five independent preparation samples per policy include host work, allocation,
  H2D/device index construction and completion synchronization. Their critical-
  rank median is added **once** to the measured four-layer median. This sum is
  separately measured preparation plus execution, not a single joint stopwatch
  measurement. RNG, first compilation and numerical qualification are excluded.
- The same calibrated cost profile is used for all 31 cases, with explicit
  `min_gain_ms=0`, 5% fractional margin, six chunks/rank and a 1,024-token floor.
  This reuse-oriented experiment does not change the shipped profile's 20 ms
  one-use headroom or fit a separate rule to each case.
- Native frontend source: `855ccd77ba9efed22441fd454f7cc0206a068e94`; all 3,285
  source hashes were verified. Software versions and all 126 cuLA runtime/
  benchmark source hashes are in the machine-readable report.

## Full matrix: preparation once plus four complete forward/backward uses

C = contiguous, Z = full zigzag, P = planner. Times are milliseconds. Negative
P/C or P/Z means less time. All three columns include their own one-time
preparation. Exact lengths and all per-rank timing/preparation samples are in
[the report](results/csa_cp_1m_reuse_20261117.json).

| Distribution | Docs | C (ms) | Z (ms) | P (ms) | P/C | P/Z |
|---|---:|---:|---:|---:|---:|---:|
| single | 1 | 3408.38 | 2565.73 | 2563.28 | -24.79% | -0.10% |
| uniform | 64 | 960.48 | 984.59 | 960.95 | +0.05% | -2.40% |
| near-uniform | 64 | 961.58 | 986.06 | 961.32 | -0.03% | -2.51% |
| 20s2l | 22 | 2317.41 | 1988.39 | 1980.67 | -14.53% | -0.39% |
| power-law | 7 | 2335.14 | 1850.24 | 1851.98 | -20.69% | +0.09% |
| count-uniform-256 | 256 | 810.30 | 848.38 | 810.30 | +0.00% | -4.49% |
| count-uniform-1024 | 1024 | 435.27 | 529.53 | 437.57 | +0.53% | -17.37% |
| count-near-uniform-512 | 512 | 649.78 | 711.26 | 651.49 | +0.26% | -8.40% |
| count-tiny-mix-512 | 512 | 940.31 | 989.99 | 940.95 | +0.07% | -4.95% |
| count-tiny-mix-1000 | 1000 | 1023.69 | 1102.91 | 1032.42 | +0.85% | -6.39% |
| count-mixed-256-1long | 256 | 2484.14 | 2028.07 | 1866.17 | -24.88% | -7.98% |
| count-mixed-512-2long | 512 | 1663.89 | 1698.86 | 1523.27 | -8.45% | -10.34% |
| count-mixed-1000-1long | 1000 | 2154.73 | 1953.73 | 1758.40 | -18.39% | -10.00% |
| count-mixed-512-4long | 512 | 1242.80 | 1357.53 | 1215.11 | -2.23% | -10.49% |
| count-mixed-512-2long-ascending | 512 | 1767.82 | 1651.00 | 1584.39 | -10.38% | -4.03% |
| count-mixed-512-2long-descending | 512 | 1765.87 | 1633.98 | 1550.22 | -12.21% | -5.13% |
| heldout-lognormal-257-seed20261114 | 257 | 888.56 | 926.41 | 889.28 | +0.08% | -4.01% |
| heldout-lognormal-1000-seed20261114 | 1000 | 905.68 | 1097.87 | 913.10 | +0.82% | -16.83% |
| heldout-jittered-768-seed20261114 | 768 | 510.22 | 591.78 | 512.01 | +0.35% | -13.48% |
| heldout-tiny-tail-768-seed20261114 | 768 | 1633.54 | 1534.28 | 1483.46 | -9.19% | -3.31% |
| heldout-trimodal-513-seed20261114 | 513 | 2002.32 | 1938.00 | 1782.25 | -10.99% | -8.04% |
| heldout-lognormal-257-seed20261115 | 257 | 978.27 | 1005.61 | 989.69 | +1.17% | -1.58% |
| heldout-lognormal-1000-seed20261115 | 1000 | 873.67 | 950.48 | 889.38 | +1.80% | -6.43% |
| heldout-jittered-768-seed20261115 | 768 | 508.49 | 588.62 | 509.94 | +0.28% | -13.37% |
| heldout-tiny-tail-768-seed20261115 | 768 | 1884.79 | 1727.32 | 1557.61 | -17.36% | -9.82% |
| heldout-trimodal-513-seed20261115 | 513 | 2098.40 | 1942.14 | 1786.93 | -14.84% | -7.99% |
| heldout-lognormal-257-seed20261116 | 257 | 969.86 | 998.21 | 972.45 | +0.27% | -2.58% |
| heldout-lognormal-1000-seed20261116 | 1000 | 901.28 | 1091.02 | 911.61 | +1.15% | -16.44% |
| heldout-jittered-768-seed20261116 | 768 | 508.47 | 589.02 | 509.46 | +0.20% | -13.51% |
| heldout-tiny-tail-768-seed20261116 | 768 | 2133.99 | 1897.15 | 1739.42 | -18.49% | -8.31% |
| heldout-trimodal-513-seed20261116 | 513 | 2318.20 | 2047.38 | 1880.72 | -18.87% | -8.14% |

## Preparation and steady execution

The four-layer execution below is directly measured, not extrapolated from a
single layer. Only the one-time preparation is amortized. Full preparation
includes more than the host planner, so these fractions are conservative bounds
on planning alone.

| Distribution | C steady | Z steady | P steady | C prep | Z prep | P prep | P prep / total |
|---|---:|---:|---:|---:|---:|---:|---:|
| single | 3406.09 | 2563.48 | 2554.28 | 2.29 | 2.26 | 8.99 | 0.35% |
| uniform | 958.81 | 979.28 | 959.01 | 1.67 | 5.31 | 1.94 | 0.20% |
| near-uniform | 959.83 | 980.69 | 959.33 | 1.74 | 5.37 | 1.98 | 0.21% |
| 20s2l | 2315.08 | 1984.86 | 1972.35 | 2.34 | 3.53 | 8.32 | 0.42% |
| power-law | 2332.86 | 1847.80 | 1843.61 | 2.28 | 2.44 | 8.37 | 0.45% |
| count-uniform-256 | 807.31 | 834.82 | 806.58 | 2.99 | 13.56 | 3.73 | 0.46% |
| count-uniform-1024 | 428.13 | 485.07 | 428.30 | 7.14 | 44.47 | 9.27 | 2.12% |
| count-near-uniform-512 | 645.40 | 688.03 | 645.86 | 4.39 | 23.23 | 5.64 | 0.87% |
| count-tiny-mix-512 | 934.99 | 965.57 | 934.68 | 5.32 | 24.42 | 6.27 | 0.67% |
| count-tiny-mix-1000 | 1015.25 | 1054.56 | 1013.74 | 8.45 | 48.35 | 18.68 | 1.81% |
| count-mixed-256-1long | 2480.44 | 1921.85 | 1856.42 | 3.70 | 106.22 | 9.76 | 0.52% |
| count-mixed-512-2long | 1658.97 | 1574.26 | 1509.76 | 4.93 | 124.60 | 13.51 | 0.89% |
| count-mixed-1000-1long | 2146.26 | 1799.81 | 1737.25 | 8.47 | 153.92 | 21.15 | 1.20% |
| count-mixed-512-4long | 1237.77 | 1232.24 | 1201.07 | 5.03 | 125.29 | 14.03 | 1.16% |
| count-mixed-512-2long-ascending | 1761.48 | 1624.16 | 1569.54 | 6.35 | 26.85 | 14.84 | 0.94% |
| count-mixed-512-2long-descending | 1759.69 | 1607.64 | 1535.84 | 6.18 | 26.34 | 14.38 | 0.93% |
| heldout-lognormal-257-seed20261114 | 885.48 | 912.48 | 885.42 | 3.08 | 13.93 | 3.86 | 0.43% |
| heldout-lognormal-1000-seed20261114 | 897.84 | 943.19 | 896.65 | 7.84 | 154.68 | 16.45 | 1.80% |
| heldout-jittered-768-seed20261114 | 503.74 | 556.57 | 503.97 | 6.48 | 35.21 | 8.04 | 1.57% |
| heldout-tiny-tail-768-seed20261114 | 1626.70 | 1501.05 | 1463.85 | 6.84 | 33.23 | 19.62 | 1.32% |
| heldout-trimodal-513-seed20261114 | 1996.91 | 1811.14 | 1767.01 | 5.40 | 126.86 | 15.24 | 0.86% |
| heldout-lognormal-257-seed20261115 | 975.37 | 990.92 | 979.49 | 2.90 | 14.69 | 10.19 | 1.03% |
| heldout-lognormal-1000-seed20261115 | 865.46 | 907.50 | 870.79 | 8.21 | 42.98 | 18.59 | 2.09% |
| heldout-jittered-768-seed20261115 | 501.90 | 552.96 | 501.77 | 6.59 | 35.66 | 8.17 | 1.60% |
| heldout-tiny-tail-768-seed20261115 | 1877.65 | 1577.75 | 1539.70 | 7.14 | 149.57 | 17.91 | 1.15% |
| heldout-trimodal-513-seed20261115 | 2092.00 | 1800.28 | 1771.37 | 6.40 | 141.86 | 15.57 | 0.87% |
| heldout-lognormal-257-seed20261116 | 966.61 | 984.06 | 966.19 | 3.25 | 14.15 | 6.26 | 0.64% |
| heldout-lognormal-1000-seed20261116 | 893.35 | 938.67 | 892.92 | 7.94 | 152.36 | 18.68 | 2.05% |
| heldout-jittered-768-seed20261116 | 501.89 | 553.54 | 501.43 | 6.57 | 35.48 | 8.03 | 1.58% |
| heldout-tiny-tail-768-seed20261116 | 2126.98 | 1747.55 | 1719.47 | 7.01 | 149.61 | 19.95 | 1.15% |
| heldout-trimodal-513-seed20261116 | 2312.82 | 1908.47 | 1865.58 | 5.38 | 138.91 | 15.14 | 0.81% |

Planner full preparation averages 11.696 ms per layout, or 2.924 ms per compatible layer. Its mean fraction is 1.036%, maximum 2.118%, and aggregate fraction 0.929%. These are distinct from the planner's performance gap to the fastest baseline.

## Numerical qualification and memory

Each of 31 cases checks 3 policies x 4 layers x 8 ranks = 96 layer/rank results,
for **2,976 results** and **23,808 full-tensor output/gradient comparisons**.
After later layers have used shared scratch, every layer is compared with a
fresh standalone contiguous invocation using identical inputs. No element
sampling is used: differences are computed in FP32 and statistics accumulated
in FP64 with bounded temporary storage. Only one layer's saved comparison
results are retained at a time, on device; no large tensor staging through CPU
is needed. All checks must be finite, relative RMS < 1%, and auxiliary loss
difference < 1e-5; observed maxima are below.

| Tensor | Worst relative RMS | Worst absolute difference |
|---|---:|---:|
| output | 0.161691% | 0.0046234131 |
| dq | 0.060764% | 0.0032958984 |
| dkv | 0.072408% | 0.25 |
| d_index_q | 0.087008% | 2.7284841e-12 |
| d_weights | 0.070834% | 4.9476512e-09 |
| d_pool_kv | 0.096349% | 2.910383e-11 |
| d_pool_score | 0.105392% | 1.4551915e-11 |
| d_ape | 0.001433% | 1.3727686e-11 |

Maximum global auxiliary loss difference: **3.4924597e-10**. This is the indexer's auxiliary loss, not model training loss. The contiguous self-comparison is included to expose native nondeterministic variation.

Three small 4,096-token fixtures additionally run independent FP32 attention,
teacher/KL and pooling-gradient checks for each policy/layer/rank, plus exact
routing/inverse/autograd markers. The oracle uses each selected key set; it does
not independently prove the selector's discrete ranking. All 548 focused host
tests and repository lint/format checks pass.

Timing-pass memory peaks include inputs, four layers' retained forward states,
outputs/gradients and all three prebuilt policy metadata/workspace bundles.
They exclude the separate numerical oracle pass and are **harness peaks**, not
isolated deployment peaks or workspace-only usage. Reserved memory is allocator
history and is not attributed as an individual policy's live storage.

| Policy | Peak allocated range (GiB/rank) | Peak reserved range (GiB/rank) |
|---|---:|---:|
| contiguous | 140.90–161.72 | 144.02–166.49 |
| zigzag | 140.90–161.72 | 144.02–166.49 |
| planner | 140.90–161.72 | 144.02–166.49 |

## Reproduction and relationship to earlier tables

```bash
OMP_NUM_THREADS=4 torchrun --standalone --nproc-per-node=8 \
  -m benchmarks.bench_csa_cp_reuse \
  --profile benchmarks/profiles/thd_selection_b300_cp8.json \
  --output-dir results/csa-1m-reuse --layers 4 \
  --warmups 2 --repeats 7 --prepare-repeats 5 \
  --min-gain-ms 0 --source-revision <tested-commit>
```

Use `--check-only` for the small independent FP32 fixtures. Dependencies are the
qualified cuDNN build and cuLA's hybrid extra; under CUDA 13 install CuTeDSL with
`nvidia-cutlass-dsl[cu13]`.

This full 31-case 1M rerun complements the earlier
[24-case cross-window study](CSA_CP_THREE_WAY_RESULTS.md), which contains eight
distributions at each of 128K, 256K and 1M. It supersedes reuse projections only
for this measured four-instance 1M setup. It does not relabel historical H20,
connected model, CP64 metadata-only or different-shape measurements as new runs.
