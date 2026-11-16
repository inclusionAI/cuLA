# B300 CP8: planner cost refinement with four-layer plan reuse

All **31 historical THD distributions** were rerun on eight B300 GPUs using
cuLA `932b3b4b9e00a05ba4f93aab53f229953e7b5eb1`. Each pack contains exactly 1,048,576 tokens.
Contiguous, complete per-document zigzag and planner use the same four sets
of synthetic layer inputs. Each policy prepares one host plan and one device
workspace, then shares them across **four independent complete CSA instances**.
These are operator instances, not a connected training model.

Across all 31 cases, the **steady four-layer execution** takes 11.89% less time
than fixed contiguous and 3.00% less than fixed zigzag. The largest gap to the
faster measured contiguous/zigzag policy is **+0.133%**.

With one preparation included, the original three-sample medians give
**+1.070%** worst gap, 31/31 within 3%, and aggregate reductions of 11.40% / 6.41%
versus contiguous / zigzag. Preparation has real Python GC tails: the follow-up
below measures 32 preparations per policy in three cases and explains why
these median-based totals are **not a tail-latency guarantee**. In particular,
part of the aggregate advantage over zigzag comes from its preparation cost.
These are measured-suite results, not a guarantee for arbitrary THD layouts.
Subpercent differences can reflect timing variation.

The [preceding seven-sample cohort](CSA_CP_1M_REUSE_RESULTS.md) is preserved.
This cohort uses the refined transport profile and fewer timing repetitions;
compare policies within each cohort rather than attributing small differences
between separate runs to the code change.

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
- One warmup and three measured four-layer passes per policy. Policy order
  rotates and reverses; timing is the median of the maximum rank per sample.
  CUDA events include host launch gaps; wall samples are retained too.
- Three independent preparation samples per policy include host work, allocation,
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
[the report](results/csa_cp_1m_local_copy_20261117.json).

| Distribution | Docs | C (ms) | Z (ms) | P (ms) | P/C | P/Z |
|---|---:|---:|---:|---:|---:|---:|
| single | 1 | 3410.11 | 2568.86 | 2559.61 | -24.94% | -0.36% |
| uniform | 64 | 958.68 | 983.30 | 958.89 | +0.02% | -2.48% |
| near-uniform | 64 | 959.76 | 982.61 | 959.82 | +0.01% | -2.32% |
| 20s2l | 22 | 2313.52 | 1984.73 | 1984.67 | -14.21% | -0.00% |
| power-law | 7 | 2334.34 | 1846.61 | 1851.27 | -20.69% | +0.25% |
| count-uniform-256 | 256 | 808.35 | 845.48 | 809.04 | +0.09% | -4.31% |
| count-uniform-1024 | 1024 | 434.86 | 530.76 | 437.84 | +0.68% | -17.51% |
| count-near-uniform-512 | 512 | 649.82 | 711.25 | 650.88 | +0.16% | -8.49% |
| count-tiny-mix-512 | 512 | 938.00 | 989.90 | 939.29 | +0.14% | -5.11% |
| count-tiny-mix-1000 | 1000 | 1020.99 | 1100.23 | 1029.34 | +0.82% | -6.44% |
| count-mixed-256-1long | 256 | 2465.16 | 1935.22 | 1865.86 | -24.31% | -3.58% |
| count-mixed-512-2long | 512 | 1663.37 | 1701.67 | 1521.85 | -8.51% | -10.57% |
| count-mixed-1000-1long | 1000 | 2154.67 | 1961.06 | 1757.63 | -18.43% | -10.37% |
| count-mixed-512-4long | 512 | 1241.58 | 1368.70 | 1214.86 | -2.15% | -11.24% |
| count-mixed-512-2long-ascending | 512 | 1763.66 | 1756.38 | 1583.82 | -10.20% | -9.82% |
| count-mixed-512-2long-descending | 512 | 1764.81 | 1754.52 | 1549.50 | -12.20% | -11.69% |
| heldout-lognormal-257-seed20261114 | 257 | 886.97 | 925.62 | 888.61 | +0.18% | -4.00% |
| heldout-lognormal-1000-seed20261114 | 1000 | 903.79 | 987.89 | 909.90 | +0.68% | -7.89% |
| heldout-jittered-768-seed20261114 | 768 | 510.60 | 591.99 | 511.75 | +0.23% | -13.55% |
| heldout-tiny-tail-768-seed20261114 | 768 | 1631.93 | 1535.97 | 1480.08 | -9.31% | -3.64% |
| heldout-trimodal-513-seed20261114 | 513 | 2001.85 | 1950.13 | 1782.77 | -10.94% | -8.58% |
| heldout-lognormal-257-seed20261115 | 257 | 977.33 | 1004.54 | 984.40 | +0.72% | -2.00% |
| heldout-lognormal-1000-seed20261115 | 1000 | 873.44 | 948.82 | 882.79 | +1.07% | -6.96% |
| heldout-jittered-768-seed20261115 | 768 | 506.62 | 588.45 | 508.23 | +0.32% | -13.63% |
| heldout-tiny-tail-768-seed20261115 | 768 | 1880.67 | 1606.93 | 1556.95 | -17.21% | -3.11% |
| heldout-trimodal-513-seed20261115 | 513 | 2097.85 | 1945.74 | 1784.15 | -14.95% | -8.30% |
| heldout-lognormal-257-seed20261116 | 257 | 968.09 | 994.82 | 970.45 | +0.24% | -2.45% |
| heldout-lognormal-1000-seed20261116 | 1000 | 901.17 | 983.59 | 910.15 | +1.00% | -7.47% |
| heldout-jittered-768-seed20261116 | 768 | 508.12 | 590.11 | 508.81 | +0.14% | -13.78% |
| heldout-tiny-tail-768-seed20261116 | 768 | 2132.37 | 1903.73 | 1734.60 | -18.65% | -8.88% |
| heldout-trimodal-513-seed20261116 | 513 | 2316.14 | 2050.95 | 1876.05 | -19.00% | -8.53% |

## Preparation and steady execution

The four-layer execution below is directly measured, not extrapolated from a
single layer. Only the one-time preparation is amortized. Full preparation
includes more than the host planner, so these fractions are conservative bounds
on planning alone.

| Distribution | C steady | Z steady | P steady | C prep | Z prep | P prep | P prep / total |
|---|---:|---:|---:|---:|---:|---:|---:|
| single | 3407.71 | 2566.47 | 2551.33 | 2.40 | 2.39 | 8.28 | 0.32% |
| uniform | 956.95 | 977.91 | 956.92 | 1.73 | 5.39 | 1.96 | 0.20% |
| near-uniform | 958.16 | 977.91 | 957.99 | 1.60 | 4.71 | 1.83 | 0.19% |
| 20s2l | 2311.27 | 1981.56 | 1976.79 | 2.25 | 3.16 | 7.88 | 0.40% |
| power-law | 2332.07 | 1844.34 | 1842.93 | 2.27 | 2.27 | 8.33 | 0.45% |
| count-uniform-256 | 805.76 | 832.54 | 805.87 | 2.59 | 12.93 | 3.17 | 0.39% |
| count-uniform-1024 | 427.78 | 485.18 | 428.35 | 7.08 | 45.58 | 9.49 | 2.17% |
| count-near-uniform-512 | 645.40 | 687.29 | 645.43 | 4.43 | 23.95 | 5.45 | 0.84% |
| count-tiny-mix-512 | 933.48 | 965.56 | 933.54 | 4.52 | 24.34 | 5.75 | 0.61% |
| count-tiny-mix-1000 | 1013.21 | 1052.97 | 1011.69 | 7.78 | 47.27 | 17.66 | 1.72% |
| count-mixed-256-1long | 2462.01 | 1919.68 | 1856.24 | 3.15 | 15.54 | 9.62 | 0.52% |
| count-mixed-512-2long | 1658.72 | 1572.71 | 1508.51 | 4.65 | 128.96 | 13.33 | 0.88% |
| count-mixed-1000-1long | 2145.60 | 1800.81 | 1736.11 | 9.07 | 160.26 | 21.53 | 1.22% |
| count-mixed-512-4long | 1235.78 | 1232.12 | 1200.35 | 5.80 | 136.59 | 14.52 | 1.20% |
| count-mixed-512-2long-ascending | 1757.00 | 1624.53 | 1569.15 | 6.66 | 131.85 | 14.67 | 0.93% |
| count-mixed-512-2long-descending | 1758.32 | 1608.22 | 1534.96 | 6.50 | 146.30 | 14.54 | 0.94% |
| heldout-lognormal-257-seed20261114 | 883.82 | 911.43 | 884.70 | 3.15 | 14.20 | 3.92 | 0.44% |
| heldout-lognormal-1000-seed20261114 | 895.96 | 943.36 | 893.40 | 7.84 | 44.52 | 16.49 | 1.81% |
| heldout-jittered-768-seed20261114 | 503.92 | 555.88 | 503.77 | 6.68 | 36.11 | 7.98 | 1.56% |
| heldout-tiny-tail-768-seed20261114 | 1625.32 | 1502.69 | 1460.09 | 6.61 | 33.28 | 19.98 | 1.35% |
| heldout-trimodal-513-seed20261114 | 1996.13 | 1810.02 | 1767.19 | 5.73 | 140.11 | 15.58 | 0.87% |
| heldout-lognormal-257-seed20261115 | 974.10 | 988.30 | 974.95 | 3.23 | 16.24 | 9.46 | 0.96% |
| heldout-lognormal-1000-seed20261115 | 865.84 | 905.35 | 865.45 | 7.60 | 43.47 | 17.34 | 1.96% |
| heldout-jittered-768-seed20261115 | 500.70 | 552.94 | 500.89 | 5.92 | 35.51 | 7.34 | 1.44% |
| heldout-tiny-tail-768-seed20261115 | 1873.66 | 1574.57 | 1538.67 | 7.01 | 32.36 | 18.27 | 1.17% |
| heldout-trimodal-513-seed20261115 | 2092.25 | 1800.25 | 1768.75 | 5.60 | 145.48 | 15.40 | 0.86% |
| heldout-lognormal-257-seed20261116 | 965.34 | 981.52 | 964.29 | 2.76 | 13.30 | 6.16 | 0.63% |
| heldout-lognormal-1000-seed20261116 | 893.42 | 940.15 | 893.08 | 7.74 | 43.44 | 17.07 | 1.88% |
| heldout-jittered-768-seed20261116 | 502.22 | 554.02 | 501.43 | 5.90 | 36.09 | 7.38 | 1.45% |
| heldout-tiny-tail-768-seed20261116 | 2125.61 | 1747.08 | 1715.28 | 6.77 | 156.65 | 19.31 | 1.11% |
| heldout-trimodal-513-seed20261116 | 2310.92 | 1907.19 | 1861.13 | 5.23 | 143.77 | 14.92 | 0.80% |

Planner full preparation averages 11.439 ms per layout, or 2.860 ms per compatible layer. Its mean fraction is 1.009%, maximum 2.168%, and aggregate fraction 0.910%. These are distinct from the planner's performance gap to the fastest baseline.

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
| output | 0.161674% | 0.0046234131 |
| dq | 0.060327% | 0.0047607422 |
| dkv | 0.072825% | 0.25 |
| d_index_q | 0.073875% | 2.742695e-12 |
| d_weights | 0.061458% | 5.1222742e-09 |
| d_pool_kv | 0.093682% | 2.910383e-11 |
| d_pool_score | 0.105180% | 1.4551915e-11 |
| d_ape | 0.001471% | 1.3955059e-11 |

Maximum global auxiliary loss difference: **4.0745363e-10**. This is the indexer's auxiliary loss, not model training loss. The contiguous self-comparison is included to expose native nondeterministic variation.

Three small 4,096-token fixtures additionally run independent FP32 attention,
teacher/KL and pooling-gradient checks for each policy/layer/rank, plus exact
routing/inverse/autograd markers. The oracle uses each selected key set; it does
not independently prove the selector's discrete ranking. All 616 focused host
tests and repository lint/format checks pass.

Timing-pass memory peaks include inputs, four layers' retained forward states,
outputs/gradients and all three prebuilt policy metadata/workspace bundles.
They exclude the separate numerical oracle pass and are **harness peaks**, not
isolated deployment peaks or workspace-only usage. Reserved memory is allocator
history and is not attributed as an individual policy's live storage.

| Policy | Peak allocated range (GiB/rank) | Peak reserved range (GiB/rank) |
|---|---:|---:|
| contiguous | 140.90–161.72 | 144.02–166.44 |
| zigzag | 140.90–161.72 | 144.02–166.44 |
| planner | 140.90–161.72 | 144.02–166.44 |

## Reproduction and relationship to earlier tables

```bash
OMP_NUM_THREADS=4 torchrun --standalone --nproc-per-node=8 \
  -m benchmarks.bench_csa_cp_reuse \
  --profile benchmarks/profiles/thd_selection_b300_cp8.json \
  --output-dir results/csa-1m-reuse --layers 4 \
  --warmups 1 --repeats 3 --prepare-repeats 3 \
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

## What the profiling changed

The original model counted remote bytes and explicit permutations, but a
nonidentity eager all-to-all also handles rows retained by their original rank.
The optional `local_copy_gbps` term now charges those diagonal bytes against the
local transfer resource. It does not create a network edge or peer startup.
Global identity still skips the collective and its copy costs completely.
Older profiles without the coefficient retain exactly the same scores and
selected layouts: all 31 layouts were compared against the preceding source.

The B300 transport profile was recalibrated independently using 20 synthetic
routing fixtures, with 15 held-out fixtures. Packing was recalibrated too;
the observed changes must not be attributed solely to the diagonal term.
No THD case name, document-count lookup, precision reduction, timing feedback,
or implicit plan cache is used in production selection.

Nsight Systems explains the formerly unfavorable 1000-document lognormal case:

| Selection region | Contiguous | Old planner |
|---|---:|---:|
| Critical native selection activity | 9.280 ms | 6.993 ms |
| Whole selection region, including routing/restoration | 9.634 ms | 11.415 ms |

These are diagnostic trace measurements, not four-layer step timings. Native
activity is the maximum per-rank summed duration; whole-region time is the
span from the first GPU operation to the last across ranks. Communication
kernel duration can include waiting for another rank. Do not add independent
per-category rank maxima as though they were a single critical path.

The compute balancing worked, but routing consumed its savings. With the
refreshed model, the three borderline lognormal layouts retain source ownership:

| Selection-only diagnostic | Old planner | Refined planner |
|---|---:|---:|
| 257 docs, seed 20261115 | 11.255 ms | 9.466 ms |
| 1000 docs, seed 20261115 | 11.427 ms | 9.656 ms |
| 1000 docs, seed 20261116 | 13.300 ms | 11.969 ms |

The diagnostic alternates contiguous, zigzag, old and refined plans in one
process, with two warmups and seven samples. It measures the public selection
call only; the full operator qualification above is a separate experiment.
Different pointers and timing variation can produce different times for
identical layouts, which are not algorithmic gains.

## Alternatives measured and rejected

Bypassing NCCL for self-owned rows required additional packing/reassembly.
Across five shapes it slowed full-zigzag transport by about 7–32%; the favorable
lognormal planner case improved transport by about 6%, without a consistent
benefit elsewhere. The production redistributor was therefore left unchanged.

Routing all attention queries can redistribute attention work as well as
selection, but also moves the much larger attention Q, outputs and gradients.
The existing benchmark's query-wide paths were compared with selection-only
routing using the same inputs and complete forward/backward:

| 1M distribution | Selection-only | Query-wide zigzag | Query-wide balanced |
|---|---:|---:|---:|
| 1000 docs + 1 long | 433.097 ms | 474.763 ms | 460.816 ms |
| 512 docs + 2 long | 376.772 ms | 429.314 ms | 441.997 ms |
| Lognormal 1000, seed 20261115 | 214.863 ms | 282.827 ms | 327.134 ms |
| Uniform 1024 | 106.858 ms | 186.788 ms | 106.568 ms |

These are **one-instance** timings with two warmups and five rotating samples,
including runtime layouts and communication, excluding one-time preparation.
The uniform balanced layout is unchanged; its small timing difference is noise.
All variants passed full output/gradient comparisons and the small FP32 oracle,
but that alone is insufficient to justify adoption. On the 1000-doc mixed case,
query-wide routing increased the worst gradient RMS difference against contiguous
to 0.372%, versus 0.068% for selection-only. The increase was concentrated in
pooling gradients; changing query ownership also changes floating-point
reduction groupings. Neither speed nor numerical evidence supports widening
the public API's routing scope in this change.

[Diagnostic records](results/csa_cp_planner_diagnostics_20261117.json) retain
per-rank samples, comparison errors, configurations and source-artifact hashes.
The four-layer memory table above and query-scope peaks are harness measurements
with multiple policy workspaces alive, not isolated deployment memory budgets.

## Why qualification takes longer than planning

The previous protocol performed 132 complete CSA forward/backward pairs per
THD case: 108 for warmup/timing and 24 for complete numerical comparisons.
The refined qualification uses 72: 48 for warmup/timing and the same 24 for
numerical comparisons. It retains full tensor checks rather than sampling them.
Three timing repetitions are sufficient for screening broad differences;
close comparisons and visibly noisy cases require additional measurements.

First calls also construct and compile shape-dependent CuTeDSL specializations.
A rank-0 CPU profile during untimed warmup observed approximately 7–12 seconds
per first full-CSA path on one mixed case, with seconds in IR generation and
compilation. Those instrumented first-call times are not steady-state operator
latency or pure compiler time. Persistent compiler caches were already enabled.
The host planner itself takes milliseconds and is reused across all four layers.

For further iterations, use independent transport calibration and selection
screens first, then complete-operator checks on representative/changed layouts,
and finish with the full matrix. Remaining sparse-attention/backward kernel
work is a separate optimization problem; a stronger selector cost model cannot
remove its arithmetic or communication waits while preserving source ownership.

## Smaller-window and preparation-tail follow-up

Eight complete four-instance CSA cases at 128K and 256K were rerun on the same
cost-model revision, with one warmup, five timing passes and three preparation
samples. Full output and seven-gradient comparisons pass for every policy/rank/
layer. This is a regression subset, not a rerun of every smaller-window layout.
Times below are milliseconds; steady columns are four full forward/backward
uses, and preparation is charged once in the last three columns.

| Case | C steady | Z steady | P steady | C prep + steady | Z prep + steady | P prep + steady |
|---|---:|---:|---:|---:|---:|---:|
| 128k-single | 190.40 | 183.60 | 182.74 | 191.19 | 184.57 | 188.87 |
| 128k-mixed-22 | 141.72 | 142.02 | 141.79 | 142.51 | 143.95 | 147.94 |
| 128k-mixed-1000 | 132.03 | 153.97 | 132.01 | 140.24 | 198.88 | 142.17 |
| 128k-uniform-1024 | 35.92 | 43.63 | 35.91 | 43.35 | 88.35 | 45.17 |
| 256k-single | 468.62 | 423.93 | 419.56 | 469.79 | 425.16 | 427.18 |
| 256k-mixed-22 | 365.32 | 351.97 | 350.88 | 366.51 | 354.47 | 359.52 |
| 256k-mixed-1000 | 336.33 | 381.54 | 330.68 | 345.51 | 521.82 | 442.73 |
| 256k-uniform-1024 | 74.68 | 89.45 | 74.73 | 82.16 | 134.85 | 83.91 |

The 256K mixed-1000 preparation samples are **22.79, 116.01 and 112.06 ms**;
the three-sample median therefore produces a **28.14%** one-preparation gap to
contiguous, despite faster steady execution. Small contexts can also have
several-percent preparation gaps without such a spike. Four uses alone do not
guarantee a <3% total gap at every context and document count.

A separate diagnostic retained normal Python garbage collection and measured
16 fresh preparations per policy both before and after full numerical
qualification (32 total per policy/case). It warmed all complete CSA paths first
and recorded GC callbacks on every rank. Gen-2 collections account for the
roughly 90–130 ms tail events, in both phases; qualification is not the sole
cause. No GC disabling, forced collection between samples, or tail trimming
was used. The native/JIT Python object graph is already live in this harness.

| Case / policy | Preparation median | Mean | Maximum | Critical samples >70 ms | Of those with gen-2 GC on the critical rank |
|---|---:|---:|---:|---:|---:|
| 256k-mixed-1000 / contiguous | 8.70 | 8.78 | 9.96 | 0 | 0 |
| 256k-mixed-1000 / zigzag | 147.91 | 120.58 | 166.35 | 23 | 23 |
| 256k-mixed-1000 / planner | 22.50 | 37.62 | 130.17 | 5 | 5 |
| 1m-count-tiny-mix-1000 / contiguous | 7.58 | 10.71 | 102.19 | 1 | 1 |
| 1m-count-tiny-mix-1000 / zigzag | 145.06 | 106.77 | 168.26 | 18 | 18 |
| 1m-count-tiny-mix-1000 / planner | 17.63 | 33.63 | 130.81 | 5 | 5 |
| 1m-heldout-trimodal-513-seed20261116 / contiguous | 5.17 | 5.26 | 6.00 | 0 | 0 |
| 1m-heldout-trimodal-513-seed20261116 / zigzag | 22.18 | 64.99 | 150.40 | 13 | 13 |
| 1m-heldout-trimodal-513-seed20261116 / planner | 14.99 | 18.23 | 116.27 | 1 | 1 |

This diagnostic does not replace the original 31-case samples or manufacture a
new aggregate from selected reruns. For long-lived reused plans, the measured
31-case **steady** reductions (11.89% versus contiguous, 3.00% versus zigzag)
are the less GC-sensitive comparison. The original three-sample preparation
aggregate remains reproducible but does not establish a tail-latency bound.
Production applications should reuse preparation and measure its mean/tail in
their own process; this library must not alter the application's GC policy.

[Smaller-window and GC evidence](results/csa_cp_preparation_tail_20261117.json)
contains exact lengths, per-rank samples, source hashes, and per-generation GC
durations/counts. Source/routing metadata remains on the host; this diagnosis
introduces no D2H synchronization or numerical changes.

## Final host-only simplification

Commit `b275d5f` follows the full GPU cohort above. It avoids calculating the
traffic lower bound when mean compute already misses the migration threshold,
and reuses immutable source spans for pinned short documents. Nonnegative
transport cannot make an already infeasible compute bound feasible; contiguous
source spans are already coalesced. No mutable cross-call cache is introduced.

An alternating CPU A/B (four warmups, 20 samples per variant) preserves every
plan field and score in **394 comparisons**: the 31 1M cases, 16 smaller-window
cases and 150 randomized cases, each with both old and refined profiles.
Logical CP1/2/4/8/16/64 coverage is host metadata validation, not a 64-GPU run.

| Host-only planning | Before | After | Time reduction |
|---|---:|---:|---:|
| Mean of 31 1M cases | 6.228 ms | 5.924 ms | 4.88% |
| Mean of all 47 timed layouts | 5.433 ms | 5.183 ms | 4.60% |
| 1M uniform 1024 documents | 2.868 ms | 2.613 ms | 8.90% |
| 1M mixed 1000 documents, one long | 12.928 ms | 12.195 ms | 5.67% |

These are pure planner times, not full GPU preparation or operator latency.
The optimization does not eliminate process-wide GC tails. The 31-case GPU
numbers remain attributed to `932b3b4`; they have not been relabeled as timings
of this later CPU change. The focused 616-test host suite passes again.
[All CPU samples and source hashes](results/csa_cp_planner_cpu_20261117.json)
are retained.

The final `b275d5f` source also passes all 288 independent FP32 GPU checks
across three small four-instance CP8 fixtures. All 126 runtime/benchmark Python
source hashes match the tested checkout. These checks validate the final CPU
change without treating another small timing run as a new 1M result.
