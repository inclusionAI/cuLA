# CSA CP: three policies across 128K, 256K and 1M

This is the complete rerun after reducing host preparation work. On B300 CP8,
adaptive selection routing stays within **0.339%** of the faster fixed
contiguous/zigzag policy in steady execution across these 24 cases. All selected
layouts match the previous revision. The largest gains occur with unequal long
documents at 1M; uniform packs generally select contiguous. This is measured
coverage, not a guarantee over arbitrary THD inputs.

The workload is **one complete CSA operator forward/backward**: ratio-4 pooling,
pooled-key and attention-KV gathers, routed selection and inverse exchange,
sparse attention, teacher/KL and all activation/parameter gradients. Runtime
layout conversions and communication are included. KDA, projections, MoE,
optimizer and checkpoint recomputation are excluded. These are not timings of
a connected seven-KDA/one-CSA training block.

## Measurement and workloads

- Eight B300 GPUs, CSA H64/D576/V512, indexer H64/D128, Top-K512.
- BF16 Q/K/weights; FP32 scores, nonlinearities and communication reductions.
  TF32 and BF16 reduced-precision GEMM intermediate reductions are disabled.
  Native nondeterministic mode is unchanged.
- PyTorch 2.12.1+cu130, CUDA runtime 13.0, CuTeDSL 4.8.1, cuDNN 9.20.0.48.
  cuDNN Frontend source commit `855ccd77ba9efed22441fd454f7cc0206a068e94`.
  Frontend wheel version metadata does not identify the imported source tree.
- Two warmups, seven timed execution samples and eight fresh preparation samples.
  Policy order rotates and reverses. Each reported duration is the median of
  the per-sample maximum across all ranks. Compilation, synthetic input creation
  and correctness reference work are outside timing. Host-time outliers remain.
- The existing compute/transport profile is used with explicit `min_gain_ms=0`,
  `max_chunks_per_rank=6`, `min_chunk_tokens=1024`. This is a reuse-oriented
  throughput experiment, not a newly calibrated profile or changed default.
- Public contiguous, full per-document zigzag and planner share the same source
  ownership. Zigzag/planner redistribute selection Q/weights and restore results;
  attention retains the original ownership. A fourth legacy-contiguous fixture
  is the correctness reference and is not a table column.

Here K means 1,024 tokens and M means 1,048,576 tokens. All lengths are divisible
by four, and each pack sums exactly to the context size T.

| Distribution | Document lengths |
|---|---|
| 1 long document | T |
| 64 uniform | 64 x T/64 |
| 64 near-uniform | Alternating T/64 - T/1024 and T/64 + T/1024, 32 pairs |
| 20 short + 2 long | 20 x 1,024, followed by 2 x (T - 20,480)/2 |
| 1,024 uniform | 1,024 x T/1,024 |
| 512 near-uniform | Alternating T/512 - 4 and T/512 + 4, 256 pairs |
| 998 short + 2 long | 998 x 64 and 2 x (T - 63,872)/2; long docs after 199 shorts and at the end |
| 1,000 long-tail | 800 x 64, 199 x 256 and one remaining long document |

All tables report milliseconds; lower is better. Percentage columns are
`100 * (planner / baseline - 1)`; negative means less time. Subpercent differences
should be treated cautiously, rather than as strong evidence of a speedup.

## 128K steady execution

| Distribution | Contiguous | Zigzag | Planner | P time vs C | P time vs Z |
|---|---:|---:|---:|---:|---:|
| 1 long document | 47.69 | 46.27 | 45.83 | -3.91% | -0.97% |
| 64 uniform | 20.00 | 20.67 | 19.99 | -0.04% | -3.29% |
| 64 near-uniform | 20.03 | 20.76 | 20.02 | -0.03% | -3.53% |
| 20 short + 2 long | 35.41 | 35.36 | 35.48 | +0.19% | +0.34% |
| 1,024 uniform | 9.32 | 11.10 | 9.28 | -0.37% | -16.34% |
| 512 near-uniform | 10.00 | 11.20 | 10.01 | +0.12% | -10.60% |
| 998 short + 2 long | 33.10 | 38.75 | 33.09 | -0.05% | -14.63% |
| 1,000 long-tail | 32.33 | 37.88 | 32.33 | +0.01% | -14.66% |

## 256K steady execution

| Distribution | Contiguous | Zigzag | Planner | P time vs C | P time vs Z |
|---|---:|---:|---:|---:|---:|
| 1 long document | 115.74 | 105.47 | 104.23 | -9.94% | -1.17% |
| 64 uniform | 50.32 | 51.76 | 50.36 | +0.08% | -2.70% |
| 64 near-uniform | 50.35 | 51.71 | 50.30 | -0.09% | -2.72% |
| 20 short + 2 long | 90.67 | 88.23 | 87.59 | -3.40% | -0.72% |
| 1,024 uniform | 18.96 | 22.55 | 18.94 | -0.15% | -16.01% |
| 512 near-uniform | 21.74 | 24.21 | 21.77 | +0.14% | -10.06% |
| 998 short + 2 long | 83.71 | 95.86 | 82.07 | -1.96% | -14.39% |
| 1,000 long-tail | 98.91 | 115.31 | 92.09 | -6.89% | -20.14% |

## 1M steady execution

| Distribution | Contiguous | Zigzag | Planner | P time vs C | P time vs Z |
|---|---:|---:|---:|---:|---:|
| 1 long document | 852.26 | 637.36 | 637.78 | -25.17% | +0.07% |
| 64 uniform | 239.24 | 244.40 | 239.44 | +0.09% | -2.03% |
| 64 near-uniform | 239.18 | 244.56 | 239.23 | +0.02% | -2.18% |
| 20 short + 2 long | 578.44 | 492.49 | 490.87 | -15.14% | -0.33% |
| 1,024 uniform | 106.93 | 120.95 | 106.89 | -0.04% | -11.63% |
| 512 near-uniform | 153.33 | 162.94 | 153.63 | +0.19% | -5.71% |
| 998 short + 2 long | 567.96 | 550.56 | 494.71 | -12.90% | -10.14% |
| 1,000 long-tail | 800.96 | 715.78 | 591.82 | -26.11% | -17.32% |

## Preparation plus one use

The following is **median fresh preparation + median execution**, not a directly
timed cold end-to-end sample. Preparation includes common metadata, host planning,
workspace/index construction, H2D copies and a completion sync. It is rebuilt,
not replaced by a cache lookup. First compilation is still excluded.

### 128K

| Distribution | Contiguous | Zigzag | Planner |
|---|---:|---:|---:|
| 1 long document | 48.41 | 47.17 | 53.27 |
| 64 uniform | 20.90 | 24.40 | 21.10 |
| 64 near-uniform | 21.07 | 25.31 | 21.31 |
| 20 short + 2 long | 36.17 | 37.22 | 43.18 |
| 1,024 uniform | 15.78 | 54.64 | 17.80 |
| 512 near-uniform | 13.56 | 33.89 | 14.72 |
| 998 short + 2 long | 40.23 | 82.95 | 42.34 |
| 1,000 long-tail | 39.66 | 83.09 | 41.45 |

### 256K

| Distribution | Contiguous | Zigzag | Planner |
|---|---:|---:|---:|
| 1 long document | 116.81 | 106.57 | 111.94 |
| 64 uniform | 51.31 | 55.48 | 51.57 |
| 64 near-uniform | 51.34 | 55.48 | 51.51 |
| 20 short + 2 long | 91.79 | 90.53 | 96.11 |
| 1,024 uniform | 26.19 | 67.67 | 28.22 |
| 512 near-uniform | 25.42 | 47.63 | 26.58 |
| 998 short + 2 long | 92.58 | 191.44 | 106.15 |
| 1,000 long-tail | 107.78 | 263.45 | 115.51 |

### 1M

| Distribution | Contiguous | Zigzag | Planner |
|---|---:|---:|---:|
| 1 long document | 854.50 | 639.80 | 647.29 |
| 64 uniform | 240.69 | 248.73 | 241.09 |
| 64 near-uniform | 240.69 | 248.94 | 240.94 |
| 20 short + 2 long | 580.61 | 495.68 | 500.99 |
| 1,024 uniform | 114.18 | 168.40 | 116.17 |
| 512 near-uniform | 158.24 | 187.40 | 159.63 |
| 998 short + 2 long | 577.91 | 599.07 | 521.10 |
| 1,000 long-tail | 813.63 | 817.01 | 620.98 |

## What the comparison shows

The aggregate below sums one execution of each of the eight distributions at
each context length. It assumes equal case frequency, not a production workload
mix. Values are percentage time reductions relative to each fixed policy;
negative reductions mean a slowdown.

| Context | Steady vs contiguous | Steady vs zigzag | Prepare + one use vs contiguous | Prepare + one use vs zigzag |
|---|---:|---:|---:|---:|
| 128K | 0.89% | 7.19% | -8.23% | 34.35% |
| 256K | 4.35% | 8.60% | -4.33% | 33.10% |
| 1M | 16.50% | 6.77% | 14.87% | 10.48% |

The same planner and coefficients handle all three lengths. Single long documents
benefit from balancing causal work; many equal documents are already balanced
under contiguous ownership. Mixed packs benefit from redistributing expensive
long-document work while keeping cheap short documents local. At 1M, the
998-short/two-long and 1,000-document long-tail cases reduce steady time by
**10.14% and 17.32% versus full zigzag**, respectively. The 64-document cases
choose contiguous; the 22-document 1M case performs close to zigzag.

Preparation remains a limitation for one-use layouts. At 128K/256K, aggregate
preparation-plus-one-step is still slower than contiguous. At 1M, the single
long document and 22-document pack remain slightly slower than zigzag under
one-use accounting. The largest one-use gap to the faster fixed policy is
**19.40%** (128k-mixed-22), compared with 50.77% in the previous run.
These are separately measured cohorts; the controlled host A/B below isolates
the preparation change. Neither cohort establishes universally optimal one-use
selection or a <3% preparation guarantee. Use a calibrated preparation budget
and account for expected layout reuse.

## Equivalent preparation optimization

The update removes redundant metadata work and prunes candidates with a proved
model-level bound; it introduces no retained cache:

1. Compute pinned per-rank loads once per invocation and copy the small arrays
   into each search grid. Visit only movable documents in adaptive grids.
2. Skip long-document candidates when immutable pinned work alone exceeds the
   migration threshold. Full zigzag remains independently eligible.
3. Price a source-local span directly, preserving source/receive permutation
   detection and floating-point accumulation order.
4. Check full-zigzag grain feasibility before transport estimation, retaining
   the original-source-fragment exception. No feasible candidate is discarded.
5. Check exact whole-document/remainder work before the DSA binary split search.
   Oversized intervals still use the same maximal budget-fitting prefix.

Candidate definitions, coefficients, tie breaks, selected layouts, native window
boundaries, scratch sizes and GPU execution are unchanged. The diagnostic list
of evaluated scores can omit candidates rejected by the new lower bound.
All extra state is local to preparation; no mutable global cache, collective,
cast or D2H read is introduced.

The following host-only A/B uses rank 7's complete `CsaCPSelectionPlan` constructor,
20 samples per variant, alternating before/after order after two warmups. It
excludes GPU workspace preparation. The baseline imports source from `ea71df3`;
the otherwise unchanged caller binds that revision's planner and DSA builder.
The raw report covers all 24 planner shapes plus fixed-policy controls.

| Host-only case | Before (ms) | After (ms) | Time reduction |
|---|---:|---:|---:|
| 128k-single-planner | 6.177 | 6.052 | +2.0% |
| 256k-single-planner | 6.190 | 6.082 | +1.8% |
| 1024k-uniform-64-planner | 0.397 | 0.361 | +9.1% |
| 256k-mixed-22-planner | 6.795 | 6.507 | +4.2% |
| 128k-uniform-1024-planner | 4.494 | 4.157 | +7.5% |
| 128k-uniform-1024-zigzag | 38.705 | 37.026 | +4.3% |
| 128k-mixed-1000-planner | 21.876 | 3.862 | +82.3% |
| 1024k-long-tail-1000-planner | 24.956 | 16.904 | +32.3% |

Full fresh preparation, including GPU metadata/workspace work, is shown next.
These before/after medians come from separate complete-CSA runs. Small values
have visible run-to-run variation; the table does not imply every case improved.
Host-time spikes remain in the raw samples. For example, the new 1M long-tail planner preparation ranges from 28.45 to 150.10 ms.

| Context | Distribution | Before preparation | After preparation | Time reduction |
|---|---|---:|---:|---:|
| 128K | 1 long document | 7.55 | 7.45 | +1.4% |
| 128K | 64 uniform | 1.24 | 1.11 | +10.8% |
| 128K | 20 short + 2 long | 7.61 | 7.70 | -1.3% |
| 128K | 1,024 uniform | 8.97 | 8.52 | +5.1% |
| 128K | 998 short + 2 long | 28.31 | 9.26 | +67.3% |
| 128K | 1,000 long-tail | 12.13 | 9.12 | +24.8% |
| 256K | 1 long document | 7.79 | 7.71 | +1.0% |
| 256K | 64 uniform | 1.28 | 1.21 | +5.0% |
| 256K | 20 short + 2 long | 8.77 | 8.52 | +2.9% |
| 256K | 1,024 uniform | 8.97 | 9.28 | -3.5% |
| 256K | 998 short + 2 long | 32.67 | 24.09 | +26.3% |
| 256K | 1,000 long-tail | 31.81 | 23.42 | +26.4% |
| 1M | 1 long document | 9.32 | 9.51 | -2.0% |
| 1M | 64 uniform | 1.72 | 1.65 | +4.3% |
| 1M | 20 short + 2 long | 9.80 | 10.12 | -3.3% |
| 1M | 1,024 uniform | 9.63 | 9.29 | +3.6% |
| 1M | 998 short + 2 long | 34.91 | 26.39 | +24.4% |
| 1M | 1,000 long-tail | 38.68 | 29.16 | +24.6% |

## Precision, review and provenance

- **518 focused CPU tests pass**, including exact budget-boundary checks with
  unaligned causal offsets, local-reorder pricing and grain-check equivalence.
- **660 planner comparisons** preserve the selected layout, metrics and score
  exactly versus `ea71df3`. Every retained candidate score is identical; omitted
  candidates are verified to miss the migration margin: 55 shapes
  (31 historical + 24 window cases) x CP8/16/64 x zero/original headroom x
  zero/1,024-token floors. CP64 is host metadata validation, not a GPU benchmark.
- **300 randomized DSA plans and 69,281 windows** match the baseline exactly.
- **128 native subgroup checks pass** for BF16/FP32 weights, unequal sources,
  empty destinations and retained outputs; canonical IDs and scores match
  unrouted selection exactly in these fixtures.
- All **24 shapes x four paths x eight ranks** pass full-tensor output and
  seven-gradient comparisons against legacy contiguous, plus exact row-marker
  forward/inverse/autograd checks. Three small fixtures under all four paths
  pass independent FP32 attention/teacher/KL/pooling checks. Those FP32 checks
  use the selected key set and do not independently establish discrete ranking.

Maximum relative RMS over all public paths/ranks/cases:

| Tensor | Maximum relative RMS |
|---|---:|
| Output | 0.21341% |
| dQ | 0.06697% |
| dKV | 0.10307% |
| Indexer dQ | 0.09174% |
| Indexer dWeights | 0.06641% |
| Pool dKV | 0.13371% |
| Pool dScore | 0.13834% |
| Pool dAPE | 0.00254% |

Maximum auxiliary loss absolute difference is **1.16e-09**. No math,
dtype or native precision setting changed. These error envelopes include native
nondeterminism; small differences from the previous run are not a new precision
improvement or a bitwise-equivalence claim.

The tested source is `489779ba35e2c5f51eccce7c7ff95ed4cdd640d0`. All 125 source
files in the final timing manifest match the reviewed checkout.
Every selected policy matches the previous run, and every measured full-zigzag
baseline fits the planner's row cap. The new measurements supersede the primary
tables in this document. The [previous 24-case evidence](results/csa_cp_three_way_20261116.json)
and [original 128K/256K study](CSA_CP_WINDOW_RESULTS.md) remain available.

[Current machine-readable results](results/csa_cp_preparation_20261116.json)
retain critical timing/preparation samples, selected policies, precision envelopes,
source-record hashes, cohort source hashes and the counterbalanced host A/B.
The allocator diagnostics in the underlying harness include four coexisting
fixtures; they are not isolated per-policy memory claims.

## Reproduction

```bash
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m benchmarks.bench_csa_cp_windows \
  --profile benchmarks/profiles/thd_selection_b300_cp8.json \
  --output-dir results/csa-three-way --tokens 131072 262144 1048576 \
  --cases single uniform-64 near-uniform-64 mixed-22 \
          uniform-1024 near-uniform-512 mixed-1000 long-tail-1000 \
  --min-gain-ms 0 --min-chunk-tokens 1024 --max-chunks-per-rank 6 \
  --warmups 2 --repeats 7 --prepare-repeats 8
```

See the [public API contract](CSA_CP_SELECTION.md) for explicit plan/workspace
lifetimes, source ownership and downstream integration.
