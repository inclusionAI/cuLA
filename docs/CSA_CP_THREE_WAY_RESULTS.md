# CSA CP: three policies across 128K, 256K and 1M

On B300 CP8, adaptive selection routing stays within **0.576%** of the faster
fixed contiguous/zigzag policy in steady execution across these 24 cases. The
largest gains occur at 1M with unequal long documents. Uniform packs usually
select contiguous; a single long document performs similarly to zigzag.
This is measured coverage, not a guarantee over arbitrary THD inputs.

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
| 1 long document | 47.60 | 46.23 | 45.82 | -3.74% | -0.88% |
| 64 uniform | 20.11 | 20.66 | 20.04 | -0.37% | -3.02% |
| 64 near-uniform | 20.08 | 20.79 | 20.06 | -0.13% | -3.53% |
| 20 short + 2 long | 35.44 | 35.23 | 35.44 | -0.00% | +0.58% |
| 1,024 uniform | 9.29 | 11.10 | 9.30 | +0.09% | -16.28% |
| 512 near-uniform | 10.01 | 11.21 | 10.03 | +0.14% | -10.52% |
| 998 short + 2 long | 33.07 | 38.75 | 33.07 | -0.00% | -14.64% |
| 1,000 long-tail | 32.37 | 37.90 | 32.35 | -0.06% | -14.64% |

## 256K steady execution

| Distribution | Contiguous | Zigzag | Planner | P time vs C | P time vs Z |
|---|---:|---:|---:|---:|---:|
| 1 long document | 115.29 | 105.58 | 104.15 | -9.67% | -1.36% |
| 64 uniform | 50.34 | 51.73 | 50.33 | -0.00% | -2.70% |
| 64 near-uniform | 50.34 | 51.77 | 50.35 | +0.01% | -2.75% |
| 20 short + 2 long | 90.69 | 87.95 | 87.39 | -3.64% | -0.64% |
| 1,024 uniform | 18.96 | 22.56 | 18.94 | -0.10% | -16.02% |
| 512 near-uniform | 21.79 | 24.21 | 21.76 | -0.12% | -10.12% |
| 998 short + 2 long | 83.64 | 96.05 | 82.15 | -1.78% | -14.47% |
| 1,000 long-tail | 99.36 | 115.25 | 92.20 | -7.20% | -20.00% |

## 1M steady execution

| Distribution | Contiguous | Zigzag | Planner | P time vs C | P time vs Z |
|---|---:|---:|---:|---:|---:|
| 1 long document | 853.46 | 638.50 | 637.55 | -25.30% | -0.15% |
| 64 uniform | 239.31 | 244.20 | 239.05 | -0.11% | -2.11% |
| 64 near-uniform | 239.06 | 244.37 | 239.25 | +0.08% | -2.10% |
| 20 short + 2 long | 576.29 | 493.97 | 491.52 | -14.71% | -0.50% |
| 1,024 uniform | 106.86 | 121.14 | 106.80 | -0.05% | -11.84% |
| 512 near-uniform | 153.78 | 162.94 | 153.50 | -0.18% | -5.79% |
| 998 short + 2 long | 566.17 | 549.72 | 493.44 | -12.85% | -10.24% |
| 1,000 long-tail | 801.37 | 715.45 | 591.85 | -26.14% | -17.28% |

## Preparation plus one use

The following is **median fresh preparation + median execution**, not a directly
timed cold end-to-end sample. Preparation includes common metadata, host planning,
workspace/index construction, H2D copies and a completion sync. It is rebuilt,
not replaced by a cache lookup. First compilation is still excluded.

### 128K

| Distribution | Contiguous | Zigzag | Planner |
|---|---:|---:|---:|
| 1 long document | 48.33 | 47.14 | 53.37 |
| 64 uniform | 21.07 | 24.84 | 21.28 |
| 64 near-uniform | 21.06 | 24.95 | 21.24 |
| 20 short + 2 long | 36.25 | 37.16 | 43.04 |
| 1,024 uniform | 16.07 | 58.33 | 18.27 |
| 512 near-uniform | 13.76 | 35.29 | 14.98 |
| 998 short + 2 long | 40.72 | 84.47 | 61.39 |
| 1,000 long-tail | 39.95 | 83.74 | 44.48 |

### 256K

| Distribution | Contiguous | Zigzag | Planner |
|---|---:|---:|---:|
| 1 long document | 116.41 | 106.72 | 111.94 |
| 64 uniform | 51.40 | 55.87 | 51.61 |
| 64 near-uniform | 51.40 | 56.02 | 51.62 |
| 20 short + 2 long | 91.84 | 90.29 | 96.16 |
| 1,024 uniform | 25.87 | 69.42 | 27.91 |
| 512 near-uniform | 25.73 | 48.89 | 26.78 |
| 998 short + 2 long | 93.22 | 188.33 | 114.82 |
| 1,000 long-tail | 109.06 | 262.29 | 124.02 |

### 1M

| Distribution | Contiguous | Zigzag | Planner |
|---|---:|---:|---:|
| 1 long document | 855.70 | 640.79 | 646.87 |
| 64 uniform | 240.80 | 248.86 | 240.77 |
| 64 near-uniform | 240.57 | 249.16 | 241.00 |
| 20 short + 2 long | 578.53 | 497.20 | 501.32 |
| 1,024 uniform | 114.35 | 169.54 | 116.43 |
| 512 near-uniform | 158.33 | 188.80 | 159.24 |
| 998 short + 2 long | 576.80 | 597.98 | 528.35 |
| 1,000 long-tail | 815.55 | 867.25 | 630.53 |

## What the comparison shows

The aggregate below sums one execution of each of the eight distributions at
each context length. It assumes equal case frequency, not equal runtime weights
or a measured production mix. Values are percentage time reductions relative
to the specified fixed policy; a negative reduction means a slowdown.

| Context | Steady vs contiguous | Steady vs zigzag | Prepare + one use vs contiguous | Prepare + one use vs zigzag |
|---|---:|---:|---:|---:|
| 128K | 0.90% | 7.11% | -17.22% | 29.77% |
| 256K | 4.36% | 8.62% | -7.07% | 31.10% |
| 1M | 16.50% | 6.85% | 14.41% | 11.42% |

At 1M, the two-long/998-short mixture improves steady time by **12.85% versus
contiguous and 10.24% versus zigzag**. The 1,000-document long tail improves by
**26.14% and 17.28%**, respectively. These two cases choose locality-preserving
six-piece-per-rank candidates. Balancing causal selection work while avoiding
unnecessary movement of short documents beats applying full zigzag everywhere.

The dozens-of-documents cases add a useful distinction: 64 uniform/near-uniform
documents choose contiguous at all three lengths, avoiding the full-zigzag cost
for already-balanced work. The 20-short/two-long pack at 1M improves **14.71%**
over contiguous and is close to zigzag. At 128K, however, the same document-count
pattern is **0.576% slower than zigzag**. Layout choice is not perfectly predicted;
the larger aggregate gain at 1M does not establish monotonic improvement per case.

Fresh preparation still matters. At 1M, including it leaves planner slightly
slower than zigzag for the single document and the 22-document pack, despite
similar steady execution. At 128K/256K the aggregate one-use result remains
slower than contiguous. The worst one-use gap to the faster fixed policy across
all 24 cases is **50.77%** for the 128K two-long/998-short mixture. Thus this study
does not establish universally optimal one-use selection or a <3% preparation
budget. A reused layout and a layout used only once need different preparation
budgets. Some preparation samples have large host-time spikes; the raw minimum,
maximum and all critical samples are retained in the linked JSON.

## Precision, review and provenance

All **24 cases x four paths x eight ranks** pass row-identity/inverse checks and
full-tensor output/seven-gradient comparisons against legacy contiguous. Maximum
relative RMS across all public paths/ranks/cases is shown below; it is not a
bitwise-equivalence claim for nondeterministic native kernels.

| Tensor | Maximum relative RMS |
|---|---:|
| Output | 0.21361% |
| dQ | 0.06709% |
| dKV | 0.10311% |
| Indexer dQ | 0.07967% |
| Indexer dWeights | 0.06914% |
| Pool dKV | 0.13756% |
| Pool dScore | 0.13780% |
| Pool dAPE | 0.00265% |

Maximum auxiliary loss absolute difference is **8.15e-10**. No runtime math,
dtype, kernel or precision option changed for this coverage extension.
**444 focused CPU tests pass**, including nine new document-count/context
coverage tests. The unchanged runtime had also passed 128 native subgroup
selection checks and three small independent FP32 oracle fixtures under all
four paths, as detailed in the [original study](CSA_CP_WINDOW_RESULTS.md).

The original ten 128K/256K records are reused from that study; fourteen additional
records cover six dozens-of-documents cases at 128K/256K and all eight 1M
distributions. Runtime source is `d2cf4fd`; only benchmark fixture
construction/help changed between timing cohorts. All 125 source hashes match
for the new cohorts. The only differing source hash in the original cohort is
that benchmark file. No algorithm or profile was tuned between these cohorts.

The report preserves per-case timing/error envelopes, selected policies, all
critical timing/preparation samples, source-record hashes, and cohort source
manifests in [machine-readable results](results/csa_cp_three_way_20261116.json).
Diagnostic allocator peaks in the underlying harness include four coexisting
fixtures; they must not be interpreted as isolated per-policy memory usage.

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
