# Real-CP THD selection results (2026-11-15)

This is a **1 KDA + 1 complete CSA operator-chain** forward/backward study.
KDA and CSA use independent synthetic inputs; projections, residuals, MLP,
optimizer and the full training model are excluded. It is not a seven-KDA/one-CSA
model measurement. The 31 shapes contain exactly 1,048,576 tokens each.

## Setup and accounting

- H20-3e: two nodes, eight GPUs per node, real CP16. B300: one node, real CP8.
- cuDNN Frontend commit `855ccd77ba9efed22441fd454f7cc0206a068e94`, backend 9.20.0.48, CuTeDSL 4.8.1.
- KDA: H32, K=V=128, batch invariant enabled; CSA MLA-absorb: H64, D576, V512; indexer: H64/D128, Top-K512, ratio-4 pooling.
- Non-deterministic selection/pooling/sparse backward. The transport never changes dtypes; pool gradients retain the per-loss-rank ordering before FP32 accumulation.
- Three warmups and ten timed iterations per policy/shape. Time is the median of each iteration's maximum-rank duration.
- Timed work includes all runtime layouts and activation/gradient communication, pooling, Top-K, sparse attention, detached teacher/KL and all backwards.
- Preparation separately includes common metadata, planning, index/workspace materialization and KDA transport allocation. Fresh-layout cost adds this **once**, without amortization. RNG generation, JIT warmup and numerical oracles are excluded.
- Memory is peak allocated/reserved for the entire measured chain, **including inputs, outputs and gradients**, not workspace-only memory.

## Results

C = contiguous; Z = complete per-document zigzag; P = calibrated selection planner.
All table times are milliseconds. The machine-readable [results](results/thd_selection_20261115.json)
include per-policy preparation, memory, errors and original record hashes.

| Case | Docs | H20 C | H20 Z | H20 P | B300 C | B300 Z | B300 P |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20s2l | 22 | 1770.25 | 1361.15 | 1326.95 | 606.72 | 525.02 | 523.09 |
| count-mixed-1000-1long | 1000 | 1738.30 | 1107.86 | 1078.85 | 564.92 | 480.15 | 466.27 |
| count-mixed-256-1long | 256 | 1933.60 | 1223.23 | 1197.76 | 646.42 | 509.91 | 494.01 |
| count-mixed-512-2long | 512 | 1235.56 | 942.93 | 932.91 | 441.96 | 421.93 | 407.34 |
| count-mixed-512-2long-ascending | 512 | 1240.10 | 947.32 | 938.54 | 466.90 | 434.59 | 421.41 |
| count-mixed-512-2long-descending | 512 | 1241.66 | 936.86 | 933.32 | 466.69 | 430.64 | 412.11 |
| count-mixed-512-4long | 512 | 972.04 | 880.24 | 874.13 | 338.36 | 338.60 | 338.46 |
| count-near-uniform-512 | 512 | 452.80 | 489.87 | 452.81 | 174.68 | 185.55 | 174.55 |
| count-tiny-mix-1000 | 1000 | 804.45 | 819.81 | 803.92 | 277.50 | 288.30 | 277.98 |
| count-tiny-mix-512 | 512 | 685.93 | 716.08 | 685.85 | 252.41 | 260.91 | 252.61 |
| count-uniform-1024 | 1024 | 304.09 | 345.73 | 303.89 | 119.41 | 133.69 | 119.48 |
| count-uniform-256 | 256 | 549.62 | 584.04 | 549.45 | 213.00 | 220.55 | 213.09 |
| jittered-768-s14 | 768 | 358.52 | 397.64 | 358.48 | 139.60 | 152.69 | 139.54 |
| jittered-768-s15 | 768 | 358.78 | 398.11 | 358.70 | 139.20 | 152.08 | 138.97 |
| jittered-768-s16 | 768 | 359.07 | 398.91 | 359.14 | 139.43 | 152.10 | 139.20 |
| lognormal-1000-s14 | 1000 | 685.00 | 700.99 | 684.54 | 242.74 | 255.30 | 242.80 |
| lognormal-1000-s15 | 1000 | 763.70 | 760.71 | 763.37 | 238.08 | 248.49 | 238.69 |
| lognormal-1000-s16 | 1000 | 736.00 | 735.10 | 736.39 | 243.51 | 255.28 | 244.25 |
| lognormal-257-s14 | 257 | 666.27 | 694.94 | 666.14 | 241.62 | 249.37 | 241.79 |
| lognormal-257-s15 | 257 | 786.76 | 774.97 | 786.91 | 265.90 | 270.33 | 266.58 |
| lognormal-257-s16 | 257 | 792.44 | 789.95 | 792.44 | 260.86 | 265.92 | 261.18 |
| tiny-tail-768-s14 | 768 | 1157.98 | 905.56 | 887.67 | 436.30 | 406.08 | 397.62 |
| tiny-tail-768-s15 | 768 | 1348.06 | 970.18 | 959.86 | 498.77 | 423.77 | 416.50 |
| tiny-tail-768-s16 | 768 | 1590.68 | 1054.16 | 1043.18 | 559.72 | 466.85 | 460.22 |
| trimodal-513-s14 | 513 | 1590.94 | 1143.29 | 1123.55 | 528.07 | 482.18 | 471.44 |
| trimodal-513-s15 | 513 | 1565.77 | 1154.85 | 1138.55 | 550.40 | 479.26 | 475.16 |
| trimodal-513-s16 | 513 | 1917.07 | 1225.09 | 1195.03 | 606.03 | 509.75 | 498.81 |
| near-uniform | 64 | 657.86 | 691.02 | 657.87 | 261.57 | 266.84 | 261.59 |
| power-law | 7 | 1788.44 | 1200.61 | 1195.98 | 612.98 | 493.15 | 492.85 |
| single | 1 | 2908.84 | 1934.28 | 1926.76 | 879.36 | 667.50 | 664.84 |
| uniform | 64 | 653.51 | 690.32 | 657.16 | 261.74 | 267.41 | 261.72 |

| Metric | H20 CP16 | B300 CP8 |
|---|---:|---:|
| Largest P slowdown vs faster C/Z, steady | 1.541% | 0.304% |
| Largest P slowdown vs faster C/Z, prepare + one iteration | 4.038% | 5.188% |
| Best C/P speedup | 1.614x | 1.323x |
| Mean P preparation | 34.118 ms | 18.972 ms |
| Mean extra P vs C preparation / mean P steady time | 3.038% | 2.518% |
| All P preparation / all P steady time | 4.011% | 5.647% |
| contiguous max allocated GiB | 44.91 | 72.95 |
| zigzag max allocated GiB | 46.20 | 73.43 |
| planner max allocated GiB | 46.02 | 75.91 |

These results do **not** show universal optimality or a strict <3% preparation
guarantee. The complete planner preparation is larger than its net incremental
cost over contiguous, and the fresh-layout metric is more conservative than
steady-state throughput. Preparation was sampled once per case, so host jitter
is visible. Profile margins are explicit: 5% plus 40 ms on H20 and 20 ms on B300.

## Precision and calibration

All 186 policy/shape/platform results passed finite-value checks. Forward, inverse
and both autograd row-marker checks were bitwise exact. Every policy used the
same source input fingerprints and compared all 14 output/gradient tensors
against a lossless CPU copy of contiguous results. KDA output and all five KDA
gradients had zero difference. Worst CSA relative RMS was 0.145868% on H20 and
0.161238% on B300; maximum auxiliary KL absolute differences were 3.2014e-10
and 1.7462e-10. These are auxiliary indexer losses, not training loss results.

Four additional small FP32 attention/teacher/KL/pooling-gradient fixtures per
platform were tested under all three policies. The reference uses the selected
key set and checks causality, document isolation, uniqueness and tail coverage;
it does not independently prove the selector's discrete ranking.

Compute coefficients used seven calibration shapes and their C/Z runs.
The other 24 matrix cases were not used for compute fitting. Independent
communication sweeps held out sizes/topologies: absolute percentage error
averaged 7.14% on H20 and 10.53% on B300. Four additional mixed-link H20
checks had 2.01--3.73% error. These are effective-cost models, not physical
network simulators. No final E2E result is used as a case-specific lookup.

## Reproduction

Install the compatible cuDNN build and cuLA's `cudnn-hybrid` extra first. Under
CUDA 13, preserve the `[cu13]` extra when pinning `nvidia-cutlass-dsl`. Use the
appropriate NCCL/rendezvous configuration for the machine; rank-to-node mapping
must match the explicit profile. For the single-node B300 configuration:

```bash
OMP_NUM_THREADS=4 torchrun --standalone --nproc-per-node=8 -m benchmarks.bench_csa_layout_matrix \
  --suite validation --heads 64 --warmups 3 --repeats 10 \
  --planner-strategy selection \
  --planner-cost-model benchmarks/profiles/thd_selection_b300_cp8.json \
  --output-dir results/b300-cp8
python -m benchmarks.summarize_csa_layout_matrix results/b300-cp8 --output results/b300-summary.json
```

For H20, launch on two eight-GPU nodes with `--nnodes=2`, distinct node ranks
and shared rendezvous; select `thd_selection_h20_cp16.json`. The runner first
executes small FP32 checks, then all 31 cases, keeping contiguous first and
alternating the other two policies. `--list-cases` lists shapes without CUDA.

## Original submission source review (2026-11-15)

The full matrix above precedes the final CPU metadata cleanup. This cleanup
rejects noninteger transport intervals, removes an obsolete benchmark import,
fixes preparation accounting in the public summarizer and replaces a quadratic
DSA window candidate sum with an integer running total. Planner decisions and
numerical kernels are unchanged. That submission has separate regression
validation recorded below; the historical matrix is not relabeled as a new run.

Local CPU preparation microbenchmark (10 measured samples after one warmup):

| Documents | Old window preparation ms | New ms | Metadata |
|---|---:|---:|---|
| 22 | 0.087 | 0.083 | field-for-field equal |
| 1000 | 11.527 | 2.693 | field-for-field equal |
| 4096 | 153.647 | 7.648 | field-for-field equal |
| 16384 | 2404.650 | 25.638 | field-for-field equal |

These CPU-only measurements are not GPU or whole-chain speedups. The change
removes repeated scans of already-grouped documents; it does not change scratch
capacity, math precision or kernel launch boundaries.
## Original submission GPU regression

On both H20 CP16 and B300 CP8, the original submission source passed four small FP32 fixtures
and three representative 1M shapes (single long, 1000 documents with one long,
and 1000-document tiny mixture), each under C/Z/P: **42 complete CSA rows**.
The 1M rerun used three warmups and five measurements. A separate six-case KDA
A/B on each platform compared native document-local execution with the forced
boundary route, including weak decay and isolated ranks with a crossing peer:
**24 A/B rows**, with zero difference in output and all five gradients.

All 181 tested source/config files matched the original submission. The 62 original
planner decisions still match every measured rank's policy, row count, span
count and remote rows. All **2,232** DSA rank plans across the full 31 x 3 x
(CP16 + CP8) matrix have field-for-field identical window metadata before and
after the integer running-total optimization. The [regression record](results/thd_selection_review_20261115.json)
includes source hashes, correctness maxima and per-case timings.

337 focused CPU tests and both repository pre-commit hooks passed. Review
covered the complete proposed diff and its distributed/hybrid call paths:
coverage and ownership, collective participation, inverse autograd, storage
lifetimes, context reuse, cost-model bounds and benchmark accounting. No
handwritten GPU kernel, precision mode or math approximation is changed.
This is not a claim of a complete race audit of every unchanged kernel in cuLA.


## Post-review validation and timing protocol

The subsequent review fixes add group-wide source-alignment rejection, finite
calibration-input validation, strict JSON failure reports, and the SM90
independent-shard option preflight. The independent-shard KDA benchmark now
warms both routes and alternates baseline/candidate order for every measured
pair. Each raw sample records its pair index, actual execution order and
position; new reports identify this as `paired-alternating-v1`. Numerical
output/gradient comparison runs separately after both routes' timed samples.

The historical JSON result files and tables above are unchanged. In particular,
the original KDA A/B timings ran all baseline samples before all candidate
samples; they are not measurements under the new paired protocol. The recorded
source hashes describe those historical runs, not the later review fixes.
No fresh GPU, NCCL or performance rerun accompanies these fixes. Their
validation consists of focused CPU regressions and repository-wide Ruff
lint/format checks; a new paired GPU run is needed for updated timing claims.
