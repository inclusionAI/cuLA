# CSA CP selection at 128K and 256K

The [expanded three-way comparison](CSA_CP_THREE_WAY_RESULTS.md) covers 1M
contexts and 22-/64-document packs, for 24 cases, and has been rerun after the
equivalent host-preparation optimization.
The historical ten-case statistics below remain scoped to this original study.

This tests the public `cudnn_csa_select_cp` API on eight B300 GPUs. The workload
is **one complete CSA operator forward/backward**, including ratio-4 pooling,
pooled-key and attention-KV gathers, selection routing/restoration, sparse
attention, teacher/KL, and all activation/parameter gradients. It excludes
KDA, projections, MoE, optimizer and checkpoint recomputation. These numbers
must not be substituted for a connected seven-KDA/one-CSA training block.

## Protocol

- CSA H64/D576/V512; indexer H64/D128; Top-K512; BF16 Q/K/weights; FP32 scores,
  nonlinearities and communication reductions. TF32 and BF16 reduced-precision
  GEMM intermediate reductions are disabled. Native nondeterministic mode.
- PyTorch 2.12.1+cu130, CUDA runtime 13.0, CuTeDSL 4.8.1. cuDNN Frontend source
  `855ccd77ba9efed22441fd454f7cc0206a068e94`: all 3,285 checked Python/CUDA/C++
  source files match that commit. The installed frontend distribution version
  in the JSON is wheel metadata, not the identity of the imported source tree.
- Five distributions per context: one document, 1,024 equal documents, 512
  slightly unequal documents, 998 short plus two long documents, and a
  1,000-document short/medium/long-tail mixture. `bench_csa_cp_windows.cases`
  specifies exact lengths without a tokenizer or private dataset.
- Two warmups, seven execution samples and eight fresh preparation samples.
  Policy order is rotated and reversed. Tables use the median of each sample's
  maximum-rank duration. Preparation includes CPU planning, workspace/index
  construction, H2D copies and a completion sync. It is rebuilt, not cached.
  RNG generation, JIT compilation and correctness oracles are outside timing.
- Throughput experiment: existing compute/transport coefficients, explicit
  `min_gain_ms=0`, `max_chunks_per_rank=6`, and `min_chunk_tokens=1024`.
  This tests reuse-oriented planning; **zero is not a recommendation for a
  fresh layout used once**. It does not replace the shipped 1M profile.

## Complete CSA results

C = public contiguous; Z = public complete per-document zigzag; P = public
planner. All times are milliseconds. The last column compares **preparation
plus one step** against the faster C/Z under that same accounting.

| Case | C step | Z step | P step | C preparation | P preparation | P one-use gap |
|---|---:|---:|---:|---:|---:|---:|
| 128k-long-tail-1000 | 32.37 | 37.90 | 32.35 | 7.58 | 12.13 | +11.34% |
| 128k-mixed-1000 | 33.07 | 38.75 | 33.07 | 7.64 | 28.31 | +50.77% |
| 128k-near-uniform-512 | 10.01 | 11.21 | 10.03 | 3.74 | 4.96 | +8.92% |
| 128k-single | 47.60 | 46.23 | 45.82 | 0.73 | 7.55 | +13.21% |
| 128k-uniform-1024 | 9.29 | 11.10 | 9.30 | 6.78 | 8.97 | +13.70% |
| 256k-long-tail-1000 | 99.36 | 115.25 | 92.20 | 9.70 | 31.81 | +13.72% |
| 256k-mixed-1000 | 83.64 | 96.05 | 82.15 | 9.59 | 32.67 | +23.16% |
| 256k-near-uniform-512 | 21.79 | 24.21 | 21.76 | 3.94 | 5.02 | +4.10% |
| 256k-single | 115.29 | 105.58 | 104.15 | 1.12 | 7.79 | +4.89% |
| 256k-uniform-1024 | 18.96 | 22.56 | 18.94 | 6.91 | 8.97 | +7.89% |

The worst measured steady-state gap against faster C/Z is **+0.142%**. Single
long documents improve over contiguous by about **3.7% at 128K** and **9.7% at
256K**. The 256K long-tail case improves about **7.2%**. These are observed
candidate comparisons, not a proof of optimality over arbitrary THD inputs.

**Fresh preparation is still a limitation.** Searching can cost more than the
execution it saves, particularly when a mixed layout ultimately returns to
contiguous. The complete preparation-plus-one-step result is not uniformly
near the best fixed policy, and this standalone CSA study does not establish
an average <3% preparation bound. Host-time spikes remain in the raw critical
samples; no samples or garbage-collection time were removed. For example, the
256K long-tail planner preparation has a 31.81 ms median and a 132.93 ms maximum.
Do not infer a stable preparation budget from a single sample.

## Sensitivity and next cost-model work

Two additional experiments are retained in the machine-readable results:

- Keeping the historical **20 ms** absolute headroom selected contiguous in
  all ten cases, missing useful single-long-document routing.
- Reducing search to **two chunks/rank with 2 ms headroom** retained the single
  long-document gains, but the 256K two-long/998-short mixture selected a route
  **2.84% slower** than contiguous. Full search found a faster route. Preparation
  for that exploratory run has only three samples and is not directly comparable
  to the final eight-sample protocol.

These are sensitivity experiments, **not independent short-context coefficient
calibration**. Neither a smaller fixed headroom nor fewer search grids is a
universal fix. A deployment should budget expected reuse and measure its message
sizes and kernel granularity. Further improvement should price the preparation
of each candidate and the value of additional search, rather than adding a
separate algorithm or case-name lookup for each context length. The interface
keeps costs, capacity, search granularity and minimum fragment size explicit.

The fragment floor prevents increasingly fine splitting at large CP. At
128K/CP64, a 1,024-token floor caps an adaptive partition at 128 pieces rather
than the requested 384. Original short source fragments remain legal; a
candidate violating the floor after capacity repair is rejected. **CP64 here
is CPU metadata validation only**, not a 64-GPU performance measurement.

## Correctness and review

- **435 focused CPU tests pass**, including API validation, transport inverses,
  empty targets, capacities, metadata parity at 128K/256K/1M and CP64 floors.
- **124 historical plans are exactly unchanged** at the default zero floor:
  31 distributions × CP8/CP16 × zero/original headroom, compared field-for-field
  against the previous PR head. This is host-plan parity, not a new H20 run.
- **128 native subgroup checks pass**, covering BF16 and FP32 weights, unequal
  source lengths, empty destinations, exact IDs/scores versus unrouted native
  selection, and retained results across workspace reuse. Maximum score
  difference is zero in these fixtures.
- All ten final context/distribution cases pass full-tensor comparison of the
  output and seven gradients against the legacy contiguous implementation.
  Largest relative RMS is **0.214%**; maximum auxiliary loss absolute difference
  is **4.66e-10**. The native nondeterministic computation is not bitwise stable;
  this does not change its dtype or numerical approximation contract.
- Three small fixtures under all four paths pass independent FP32 attention,
  teacher/KL and pooling-gradient checks. These references use the selected key
  set; they do not independently prove native discrete ranking.
- The 125 source files in the final timing manifest match the reviewed checkout.
  Review checked group participation, unique inverse-permutation writes, source
  ownership, empty destinations, retained outputs, explicit stream dependencies,
  dtype validation and absence of D2H metadata reads. No handwritten GPU kernel
  or native precision mode is changed.

Allocator peaks in the raw harness include four coexisting comparison fixtures;
they are **not standalone per-policy memory claims**. The JSON separately
records candidate-workspace bytes for each rank.

[Machine-readable results, error envelopes, samples and source hashes](results/csa_cp_windows_20261116.json)

## Reproduction

```bash
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m benchmarks.bench_csa_cp_windows \
  --profile benchmarks/profiles/thd_selection_b300_cp8.json \
  --output-dir results/csa-windows --tokens 131072 262144 \
  --cases single uniform-1024 near-uniform-512 mixed-1000 long-tail-1000 \
  --min-gain-ms 0 --min-chunk-tokens 1024 --max-chunks-per-rank 6 \
  --warmups 2 --repeats 7 --prepare-repeats 8

python -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m benchmarks.check_csa_cp_selection --weights-dtype fp32
```

See the [integration contract](CSA_CP_SELECTION.md) for plan/workspace lifetimes,
source ownership and explicit preparation. Calibrate for the actual deployment;
this experiment does not install a universal small-context profile.
