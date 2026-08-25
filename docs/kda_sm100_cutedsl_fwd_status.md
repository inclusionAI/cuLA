# SM100 CuTeDSL KDA forward status

Branch: `icavan/cutedsl-sm100-fwd`

Target machine: `aistudio-58650011-ssctl`, NVIDIA GB200, physical GPU 1
(`CUDA_VISIBLE_DEVICES=1`).

## Implemented candidates

- A csrc-aligned persistent CuTeDSL `recompute_w_u`: 384 threads, separate
  A/K/G/V and beta pipelines, co-produced K/V MMA-ready barrier, dp-lane
  aliased W/U TMEM accumulators, direct CUDA-core W/U stores, and vectorized
  KG stores. Equal-length, packed-uniform, and varlen dispatch are supported.
- A preprocessed specialization that consumes `k_scaled` and skips the fp32
  cumulative-gate load and duplicate `kg` store.
- An FP16 `k * exp2(gk)` workspace specialization. It keeps the same two-byte
  footprint as BF16, lets WU apply beta before the final BF16 MMA-operand
  rounding, and avoids adding a multiply to the K123 critical path.
- Fused raw-gate K1/K2/K3 intra candidate plus standalone fp32 Akk inverse.
- A csrc-boundary CuTeDSL intra entry point that consumes the same precomputed
  FP32 `gk` tensor as `chunk_kda_fwd_intra_cuda`; no gate activation or cumsum
  is fused into this correctness baseline. Its FP32-workspace inverse uses the
  csrc single-accumulator K=32 TF32 Schur order.
- A zero-copy packed-uniform route through the equal-length kernel.
- Same-input benchmarks against the repository SM100 csrc kernels.

The public csrc dispatch remains unchanged. The standalone recompute-WU port is
bitwise equal to csrc for W, U, and KG in the FP32- and BF16-beta tests. At the
representative T=8192 shape it is about 4% faster than csrc.

The csrc-boundary intra plus recompute-WU path is also bitwise equal for Aqk,
Akk, KG, W, and U. At `B=2,T=8192,H=64,K=V=128` it runs in 0.8645 ms versus
0.9157 ms for csrc, or 1.059x csrc throughput.

## GB200 results

All timings below use physical GPU 1. The representative shape is BF16
`B=2, H=64, K=V=128, chunk_size=64`.

### Modular recompute WU

| T | csrc (ms) | CuTeDSL (ms) | csrc / CuTeDSL |
|---:|---:|---:|---:|
| 512 | 0.0328 | 0.0704 | 0.466x |
| 1024 | 0.0517 | 0.0782 | 0.662x |
| 4096 | 0.1702 | 0.1613 | 1.055x |
| 8192 | 0.3261 | 0.3129 | 1.042x |
| 16384 | 0.6376 | 0.6222 | 1.025x |
| 32768 | 1.2618 | 1.2534 | 1.007x |

The standalone API computes the fp32 gate transform and writes `kg`, exactly as
the csrc baseline. The CuTeDSL runtime has a visible fixed launch/descriptor
cost at T=512 and T=1024; from T=4096 through T=32768 the aligned persistent
kernel meets or exceeds csrc throughput. The former staged-output/store-warp
CuTeDSL implementation has been removed.

### Csrc-boundary intra plus recompute WU

This comparison consumes the same precomputed FP32 `gk` tensor on both sides
and includes intra, the Akk inverse, and recompute WU. It uses three warmup
iterations and 20 CUDA-Event-timed iterations.

| Shape | csrc (ms) | CuTeDSL (ms) | csrc / CuTeDSL |
|---|---:|---:|---:|
| B=2, T=8192, H=64 | 0.9157 | 0.8645 | 1.059x |

For the measured input, `torch.equal` passes for every complete Aqk, Akk, KG,
W, and U tensor; each tensor has zero mismatched elements and zero maximum
absolute difference.

### Raw-gate fused intra plus specialized WU

The table below measures the complete csrc gate + intra + recompute chain
against CuTeDSL fused K1/K2/K3 + fp32 inverse + FP16-workspace preprocessed
W/U. Each point uses 10 warmup iterations and 30 measured iterations.

| Shape | csrc (us) | CuTeDSL (us) | csrc / CuTeDSL |
|---|---:|---:|---:|
| equal, T=4096 | 554.7 | 554.3 | 1.001x |
| equal, T=8192 | 1072.9 | 1065.4 | 1.007x |
| equal, T=16384 | 2101.4 | 2139.0 | 0.982x |
| varlen, lengths=4096,4096 | 561.6 | 558.9 | 1.005x |

Representative relative RMSE against csrc at T=8192 is:

- `Aqk`: 4.112e-4
- `Akk`: 7.065e-6
- `w`: 1.020e-3
- `u`: 4.838e-5

The W improvement comes from replacing only the internal `k_scaled` workspace
with FP16. Computing `bf16((k * beta) * exp2(gk))` directly in K123 improves W
parity to 4.233e-5 but costs about 30 us at T=8192, so it is retained only as
an experimental accuracy variant. The selected path does not add arithmetic
or bytes to K123.

### FP64 precision criterion

`tests/test_kda_sm100_intra_fused_cutedsl.py` builds Aqk, the unit-lower
inverse, W, and U from the same BF16 inputs in FP64. It compares both csrc and
CuTeDSL against that shared oracle and requires every CuTeDSL relative RMSE to
be no greater than the csrc error (apart from a 1e-6 comparison epsilon).

The strict check uses the same precomputed FP32 `gk` for both implementations
and now requires bitwise equality to csrc for Aqk, Akk, KG, W, and U. It passes
with both FP32 and BF16 beta at `B=1,T=256,H=4` and
`B=2,T=512,H=8`, and the representative BF16-beta benchmark passes at
`B=2,T=8192,H=64`. The standalone recompute-WU test independently requires
bitwise equality for W, U, and KG.

The source audit found why the earlier fused version missed the precision
target: it recomputed gate activation and the chunk scan with a different
reduction tree before intra. That changed the FP32 cumulative gate before any
TF32 MMA. Fusion and FP16 workspace rounding are therefore excluded from the
new baseline.

The former mismatch was isolated to the final lower-left 32x32 Schur block.
The CuTeDSL baseline split the K=32 product into two independently initialized
K=16 accumulators and then added their FP32 results. csrc instead issues the
four K=8 TF32 MMA steps into one accumulator. Replacing the split reduction
with the csrc order removes the different FP32 rounding point and makes the
complete csrc-boundary chain bitwise equal. The experimental in-CTA inverse is
still excluded from this claim and the boundary API rejects selecting it.

The csrc-boundary specialization also writes the complete 64x64 Aqk tile and
explicitly zeros the causal upper triangle. The earlier lower-tile-only store
left those global elements dependent on `torch.empty` allocator contents.
CuTe tensor-wrapper and varlen-padding caches now validate weak-reference
identity as well as tensor version, preventing Python object-ID reuse from
selecting a stale device pointer during long sequential workloads.

### Profile decomposition

The aligned recompute-WU kernel uses one 384-thread CTA per SM, 168 registers
per thread at launch, and about 199 KB dynamic shared memory. Nsight Compute
confirmed that the csrc launch is not a CUDA cluster; removing the accidental
single-CTA cluster launch from CuTeDSL removed about 4% at T=8192. Pipeline
barrier initialization is deferred so all seven pipelines share one CTA sync,
matching the single post-construction `__syncthreads()` in csrc.

## Appendix A: csrc-boundary bitwise and determinism stress

The bitwise-aligned csrc-boundary path was replayed 10,000,000 times on
physical GPU 1. The case uses BF16 beta at `B=1,T=256,H=4,K=V=128`. Before
stress replay, the harness compares every complete Aqk, Akk, KG, W, and U
tensor against csrc with `torch.equal`. It then captures CuTeDSL intra, the
Akk inverse, recompute WU, and the complete output comparison against csrc in
one CUDA Graph. Every replay therefore checks every output element.

```bash
CUDA_VISIBLE_DEVICES=1 python \
  benchmarks/stress_kda_sm100_csrc_boundary_determinism.py \
  --iterations 10000000 --checkpoint 1000000 \
  --report-json /tmp/kda_sm100_csrc_boundary_10m.json
```

Bitwise alignment before graph replay:

| Output | `torch.equal` | Mismatched elements | Max absolute difference |
|---|---:|---:|---:|
| Aqk | true | 0 | 0 |
| Akk | true | 0 | 0 |
| KG | true | 0 | 0 |
| W | true | 0 | 0 |
| U | true | 0 | 0 |

Determinism result:

- Iterations: 10,000,000
- Exact element mismatches accumulated across all replays: 0
- Elapsed time: 271.881 seconds
- Throughput: 36,780.9 iterations/second
- Status: passed

The FP32-beta specialization is covered separately by the strict bitwise
pytest cases. The 10,000,000-replay stress above uses the representative BF16
beta specialization.

## Appendix B: experimental raw-gate varlen determinism stress

The complete CuTeDSL varlen forward chain was replayed 10,000,000 times on
physical GPU 1. The case uses four deliberately unaligned sequence lengths
`[65, 127, 193, 255]`, `H=8`, and therefore exercises partial chunks and the
non-pure varlen path.

```bash
CUDA_VISIBLE_DEVICES=1 python benchmarks/stress_kda_sm100_varlen_determinism.py \
  --iterations 10000000 --checkpoint 1000000 \
  --report-json /tmp/kda_sm100_varlen_10m.json
```

The complete forward plus exact-output comparison is captured in one CUDA
Graph. Every replay compares every element of `k_scaled`, `kg`, `q_scaled`,
`gk_last_exp`, `Aqk`, `Akk`, `w`, and `u` against the first-run golden output;
the device-side mismatch counter is checked every 1,000,000 iterations.

- Iterations: 10,000,000
- Exact element mismatches: 0
- Elapsed time: 471.316 seconds
- Throughput: 21,217.2 iterations/second
- Aqk/Akk upper-triangle max absolute value: 0

Accuracy against the csrc path for the same inputs:

| Output | Relative RMSE | Max absolute error |
|---|---:|---:|
| k_scaled | 3.894e-8 | 2.384e-7 |
| kg | 4.572e-5 | 4.883e-4 |
| q_scaled | 6.323e-7 | 3.815e-6 |
| gk_last_exp | 4.398e-9 | 4.470e-8 |
| Aqk | 4.397e-4 | 1.221e-4 |
| Akk | 1.466e-5 | 4.883e-4 |
| w | 2.163e-3 | 9.766e-4 |
| u | 9.090e-5 | 9.766e-4 |

## Rejected experiments

- High-occupancy small-tile WU: slower than the warp-specialized baseline.
- Fully persistent WU scheduling: correct after alias fences, but slower due to
  insufficient latency hiding.
- Shared chunk-end exponent with reciprocal: exact division was much slower;
  approximate reciprocal plus synchronization also regressed.
- PDL chaining of K123, inverse, and WU: no measurable overlap benefit.
- In-CTA fused inverse with a single G buffer: serialized G staging and
  regressed the full chain to about 2.04 ms.
- In-CTA fused inverse while aliasing the Aqk stage: numerically correct, but
  serializes inverse work across four persistent chunks and regresses T=8192
  to 2068.8 us.
- Eight chunks per persistent workgroup: T=16384 regresses to 3014.9 us.
- Two chunks per workgroup: violates the current phase/pre-arrive protocol and
  fails the dependent inverse launch; four remains required.
- Fused pairwise cumsum plus exact W operand: reduces the fused Aqk FP64
  regression from 0.164% to 0.104% and makes W equal to csrc, but still changes
  the csrc gate boundary and runs at 1143.7 us. It is no longer the correctness
  baseline.
- Precomputed-FP32-gk diagnostic with the old fused shell: passes the strict
  FP64 gate but took 1675.2 us at T=8192. The completed csrc-boundary port
  skips fused K1 output work and now measures 864.5 us at the same target
  shape.

## Remaining validation

- Replace the experimental fused K2 warp-MMA/shared-memory path with the csrc
  TMEM/UMMA residency before claiming that fused raw-gate candidate is also a
  one-to-one implementation.
- Extend the performance matrix to additional batch/head combinations before
  changing the default public dispatch.
- Keep the csrc fallback for shapes that do not satisfy the current K=V=128,
  chunk-size=64 specialization constraints.

The SM100 tests cover float32/bf16 beta, the preprocessed path, the FP64 oracle
criterion, and bitwise equality of the packed-uniform fast path with the equal
kernel.
