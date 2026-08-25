# SM100 CuTeDSL KDA forward status

Branch: `icavan/cutedsl-sm100-fwd`

Target machine: `aistudio-58650011-ssctl`, NVIDIA GB200, physical GPU 1
(`CUDA_VISIBLE_DEVICES=1`).

## Implemented candidates

- Warp-specialized CuTeDSL `recompute_w_u`, including equal-length, packed
  uniform, and varlen dispatch.
- A preprocessed specialization that consumes `k_scaled` and skips the fp32
  cumulative-gate load and duplicate `kg` store.
- An FP16 `k * exp2(gk)` workspace specialization. It keeps the same two-byte
  footprint as BF16, lets WU apply beta before the final BF16 MMA-operand
  rounding, and avoids adding a multiply to the K123 critical path.
- Fused raw-gate K1/K2/K3 intra candidate plus standalone fp32 Akk inverse.
- A zero-copy packed-uniform route through the equal-length kernel.
- Same-input benchmarks against the repository SM100 csrc kernels.

The public csrc dispatch remains unchanged. The FP16-workspace candidate is
faster at the representative T=8192 shape and for packed-uniform T=4096+4096,
but the T=16384 point is still 1.8% slower and therefore blocks a global
dispatch change.

## GB200 results

All timings below use physical GPU 1. The representative shape is BF16
`B=2, H=64, K=V=128, chunk_size=64`.

### Modular recompute WU

| T | csrc (ms) | CuTeDSL (ms) | csrc / CuTeDSL |
|---:|---:|---:|---:|
| 4096 | 0.1702 | 0.1849 | 0.920x |
| 8192 | 0.3280 | 0.3586 | 0.915x |
| 16384 | 0.6410 | 0.7070 | 0.907x |

The standalone API computes the fp32 gate transform and writes `kg`; it remains
8-9% slower than csrc. The forward path instead uses the preprocessed
specialization: fused intra already produced `k_scaled` and `kg`, so W/U skips
the duplicate fp32 gate load, exponentiation, and `kg` store. This is the path
used in the parity table below.

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
be at most 1.05 times the csrc error. The FP16-workspace candidate passes all
four checks; this replaces csrc-output parity as the precision acceptance
criterion.

### Profile decomposition

At T=16384, Nsight Systems reports K123 / inverse / WU averages of
1386.0 / 216.1 / 519.4 us. Nsight Compute reports K123 at about 79% memory
throughput (L1/TEX is the dominant path), 48% achieved occupancy, and one CTA
per SM due to both registers and roughly 220 KB shared memory. WU reaches only
25% theoretical occupancy with two CTAs per SM and is latency limited. This is
why launch fusion and extra rounding do not close the long-sequence slope.

## Varlen determinism stress

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

## Remaining validation

- Replace the K2 warp-MMA/shared-memory schedule with the csrc-style TMEM/UMMA
  residency, or reduce WU latency, to recover the remaining 37.6 us at T=16384.
- Extend the performance matrix to additional batch/head combinations before
  changing the default public dispatch.
- Keep the csrc fallback for shapes that do not satisfy the current K=V=128,
  chunk-size=64 specialization constraints.

The SM100 tests cover float32/bf16 beta, the preprocessed path, the FP64 oracle
criterion, and bitwise equality of the packed-uniform fast path with the equal
kernel.
