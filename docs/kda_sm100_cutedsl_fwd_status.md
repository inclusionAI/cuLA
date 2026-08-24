# SM100 CuTeDSL KDA forward status

Branch: `icavan/cutedsl-sm100-fwd`

Target machine: `aistudio-58650011-ssctl`, NVIDIA GB200, physical GPU 1
(`CUDA_VISIBLE_DEVICES=1`).

## Implemented candidates

- Warp-specialized CuTeDSL `recompute_w_u`, including equal-length, packed
  uniform, and varlen dispatch.
- A preprocessed specialization that consumes `k_scaled` and skips the fp32
  cumulative-gate load and duplicate `kg` store.
- Fused raw-gate K1/K2/K3 intra candidate plus standalone fp32 Akk inverse.
- Same-input benchmarks against the repository SM100 csrc kernels.

The public csrc dispatch remains unchanged while the CuTeDSL candidate is
validated on a broader shape matrix. The fused forward chain now matches the
csrc implementation within 0.1% at the representative T=8192 shape and within
2.0% across the measured equal-length matrix.

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
against CuTeDSL fused K1/K2/K3 + fp32 inverse + preprocessed W/U. Each point
uses 10 warmup iterations and 100 measured iterations.

| Shape | csrc (us) | CuTeDSL (us) | csrc / CuTeDSL |
|---|---:|---:|---:|
| equal, T=4096 | 553.4 | 554.1 | 0.999x |
| equal, T=8192 | 1068.0 | 1068.6 | 0.999x |
| equal, T=16384 | 2099.0 | 2141.1 | 0.980x |
| varlen, lengths=4096,4096 | 559.3 | 565.5 | 0.989x |

Representative relative RMSE against csrc at T=8192 is:

- `Aqk`: 4.112e-4
- `Akk`: 7.065e-6
- `w`: 2.836e-3
- `u`: 4.838e-5

The final parity improvement came from sharing each chunk's 64 beta values in
SMEM across the ten MMA tile warps. Other retained changes vectorize K123 and
inverse loads/stores, clear only the six inverse upper tiles, reuse prefix
results, hoist gate invariants, and use ordinary stream ordering between K123
and the inverse instead of PDL fences.

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
- Elapsed time: 471.308 seconds
- Throughput: 21,217.6 iterations/second
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

## Remaining validation

- Extend the performance matrix to additional batch/head combinations before
  changing the default public dispatch.
- Keep the csrc fallback for shapes that do not satisfy the current K=V=128,
  chunk-size=64 specialization constraints.

`tests/test_kda_sm100_recompute_wu_cutedsl.py` passes on the target GB200 for
float32 and bf16 beta in the generic path and for the preprocessed path against
a Torch reference.
