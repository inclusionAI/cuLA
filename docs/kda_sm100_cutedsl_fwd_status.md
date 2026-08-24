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

The public csrc dispatch remains unchanged because the measured CuTeDSL paths
do not yet meet the no-regression performance requirement.

## GB200 results

All timings below use physical GPU 1. The representative shape is BF16
`B=2, H=64, K=V=128, chunk_size=64`.

### Modular recompute WU

| T | csrc (ms) | CuTeDSL (ms) | csrc / CuTeDSL |
|---:|---:|---:|---:|
| 4096 | 0.1700 | 0.1974 | 0.861x |
| 8192 | 0.3272 | 0.3829 | 0.855x |
| 16384 | 0.6399 | 0.7556 | 0.847x |

The best measured register split is 216 registers for the four CUDA compute
warps and 32 for the load/MMA/store warps, while retaining two CTAs per SM.

### Raw-gate fused intra plus specialized WU

For `B=2, T=8192, H=64`, the csrc gate + intra + recompute chain takes
1.0692 ms. The CuTeDSL fused K1/K2/K3 + fp32 inverse + specialized WU takes
1.3162 ms (0.81x). Representative relative RMSE against csrc is:

- `Aqk`: 4.112e-4
- `Akk`: 7.065e-6
- `w`: 2.836e-3
- `u`: 4.838e-5

## Rejected experiments

- High-occupancy small-tile WU: slower than the warp-specialized baseline.
- Fully persistent WU scheduling: correct after alias fences, but slower due to
  insufficient latency hiding.
- Shared chunk-end exponent with reciprocal: exact division was much slower;
  approximate reciprocal plus synchronization also regressed.
- PDL chaining of K123, inverse, and WU: no measurable overlap benefit.
- In-CTA fused inverse with a single G buffer: serialized G staging and
  regressed the full chain to about 2.04 ms.

## Required next step

Closing the remaining gap needs a dataflow change rather than another launch
or register tweak. The promising direction is a direct CuTeDSL port of the
csrc persistent intra schedule, with inverse and WU sharing the same per-chunk
SMEM/TMEM residency so the inverted Akk matrix is not written and reread from
global memory. Default API dispatch must remain on csrc until that path is at
least 1.0x across the benchmark matrix.
