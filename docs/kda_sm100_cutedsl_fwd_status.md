# Pure CuTeDSL SM100 KDA forward

The SM100/SM103 modular KDA forward no longer builds or loads a C++/CUDA
extension. Intra-chunk Aqk/Akk, the csrc-ordered TF32 inverse, recompute W/U,
KG, and optional QG are implemented in CuTeDSL. The remaining native cuLA
extension is SM90-only.

## Compatibility and dispatch

The public `chunk_kda_fwd_intra` API supports the production specialization:

- BF16 Q/K/V, BF16 or FP32 beta;
- `K=V=128` and chunk size 64;
- MHA and grouped-value attention;
- fixed length, packed varlen, partial chunks, and unaligned fixed-length tails;
- recompute and `disable_recompute=True` QG output;
- the default csrc-compatible TF32 inverse.

Aligned fixed-length inputs with batch size at least two use one composed host
dispatch for the exact intra, inverse, and recompute-WU kernels. Batch-one,
partial, and packed-varlen inputs use the same exact CuTeDSL kernels through
the modular host path. Batch one deliberately avoids a composed-host compiler
failure observed in CUTLASS DSL 4.6.2 and 4.6.3; it does not fall back to C++.

`unified_gref=True` and `use_tf32_inverse=False` are rejected explicitly
because neither option belongs to the bitwise-compatible production path.

## Mathematical and bitwise contract

The port preserves the former csrc kernel boundary: it consumes the same FP32
activated, chunk-cumulative `gk` and does not fuse gate activation or cumsum.
It also preserves the four K=8 TF32 MMA steps in one inverse accumulator, beta
rounding, BF16 conversion points, causal masking, GVA head mapping, and partial
chunk behavior.

Before the C++ sources were removed, 21 direct migration tests passed on
SM100. Complete tensors were compared with `torch.equal`, including causal
upper triangles, for FP32/BF16 beta, MHA, GVA, packed partial-varlen, unaligned
fixed tails, QG, KG, W, U, Aqk, and Akk. All mismatched-element counts and
maximum absolute differences were zero.

Six representative former-csrc outputs are retained as SHA-256 regression
vectors in `tests/test_kda_sm100_csrc_golden.py`. They cover both beta dtypes
across MHA, GVA partial-varlen, and unaligned tails. Pure CuTeDSL CI regenerates
the full outputs and compares every byte to those frozen csrc vectors.

## Performance against the removed csrc implementation

The table measures the complete public forward boundary: exact intra,
standalone exact inverse, and recompute-WU. Every measured output was bitwise
equal to csrc. Values are CUDA-event medians in milliseconds on GB200; ratios
greater than one favor CuTeDSL.

| Beta | B | T | HQK/HV | csrc | CuTeDSL | csrc / CuTeDSL |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 2 | 512 | 64/64 | 0.0803 | 0.0809 | 0.993x |
| BF16 | 2 | 1024 | 64/64 | 0.1379 | 0.1320 | 1.044x |
| BF16 | 2 | 4096 | 64/64 | 0.4696 | 0.4362 | 1.076x |
| BF16 | 2 | 8192 | 64/64 | 0.9152 | 0.8896 | 1.029x |
| FP32 | 2 | 512 | 64/64 | 0.0804 | 0.0810 | 0.992x |
| FP32 | 2 | 1024 | 64/64 | 0.1380 | 0.1326 | 1.040x |
| FP32 | 2 | 4096 | 64/64 | 0.4692 | 0.4263 | 1.101x |
| FP32 | 2 | 8192 | 64/64 | 0.9164 | 0.8564 | 1.070x |
| BF16 | 1 | 4096 | 16/64 | 0.2468 | 0.2379 | 1.038x |

The 512-token difference is within one percent and treated as measurement
noise. The main training lengths are 2.9% to 10.1% faster. The acceleration
comes from the CuTeDSL kernel/host implementation; gate activation and cumsum
are intentionally not fused into this bitwise baseline.

## Determinism

The stress harness captures the complete pure CuTeDSL chain plus an exact
device-side comparison in one CUDA Graph. Every replay compares every output
element. Run it with:

```bash
python benchmarks/stress_kda_sm100_cutedsl_determinism.py \
  --iterations 10000000 --checkpoint 1000000 \
  --report-json /tmp/kda_sm100_cutedsl_10m.json
```

The packed partial-varlen raw-gate candidate has a separate harness:

```bash
python benchmarks/stress_kda_sm100_varlen_determinism.py \
  --iterations 10000000 --checkpoint 1000000 \
  --report-json /tmp/kda_sm100_varlen_10m.json
```

Both harnesses retain a first-run golden output, reject NaN/Inf, compare every
output bit on every replay, and report checkpoint progress plus a JSON result.

## Validation gates

- frozen former-csrc bitwise vectors;
- FP64 mathematical oracle bounds;
- repeated-call and CUDA Graph determinism;
- default-stream and concurrent-stream isolation;
- output lifetime, pool reuse, and bounded retained memory;
- fixed, GVA, QG, packed-varlen, and tail routing;
- installation without an SM100 native extension;
- SM90 extension build retained independently.
