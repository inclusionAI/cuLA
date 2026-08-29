# Gated DeltaNet-2 Prefill API on Hopper SM90

cuLA provides a fully fused, packed variable-length Gated DeltaNet-2 (GDN2)
forward-prefill kernel for NVIDIA Hopper GPUs. The implementation is written in
CuTe DSL and has one product backend:
`sm90a_cutedsl_gdn2_prefill_v1`. Unsupported inputs fail explicitly; the public
entry point does not fall back to Triton or another cuLA kernel.

## Requirements

- NVIDIA compute capability 9.0 (Hopper SM90, including H20 and H100)
- Python 3.12 or newer
- CUDA and PyTorch versions supported by the surrounding cuLA installation
- `nvidia-cutlass-dsl>=4.5.1,<4.7`. `is_sm90_gdn2_available()` and the
  dispatch error enforce exactly this range and report the backend
  unavailable outside it. Both endpoints are exercised on H20: 4.5.1 and
  4.6.2 each pass the full product suite. This range is narrower than the
  project-wide dependency in `pyproject.toml`, because the kernel needs
  `cutlass.cute.nvgpu.OperandMajorMode`, which 4.4.x does not provide — the
  backend cannot even be imported there. The upper bound matches the
  repository-wide CuTeDSL contract in `cula/ops/_mlir_compat.py`, which is
  enforced independently on every private-dialect access; GDN2 only raises
  the floor, and reads the installed version through that same gateway.

## API

```python
from cula.gdn2 import chunk_gdn2
```

The public tensors use packed-token layouts:

| Argument | Dtype | Shape | Meaning |
|---|---|---|---|
| `q` | BF16 | `[T,16,128]` | queries |
| `k` | BF16 | `[T,16,128]` | keys |
| `v` | BF16 | `[T,Hv,128]` | values |
| `g` | FP32 | `[T,16,128]` | finite log decay in `[-5, 0]` |
| `b` | BF16 | `[T,16,128]` | erase gate in `[0,1]` |
| `w` | BF16 | `[T,Hv,128]` | write gate in `[0,1]` |
| `cu_seqlens` | INT64 | `[N+1]` | packed sequence prefixes |
| `initial_state` | FP32 | `[N,Hv,128,128]` | optional state in public `[V,K]` orientation |

`Hv` may be 16, 32, or 64. This gives MHA (`Hv=16`), GVA2 (`Hv=32`),
or GVA4 (`Hv=64`). GQA and other head relationships are outside the first
product contract. `N` must be in `[1,32]`, and every sequence must contain at
least one token.

All tensors must be contiguous, CUDA-resident on the same device, and at least
16-byte aligned. The default path validates tensor metadata without copying
device values to the host. The caller must ensure:

- `cu_seqlens[0] == 0`;
- `cu_seqlens[-1] == T`;
- offsets are strictly increasing;
- `g` is finite and in `[-5, 0]` elementwise (the blockwise-rebased
  factorization bound; see
  [GDN2 SM90 stable factorization](gdn2_sm90_stable_factor.md));
- `b` and `w` are finite and in `[0,1]`.

Pass `validate_inputs=True` for a synchronous diagnostic check of those value
preconditions. Do not enable it on a latency-sensitive steady-state path.

The output is BF16 `[T,Hv,128]`. With `output_final_state=True`, the function
returns `(output, final_state)`, where `final_state` is FP32
`[N,Hv,128,128]` in the same public `[V,K]` orientation accepted by
`initial_state`. Callers may provide contiguous `output` and `output_state`
buffers.

## Numerical validation

The SM90 correctness suite uses an independent tokenwise PyTorch FP32
recurrence and requires finite values from both implementations. The frozen
comparison policies are:

| Result | Product dtype | Reference accumulation | `rtol` | `atol` |
|---|---|---|---:|---:|
| Output | BF16 | FP32, cast to BF16 for comparison | `0.01` | `0.01` |
| Final state | FP32 | FP32 | `0.001` | `0.005` |

An unexpected `NaN` or `Inf` in either output or final state fails the test
before tolerance comparison. Repeated product launches with identical inputs,
initial state, seed, and configuration must be bitwise exact.

## Example

```python
import torch

from cula.gdn2 import chunk_gdn2

device = "cuda"
lengths = (65, 1)
total_tokens = sum(lengths)
query_heads = 16
value_heads = 32  # GVA2

q = torch.randn(
    total_tokens,
    query_heads,
    128,
    device=device,
    dtype=torch.bfloat16,
)
k = torch.randn_like(q)
v = torch.randn(
    total_tokens,
    value_heads,
    128,
    device=device,
    dtype=torch.bfloat16,
)
g = -torch.rand_like(q, dtype=torch.float32) * 0.05
b = torch.rand_like(q)
w = torch.rand_like(v)
cu_seqlens = torch.tensor([0, 65, 66], device=device, dtype=torch.int64)

output, final_state = chunk_gdn2(
    q,
    k,
    v,
    g,
    b,
    w,
    cu_seqlens=cu_seqlens,
    output_final_state=True,
)
```

Pass a returned `final_state` as `initial_state` in a later call to continue
the recurrence without changing orientation.

## Unsupported behavior

- compute capability other than 9.0;
- `Hq` other than 16 or `Hv` outside `{16,32,64}`;
- `N > 32`, zero-length sequences, or `T > 2^31-1`;
- key or value size other than 128;
- FP16, FP8, or FP32 Q/K/V;
- GQA, decode, backward, or intermediate state checkpoints;
- implicit fallback to FLA Triton or another backend.

Unsupported metadata fails before compilation or launch with `ValueError`,
`TypeError`, `NotImplementedError`, or `RuntimeError`.

## Canonical benchmark

From the repository root, run:

```bash
python benchmarks/bench_gdn2_prefill.py \
  --implementation both \
  --output gdn2-sm90-benchmark.json
```

The five immutable rows cover MHA, GVA2, GVA4, packed variable-length input,
all four initial/final-state modes, `T={64,1024,4096}`, and `N={1,20}`.
Compilation is recorded separately and excluded from CUDA-event timing. The
FLA comparison includes any GVA head expansion inside its timed public logical
call. Use `--list-matrix` to inspect the rows without running CUDA work.

The standalone benchmark is a developer diagnostic. Release performance claims
must additionally use fresh processes, fresh caches, alternating implementation
order, exact source/input identities, and independently replayable raw timing
receipts.

## Deterministic stress and Compute Sanitizer

Run the deterministic SM90 stress matrix in one process:

```bash
python tests/gdn2/stress_gdn2_sm90.py \
  --iterations 100000 \
  --warmup 1 \
  --device 0 \
  --output gdn2-sm90-stress.json
```

The six-row round-robin matrix covers all six product compile
specializations, MHA/GVA2/GVA4, all four state modes, `T={64,1024,4096}`,
`N=32`, and irregular packed tails. Every launch is compared bitwise with its
fixed-input output/final-state baseline using CUDA-stream-ordered checks. The
harness also accumulates non-finite counts and verifies input hashes,
redzones, backend identity, GPU identity, and the exact launch count.

Run repeated stress under all four applicable NVIDIA Compute Sanitizer tools:

```bash
tests/gdn2/run_compute_sanitizer_sm90.sh \
  gdn2-sm90-sanitizers \
  120
```

The runner uses separate empty compiler caches and records the exact command,
return code, stdout/stderr, JSON receipt, and SHA-256 evidence manifest for
each of:

- `memcheck --leak-check full`;
- `initcheck`;
- `synccheck --check-warpgroup-mma yes`;
- `racecheck --racecheck-report hazard --racecheck-memcpy-async yes
  --racecheck-trace-sync yes`.

The runner takes its interpreter from `PYTHON` and falls back to `python3`.
Set it explicitly to the interpreter whose `nvidia-cutlass-dsl` is inside the
supported range — a bare `python3` may resolve to a system interpreter with a
different, unsupported version, in which case the backend now fails closed
rather than silently producing evidence on an unvalidated toolchain.

For a source-bound release run, set `GDN2_SOURCE_MANIFEST` to the frozen
source manifest and `GDN2_REQUIRED_GPU_UUID` to the expected `GPU-...`
identity before invoking the sanitizer runner.

For the kernel decomposition and scheduling policy, see
[GDN2 SM90 Pipeline](gdn2_sm90_pipeline.md).
