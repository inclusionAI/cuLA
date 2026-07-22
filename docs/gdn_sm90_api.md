# GDN Prefill API on Hopper SM90

cuLA provides a packed variable-length Gated DeltaNet (GDN) prefill kernel for
NVIDIA Hopper GPUs. The implementation is written in CuTe DSL, launches one
512-thread CTA for each `(sequence, output_head)` pair, and does not load the
cuLA C++ extension or require FlashInfer headers at runtime.

## Requirements

- NVIDIA compute capability 9.0 (Hopper SM90, including H20 and H100)
- Python 3.10 or newer
- CUDA and PyTorch versions supported by the surrounding cuLA installation
- `nvidia-cutlass-dsl==4.5.1`

The SM90 GDN entry point checks the GPU capability and installed CuTe DSL
version before compilation. It does not fall back to another implementation
when either requirement is missing.

## API

```python
from cula.gdn import chunk_gated_delta_rule
```

The input tensors use packed-token layouts:

| Argument | Dtype | Shape | Meaning |
|---|---|---|---|
| `q` | BF16 | `[total_tokens, Hq, 128]` | queries |
| `k` | BF16 | `[total_tokens, Hk, 128]` | keys |
| `v` | BF16 | `[total_tokens, Hv, 128]` | values |
| `g` | FP32 | `[total_tokens, Ho]` | positive, finite forget factors; `None` means one |
| `beta` | FP32 | `[total_tokens, Ho]` | finite update factors; `None` means one |
| `cu_seqlens` | INT32 or INT64 | `[num_sequences + 1]` | packed sequence prefixes |
| `initial_state` | FP32 | `[num_sequences, Ho, 128, 128]` | optional state in `[V,K]` orientation |

`Ho = max(Hq, Hv)`. Supported head relationships are:

- MHA: `Hq = Hk = Hv`;
- GQA: `Hq` is an integer multiple of `Hk = Hv`;
- GVA: `Hv` is an integer multiple of `Hq = Hk`.

All tensors must be contiguous, CUDA-resident on the same device, and at least
16-byte aligned. Every packed sequence must contain at least one token.
`cu_seqlens[0]` must be zero and its final value must equal `total_tokens`.
These CUDA-resident values are trusted caller preconditions on the default fast
path so that each request does not force device-to-host synchronization.

For debugging or input-pipeline validation, pass `validate_inputs=True`. This
explicitly copies `cu_seqlens` to the host and checks its endpoints and strictly
increasing contents, and synchronously verifies that `g` is finite and positive
and `beta` is finite. Do not enable this diagnostic option in latency-sensitive
steady-state execution.

The output is BF16 `[total_tokens, Ho, 128]`. With
`output_final_state=True`, the function returns `(output, final_state)` where
`final_state` is FP32 `[num_sequences, Ho, 128, 128]` in public `[V,K]`
orientation. Callers may supply preallocated contiguous `output` and
`output_state` buffers.

## Example

```python
import torch

from cula.gdn import chunk_gated_delta_rule

device = "cuda"
seq_lens = (5, 3)
total_tokens = sum(seq_lens)
heads = 2

q = torch.randn(total_tokens, heads, 128, device=device, dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
g = torch.rand(total_tokens, heads, device=device, dtype=torch.float32) * 0.1 + 0.85
beta = torch.rand(total_tokens, heads, device=device, dtype=torch.float32)
cu_seqlens = torch.tensor([0, 5, 8], device=device, dtype=torch.int64)

output, final_state = chunk_gated_delta_rule(
    q,
    k,
    v,
    g=g,
    beta=beta,
    cu_seqlens=cu_seqlens,
    output_final_state=True,
)
```

Pass a returned `final_state` as `initial_state` in a later call to continue a
sequence. The state orientation is unchanged between calls.

## Unsupported behavior

- zero-length packed sequences;
- head size other than 128;
- FP16, FP8, or FP32 Q/K/V inputs;
- Q/K L2 normalization inside this kernel;
- intermediate state checkpoints (`state_checkpoints`,
  `checkpoint_cu_starts`, or nonzero `checkpoint_every_n_tokens`);
- decode, backward, or non-SM90 execution;
- implicit fallback to a C++ or FlashInfer kernel.

Unsupported metadata fails before launch with `ValueError`, `TypeError`,
`NotImplementedError`, or `RuntimeError` rather than silently selecting a
different backend. Device-resident value constraints are caller preconditions
unless the explicit synchronous `validate_inputs=True` diagnostic is enabled.

## Canonical benchmark

From the repository root, run:

```bash
python benchmarks/bench_gdn_prefill.py --output gdn-sm90-benchmark.json
```

The default command always executes the complete canonical matrix:

- fixed length: `B={1,2}`, `T={512,1024,4096,8192,16384}`;
- variable length: `num_sequences={10,20}`,
  `total_tokens={4096,8192,16384}`, and
  `distribution={uniform,random,skewed}`;
- `Hq=Hk=Hv=64`, head size 128, BF16 Q/K/V.

Compilation and first-call setup are excluded from the reported CUDA-event
mean. Use `--list-matrix` to inspect all 28 deterministic rows without running
CUDA work.

For the thread roles, pipeline stages, shared-memory layout, and recurrent-state
dataflow, see [GDN SM90 Pipeline](gdn_sm90_pipeline.md).
