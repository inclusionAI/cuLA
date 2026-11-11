# Standalone kernel inventory and overlap assessment

This document records why each cuLA kernel family exists after separating the
CuTeDSL runtime from FLA. It reflects upstream `main` as inspected on
2026-10-26; upstream coverage and dispatch policy can change quickly.

## Decision summary

| cuLA surface | Keep? | Overlap | Why it still matters |
|---|---:|---|---|
| Lightning Attention prefill, decode, MTP and state-buffer kernels | **Yes — primary** | No production implementation was found in FlashInfer; its Lightning issue remains open. No cuDNN linear-attention engine documents Lightning Attention. | This is the clearest independent product surface: SM90 + SM100, fixed/packed prefill, decode and serving-oriented MTP/state-buffer primitives. |
| SM90 FlashKDA prefill and intracard CP | **Yes — primary** | FlashInfer's current KDA prefill dispatch documents SM100/SM103/SM120 paths; cuDNN FROST KDA starts at SM100. | Maintains a Hopper path and a single-GPU CP implementation for long sequences. |
| KDA MTP/KV-buffer/conv decode | **Keep selectively** | FlashInfer now exposes recurrent, packed and fused speculative KDA decode, including fused short convolution and accepted-token cache updates. cuDNN does not document a KDA decode engine. | Useful as decomposed kernels, for custom state layouts, and as an integration-independent reference. It is no longer differentiated as a complete serving path. |
| SM100 KDA prefill | **Keep as an alternative** | FlashInfer has CuTeDSL/Cake prefill and can dispatch to cuDNN; cuDNN FROST is the native full-operator implementation on SM100/SM103. | cuLA remains smaller, source-readable, and supports explicit atomic composition and CP experiments. It should not claim broad backend superiority without shape-by-shape data. |
| SM100 modular `intra`, `recompute_w_u`, `delta_h`, `fwd_o` | **Yes — primitives** | Their combined mathematical operator is covered by cuDNN FROST KDA forward/backward. | Stable, individually benchmarkable building blocks are valuable for algorithm research, precision regression, scheduling experiments, and embedding into a larger fused pipeline. |
| KDA WY backward primitives | **Research only** | cuDNN documents KDA forward and backward, with checkpoint recomputation; FlashInfer intentionally moved training ownership to cuDNN Frontend. | Retain only while they provide measurable value for kernel research or a needed boundary that cuDNN does not expose. Do not position them as a complete training stack. |
| FLA/Triton orchestration and vendored FLA submodule | **Remove** | FLA was framework glue and a reference implementation rather than a CuTeDSL kernel. | It made the runtime import graph larger and obscured which code was actually owned by cuLA. Reference comparisons can install FLA separately. |

## Upstream evidence

- [FlashInfer's current KDA facade](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/kda.py)
  covers prefill, decode and speculative decode, with CuTeDSL, Cake and cuDNN
  backends selected by contract and shape.
- [FlashInfer v0.7.1 release notes](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.1)
  describe packed fused KDA speculative verification with short convolution,
  accepted-token cache updates, CUDA Graph capture and FP32/BF16 state.
- [cuDNN Frontend's Python engine documentation](https://docs.nvidia.com/cudnn/latest/utilities/python_graph_and_execution_backends/)
  documents `KdaFrostEngine` and `KdaCuTileEngine`, including KDA forward,
  backward and checkpoint recomputation on Blackwell.
- [FlashInfer Lightning Attention issue #739](https://github.com/flashinfer-ai/flashinfer/issues/739)
  is still an open feature request, which is the strongest evidence that the
  cuLA Lightning family is not duplicated there today.

## Public boundary

Use `cula.kernels.kda` and `cula.kernels.lightning` for the standalone API.
The registry is intentionally static and import-light:

```python
from cula.kernels import list_kernels

for spec in list_kernels(family="kda"):
    print(spec.name, spec.level, spec.architectures, spec.status)
```

`operator` entries are complete calls, `composite` entries compose several
owned kernels, and `primitive` entries expose a deliberate intermediate
boundary. Inputs such as the cumulative FP32 KDA gate are part of a primitive's
contract; cuLA does not silently import another framework to construct them.

## Maintenance policy

1. A public kernel must import and run without FLA or Triton.
2. New primitives need a correctness test for their tensor contract and a
   benchmark that justifies preserving the boundary.
3. An overlapped complete operator is retained only if cuLA wins an important
   architecture/shape, exposes a missing primitive, or materially simplifies
   integration.
4. Reference frameworks belong in optional tests and benchmarks, never in
   `cula/` or as a git submodule.
