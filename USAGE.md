# Usage Guide

This document provides usage notes and examples for cuLA kernels.

---

## KDA

cuLA provides two KDA kernel implementations targeting different GPU architectures:

| Kernel | GPU | Import |
|---|---|---|
| Modular Forward | Blackwell (SM100) | `from cula.kda import chunk_kda` |
| Fused Forward | Hopper (SM90) | `from cula.kda import kda_prefill_hopper` |

Both are drop-in replacements for [FLA](https://github.com/fla-org/flash-linear-attention)'s `chunk_kda` — just change the import.

**General Notes**

- **`safe_gate=True`** is required to leverage TensorCore (M=16) acceleration.
- **`beta`** and **`initial_state`** must be **`float32`**.
- **`cu_seqlens`** (for variable-length sequences) must be **`int32`**.

---

### Modular Forward (SM100 — Blackwell)

The modular forward kernel replaces sub-kernels of KDA in FLA (chunk_intra, chunk_delta_h, fwd_o, etc.) for easy integration with [Kimi CP](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/cp/KCP.md).

#### Example

```python
import torch
from cula.kda import chunk_kda

B, T, H, K, V = 2, 2048, 32, 128, 128
device = 'cuda'

q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16, requires_grad=True)
g = torch.randn(B, T, H, K, device=device, dtype=torch.float32) * 0.1
beta = torch.randn(B, T, H, device=device, dtype=torch.float32).sigmoid()
A_log = torch.randn(H, device=device, dtype=torch.float32) * 0.01
dt_bias = torch.zeros(H * K, device=device, dtype=torch.float32)
init_state = torch.zeros(B, H, K, V, device=device, dtype=torch.float32)

o, final_state = chunk_kda(
    q=q, k=k, v=v, g=g, beta=beta,
    A_log=A_log, dt_bias=dt_bias,
    initial_state=init_state,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    safe_gate=True,
    lower_bound=-5.0,
)

# Backward is supported
o.backward(torch.randn_like(o))

print(f'Output shape: {o.shape}')             # [2, 2048, 32, 128]
print(f'Final state shape: {final_state.shape}')  # [2, 32, 128, 128]
```

**Notes**

- The backward pass is currently supported via FLA's implementation; further optimizations are on the roadmap.
- Compatible with [Kimi CP](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/cp/KCP.md) via the `cp_context` parameter, same as in FLA.

---

### Fused Forward (SM90 — Hopper)

The fused forward kernel fuses intra-chunk attention, inter-chunk state propagation, and output computation into a single kernel for maximum throughput. **Forward-only; backward is not yet implemented.**

#### Example

```python
import torch
from cula.kda import kda_prefill_hopper

B, T, H, K, V = 2, 2048, 32, 128, 128
device = 'cuda'

q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
v = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16)
g = torch.randn(B, T, H, K, device=device, dtype=torch.float32) * 0.1
beta = torch.randn(B, T, H, device=device, dtype=torch.float32).sigmoid()
A_log = torch.randn(H, device=device, dtype=torch.float32) * 0.01
dt_bias = torch.zeros(H * K, device=device, dtype=torch.float32)
init_state = torch.zeros(B, H, K, V, device=device, dtype=torch.float32)

o, final_state = kda_prefill_hopper(
    q=q, k=k, v=v, g=g, beta=beta,
    A_log=A_log, dt_bias=dt_bias,
    initial_state=init_state,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    safe_gate=True,
    lower_bound=-5.0,
)

print(f'Output shape: {o.shape}')             # [2, 2048, 32, 128]
print(f'Final state shape: {final_state.shape}')  # [2, 32, 128, 128]
```

**Notes**

- Mainly **suitable for large-batch inference**; performance is limited when both batch size and head count are small, because we do not parallelize over the sequence-length dimension.
- **Matrix inversion uses fp16 precision**, which is faster and occupies less shared memory but introduces minor numerical differences compared to tf32 inversion.
- **Intra-subchunk attention uses g-first as anchor**, which causes some numerical differences compared with the FLA Triton implementation (FLA uses g-half as anchor in the diagonal).
