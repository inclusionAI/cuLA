# `cula.kda` — KDA (Kimi Delta Attention) operators

Public entry points for KDA. For full examples (varlen, intracard CP, per-backend notes), see [`USAGE.md`](../../USAGE.md).

## API

| Symbol | Description | Arch | Direction |
|--------|-------------|------|-----------|
| `chunk_kda` | Chunked KDA prefill | SM100 (Blackwell) | Forward + backward |
| `kda_prefill_hopper` / `_opt` / `_auto` | KDA prefill (CUDA C++ fused) | SM90 (Hopper) | Forward only |
| `flashkda_prefill` | KDA prefill (CuTeDSL K1+K2, intracard-CP capable) | SM90 (Hopper) | Forward only |
| `kda_prefill` | Auto backend dispatch (FlashKDA → fully_fused) | SM90 (Hopper) | Forward only |
| `kda_decode` | Single-token decode | SM100 | Forward |
| `fused_sigmoid_gating_delta_rule_update` | Decode state update | SM100 | Forward |

`chunk_kda`, `kda_prefill_hopper*`, and `flashkda_prefill` each call a fixed implementation.
`kda_prefill` picks an SM90 backend at runtime (disable with `CULA_BACKEND_FLASHKDA=0` / `CULA_BACKEND_FULLY_FUSED=0`).

## Minimal example

```python
import torch
from cula.kda import kda_prefill

B, T, H, D = 1, 1024, 4, 128
q = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
k, v, g = torch.randn_like(q), torch.randn_like(q), torch.randn_like(q)
beta = torch.randn(B, T, H, dtype=torch.bfloat16, device="cuda").sigmoid()
A_log = torch.randn(H, dtype=torch.float32, device="cuda")
dt_bias = torch.randn(H * D, dtype=torch.float32, device="cuda")

with torch.inference_mode():
    o, ht = kda_prefill(q, k, v, g, beta, A_log=A_log, dt_bias=dt_bias)
```
