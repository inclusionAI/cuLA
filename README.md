<div align="center">

# cuLA — CUDA Linear Attention

**High-performance CUDA kernels for linear attention variants, written in [CuTe DSL](https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL) and CUTLASS C++.**

</div>

## Introduction

Linear attention mechanisms reformulate the standard attention computation by replacing the softmax with a feature map $\phi$, enabling the output to be expressed via a recurrent state:

$$S_i = S_{i-1} + \phi(k_i) v_i^T, \quad o_i = \phi(q_i)^T S_i$$

This recurrence reduces the complexity from $O(N^2)$ (standard attention) to $O(N)$, making linear attention particularly attractive for long-sequence modeling in LLMs. Recent variants — such as [GLA](https://arxiv.org/abs/2312.06635), [KDA](http://arxiv.org/abs/2510.26692), [GDN](https://arxiv.org/abs/2412.06464), and [Lightning Attention](https://arxiv.org/abs/2405.17381) — further enhance expressiveness with gating, delta updates, and chunkwise decomposition.

**cuLA** provides hand-tuned CUDA implementations of these linear attention variants, targeting NVIDIA Blackwell (SM100) and Hopper (SM90) GPUs. It is designed as a submodule of [flash-linear-attention (FLA)](https://github.com/fla-org/flash-linear-attention), sharing the same interface — adopting cuLA requires only a one-line import change. For ease of maintenance, cuLA is currently developed as a standalone library; the end goal is for users to seamlessly access these kernels through FLA. Since FLA already has a kernel dispatch mechanism in place, integration will be ready soon.

## Installation

```bash
git clone <repo-url>
git submodule update --init --recursive

# Install PyTorch (CUDA 13.0)
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130

# Install flash-linear-attention for benchmark repro
cd third_party/flash-linear-attention
pip install -e .
cd ../..

# Install cuLA
pip install -e . --no-build-isolation
```

> **Requirements:** Python 3.12+, CUDA Toolkit 13.0, NVCC 12.9+ (SM100a support), PyTorch 2.9.1+

## Quick Start

### KDA (Kimi Delta Attention)

cuLA is a drop-in replacement for [FLA](https://github.com/fla-org/flash-linear-attention) — just change the import:

```python
import torch
from cula.kda.chunk import chunk_kda  # <-- one-line change from fla.ops.kda

B, T, H, K, V = 2, 2048, 4, 128, 128
device = 'cuda'

q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16, requires_grad=True)
g = torch.randn(B, T, H, K, device=device, dtype=torch.float32) * 0.1   # gate (log space)
beta = torch.randn(B, T, H, device=device, dtype=torch.float32).sigmoid()
A_log = torch.randn(H, device=device, dtype=torch.float32) * 0.01
dt_bias = torch.zeros(H * K, device=device, dtype=torch.float32)
init_state = torch.zeros(B, H, K, V, device=device, dtype=torch.float32)

# Forward
o, final_state = chunk_kda(
    q=q, k=k, v=v, g=g, beta=beta,
    scale=1.0,
    A_log=A_log,
    dt_bias=dt_bias,
    initial_state=init_state,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    safe_gate=True,
    lower_bound=-5.0,
)

# Backward
do = torch.randn_like(o)
o.backward(do)

print(f'Output shape: {o.shape}')             # [2, 2048, 4, 128]
print(f'Final state shape: {final_state.shape}')  # [2, 4, 128, 128]
```

**Notes:**
- `safe_gate=True` is required (leverages M=16 TensorCore acceleration).
- `beta` and `initial_state` must be `float32`.
- `cu_seqlens` (for variable-length sequences) must be `int32`.
- Matrix inversion uses FP16 precision.

## Benchmarks

See [BENCHMARK.md](BENCHMARK.md) for detailed results.

All benchmarks run on a single **NVIDIA GB200** GPU with **CUDA Toolkit 13.0**, **PyTorch 2.9.1**, **Triton 3.5.1**.
FLA baseline: [flash-linear-attention v0.4.2](https://github.com/fla-org/flash-linear-attention/releases/tag/v0.4.2).

**Highlights:**
- **KDA:** up to **1.4x** speedup over FLA Triton on fixed-length and variable-length sequences.
- **Lightning Attention Prefill:** up to **2.4x** speedup (B=2).
- **Lightning Attention Varlen:** **avg 1.6x** speedup across 60+ configs (uniform/skewed/random).

To regenerate benchmarks:

```bash
python benchmarks/generate_benchmark_md.py
```

## Tests

```bash
# End-to-end test (forward + backward) against FLA Triton implementation
python -m pytest tests/test_kda_e2e_compare_fla.py -v

# End-to-end test against naive KDA reference
python -m pytest tests/test_kda.py -v

# Tests for KDA fused forward
python -m pytest tests/test_kda_fully_fused.py -v
```

<details>
<summary>Sample test output</summary>

```
tests/test_kda_e2e_compare_fla.py::test_safe_gate_chunk[B1-T63-H1-D128-...]    PASSED
tests/test_kda_e2e_compare_fla.py::test_safe_gate_chunk[B2-T500-H3-D128-...]   PASSED
tests/test_kda_e2e_compare_fla.py::test_safe_gate_chunk[B2-T1000-H3-D128-...]  PASSED
tests/test_kda_e2e_compare_fla.py::test_safe_gate_chunk[B3-T1024-H4-D128-...]  PASSED
tests/test_kda_e2e_compare_fla.py::test_safe_gate_chunk[B4-T1024-H4-D128-...]  PASSED
tests/test_kda_e2e_compare_fla.py::test_safe_gate_chunk[B4-T2048-H8-D128-...]  PASSED
tests/test_kda_e2e_compare_fla.py::test_safe_gate_chunk_varlen[...]             PASSED
...
======================= 17 passed in 40.95s =======================
```

</details>

CUDA kernel tuning is significantly more labor-intensive than Triton — contributions from the open-source community are warmly welcomed!

## Roadmap

* [ ] Integrate into [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) via FLA's kernel dispatch mechanism
* [ ] More fusions.

**Train**

* [x] Modular KDA Forward (sm100, compatible with Kimi CP)
  * [x] kda chunk intra
  * [x] chunk gated delta h
  * [ ] recompute wu
  * [x] chunk fwd o

* [ ] Modular GDN Forward / Backward Kernels (compatible with Kimi CP)

* [ ] More backward supports

* [ ] Kernel-level compute-communication overlapping CP linear attention kernels (via **nvshmem**)

**Inference**

* [x] Lightning prefill kernel (sm100)

* [x] Lightning decode kernel (sm90 & sm100)

* [x] Fused KDA prefill kernel (sm90)

* [ ] Fused KDA prefill kernel (sm100)

* [ ] Small B/H optimizations

* [ ] MTP support

* [ ] More aggressive fusion of small neighbor kernels like cumsum for inference scenarios.

## Acknowledgements

This project is inspired by [flash-linear-attention](https://github.com/fla-org/flash-linear-attention), [CUTLASS](https://github.com/NVIDIA/cutlass) and [CuTe DSL](https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL). We thank [FLA-org](https://github.com/fla-org) and NVIDIA for their great work.

## Contact

If you're interested in an internship or job opportunity, feel free to reach out: **shuyan.ycf@antgroup.com**  / **chaofanyu@gmail.com**

No CUDA experience is required as long as you're a quick learner.
