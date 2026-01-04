# FlashLA — Flash Linear Attention

FlashLA is a lightweight, high-performance linear attention library inspired by ideas from FlashAttention but specialized and engineered for linear attention mechanisms.

## Features

- O(N) time and memory approximate attention for long sequences (instead of O(N^2)).
- Supports causal (autoregressive) and non-causal modes.
- Numerical-stability controls (normalization, log-space ops, safe modes).

## Installation

- From PyPI (if published):

```bash
pip install flashla
```

- From a prebuilt local wheel (recommended if available):

```bash
# create and activate a virtual environment (example)
python -m venv /path/to/venv
. /path/to/venv/bin/activate

# install the wheel produced in the project's `dist/` directory
pip install dist/*.whl
```

- Build and install from source (requires CUDA toolchain + matching PyTorch):

```bash
# create and activate a virtual environment (example)
python -m venv /path/to/venv
. /path/to/venv/bin/activate

# ensure build tooling and setuptools_scm are available in the build environment
pip install -U build wheel setuptools setuptools_scm

# build using non-isolated mode so the local `torch` and CUDA toolchain are used
python -m build --no-isolation

# install the produced wheel
pip install dist/*.whl
```

Notes:
- Building the native CUDA extension requires a working CUDA toolkit (nvcc) and a compatible `torch` installed in the build environment. Use a non-isolated build (`--no-isolation`) if you rely on a locally installed `torch`/CUDA.
- Editable installs (`pip install -e .`) are useful for development, but you may still need to compile the extension manually after changes.
- Ensure git submodules are initialized before building from source:

```bash
git submodule update --init --recursive
```

For CI or isolated builds, `setuptools_scm` is declared in `pyproject.toml` and will provide dynamic versions; if Git metadata is unavailable a fallback version will be used.

## Quick start

WIP

## Performance and benchmarks

## Mathematical background (brief)

Linear attention rewrites the attention kernel using a feature map phi:

$$A_{ij} = \\phi(q_i)^T\\phi(k_j)$$

so the output can be expressed as

$$out_i = \\sum_j \\mathrm{softmax\\_approx}(q_i^T k_j) v_j \\approx \\phi(q_i)^T \\left(\\sum_j \\phi(k_j) v_j^T\\right)$$

This transforms the O(N^2) pairwise computation into two O(N) accumulation operations.
