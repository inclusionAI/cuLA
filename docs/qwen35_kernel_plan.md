# Qwen3.5 Kernel Landing Plan Inside cuLA

This note records the internal landing structure for Qwen3.5 linear-attention
support added directly inside `cuLA`.

## New Python package surface

- `cula/qwen35/__init__.py`
- `cula/qwen35/common.py`
- `cula/qwen35/runtime.py`

## New CuTe op entry files

- `cula/ops/qwen35_conv1d_prefill.py`
- `cula/ops/qwen35_conv1d_decode.py`
- `cula/ops/qwen35_scalar_kda_prefill.py`
- `cula/ops/qwen35_scalar_kda_decode.py`

## Intended ownership

- `common.py`
  shared constants, local-head config, shape validation
- `runtime.py`
  compile-cache, stream-cache, prefill/decode dispatch boundaries
- `qwen35_conv1d_*`
  depthwise causal conv1d + silu
- `qwen35_scalar_kda_*`
  scalar-gated delta-rule prefill/decode kernels

## What should be reused from existing cuLA code

- runtime compile-cache patterns from `cula/ops/kda_decode.py`
- device helpers from `cula/utils.py`
- operator boundary style from `cula/kda/chunk.py`

## What should stay isolated at first

- no direct mutation of the generic `chunk_kda` public entry
- no pybind work until Python/CuTe path is numerically correct
- no conv + kda fusion until standalone kernels are validated
