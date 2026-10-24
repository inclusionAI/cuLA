# Coding Style Guide

cuLA is a Python package whose GPU kernels are written in CuTeDSL, with a
small number of Triton support kernels. The package does not build a native
C++/CUDA extension.

## General principles

- Prefer clear Python host code and explicit CuTeDSL data movement.
- Keep architecture-specific code under `cula/ops/<operator>/sm90/` or
  `cula/ops/<operator>/sm100/`.
- Preserve the public FLA-compatible tensor layout, dtype, and state-layout
  contracts at wrapper boundaries.
- Validate unsupported architectures, shapes, dtypes, layouts, and modes
  before launching a kernel.
- Document intentional precision changes and reduction-order differences.

## Python

- Use four spaces, type annotations for public APIs, and `snake_case` names.
- Keep imports sorted with Ruff and avoid eager imports of large kernel modules
  from package `__init__.py` files.
- Use `PascalCase` for classes and `SCREAMING_SNAKE_CASE` for module constants.
- Raise `TypeError` for dtype/type mismatches, `ValueError` for invalid values
  or shapes, and `NotImplementedError` for valid but unsupported modes.
- Keep compile caches bounded and key them by every property that affects the
  generated kernel.

## CuTeDSL kernels

- Make grid shape, warp roles, pipeline stages, and shared-memory ownership
  explicit near the kernel entry point.
- Keep barrier arrival counts and phase changes next to the producer/consumer
  protocol they protect.
- Do not reuse shared memory until every prior consumer has completed and the
  required async-proxy fence has been issued.
- Prefer vectorized, aligned global-memory accesses and state the required
  alignment in host validation.
- Use descriptive memory-space prefixes for tensors (`g`, `s`, `r`) when that
  matches the surrounding CuTe/CUTLASS conventions.
- Keep numerical conversion points visible. Avoid implicit precision loss in
  accumulators, recurrence state, or transition products.

## Tests and benchmarks

- Add a focused test for every public behavior change and each corrected race
  or boundary condition.
- Compare against an independent reference where possible, and distinguish
  bitwise requirements from tolerance-based numerical contracts.
- Include tail, varlen, initial-state, output-state, and non-default-stream
  coverage when the changed path supports them.
- Mark architecture-specific and long-running cases with the repository's
  pytest markers so the default suite stays bounded.
- Benchmark the complete public boundary, warm up JIT compilation first, and
  report the device, shapes, dtype, and aggregation method.

## Tooling

Before committing, run:

```bash
pre-commit run --all-files
```

Run CPU-only checks locally and CUDA tests in the designated remote container.
