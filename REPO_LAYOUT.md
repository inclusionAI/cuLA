# Repository Layout

Legend: `[exp]` experimental / unwired · `[non-KDA]` other operator.

```
cuLA/
├── cula/                              # Python package (pip install -e .)
│   ├── __init__.py
│   ├── backends.py                    # Generic backend registry, probing, verification, dispatch
│   ├── cudac.py                       # Lazy proxy for the per-architecture CUDA extension
│   ├── utils.py                       # Architecture, stream-buffer, and cu_seqlens helpers
│   │
│   ├── kda/                           # KDA public API, wrappers, autograd, routing, and Triton support kernels
│   │   ├── __init__.py                # Lazy exports for chunk, prefill, and decode APIs
│   │   ├── backends/                  # kda_prefill runtime dispatch
│   │   │   ├── flashkda.py            # Preferred SM90 CuTeDSL backend + verifier
│   │   │   ├── fully_fused.py         # SM90 CUDA C++ backend + verifier
│   │   │   └── _common.py             # Shared dispatch input checks
│   │   ├── auto_route.py              # Fully-fused SM90 basic/optimized router
│   │   ├── flashkda.py                # SM90 CuTeDSL K1+K2 wrapper, autograd shell, and CP selection
│   │   ├── hopper_fused_fwd.py        # SM90 CUDA C++ basic wrapper
│   │   ├── hopper_fused_fwd_opt.py    # SM90 CUDA C++ optimized/CP wrapper
│   │   ├── chunk.py                   # SM100 modular chunk API + autograd
│   │   ├── chunk_fwd.py               # SM100 forward orchestration
│   │   ├── chunk_intra.py             # C++ forward intra + Triton backward intra
│   │   ├── chunk_bwd.py               # Triton + FLA + CuTeDSL + C++ backward orchestration
│   │   ├── wy_intra.py / wy_recompute.py # Triton WY preparation kernels
│   │   └── gate_l2norm_fused.py / l2norm_qk_fused.py # SM90 Triton preprocessing kernels
│   │
│   ├── lightning/                     # [non-KDA] Lightning Attention public wrappers
│   │
│   └── ops/                           # CuTeDSL kernels and shared low-level helpers
│       ├── inv.py / ptx.py            # Shared low-level helpers
│       ├── sm100/ptx.py               # Shared SM100 PTX helpers
│       ├── kda/
│       │   ├── cp_mode.py              # Shared intracard-CP mode vocabulary
│       │   ├── sm100/                  # Blackwell modular forward/backward kernels
│       │   │   ├── delta_h.py  fwd_o.py  bwd_wy_dqkg.py
│       │   │   ├── policy.py           # SM100 intracard-CP decision logic
│       │   │   └── cp/                 # chunk_delta_h, pre_scan, merge
│       │   ├── sm90/                   # Hopper FlashKDA K1+K2 kernels
│       │   │   ├── fwd.py  k1.py  k2.py
│       │   │   └── cp/                 # driver, planner, pre-scan, merge
│       │   ├── decode/                 # Single-token, packed, and MTP decode kernels
│       │   └── experimental/sm100_fused/ # [exp] unwired fully-fused SM100 prototype
│       ├── lightning/                  # [non-KDA] Lightning prefill/decode kernels
│       └── experimental/               # [non-KDA] unwired prototypes
│
├── csrc/                               # CUDA C++ / CUTLASS kernels for SM90 and SM100/SM103
│   ├── api/
│   │   ├── pybind.cu                   # Shared named binding definitions
│   │   ├── pybind_sm90.cu              # SM90-specific binding translation unit
│   │   ├── pybind_sm100.cu             # SM100/SM103-specific binding translation unit
│   │   ├── kda_sm90.cu                 # SM90 host API
│   │   └── kda_sm100.cu                # SM100/SM103 host API
│   ├── kda/sm90/                       # Hopper fully-fused prefill kernels
│   ├── kda/sm100/                      # Blackwell modular chunk kernels
│   ├── kerutils/include/               # Shared device and host helpers
│   └── cutlass/                        # CUTLASS submodule
│
├── benchmarks/                         # Accuracy and performance runners
├── tests/                              # CPU registry tests and architecture-marked GPU tests
├── docs/                               # Pipeline and backend-dispatch design notes
├── scripts/build_wheel.sh
├── third_party/flash-linear-attention/ # FLA baseline and reused operations
├── README.md  USAGE.md  REPO_LAYOUT.md  RECOMMENDED_CODING_STYLE.md
└── setup.py  pyproject.toml  LICENSE
```

## Key Directories

| Directory | Language | Description |
|-----------|----------|-------------|
| `cula/kda/` | Python / Triton | Public KDA APIs, wrappers, autograd, routing, and supporting Triton kernels. `kda_prefill` dispatches SM90 calls to FlashKDA first and then the fully-fused CUDA backend; explicit backend exports remain available. See [`cula/kda/README.md`](cula/kda/README.md). |
| `cula/backends.py` · `cula/kda/backends/` | Python | Generic backend registry plus KDA-specific availability probes, capability verifiers, priorities, and adapters. |
| `cula/ops/kda/` | Python (CuTeDSL) | CuTeDSL KDA kernels organized into SM100 modular kernels, SM90 FlashKDA K1+K2 and intracard CP, decode, and experimental code. The fully-fused SM90 implementation lives under `csrc/`, not here. |
| `csrc/kda/{sm90,sm100}/` | CUDA C++ | Hopper fully-fused prefill and Blackwell modular chunk kernels. |
| `csrc/api/` · `cula/cudac.py` | CUDA C++ / Python | Per-architecture `_cudac_sm90` and `_cudac_sm100` extensions, exposed lazily through the `cula.cudac` compatibility proxy. |
| `cula/ops/lightning/` · `cula/ops/experimental/` | Python (CuTeDSL) | `[non-KDA]` Lightning/linear-attention kernels and prototypes. |
| `cula/ops/{inv,ptx}.py` · `cula/ops/sm100/ptx.py` | Python | Shared low-level helpers used across operators. |
