# Repository Layout

Legend: `[exp]` experimental / unwired · `[non-KDA]` other operator.

```
cuLA/
├── cula/                              # Python package (pip install -e .)
│   ├── __init__.py
│   ├── utils.py                       # Architecture, stream-buffer, and cu_seqlens helpers
│   │
│   ├── kda/                           # KDA public API, wrappers, autograd, and support kernels
│   │   ├── __init__.py                # Lazy exports for chunk, prefill, and decode APIs
│   │   ├── prefill.py                 # Public FlashKDA prefill validation and routing
│   │   ├── flashkda.py                # SM90 CuTeDSL K1+K2 wrapper, autograd shell, and CP selection
│   │   ├── chunk.py                   # SM100 modular chunk API + autograd
│   │   ├── chunk_fwd.py               # SM100 forward orchestration
│   │   ├── chunk_intra.py             # CuTeDSL forward intra + Triton backward intra
│   │   ├── chunk_bwd.py               # Triton + FLA + CuTeDSL backward orchestration
│   │   └── wy_intra.py / wy_recompute.py # Triton WY preparation kernels
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
│       ├── lightning/                  # [non-KDA] Lightning prefill/decode kernels
│       │   ├── prefill.py              # Public SM90/SM100 architecture dispatch
│       │   ├── prefill_sm90.py         # SM90 validation, compile cache, and TVM-FFI launch
│       │   ├── prefill_sm100.py        # SM100 prefill implementation
│       │   ├── decode.py               # Linear Attention decode
│       │   └── sm90/                   # SM90 CuTe DSL implementation
│       │       ├── schedule.py         # Schedule constants and WGMMA fragment helpers
│       │       └── prefill_kernel.py   # Fixed and packed recurrent prefill kernel
│       └── experimental/               # [non-KDA] unwired prototypes
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
| `cula/kda/` | Python / Triton | Public KDA APIs, wrappers, autograd, routing, and supporting Triton kernels. `kda_prefill` calls FlashKDA directly without a native fallback. See [`cula/kda/README.md`](cula/kda/README.md). |
| `cula/ops/kda/` | Python (CuTeDSL) | CuTeDSL KDA kernels organized into SM100 modular kernels, SM90 FlashKDA K1+K2 and intracard CP, decode, and experimental code. |
| `cula/ops/lightning/` · `cula/ops/experimental/` | Python (CuTeDSL) | `[non-KDA]` Lightning/linear-attention kernels and prototypes. Lightning prefill dispatches to the SM90 or SM100 backend from Q's device capability. |
| `cula/ops/{inv,ptx}.py` · `cula/ops/sm100/ptx.py` | Python | Shared low-level helpers used across operators. |
