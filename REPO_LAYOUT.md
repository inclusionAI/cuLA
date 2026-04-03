# Repository Layout

```
cuLA/
├── cula/                         # Python package (pip install -e .)
│   ├── kda/                      # KDA (Kimi Delta Attention) operators
│   │   ├── chunk.py              # End-to-end chunk KDA (fwd + bwd entry point)
│   │   ├── chunk_fwd.py          # Chunk forward dispatch
│   │   ├── chunk_intra.py        # Intra-chunk forward logic
│   │   ├── blackwell_fused_fwd.py  # Fused KDA forward (SM100)
│   │   └── hopper_fused_fwd.py     # Fused KDA forward (SM90)
│   ├── lightning/                # Lightning Attention operators
│   │   └── la_decode.py          # Single-token decode kernel (CuTe DSL)
│   ├── ops/                      # CuTe DSL kernel implementations
│   │   ├── chunk_delta_h.py      # Chunk delta-H kernel
│   │   ├── fwd_o.py              # Forward output kernel
│   │   ├── lightning_attn.py     # Lightning Attention prefill kernel
│   │   ├── linear_attn.py        # Generic linear attention kernel
│   │   ├── kda_fully_fused_wip.py # WIP fully fused KDA kernel
│   └── utils.py                  # Shared utilities
│
├── csrc/                         # CUDA C++ / CUTLASS kernels
│   ├── api/                      # PyBind11 bindings
│   │   ├── pybind.cu             # Python ↔ CUDA binding entry
│   │   ├── kda_sm90.cu           # SM90 API wrappers
│   │   └── kda_sm100.cu          # SM100 API wrappers
│   ├── kda/
│   │   ├── sm90/                 # Hopper KDA kernels (CUTLASS 3.x)
│   │   │   ├── kda_fwd_sm90.cu
│   │   │   ├── kda_fwd_sm90_safe_gate.cu
│   │   │   ├── prefill_kernel.hpp
│   │   │   ├── collective/       # CUTLASS collective mainloop
│   │   │   ├── device/           # Device-level kernel wrappers
│   │   │   ├── kernel/           # Kernel-level logic
│   │   │   └── utils/            # SM90-specific helpers
│   │   └── sm100/                # Blackwell KDA kernels (CUTLASS 3.x)
│   │       ├── kda_fwd_sm100.cu
│   │       ├── kda_fwd_common.cuh
│   │       ├── kda_fwd_intra_kernel_sm100.hpp
│   │       ├── kda_fwd_intra_mainloop_sm100.hpp
│   │       ├── kda_config.hpp
│   │       ├── fwd_helpers.hpp
│   │       ├── sm100_umma_ext.hpp
│   │       └── tile_scheduler.hpp
│   └── kerutils/
│       └── include/              # Shared C++ header utilities
│
├── benchmarks/                   # Performance benchmarks
│   ├── bench_kda.py              # KDA fixed + varlen benchmark
│   ├── bench_lightning_attn.py   # Lightning Attention prefill + varlen
│   ├── bench_la_decode_vs_fla.py # Decode: la_decode vs fla fused_recurrent
│   ├── bench_kda_fused_fwd.py    # KDA fused forward benchmark
│   ├── bench_kda_chunk_intra.py  # KDA chunk intra benchmark
│   ├── bench_chunk_delta_h.py    # Chunk delta-H benchmark
│   ├── bench_fwd_o.py            # Forward output benchmark
│   ├── bench_linear_attn.py      # Linear attention benchmark
│   ├── generate_benchmark_md.py  # Auto-generate BENCHMARK_GB200.md (Blackwell)
│   ├── generate_benchmark_hopper_md.py  # Auto-generate BENCHMARK_H200.md (Hopper)
│   └── utils.py                  # Benchmark utilities
│
├── tests/                        # Unit / integration tests
│   ├── test_kda_compare_fla.py   # Modular KDA forward vs FLA Triton
│   ├── test_kda.py               # Modular KDA forward vs naive reference
│   ├── test_kda_fused_fwd.py     # Fused KDA forward tests
│   ├── test_chunk_delta_h.py     # Chunk delta-H tests
│   ├── test_fwd_o.py             # Forward output tests
│   ├── test_compare_with_fla.py  # General FLA comparison
│   ├── test_lightning_attn.py    # Lightning Attention tests
│   └── test_la_decode.py         # Decode kernel tests
│
├── docs/                         # Design documents
│   ├── chunk_delta_h_pipeline.md
│   ├── fwd_o_pipeline.md
│   └── lightning_attn_pipeline.md
│
├── third_party/
│   └── flash-linear-attention/   # FLA submodule (baseline)
│
├── BENCHMARK_GB200.md            # Auto-generated Blackwell benchmark results
├── BENCHMARK_H200.md             # Auto-generated Hopper benchmark results
├── README.md                     # Project overview
├── setup.py                      # Build configuration
├── pyproject.toml                # Project metadata
└── LICENSE
```

## Key Directories

| Directory | Language | Description |
|-----------|----------|-------------|
| `cula/ops/` | Python (CuTe DSL) | Warp-specialized GPU kernels written in CuTe DSL — compiled to CUDA at import time |
| `cula/kda/` | Python | KDA operator dispatch — selects SM90 or SM100 path, handles chunking and autograd |
| `cula/lightning/` | Python (CuTe DSL) | Lightning Attention decode kernel |
| `csrc/kda/sm90/` | CUDA C++ | Hopper KDA kernels using CUTLASS 3.x collective API |
| `csrc/kda/sm100/` | CUDA C++ | Blackwell KDA kernels using CUTLASS 3.x + UMMA extensions |
| `csrc/api/` | CUDA C++ | PyBind11 entry points exposing C++ kernels to Python |
| `benchmarks/` | Python | Performance benchmarks vs FLA Triton baselines |
| `tests/` | Python | Correctness tests (pytest) |
| `docs/` | Markdown | Internal pipeline design notes |
