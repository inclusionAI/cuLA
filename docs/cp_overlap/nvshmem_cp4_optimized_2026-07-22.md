# NVSHMEM CP=4 optimized state assembly

## Goals and status

This follow-up starts from the stable world-size-four NVSHMEM bootstrap in commit `59a245e` and targets the CP preprocessing boundary used by cuLA KDA.

| Goal | Status |
| :--- | :--- |
| Exact forward and backward agreement with FLA CP | Pass |
| Stable across fresh four-process launches | Pass |
| Faster than FLA CP preprocessing by rank-max median | Pass |
| At least 20% faster | Not met |
| Integrated Megatron production result | Out of scope for this cuLA experiment |

The result is an opt-in CP=4 research path. It is not evidence for the full Megatron training iteration or the 256K production shape.

## Retained design

The winning eager-mode path makes four changes:

1. Each rank writes its local affine recurrence state directly into NVSHMEM symmetric memory.
2. A single NVSHMEM barrier establishes publication and visibility before peer reads.
3. Ranks with multiple predecessors read the remote symmetric states directly and fuse the recurrence assembly into one Triton kernel, eliminating peer-copy staging and a second local merge pass.
4. Communication and merge run on the producer's current CUDA stream, avoiding the extra stream event handoff for this serialized boundary.

The path is enabled with:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=2
CULA_CP_COMM_USE_CURRENT_STREAM=1
CULA_CP_NVSHMEM_DIRECT_STORE_CONN1_ONLY=0
CULA_CP_NVSHMEM_FUSED_REMOTE_MERGE=1
CULA_CP_NVSHMEM_READY_WAIT=0
```

The optimized remote merge is restricted to eager execution, one global sequence, non-transposed state layout, and world size at most four. Other configurations retain the existing path.

## Correctness

Environment: one host with four NVIDIA GB200 GPUs, PyTorch `2.10.0+cu130`, CUDA 13.0, NVSHMEM 3.4.5, and CUTLASS DSL 4.4.2.

The deterministic oracle compares the cuLA NVSHMEM result with FLA CP at the same output/backward boundary. It checks output and active Q/K/V/g/beta gradients.

| World size | Global sequence | Heads | Fresh launches | Maximum absolute difference |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 1,024 | 4 | 3/3 pass | 0.0 |
| 4 | 4,096 | 8 | pass | 0.0 |
| 4 | 8,192 | 16 | pass | 0.0 |

The opt-in integration suite, which also includes a CP=2 regression, reports `3 passed`.

## Performance

Measurements use BF16, 20 warmups, 60 iterations, five in-process repeats, and the median of each repeat's rank-max iteration median. Backend order is rotated between shapes.

| Shape | FLA CP preprocessing | NVSHMEM optimized | Improvement |
| :--- | ---: | ---: | ---: |
| `4096x8` | 0.6317 ms | 0.5680 ms | 10.08% |
| `8192x16` | 0.6112 ms | 0.5662 ms | 7.36% |

The optimization reverses the stable pre-optimization result, where NVSHMEM was 12.5-13.7% slower than FLA at these cells. It does not meet the 20% target. Further gains require reducing or fusing the local preprocessing kernel, because transport and state assembly no longer account for enough of the end-to-end boundary.

## Rejected experiments

- Chained predecessor publication serialized the CP ranks and regressed both cells.
- Per-peer ready signals were correct but slower than the collective barrier on this one-host topology.
- Connection counts three and four were slower than two.
- Eight-warp remote-merge candidates regressed the 8192x16 cell.
- A CUDA Graph comparison did not produce a valid FLA baseline and is not included.

## Reproduction

```bash
benchmarks/run_cp4_nvshmem_optimized.sh
```

Set `PYTHON` if the repository virtual environment is not at `.venv/bin/python`. The machine must provide four visible GPUs and `nvshmem4py`.
