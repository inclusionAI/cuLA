# NVSHMEM CP=4 stability follow-up

## Result

The NVSHMEM CP path now completes reliably at world size four on one GB200 host. The fix establishes two dependencies that were implicit in the CP=2 implementation:

1. NVSHMEM initialization and symmetric cache construction now finish collectively before any rank can publish or wait on a readiness epoch.
2. The communication stream explicitly waits for the producer stream before publishing a state computed directly in symmetric memory.

Before the fix, ranks could leave asynchronous NVSHMEM initialization at different times. Later ranks reached `wait_ready(peer=0, epoch=1)` while rank 0 had not completed symmetric cache construction, causing an unbounded wait. Increasing `NVSHMEM_SYMMETRIC_SIZE` did not address this ordering bug.

## Correctness and stability

Environment:

- 4x NVIDIA GB200, one host
- PyTorch `2.10.0+cu130`
- NVSHMEM `3.4.5`
- CUTLASS DSL `4.4.2`
- `NVSHMEM_DISABLE_CUDA_VMM=1`
- `NVSHMEM_SYMMETRIC_SIZE=256M`
- `CUDA_DEVICE_MAX_CONNECTIONS=1`

The deterministic oracle compares the FLA CP preprocessing path and the NVSHMEM path at the same output and backward boundary.

| World size | Global sequence | Heads | Process launches | Result | Maximum absolute difference |
| ---: | ---: | ---: | ---: | :--- | ---: |
| 2 | 1,024 | 4 | 1 | pass | 0.0 |
| 4 | 1,024 | 4 | 5 | 5/5 pass | 0.0 |
| 4 | 4,096 | 8 | 1 | pass | 0.0 |

The maximum covers output and active Q/K/V/g/beta gradients. The opt-in pytest, including CP=2 and CP=4 subprocess checks, reports `3 passed`.

```bash
CULA_RUN_NVSHMEM_TESTS=1 \
NVSHMEM_DISABLE_CUDA_VMM=1 \
NVSHMEM_SYMMETRIC_SIZE=256M \
NVSHMEM_IB_ENABLE=0 \
.venv/bin/python -m pytest tests/test_cp_predecessor_nvshmem.py -q
```

## Performance status

The stability fix makes CP=4 measurable, but the current multi-peer NVSHMEM implementation is slower than FLA's collective preprocessing path. Measurements use BF16, 20 warmups, 60 iterations, five in-process repeats, and the median repeat rank-max iteration median.

| Shape | FLA CP preprocessing | NVSHMEM multi-peer | NVSHMEM change |
| :--- | ---: | ---: | ---: |
| `4096x8` | 0.6088 ms | 0.6921 ms | +13.68% |
| `8192x16` | 0.6011 ms | 0.6764 ms | +12.53% |

The CP=4 backend is therefore correctness- and stability-ready for further optimization, but it is not ready to replace FLA on performance grounds. The next performance hypothesis is a chained predecessor-state handoff that publishes the already-merged V-state, avoiding rank 2 and rank 3 fetching and merging all earlier affine states.
