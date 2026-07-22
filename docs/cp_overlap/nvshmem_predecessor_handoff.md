# NVSHMEM CP overlap

## Scope

This opt-in backend overlaps KDA CP preprocessing with an NVSHMEM symmetric-memory
state exchange. The strict public validation covers the CP=2, predecessor-only
topology. It does not replace general FLA CP communication.

## Locked CP=2 performance

The implementation was extracted from the private checkpoint at commit
`6823a3a`. The locked CP=2 matrix used BF16, 20 warmup iterations, 60 measured
iterations, and five process repeats per cell. The score is the median of the
five per-run mean rank latencies. All six cells had five valid runs, the
fairness checks passed, and there were no timeouts.

| Cell | cuLA NVSHMEM overlap | NCCL symmetric-memory baseline | FLA CP KDA | vs NCCL | vs FLA |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph 4096x8 | 0.1517 ms | 0.1750 ms | 0.2400 ms | -13.3% | -36.8% |
| non-graph 4096x8 | 0.5700 ms | 0.6731 ms | 0.7571 ms | -15.3% | -24.7% |
| graph 8192x8 | 0.2531 ms | 0.2763 ms | 0.3663 ms | -8.4% | -30.9% |
| non-graph 8192x8 | 0.5857 ms | 0.6762 ms | 0.7670 ms | -13.4% | -23.6% |
| graph 8192x16 | 0.3154 ms | 0.3601 ms | 0.4643 ms | -12.4% | -32.1% |
| non-graph 8192x16 | 0.5767 ms | 0.6770 ms | 0.7910 ms | -14.8% | -27.1% |

Negative percentages denote lower latency. NVSHMEM overlap won all six cells.
The Megatron measurements collected by the original harness are omitted because
they measured full training iterations and are not comparable with this
operation-level benchmark.

## Fresh GB200 validation

The upstream-ready branch was revalidated on two GB200 GPUs with NVSHMEM 3.4.5. Forward
output and active Q/K/V/g/beta gradients matched the FLA-preprocessing baseline
exactly (`max_abs_diff=0.0`). A five-repeat non-graph check at 8192x8 produced a
`0.5636 ms` median rank-max latency, versus `0.6076 ms` for cuLA with FLA CP
preprocessing on the same branch and run protocol.
See
[`nvshmem_predecessor_handoff_report.json`](nvshmem_predecessor_handoff_report.json)
for the repeat values and environment.

## Reproduce

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVSHMEM_DISABLE_CUDA_VMM=1
export NVSHMEM_SYMMETRIC_SIZE=256M
export NVSHMEM_IB_ENABLE=0
export CULA_CP_OVERLAP=1
export CULA_CP_ALLOW_FALLBACK=0
export CULA_CP_NVSHMEM_READY_WAIT=1

torchrun --standalone --nproc-per-node=2 \
  benchmarks/check_cp_predecessor_nvshmem.py

for backend in fla_full fla nvshmem; do
  torchrun --standalone --nproc-per-node=2 \
    benchmarks/bench_cp_predecessor_nvshmem.py \
    --backend="$backend" --sequence-length=8192 --heads=8 \
    --warmup=20 --iterations=60 --repeats=5
done
```

`NVSHMEM_DISABLE_CUDA_VMM=0` failed during NVSHMEM initialization with
`cuMemCreate` status 800 on this GB200/NVSHMEM 3.4.5 environment. cuLA does not
override the setting because other NVSHMEM installations may require a
different policy.
