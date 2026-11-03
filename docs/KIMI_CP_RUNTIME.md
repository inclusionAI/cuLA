# cuDNN KDA Kimi CP runtime

## Scope

cuLA owns the distributed execution layer around cuDNN Frontend KDA.  cuDNN
owns local KDA forward/backward, span summaries, and the IEEE FP32 affine state
merge.  Frameworks own model-specific tensor layouts, gate parameterization,
and construction of the global packed sequence metadata.

The public surface intentionally has three KDA-specific layers:

- `KimiCPPlan`: immutable CPU metadata for one contiguous packed-THD shard.
- `KimiCPBoundaryTransport`: the narrow ordered-state propagation contract,
  with `AllGatherKimiCPTransport` as the exact prepared implementation.
- `cudnn_kda_cp`: the differentiable distributed KDA call.

There is no generic backend hierarchy.  The runtime is KDA-specific and uses
the cuDNN contracts directly. A future hierarchical or scan implementation can
replace only boundary-state propagation without changing planning, local KDA,
autograd, or the framework adapter.

## Ownership boundary

```text
framework adapter
  layout/gate canonicalization, global cu_seqlens, process group
          |
          v
cuLA Kimi CP runtime
  shard plan, boundary-only summaries, boundary transport, distributed autograd
          |
          v
cuDNN Frontend
  KDA summary, KDA summary_bwd, IEEE kimi_cp_merge, local KDA fwd/bwd
```

Framework-specific compatibility does not belong in the runtime.  In
particular, flattened Megatron `dt_bias`, non-safe gate preprocessing, and
`[1,T,H,D]` wrappers must be canonicalized before calling cuLA.

## Packed shard plan

`KimiCPPlan.from_global_cu_seqlens` accepts CPU global boundaries and either a
balanced partition or explicit global shard offsets.  It derives:

- clipped local `cu_seqlens`;
- first and last local sequence IDs;
- the first and last boundary token spans;
- the number of preceding and following ranks in those sequence chains.

Plan construction is CPU-only and must happen once per packed microbatch, not
once per layer.  The same plan is reusable by every KDA layer.

Only the first and last local spans can cross rank boundaries under the
contiguous-shard contract.  Forward therefore summarizes only the last local
span; backward summarizes only the first.  This is the boundary-only strategy
validated by the Ant Megatron 256K workload (20 short samples plus two long
samples), where it removed the full-local-summary overhead.

## Forward and backward

Forward:

1. Produce FP32 `(H, M)` for the last local span.
2. Pack H/M into one equal-shaped collective payload and all-gather it.
3. Select only preceding ranks belonging to the first local sequence.
4. Compose their summaries in rank order with cuDNN `kimi_cp_merge`.
5. Seed only the first local sequence and run native cuDNN KDA.

Backward performs the symmetric reverse chain:

1. Produce `(G, M^T)` for the first local span.
2. Convert the transition to the forward orientation and all-gather it.
3. Select following ranks belonging to the last local sequence.
4. Merge in reverse order and seed only the last final-state gradient.
5. Let cuDNN autograd compute local tensor and gate gradients.

The state payload and transition remain FP32.  cuLA exposes no merge-precision
switch; the cuDNN API uses its fixed IEEE FP32 state chain.

## Current transport and rejected chain experiment

The runtime uses `all_gather_into_tensor`.  It has one collective per direction
and lets cuDNN merge all preceding or following summaries in one batch.  For
`H=32`, `K=V=128`, and CP64, the gathered `(H,M)` storage is 256 MiB per rank.

An order-preserving P2P chain was prototyped as a bounded-memory alternative.
It reduced communication storage to one 2 MiB `H` state and was numerically
exact: one P63 IEEE merge and 63 same-order P1 IEEE merges produced bitwise
identical final and per-piece initial states in both directions.  It was not
retained because 63 real NCCL hops plus P1 merges took 7.346 ms on the four-GPU
emulation, versus 3.024 ms for the 4 MiB ring path plus 0.305 ms for one P63
merge (3.329 ms estimated total).  Unbatched dependent P2P also creates lazy
rank-pair communicators and is a poor fit for CUDA Graph capture.

A tree or recursive-doubling scan can reduce the hop count, but reassociates
FP32 matrix products and therefore changes the numerical contract.  It should
not replace the current path without an explicit accuracy study and real
multi-node CP64 evidence.

## 1M logical-CP64 KDA E2E emulation

The four-GPU L20C environment cannot measure real 64-GPU scaling. The E2E
emulator therefore maps representative logical ranks `0/21/42/63` onto four
physical processes, uses 16,384 local tokens per logical rank, and executes 63
real NCCL ring hops with the 4 MiB FP32 `(H,M)` payload in each direction. The
local summary, IEEE merge, KDA forward, backward summary, reverse merge, and
local autograd are real cuDNN operations. No DSA stage is included.

Shape: 1,048,576 global tokens, logical CP64, `H=32`, `K=V=128`. One complete
warmup was excluded, followed by three steady-state samples.

| E2E stage | Critical-stage median |
|---|---:|
| Forward summary | 0.903 ms |
| Forward 63-hop communication | 3.008 ms |
| Forward FP32 merge | 0.293 ms |
| Local KDA forward | 1.054 ms |
| Backward summary | 1.083 ms |
| Backward 63-hop communication | 3.216 ms |
| Backward FP32 merge | 0.403 ms |
| Local KDA backward | 3.261 ms |

The measured critical-path median was **12.953 ms GPU time** and **16.102 ms
wall time**. Peak allocated memory was 3.049 GiB per physical process; peak
reserved memory was 3.15–3.27 GiB. Outputs and all five local input gradients
were finite on every representative rank. Proxy loss, output norm, and gradient
norms were identical across all three repeats on each rank.

Communication accounts for roughly 6.22 ms of the stage envelope and is the
largest optimization target. These numbers model the serial CP64 dependency
length on the local four-GPU fabric; they do not model 64-GPU topology,
cross-node contention, or aggregate bisection bandwidth. The reproducible
harness is `benchmarks/large_cp_emulation/bench_kda.py`. The logical topology
and benchmark-only transport live beside it rather than in the production CP
runtime.

## Prepared large-CP transport

The convenience path allocates its packed local and gathered summary tensors
on each forward and backward call. At large CP this creates avoidable allocator
traffic: for `H=32`, `K=V=128`, and CP64, the gathered tensor is 256 MiB and the
local payload is 4 MiB.

`AllGatherKimiCPTransport.prepare` allocates those FP32 buffers once. The same
transport can be reused by every KDA layer that executes sequentially on one
CUDA stream because the buffers are not retained by the local autograd graph.
Use one prepared transport per concurrently executing stream. Shape, device,
world-size, dtype, and contiguity are validated before every collective.
`required_workspace_bytes` reports the exact allocation before materializing
it; the standard CP64 shape above requires 260 MiB.

```python
from cula.distributed import (
    AllGatherKimiCPTransport,
    build_kimi_cp_plan,
    cudnn_kda_cp,
)

plan = build_kimi_cp_plan(global_cu_seqlens_cpu, group=cp_group)
transport = AllGatherKimiCPTransport.prepare(
    plan,
    heads=32,
    key_dim=128,
    value_dim=128,
    device=q.device,
)

# Reuse `plan` and `transport` across compatible KDA layers.
output = cudnn_kda_cp(
    q,
    k,
    v,
    g,
    beta,
    local_cu_seqlens,
    plan=plan,
    group=cp_group,
    boundary_transport=transport,
)
```

The transport contract returns only the FP32 state entering the first local
sequence (forward) or leaving the last local sequence (backward). It does not
expose framework layouts or local KDA details. This is the intended extension
point for future topology-aware CP64 implementations.

CUDA Graph is an execution mode, not a CP algorithm. The prepared transport
provides stable collective addresses; full graph capture also depends on the
process group and cuDNN merge implementation supporting capture. A future
cuDNN correction API can likewise replace the current summary-plus-full-local
pass implementation behind the same call.

## Validation ladder

1. CPU plan tests, including the 256K 20-short/2-long partition.
2. Helper tests for boundary selection and mandatory FP32 payloads.
3. Two-rank distributed comparison with monolithic cuDNN forward and gradients.
4. CP4/CP8 128K and 256K packed-THD accuracy/performance regression.
5. CP16/CP32/CP64 accuracy, peak memory, collective scaling, and loss checks.
6. Warm CUDA Graph capture/replay after reusable workspace is introduced.
