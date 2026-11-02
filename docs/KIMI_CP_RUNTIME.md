# cuDNN KDA Kimi CP runtime

## Scope

cuLA owns the distributed execution layer around cuDNN Frontend KDA. cuDNN
owns local KDA forward/backward, span summaries, and ordered FP32 affine-summary
composition inside the KDA apply operations. Frameworks own model-specific
tensor layouts, gate parameterization, and construction of the global packed
sequence metadata.

The public surface intentionally has three KDA-specific layers:

- `KimiCPPlan`: immutable CPU metadata for one contiguous packed-THD shard.
- `KimiCPBoundaryTransport`: the narrow ordered-summary propagation contract,
  with `AllGatherKimiCPTransport` as the exact prepared implementation.
- `cudnn_kda_cp`: the differentiable distributed KDA call.

There is no generic backend hierarchy.  The runtime is KDA-specific and uses
the cuDNN contracts directly. A future hierarchical or scan implementation can
replace only boundary-summary propagation without changing planning, local KDA,
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
  KDA summaries, concatenated-summary composition, local KDA fwd/bwd
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

1. Produce dense FP32 `(H, M)` for the last local span.
2. All-gather H and M into separate dense typed buffers.
3. Select only preceding ranks belonging to the first local sequence.
4. Pass them as `preceding_state_update` and `preceding_transition`.
5. Let cuDNN compose the rows and seed the first local sequence.

Backward performs the symmetric reverse chain:

1. Produce `(G, M^T)` and reusable backward context for the first local span.
2. All-gather the transition in its stored-domain orientation.
3. Select following ranks belonging to the last local sequence.
4. Pass them as `following_state_grad` and `following_transition`.
5. Let cuDNN compose the reverse chain and compute local tensor and gate gradients.

The state payload and transition remain FP32. cuLA exposes no merge-precision
switch; the cuDNN apply API owns the fixed high-precision state chain. The
public cuDNN Frontend 1.32.2 wheel does not contain this API. Until a compatible
release is published, use the tested source revision
`9e056425989e75f406a5ce3ad58cb563e4dff8b5` as shown in `USAGE.md`; the runtime
also checks the required call signatures and fails early on older builds.

## Current transport and rejected chain experiment

The first implementation uses two typed `all_gather_into_tensor` calls per
direction. Keeping H and M separate preserves the dense layout required by
cuDNN without a CP-sized repack buffer; total communication bytes are unchanged.
It still materializes every rank's summary on every rank, so CP32/CP64 profiling
should decide whether a topology-aware exact-order transport is worth adding. A
tree or scan must not silently reassociate FP32 products and change the
numerical contract.

## Prepared large-CP transport

The convenience path allocates its gathered H and M tensors on each forward and
backward call. At large CP this creates avoidable allocator traffic: for
`H=32`, `K=V=128`, and CP64, the two gathered tensors total 256 MiB.

`AllGatherKimiCPTransport.prepare` allocates those FP32 buffers once. The same
transport can be reused by every KDA layer that executes sequentially on one
CUDA stream because the buffers are not retained by the local autograd graph.
Use one prepared transport per concurrently executing stream. Shape, device,
world-size, dtype, and contiguity are validated before every collective.
`required_workspace_bytes` reports the exact allocation before materializing
it; the standard CP64 shape above requires 256 MiB.

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

The transport contract returns only the ordered FP32 `(state_update,
transition)` rows belonging to the boundary sequence. It does not compose the
state or expose framework layouts and local KDA details. This is the intended
extension point for future topology-aware CP64 implementations.

CUDA Graph is an execution mode, not a CP algorithm. The prepared transport
provides stable collective addresses; full graph capture also depends on the
process group and cuDNN summary-apply implementation supporting capture.

## Validation ladder

1. CPU plan tests, including the 256K 20-short/2-long partition.
2. Helper tests for boundary selection and mandatory FP32 payloads.
3. Two-rank distributed comparison with monolithic cuDNN forward and gradients.
4. CP4/CP8 128K and 256K packed-THD accuracy/performance regression.
5. CP16/CP32/CP64 accuracy, peak memory, collective scaling, and loss checks.
6. Warm CUDA Graph capture/replay after reusable workspace is introduced.
