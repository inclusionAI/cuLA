# cuDNN KDA Kimi CP runtime

## Scope

cuLA owns the distributed execution layer around cuDNN Frontend KDA.  cuDNN
owns local KDA forward/backward, span summaries, and the IEEE FP32 affine state
merge.  Frameworks own model-specific tensor layouts, gate parameterization,
and construction of the global packed sequence metadata.

The public surface intentionally has two concepts:

- `KimiCPPlan`: immutable CPU metadata for one contiguous packed-THD shard.
- `cudnn_kda_cp`: the differentiable distributed KDA call.

There is no generic backend hierarchy.  The runtime is KDA-specific and uses
the cuDNN contracts directly.  Transport and workspace may become replaceable
implementation details when a second implementation is justified by measured
CP32/CP64 results.

## Ownership boundary

```text
framework adapter
  layout/gate canonicalization, global cu_seqlens, process group
          |
          v
cuLA Kimi CP runtime
  shard plan, boundary-only summaries, collective, ordered state chain,
  distributed autograd
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

## Current transport and intended evolution

The first implementation uses `all_gather_into_tensor`.  It is simple, has one
collective per direction, and matches the validated Megatron path.  It also
materializes every rank's summary on every rank, so CP32/CP64 profiling should
decide whether to add an ordered tree or hierarchical scan.

CUDA Graph is an execution mode, not a CP algorithm.  Graph-ready follow-up
work should add explicit reusable collective/state buffers without changing
the public operator.  A future cuDNN correction API can likewise replace the
current summary-plus-full-local-pass implementation behind the same call.

## Validation ladder

1. CPU plan tests, including the 256K 20-short/2-long partition.
2. Helper tests for boundary selection and mandatory FP32 payloads.
3. Two-rank distributed comparison with monolithic cuDNN forward and gradients.
4. CP4/CP8 128K and 256K packed-THD accuracy/performance regression.
5. CP16/CP32/CP64 accuracy, peak memory, collective scaling, and loss checks.
6. Warm CUDA Graph capture/replay after reusable workspace is introduced.
