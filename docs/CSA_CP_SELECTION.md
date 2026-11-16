# CSA context-parallel selection API

The API separates one-time THD planning and device preparation from per-layer
execution. `cudnn_csa_select_cp` consumes a prepared workspace to redistribute
queries/weights, run cuDNN block selection, and perform the inverse exchange.
It returns document-local block IDs and FP32 scores in the caller's original
token order. Downstream attention implementations can use the planner without
copying benchmark code.

The interface stops at selection. Projection weights, pooling transforms,
sparse attention, and the indexer loss remain independently composable.
`cudnn_index_pool_cp` can pool before gathering; the caller supplies global
pooled keys after its normalization/position transform. Attention and KDA
keep source ownership.

## Integration

```python
import torch.distributed as dist
from cula.distributed import THDPlannerConfig, THDSelectionCostModel
from cula.hybrid import (
    CsaCPSelectionPlan,
    CsaCPSelectionWorkspace,
    cudnn_csa_select_cp,
)

# Identical host-visible boundaries on all ranks, prepared outside forward.
plan = CsaCPSelectionPlan.from_cu_seqlens(
    global_cu_seqlens_cpu,
    rank=dist.get_rank(cp_group),
    world_size=dist.get_world_size(cp_group),
    top_k=512,
    policy="planner",
    cost_model=THDSelectionCostModel(**calibrated_coefficients),
    config=THDPlannerConfig(min_chunk_tokens=1024),
    candidate_budget_bytes=512 << 20,
)
workspace = CsaCPSelectionWorkspace.from_plan(plan, device, group=cp_group)

# Each compatible layer supplies fresh activations but shares the prepared
# workspace. Produce layer_inputs lazily as the model runs its layers.
selected_by_layer = []
for qi, global_kbar, index_weights in layer_inputs:
    selected = cudnn_csa_select_cp(
        qi,                    # BF16 [source_rows, index_heads, index_dim]
        global_kbar[:, None],   # BF16 [global_blocks, 1, index_dim]
        index_weights,         # BF16 or FP32 [source_rows, index_heads]
        workspace=workspace,
    )
    # Retain the results needed by each layer's attention/backward.
    selected_by_layer.append(selected)
    # selected.block_ids: int32 [source_rows, 512], document-local, -1 padding
    # selected.scores: FP32 [source_rows, 512], paired raw scores
```

Both factories run outside the layer loop. The execution function requires
`workspace`; it has no implicit planning or workspace-construction fallback.
It reuses routing indices and native scratch, while per-layer payload transport,
key packing and owned result storage can still allocate. Nondeterministic
selection is the default.

`plan.policy` and `plan.evaluated_scores` expose the chosen candidate and modeled
costs. Fixed `contiguous` and complete per-document `zigzag` policies need no
cost model. There is no implicit hardware detection, calibration, cache, or
device-to-host metadata readback. Planning remains on the host; workspace
materialization uploads compact interval metadata and builds the large integer
index/permutation arrays on the target device. This avoids constructing and
copying token-sized CPU arrays. Device operations use explicit output sizes.

## Reuse across layers

Prepare once at microbatch ingress and pass the same immutable plan to every
compatible CSA layer. The plan contains geometry, routing and scratch sizing,
not layer activations or selected block IDs. Each invocation supplies its own
Q, pooled keys and weights and computes fresh selection results.

Reuse requires unchanged document boundaries, source ownership, CP group/rank
mapping, block size, Top-K, row capacity and scratch policy. The calibrated costs
must remain applicable to the indexer shape, dtypes and topology. A new batch
with different boundaries needs a new plan even when its total token count
matches. Layers with different selection geometry need separate plans.

Sequential invocations on the same device/group/stream can also share one
`CsaCPSelectionWorkspace`. Results own their storage and survive later calls;
keep the per-layer results needed by backward. Concurrent invocations share
only the host plan and need separate mutable workspaces or explicit ordering.
This does not cache activations or sparse selections across layers.

For R compatible uses, one-time host planning contributes `P_host / R` per use.
If device metadata and scratch are also reused, their preparation is amortized
too. Runtime routing, selection, attention and backward still execute per layer.
The [reuse analysis](CSA_CP_PLAN_REUSE.md) separates this accounting from the
single-CSA measurements and from measured multilayer model performance.
The [31-case 1M rerun](CSA_CP_1M_REUSE_RESULTS.md) directly measures four complete
CSA instances sharing one plan/workspace, with all forwards preceding reverse
backwards and full per-layer output/gradient comparisons.

## Layout, lifetime and numerical contract

- Documents are block-aligned, with pooling block size four by default. Source
  ownership divides the packed stream into block-balanced contiguous shards.
  `plan.source_offsets` gives the exact cuts, which can have unequal lengths.
  There must be at least one block per source rank.
- All ranks use identical global metadata, costs and constraints. Rank IDs and
  topology maps are relative to the supplied group; subgroups are supported.
  Every rank enters the same exchanges, including empty selection destinations.
- Inputs are contiguous on the workspace device. Trailing dimensions, dtypes,
  scale and deterministic mode agree across peers. No casts or head padding
  are implicit; native backend head/dimension restrictions still apply.
- `cost_model.bytes_per_query` must match outgoing Q/weights and returning
  int32 IDs/FP32 scores: `H*D*2 + H*weights.element_size() + top_k*8`.
  This checks payload shape, not applicability of fitted kernel/bandwidth costs.
- IDs stay local to the original document. Rows are restored to source order;
  output slot order is unspecified. Keep IDs and scores paired when sorting.
  Add document block offsets only for a backend requiring global IDs.
- Results own their storage and survive workspace reuse. Identity routing
  copies native outputs to provide the same lifetime as a real exchange.
- Selection is discrete and runs without autograd. Indexer training uses its
  explicit backward/loss interface, as in the complete CSA benchmark.
- Scratch is mutable. Prepare/use on one stream, or establish an explicit
  stream dependency. Overlapping invocations require separate workspaces.
  Keep the process group alive. The eager transport allocates and does not
  promise CUDA Graph capture/replay support.

## Granularity and smaller contexts

One planner handles 128K, 256K and 1M contexts. For a single long document,
causal selection work falls roughly quadratically with context length, while
moved bytes fall roughly linearly. Startup, span launches and preparation can
dominate shorter contexts.

`min_split_tokens` controls document eligibility, not resulting piece size.
`min_chunk_tokens` separately constrains newly created selection spans:

1. Adaptive grids cap their piece count to meet the floor.
2. Candidates violating it after placement/capacity repair are rejected.
3. Existing source fragments and complete short documents remain legal.
4. Contiguous remains feasible even if the requested grain is too large.
5. The planner applies the constraint to its full-zigzag candidate; explicitly
   forcing `policy="zigzag"` preserves the baseline layout.

The floor defaults to zero for backward compatibility. The example explicitly
uses 1024 tokens, a tunable constraint rather than a universally optimal value.
For 128K at CP64, six pieces per rank average about 341 tokens; a 1024-token
floor caps the adaptive partition at 128 pieces. This is a CPU planning
property, not a measured CP64 performance result.

Destination capacity defaults to twice the largest source shard. It limits
query rows, not total memory. Candidate scratch has a separate budget; global
keys, routed payloads, retained results and key packing consume additional
memory. Explicit full zigzag can exceed the default cap for tiny documents;
raise `max_tokens_per_rank` deliberately to benchmark that layout.

Calibrate for the actual shape, topology and message sizes. Historical 1M
profiles' 20/40 ms `min_gain_ms` values are one-use preparation budgets, not
universal defaults. Reusing them at shorter contexts can reject useful layouts.
Measure preparation, transport and native selection separately, then validate
their sum on held-out distributions. Use explicit calibrated parameters rather
than case-name or context-length-specific layout rules.

## Validation

- `tests/test_csa_cp_selection.py`: host metadata parity, invalid inputs,
  capacities, execution without preparation, retained results across distinct
  layer inputs, and CP64 grain constraints.
- `benchmarks/check_csa_cp_selection.py`: native selection in two disjoint
  subgroups, unequal source sizes, empty destinations, exact source mapping,
  and workspace reuse.
- `benchmarks/bench_csa_cp_windows.py`: legacy contiguous versus the public
  contiguous/zigzag/planner paths, with rotating measurement order and full
  output/seven-gradient comparisons. Small fixtures add independent FP32
  attention/indexer/pooling checks.

The performance scope is complete CSA operator F/B, including pooling,
selection, sparse attention, teacher/KL backward, layouts and communication.
It excludes projections, KDA, MoE, optimizer and checkpoint recomputation.
Comparison fixtures coexist during timing: diagnostic allocator peaks include
other variants and are not standalone per-variant memory measurements.
Workspace bytes are reported separately. First-use compilation is outside
warmed timing and is not interpreted as a planner speedup. Fresh preparation
is rebuilt after warmup, with rotated and reversed policy order. Its full
wall time includes workspace allocation, H2D metadata and a completion sync;
there is no implicit cache or plan-reuse credit. Host-time outliers are retained.

Measured B300 CP8 results and the preparation limitations are documented in
[the 128K/256K study](CSA_CP_WINDOW_RESULTS.md) and the expanded
[128K/256K/1M three-way comparison](CSA_CP_THREE_WAY_RESULTS.md), including
22- and 64-document packs.
