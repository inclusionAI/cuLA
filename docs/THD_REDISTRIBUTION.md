# Executing and reversing a THD query layout

`plan_thd_layout` and `plan_thd_selection` choose global token spans. `THDRedistributionPlan` and
`THDRedistributor`, exported from `cula.distributed`, execute that assignment
from existing contiguous CP shards and restore the original ownership/order.
The planner's modeled score is not an end-to-end latency guarantee; see the
[real CP16/CP8 measurements](THD_SELECTION_RESULTS.md).

## Contract

Each process-group rank starts with rows in
`[source_offsets[rank], source_offsets[rank + 1])`. Offsets start at zero and
are nondecreasing; source and target row counts need not be equal. The target
order is the concatenation of that rank's `THDRankLayout.spans`, in the supplied
order. A plan validates full coverage, span ownership, sequence containment,
and absence of gaps/overlap. Source offsets and layouts must be identical on
every peer; rank numbers are **process-group ranks**, not global ranks.

Preparation builds CPU interval metadata and device row permutations once.
Execution packs rows as necessary, performs a real variable-split
`all_to_all_single`, and unpacks into target span order. `restore` reverses both
the communication splits and the permutations. There is no cast, reduction,
padding, or replicated full-tensor gather in this transport. An identity
layout skips the collective only when **all ranks** are unchanged, and returns
the input tensor itself. Callers must account for this alias before modifying
a returned tensor in place.

All peers must call the exchanges in the same order with matching trailing
shapes and dtypes. Input tensors must have contiguous rows on the explicitly
selected device; `"cuda"` resolves to the current device during preparation.
An explicit device such as `torch.device('cuda', local_rank)` is also supported.
The caller owns any necessary conversion to contiguous storage. The process
group must outlive every backward using the redistributor. Backward participation
and collective order must also agree across ranks; rank-local conditional losses
must not cause peers to skip a required inverse exchange.

Prepare on the execution stream, or establish an explicit stream dependency
before using the device indices. Keep collective ordering consistent across
peers. Permutation indices are read-only after preparation; each nonidentity
exchange allocates its own packed/received outputs rather than overwriting
transport storage retained by an earlier autograd invocation.

## Integration

```python
from cula.distributed import (
    THDRedistributionPlan,
    THDRedistributor,
    plan_thd_layout,
)

# global_cu_seqlens and source_offsets are the same CPU metadata on all peers.
layout = plan_thd_layout(global_cu_seqlens, world_size=cp_world_size)
route = THDRedistributionPlan.build(layout.ranks, source_offsets, cp_rank)
exchange = THDRedistributor(route, local_q.device, group=cp_group)

target_q = exchange.distribute(local_q)
# The attention call must use this rank's span metadata, original document
# origins, and causal offsets. KV ownership is a separate contract.
target_output = attention_for_spans(target_q, layout.for_rank(cp_rank))
local_output = exchange.restore(target_output)
```

`distribute` backward invokes the inverse exchange; `restore` backward invokes
the forward exchange. First-order autograd is supported; higher-order gradients
are explicitly disabled. The full CSA benchmark also exercises the explicit
inverse APIs because native backward wrappers return gradient tensors directly.

This does not turn arbitrary attention into distributed attention: global KV
visibility, document isolation, causal masking, KV-gradient reduction and
replicated-parameter reduction remain the integration's responsibility. In
particular, do not pass a shuffled set of recurrent KDA spans into a contiguous
Kimi CP boundary plan. The CP16 experiment keeps KDA and KV/pooling ownership
contiguous, redistributes CSA query-side operands, and restores CSA outputs and
query-side gradients at the layer boundary.

For the selection-only strategy, use `plan_thd_selection` with calibrated costs
and an explicit row capacity. Route indexer Q/weights to its spans and inverse-route
the selected IDs/scores, then run attention in original source ownership. The
[planner contract](THD_LAYOUT_PLANNER.md#capacity-bounded-selection-planner)
describes this policy and its preparation/reuse budget.

## Execution and validation scope

This is an eager PyTorch/NCCL implementation with output/packing allocations.
It does not promise CUDA Graph capture, reusable tensor workspaces, asynchronous
overlap, or a topology-aware communication schedule. Tensor lifetimes can be
material at MLA widths and should be included in memory measurements.

CPU protocol tests cover unequal counts, reversed target span order, sender
packing, receiver unpacking, inverse counts and identity skipping. Real CP16
tests on two H20 nodes verify token/channel markers and both autograd directions
for all three layouts, including all 31 measured 1M distributions. Full-tensor CSA
output/gradient comparisons are in the linked report. The CSA harness requires
ratio-4-aligned document/source boundaries. In its default selection scope,
empty selection ranks skip native plan construction and selection kernels but
still participate in all exchanges. Attention stays in nonempty source shards.
The optional query-wide scope still requires nonempty attention query ranks;
these tests do not establish support for unaligned document boundaries.
