# THD context-parallel planning

cuLA separates **where selection runs** from **where recurrent state and
attention activations live**. `plan_thd_selection` balances causal indexer work
while KDA, pooling, attention forward/backward and indexer backward retain their
original contiguous CP shards. Only indexer Q/weights move; document-local
block IDs and FP32 scores return to the source query owners before attention.
The [redistributor](THD_REDISTRIBUTION.md) implements both directions and their
first-order autograd, without casting or reducing row values.

This avoids applying full per-document zigzag to every short document or moving
wide MLA attention activations just to balance selection. It is an eager
integration; it does not promise CUDA Graph capture or whole-model E2E speedups.
See the [31-distribution measurements](THD_SELECTION_RESULTS.md) for scope,
preparation costs, precision, memory and measured limitations.

## Public interfaces

- `plan_thd_selection`: calibrated, capacity-bounded selection ownership; target
  row counts can differ or be zero. Keep attention and KDA in source ownership.
- `zigzag_thd_layout`: conventional per-document early/late pairing, shared by
  the planner and benchmark baseline. Rank r receives pieces r and 2*CP-1-r
  from each document, including short documents. Empty pieces are omitted;
  adjacent pieces remain distinct execution spans.
- `plan_thd_layout`: existing balanced-token policy, for consumers that need
  exact balanced block capacities. Its default analytic objective is preserved.
  This API alone does not make shuffled KDA spans executable by a contiguous
  Kimi CP state chain.

All planning uses CPU sequence boundaries and explicit caller configuration.
There is no device probing, online timing, hidden cache, process collective or
case-name lookup. Peers must use identical global metadata, coefficients and
configuration. Plan once at microbatch ingress and reuse where appropriate.
Planning is deterministic; kernel `deterministic=False` remains the throughput
option and is independent of the planner's stable tie-breaking.

```python
from cula.distributed import plan_thd_selection, THDRedistributionPlan, THDRedistributor

layout = plan_thd_selection(
    global_cu_seqlens,
    world_size=cp_world_size,
    max_tokens_per_rank=query_capacity,
    cost_model=calibrated_model,
)
route = THDRedistributionPlan.build(layout.ranks, source_offsets, cp_rank)
exchange = THDRedistributor(route, local_index_q.device, group=cp_group)
selection_q = exchange.distribute(local_index_q)
selection_weights = exchange.distribute(local_weights)
# Selection must use layout.for_rank(cp_rank)'s original document origins,
# causal offsets and document-local block numbering, with the required keys.
selected = select_for_spans(selection_q, selection_weights, layout.for_rank(cp_rank))
local_ids = exchange.restore(selected.block_ids)
local_scores = exchange.restore(selected.scores)
```

The model assumes balanced, block-aligned contiguous source shards. Pass those
same offsets to the redistributor. `query_capacity` must be block aligned and
fit the largest source shard. It bounds query-shaped buffers, **not** gathered
keys or backend scratch. The full-CSA benchmark uses twice the source row count;
callers must budget actual backend workspace separately. Documents must be
block aligned for these planners; the current CSA harness uses ratio four.

## Candidate pool and fallback

Selection compares all feasible candidates under one cost model:

1. Contiguous source ownership.
2. Complete per-document zigzag, including documents below the split threshold.
3. Early/late pairs within long documents, relabeled onto ranks with a
   minimum-cost assignment to reduce aggregate traffic. Short source spans stay
   local in this candidate family.
4. Greedy long-span placement on grids from 1 through `max_chunks_per_rank`
   pieces per rank. Descending modeled work is placed using current compute,
   incoming/outgoing traffic and row capacity; a span splits only when no bin
   can hold it. Adjacent same-owner spans coalesce.

Only candidates 3 and 4 pin documents shorter than twice `min_split_tokens`
(default 16K). Full zigzag is independently considered and obeys the same row
cap. A zigzag that concentrates tiny documents beyond that cap is infeasible.
The minimum-cost assignment is a stable O(CP^3) primal-dual algorithm. It
minimizes aggregate traffic, so the result is still scored by the full
critical-path model rather than assumed to be best.

Migration requires both `min_gain_fraction` (default 5%) and `min_gain_ms`
(default zero) over contiguous. `min_gain_ms` is an explicit minimum predicted
saving per selection use. A caller can price one-time preparation divided by
an expected reuse count; cuLA neither guesses that count nor hides preparation
in steady-state timing. `evaluated_scores` contains only materialized/scored
candidates. The selected modeled cost cannot exceed contiguous; this is **not**
a guarantee of minimum measured GPU latency over arbitrary distributions.

## Compute and communication model

For each rank, compute cost in nanoseconds is

```text
candidate_ns * exact_causal_candidates
    + query_ns * query_rows + 1000 * span_us * execution_spans
```

The candidate count is an integer prefix-sum formula for the complete causal
blocks visible at each query. It includes original-document causal offsets,
not just each fragment's local length. Critical compute is the maximum over
ranks.

Transport sums actual rows over source/destination edges. `bytes_per_query`
includes the outgoing Q/weights and returning IDs/scores in their real dtypes.
The critical transfer term takes the maximum of:

- incoming and outgoing per-rank time at explicit intra/inter-node rates;
- the busiest aggregated peer edge, if `peer_gbps` is supplied;
- shared node injection/receive time, if `inter_node_aggregate_gbps` is supplied.

When `local_copy_gbps` is supplied, rows retained on their source rank also
consume the local per-rank transfer resource. A nonidentity eager all-to-all
still copies its diagonal, including ranks whose ownership is unchanged.
These bytes do not cross a network link and do not add remote peer startups.
Omitting the coefficient preserves older profiles. A global identity layout
still bypasses the collective entirely.

Add the busiest rank's nonempty peer count times `peer_startup_us`, actual
permutation-copy work times `packing_ns_per_query`, and `exchange_us`. Multiple
fragments on one edge do not pay multiple peer startups. Naturally ordered
source/receive ranges skip `index_select` and carry zero copy cost on that side.
Global identity skips all transport terms.

Ordered forward exchange, compute and inverse exchange costs add. The optional
`overlap_intra_inter` lets local and inter-node transfer resources overlap
inside collectives; it does not assume compute/communication overlap. Enable
it only after measuring mixed-link traffic. Coefficients are effective eager
implementation costs, not physical link specifications. The payload model is
shape-specific; different head counts, top-K, topology or software require
calibration and held-out validation.

## Admissible pruning and preparation complexity

Three lower bounds avoid building candidates that cannot meet the margin:

- Every partition has the same token/candidate sums and at least one span per
  document. Their mean is a compute lower bound. To reduce source rank i's
  compute below C, some work must leave it; no moved row can remove more work
  than its largest causal position allows. Optimistic outgoing bandwidth prices
  these minimum rows. Minimize `C + max(outgoing_bound_i(C))` at the endpoints
  and intersections of affine functions. With concurrent intra/inter links,
  their capacities add in this optimistic bound.
- Full zigzag has a stronger bound: mean compute plus exact outgoing intra/inter
  traffic. Consecutive pieces on the same node form interval runs. Intersecting
  source spans with those runs and the owner's two pieces avoids constructing
  O(documents * CP) objects. This bound omits incoming and peer bottlenecks and
  permutation copies. Local candidates run first to tighten its incumbent.
- The long-document candidate family retains pinned source spans. Its critical
  compute time cannot fall below the largest pinned per-rank compute load;
  added nonnegative work and transport cannot lower it. If that bound already
  misses the migration margin, skip the entire long-document search. Full
  zigzag is checked independently because it can move the short documents too.

Bounds are tested against exhaustive small assignments and concrete zigzag
layouts on contiguous, interleaved and irregular topologies. They cannot exclude
a strictly better modeled zigzag; they make no statement about model error.

Inside greedy placement, existing sender/receiver, peer and node bottlenecks
also bound every prospective rank's cost. Compute the rank's prospective
compute/copy price and reuse those existing transport bounds before building
its communication deltas. If this floor strictly exceeds the best placement
already examined, skip that rank. Nonnegative additions cannot improve it.
Equality must still evaluate the complete tie-break key, including locality
and row load. This changes neither the candidate family nor its assignments.

Repeated planning grids are deduplicated. Pinned spans are coalesced once;
their initial per-rank loads are computed once and copied for each grid. Adaptive
grids visit only movable documents. Owner groups are built in one pass, and
diagnostics are computed only for the
selected selection plan. Balanced-token scheduling uses a rank heap for
identical-cost pieces and indexes capacity repair by donor and piece shape.
Source-local spans have a direct intersection path; other source intersections
use binary search. A geometry-only grain check rejects infeasible full zigzag
before transport pricing, preserving the original-source-fragment exception.
These changes preserve the scores of evaluated candidates, choices and stable
tie breaks. Diagnostics can omit candidates proven unable to meet the margin.

DSA window construction checks whether all remaining rows fit before searching
for a split. It uses exact integer work, including unaligned causal offsets;
long documents still receive the same maximal budget-fitting prefixes. An
integer candidate total avoids rescanning all preceding short documents;
window boundaries, scratch requirements and numerical execution are unchanged.
Temporary solver state is local to one preparation call; there is no implicit
retained layout cache or new device-to-host metadata read. Callers can explicitly
reuse the resulting immutable plan across compatible CSA layers; see
[plan/workspace reuse and amortized accounting](CSA_CP_PLAN_REUSE.md).

## Calibrating and measuring

`benchmarks/calibrate_thd_selection.py` fits nonnegative candidate/query/span
costs from unmodeled selection runs and retains source JSON digests. Pipeline
exchange measurements use differences of maximum cumulative phase endpoints,
so KDA/selection arrival waits are not counted again as network work.

`benchmarks/bench_thd_transport.py` measures the actual Q/weight and inverse
ID/score exchanges with exact marker checks. Sparse peer pairs, node-local,
inter-node and dense exchanges exercise different message sizes. Local-retention
fixtures keep one quarter, one half or three quarters of each source shard and
move the remainder to a neighboring rank. This separates diagonal copies from
remote traffic. The 32768-row, striped and half-local fixtures are held out.
`--validation-only` measures additional
mixed-link fractions. `benchmarks/fit_thd_transport.py` fits nonnegative resource
costs without using held-out rows in its objective. For older sweeps containing
only ordered fixtures, permutation bandwidth comes from the input profile;
new locality fixtures also exercise packing. Sweeps without diagonal features
retain the older model. Unidentifiable node limits must not be
interpreted as measured physical bandwidth.

The [example profiles](../benchmarks/profiles/README.md) are optional deployment
inputs, not library defaults. The runner's `--suite validation` fixes the 31
measured THD shapes. Use `summarize_csa_layout_matrix.py` to check complete
output/gradient evidence, compatible configurations, all-rank samples and
preparation accounting before producing a comparison. Preparation includes
common metadata, layout/index/workspace construction and KDA transport buffers.
Report it separately and added to one iteration; amortize only when reuse is real.

## KDA integration

The [Kimi CP runtime](KIMI_CP_RUNTIME.md) requires contiguous source ownership.
When every internal shard boundary is a document boundary, all ranks agree that
no recurrence crosses ranks. Native local KDA/autograd then skips summaries and
boundary collectives, with unchanged operands, dtypes and batch-invariance.
One isolated rank cannot bypass exchanges required by other ranks. Manually
constructed plans conservatively default to requiring exchange.

On the native summary path, backward context is reused only if the summarized
first span covers the complete local shard. A prefix-only context does not
contain the full THD shard's per-token intermediates; full apply prepares its
own intermediates instead. The paired independent-shard benchmark checks the
fast path against forced boundary execution, including weak decay and a
crossing-document control.
