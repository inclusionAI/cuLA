# Reusing CSA plans across layers

The [31-case B300 CP8 study](CSA_CP_1M_REUSE_RESULTS.md) now directly measures
four complete CSA instances sharing one plan/workspace at 1M. The cross-window
amortization table below remains a projection from its original single-CSA
cohort; it is not relabeled as that new measurement.

A single-CSA benchmark deliberately charges fresh preparation to one operator.
A training stack can prepare once at microbatch ingress and reuse the same plan
across compatible CSA layers. More KDA or MLP layers also enlarge the denominator
of a model-step overhead ratio, but do not count as additional CSA plan uses.

## What is reusable

`CsaCPSelectionPlan` is immutable host metadata: document geometry, source and
selection ownership, exchange ranges and native scratch/window sizes. It does
not contain Q, pooled keys, weights, or a layer's selected block IDs. Those
values are supplied or recomputed on every invocation.

Reuse requires the same document boundaries, source layout, group/rank mapping,
block size, Top-K, capacity and scratch policy, with a cost profile applicable
to the layer's indexer shape, dtypes and topology. Equal total tokens alone are
insufficient. See the [API contract](CSA_CP_SELECTION.md#reuse-across-layers).

The same device workspace can serve sequential invocations on one group/stream.
Selection results own their storage; retain each layer's results needed by
backward. Concurrent invocations need separate mutable workspaces or explicit
ordering. Sharing metadata does not authorize sharing or caching activations,
layer-dependent selections or saved backward tensors.

## Amortized accounting

Let `P_host` be host plan construction, `P_device` device metadata/workspace
preparation, `S` one complete CSA forward/backward execution, and `R` compatible
uses of the layout:

- Reuse only the host plan: `S + P_device + P_host / R` per use.
- Reuse both plan and device workspace: `S + (P_host + P_device) / R` per use.
- Communication, layout conversions, selection, attention and backward remain
  in `S` and execute at every layer. Only preparation is divided by `R`.

For a policy comparison, amortize preparation for every policy, including
contiguous and full zigzag. A candidate pays for migration when
`R * (S_baseline - S_candidate) > P_candidate - P_baseline`, subject to the
configured safety margin and memory constraints. A caller can divide a
calibrated one-time preparation headroom by expected reuse when setting
`min_gain_ms`; kernel, transport and fractional-margin coefficients are not
one-time costs and must not be divided. The library does not infer a layer
count or silently change the supplied profile.

The following is an **accounting projection**, not a measured multilayer model
run. It uses the [completed 24-case B300 CP8 cohort](results/csa_cp_preparation_20261116.json)
on source `489779b`, with `P = prepare_ms` and `S = step_ms`. It assumes all
measured metadata/workspace preparation is shared and each compatible layer
has the measured single-layer cost. The complete-CSA harness also prepares
common pooling/source-layout metadata; downstream integration must reuse that
metadata too to realize the full `P / R` credit. Reusing only the CSA host plan
amortizes only that portion. Compilation and input creation remain excluded.

| Compatible CSA uses R | Worst projected gap to faster contiguous/zigzag across 24 cases |
|---:|---:|
| 1 | +19.399% |
| 4 | +5.067% |
| 8 | +2.634% |
| 16 | +1.413% |

Each row compares `S_policy + P_policy / R` with the smaller fixed-policy value
using that same R. This is not the planner's own preparation percentage: shared
baseline preparation is included too. It does not establish a universal <3%
preparation guarantee. The steady limit remains the separately measured worst
+0.339% gap.

## Lessons from MagiAttention

We inspected MagiAttention revision
[`e5bcb9f`](https://github.com/SandAI-org/MagiAttention/tree/e5bcb9f4613d84f2f42442bd8bd6227665bb37b8).
Its static dispatch separates pre-iteration geometry from execution and includes
load and affinity heuristics. Its dynamic solver retains source ownership and
balances compute with send/receive bottlenecks, but requires per-layer mask
information. The documented CPU solver has a D2H limitation for GPU-generated
masks. These are useful distinctions for CSA:

| Idea | Application to cuLA |
|---|---|
| Reusable static scheduling metadata | Plan the known causal selection geometry once; recompute layer-dependent scores and selected IDs each time. |
| Bottleneck-aware task placement | Include rank, peer and node traffic; eliminate provably losing placements before constructing their full communication deltas. |
| Local work plus remote stages | A possible next runtime optimization, requiring explicit cuDNN dependencies, buffer lifetimes and actual overlap measurements. |
| Dynamic sparse-mask scheduling | A separate problem from geometry-only selection planning; avoid importing a CPU mask round trip into this API. |

Sources: [static dispatch implementation](https://github.com/SandAI-org/MagiAttention/blob/e5bcb9f4613d84f2f42442bd8bd6227665bb37b8/magi_attention/meta/solver/dispatch_solver.py),
[dynamic solver design and limitations](https://github.com/SandAI-org/MagiAttention/blob/e5bcb9f4613d84f2f42442bd8bd6227665bb37b8/docs/source/blog/dynamic_solver.md).

The current cuLA execution has ordered exchange/compute/return phases. Its cost
model must continue to sum the exposed stages. Replacing that sum with a maximum
would assume overlap that the implementation does not provide. Adding overlap
requires a runtime change and a new calibration; it is not a cost-model shortcut.

## Implemented follow-up: cheaper rank placement

The follow-up keeps the existing candidate family and objective. During greedy
placement, current sender/receiver, peer and node bottlenecks give an optimistic
floor for each possible destination. Adding nonnegative traffic cannot lower
those bottlenecks. If that floor strictly exceeds the best rank already scored,
skip construction of its detailed communication deltas. Equality still evaluates
the complete key, preserving the locality and load tie breaks. This is an
independent cuLA optimization motivated by the scheduling study, not a port of
MagiAttention's solver or a change to native attention math.

The following measures **`plan_thd_selection` on the B300 host CPU**. CP16/CP64
are logical metadata sizes, not H20 or 16-/64-GPU execution results. Baseline is
`aafd594`; each variant has two warmups and 15 samples in alternating order.
Both use the same profile with zero absolute headroom, a 1,024-token floor,
six grids/rank and twice-source row capacity. The full report includes all 24
CP8 shapes and five representative 1M shapes at each larger logical CP.

| Logical CP | 1M distribution | Before (ms) | After (ms) | Host planning time reduction |
|---:|---|---:|---:|---:|
| 8 | one long | 6.109 | 4.802 | 21.4% |
| 8 | 20 short + 2 long | 6.649 | 5.067 | 23.8% |
| 8 | 1,000-document long-tail | 15.992 | 14.380 | 10.1% |
| 16 | one long | 20.473 | 14.382 | 29.8% |
| 16 | 20 short + 2 long | 21.884 | 14.595 | 33.3% |
| 16 | 1,000-document long-tail | 29.331 | 22.596 | 23.0% |
| 64 | one long | 284.068 | 179.442 | 36.8% |
| 64 | 20 short + 2 long | 312.317 | 175.745 | 43.7% |
| 64 | 1,000-document long-tail | 275.419 | 159.970 | 41.9% |

Unchanged control paths have ordinary timing variation, up to about 3.4% in
this cohort. These CPU reductions do not imply faster steady GPU kernels. They
reduce the one-time cost that the caller can then amortize across layers.

Validation on the final implementation:

- 541 focused CPU tests pass, including a tie case that would fail if pruning
  used `>=` rather than strict `>`.
- 34 timed cases and 240 random topology/cost cases match every baseline plan
  field exactly, including candidate scores. The broader 660 comparisons retain
  the previously established layout/score parity; 300 DSA plans and 69,281
  windows remain identical.
- Eight B300 GPUs run two disjoint CP4 subgroups. The 128 native fixtures cover
  BF16 and FP32 weights, unequal sources, empty destinations and three distinct
  sequential inputs per plan/workspace. All 256 later-layer comparisons against
  fresh workspaces have identical canonical IDs and zero score difference.
  Every retained earlier result is unchanged. This checks the selection API,
  not a connected multilayer CSA forward/backward model.
- The earlier 24-case complete-CSA GPU timing/gradient cohort remains identified
  by its source revision; it has not been relabeled as a new GPU timing run.

[Machine-readable host timings, reuse checks and amortization projection](results/csa_cp_reuse_20261116.json)
retain the samples and source hashes. Reproduce the native reuse checks with
`benchmarks.check_csa_cp_selection --reuse-layers 3` under eight distributed
ranks, once with `--weights-dtype bf16` and once with `--weights-dtype fp32`.
