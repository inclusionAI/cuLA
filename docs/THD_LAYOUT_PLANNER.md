# Cost-driven THD context-parallel layout planner

## Why the alternating prototype is not the production policy

Alternating the shard direction of adjacent long samples exactly balances the
original `20 x 1K + 2 x 514048` experiment, but it encodes properties of that
batch into the policy.  It is weak for a single dominant sequence, an odd
number of long sequences, and power-law mixtures.

`plan_thd_layout` instead consumes arbitrary CPU `global_cu_seqlens` and
balances two independent resources:

- token-linear KDA and index-pooling work;
- position-dependent causal candidate work in CSA/DSA.

Short samples remain whole when exact token capacity permits it. Long samples
are considered on several planning grids from `1 x CP` through `6 x CP`. Equal-token pieces are assigned by a
deterministic multi-resource largest-processing-time scheduler, then adjacent
pieces with the same owner are coalesced. This naturally produces an
early/late zigzag for one long sequence and cross-sample compensation when
several long sequences are available.

After compute scheduling, a capacity-repair pass moves whole spans where
possible and otherwise splits the cheapest span boundary. Every rank receives
its exact balanced block capacity (at most one block difference when the global
block count is not divisible by CP), so the plan remains compatible with
fixed-shape KDA and pooled all-gather collectives.

## Objective and fallback guarantee

For rank `r`, let `T_r` be its tokens and `C_r` its exact analytic causal
candidate count. The default robust objective is

```text
max(max(T_r) / mean(T_r), max(C_r) / mean(C_r))
    + 0.02 * max(cross_rank_boundary_count_r)
```

The boundary term prevents fine planning grids from winning on an immaterial
compute difference while introducing extra KDA state chains. The weights and
maximum grid are explicit configuration rather than hidden thresholds.

The planner always evaluates the ordinary contiguous CP partition with the
same objective and returns the lowest-scoring candidate. Therefore its result
is never worse than contiguous under the declared model. This is a model-level
guarantee; GPU profiling is still needed to calibrate how candidate work,
token work, and boundary communication overlap on a target system.

## Representative CP4 distributions

The CPU benchmark is `benchmarks/large_cp_emulation/bench_thd_planner.py`.
The following table uses 1M total tokens and the default cost model.

| THD distribution | selected grid | token max/avg | causal max/avg | selected score | contiguous score | remote boundaries |
|---|---:|---:|---:|---:|---:|---:|
| 1024 equal short samples | contiguous | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| one 1M sequence | 2 x CP | 1.000 | 1.000 | 1.080 | 1.790 | 6 |
| 20 x 1K + 2 x 514048 | 2 x CP | 1.000 | 1.000 | 1.040 | 1.560 | 4 |
| 256 x 1K + 1 x 768K | 3 x CP | 1.000 | 1.037 | 1.157 | 2.261 | 10 |
| 32 x 4K + 128K + 256K + 512K | 4 x CP | 1.000 | 1.046 | 1.166 | 2.302 | 10 |
| power-law 1K through 511K | 4 x CP | 1.000 | 1.028 | 1.188 | 2.149 | 15 |

The important result is not that one grid wins everywhere. Low-fragmentation
`2 x CP` is selected for the original and single-long cases, while irregular
mixtures spend additional boundaries only when they materially improve both
token and causal balance.

At logical CP64, the planning grid scales below the fixed 16K short-sample
threshold for long sequences. A one-million-token sequence therefore still
selects `2 x CP`, with exact token and causal balance. The original two-long
mixture selects `5 x CP` and reduces modeled causal max/average from 2.007 to
1.072; the power-law mixture also selects `5 x CP` and reduces it from 2.770
to 1.034. The threshold is used only to preserve genuinely short samples, not
to prevent a dominant long sequence from forming sub-rank zigzag pieces.

## Four-GPU DSA validation

The layout-native DSA benchmark executes real deterministic cuDNN index
pooling and DSA selection, including pooled-key all-gather and global block-ID
remap. Each number below is the median critical-path GPU time after two warmups
and three measured iterations. All cases contain 1,048,576 global tokens.
These published measurements used four index heads; current benchmark output
records `index_heads` explicitly because the default model shape is now 64.

| THD distribution | contiguous | planned | speedup | planned causal max/avg |
|---|---:|---:|---:|---:|
| one 1M sequence | 418.51 ms | 239.16 ms | 42.9% | 1.000 |
| 20 x 1K + 2 x 514048 | 166.77 ms | 109.05 ms | 34.6% | 1.000 |
| 256 x 1K + 1 x 768K | 291.80 ms | 134.35 ms | 54.0% | 1.037 |

For the original mixture, planned per-rank selection time is approximately
108 ms and remap plus collective time is 0.55--0.96 ms. Under contiguous CP,
the light ranks spend 107--120 ms waiting at the collective while the heavy
ranks finish selection. For the single-long sequence, contiguous candidate
work grows from 8.59 to 60.13 billion across ranks; the planned layout gives
every rank exactly 34.36 billion candidates. This confirms that the speedup is
removal of causal load bubbles rather than a precision or kernel change.

## Plan contract and integration boundary

`THDLayoutPlan` records the global spans, owner rank, local `cu_seqlens`,
causal offsets, per-resource loads, and all candidate scores. Planning is
CPU-only and deterministic. It must run once at microbatch ingress, before the
model layer stack, rather than permuting tokens inside every KDA or CSA layer.

The current KDA runtime still assumes one contiguous global token shard. A
separate lowering step is required before `THDLayoutPlan` becomes its default:

1. derive every sequence's ordered `(sequence, span ordinal)` chain;
2. merge forward and backward KDA state in sequence order, not physical-rank
   order;
3. produce CSA global block IDs and key-remap metadata from the same spans;
4. apply the token permutation once and retain it across all model layers.

The benchmark adapter already lets DSA consume the planned layout. Keeping the
planner independent from KDA transport is intentional: layout policy and the
collective implementation can be validated and evolved separately.
