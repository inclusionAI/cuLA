# cuDNN hybrid-attention runtime

cuLA's hybrid layer is a layout-aware integration boundary for composing
cuDNN Frontend attention kernels. It is intentionally not another graph API:
cuDNN owns the kernels and their numerical contracts; cuLA owns static packed
metadata, bounded workspace, context-parallel layout, and framework-facing
PyTorch autograd.

The first implemented path covers index pooling and DSA block selection:

1. `cudnn_index_pool` consumes native contiguous THD rows and calls cuDNN CSA
   compressor forward/backward. The FP32 intermediate and dAPE reduction
   behavior are unchanged.
2. `DsaSelectionPlan` consumes host-visible packed query/block prefixes and
   derives exact causal selection windows.
3. `cudnn_dsa_select` consumes the packed pooling output directly and calls
   cuDNN's fused compact-logit + radix Top-K path. It does not materialize a
   dense score tensor and does not pad blocks to `[B, max_blocks, D]`.

The selected ids remain local compressed-block ids for each sequence. This is
the native QSA contract and avoids expanding `[T, 512]` block ids into a
`[T, 2048]` token-id tensor (128 MiB per rank at 16K local tokens). A future
QSA execution adapter should consume these block ids directly; cuLA should not
add an expansion solely to fit a token-index API.

## Numerical contract

The hybrid layer does not expose a precision downgrade knob:

- index-pool KV/score inputs are BF16 and APE is FP32, matching cuDNN;
- DSA indexer Q/K are BF16, weights may be BF16 or FP32, and selected logits
  are FP32;
- deterministic Top-K is the default, so exact ties at the K-th boundary keep
  the smaller block id;
- deterministic selection guarantees the `(block_id, logit)` set, not radix
  slot order. CUDA Graph capture/replay preserved the sorted pairs bitwise in
  validation, while the raw slot order changed;
- windowing replays the same causal rows and visible block range. It only
  changes scratch lifetime and launch partitioning.

On the remote SM100 validation environment, a 4096-token full launch and four
1024-row causal windows produced identical selected sets and bitwise-identical
selected logits (`max_abs = 0`).

## Why the selection plan owns windows

cuDNN's THD compact-logit path uses exact packed storage, but its scratch grows
with the number of visible `(query, block)` pairs. For a long causal sequence,
that is still quadratic scratch even though no dense score tensor is exposed.
`DsaSelectionPlan` splits rows at causal offsets and reuses one caller-owned
FP32 candidate buffer. It coalesces adjacent short packed sequences when they
fit the same budget.

The default budget is 512 MiB, matching cuDNN QSA's serving-side selection
budget. It is a memory/scheduling choice, not a numerical choice. Callers can
set another explicit budget when constructing the plan.

### 128K heterogeneous THD result

Input: 20 sequences of 1,024 tokens and two sequences of 55,296 tokens,
compression ratio 4, Top-K 512.

| Selection form | Windows | Candidate scratch | Median time |
|---|---:|---:|---:|
| One packed launch | 1 | 2.8573 GiB | 5.3818 ms |
| cuLA bounded plan | 7 | 512 MiB | 5.3246 ms |

The bounded plan reduced candidate scratch by 5.715x. The measured time was
effectively unchanged; the small apparent speedup is within benchmark noise.

The same batch partitioned as four contiguous 32K-token CP shards shows the
rank imbalance that a global-only measurement hides:

| CP rank | Full scratch | Bounded scratch | Windows | Full | Bounded | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 81.969 MiB | 81.969 MiB | 1 | 0.4274 ms | 0.4293 ms | +0.45% |
| 1 | 895.969 MiB | 511.970 MiB | 2 | 1.6170 ms | 1.6589 ms | +2.59% |
| 2 | 731.969 MiB | 489.990 MiB | 2 | 1.3802 ms | 1.4048 ms | +1.78% |
| 3 | 1,215.969 MiB | 511.985 MiB | 3 | 2.0281 ms | 2.1238 ms | +4.72% |

The worst rank saves about 704 MiB of candidate scratch for a 4.72% local
selection-stage cost. This table excludes model tensors, outputs, and
collectives; it isolates the DSA compact-logit stage.

## Integration example

```python
from cula.hybrid import (
    DsaSelectionPlan,
    DsaSelectionWorkspace,
    IndexPoolPlan,
    cudnn_dsa_select,
    cudnn_index_pool,
)

pool_plan = IndexPoolPlan.from_cu_seqlens(
    cu_seqlens_cpu,
    head_dim=128,
    ratio=4,
)
cu_tokens, cu_blocks = pool_plan.materialize(qi.device)
kbar = cudnn_index_pool(
    raw_index_k,
    pool_score,
    ape,
    cu_tokens,
    cu_blocks,
    plan=pool_plan,
    deterministic=True,
)

select_plan = DsaSelectionPlan.from_cu_seqlens(
    cu_queries_cpu,
    pool_plan.cu_seqlens_comp,
    top_k=512,
)
select_workspace = DsaSelectionWorkspace.from_plan(select_plan, qi.device)
selected = cudnn_dsa_select(
    qi,
    kbar[:, None, :],
    index_weights,
    plan=select_plan,
    workspace=select_workspace,
)
```

Construct plans and workspaces during setup/warmup, not inside CUDA Graph
capture. The execution call performs no implicit layout conversion. Returned
selection tensors alias the workspace and are overwritten on its next use.

## Context-parallel ownership

`IndexPoolShardPlan` assigns each complete ratio-4 block to the rank owning its
first token. Aligned shards pool locally and exchange only compressed rows.
Misaligned shards report a bounded neighbor halo (at most three raw tokens for
ratio 4), avoiding a raw-token all-gather. The current implementation exposes
and tests this ownership/halo plan; the distributed halo executor is not yet
part of the public path.

For large CP, the next execution layer should keep tokens stationary and move
only compressed blocks or QSA partial statistics. It must preserve FP32 merge
semantics; reducing collective payload by lowering merge precision is outside
this runtime's contract.

## Current boundary

Implemented and tested:

- differentiable cuDNN index pooling;
- packed THD DSA selection with exact bounded workspace;
- deterministic local block ids and FP32 logits;
- CP block ownership and seam-halo planning.

Not yet claimed as complete:

- the distributed ratio-4 halo exchange executor;
- a public block-id-native QSA forward/backward composition;
- hybrid linear + MLA collective scheduling;
- end-to-end CUDA Graph capture tests of the complete hybrid block.

These omissions are explicit because a reusable integration layer should not
hide layout copies or claim end-to-end correctness before those paths are
measured against the monolithic references.
