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

When `candidate_budget_bytes=None`, the balanced policy starts at 512 MiB and
adds 512 MiB for each 128K of absolute causal reach. Beyond the first 128K of
causal reach, each 64K of rank-local query rows adds up to a capped 4x
throughput multiplier; the global cap is 8 GiB. Thus the 128K packed workload
retains its 512 MiB workspace, the worst 1M/CP64 rank stays at its measured
4 GiB performance knee, and critical long-sequence ranks at 1M/CP4 select the
measured 8 GiB knee. The exported
`MEMORY_EFFICIENT_CANDIDATE_BUDGET_BYTES` forces a fixed 512 MiB policy, and
`MAX_BALANCED_CANDIDATE_BUDGET_BYTES` exposes the cap. This is a
memory/scheduling choice, not a numerical choice. Allocate one workspace per
concurrent execution stream and reuse it across compatible layers; do not
allocate one multi-GiB workspace per layer.

### 128K heterogeneous THD result

Input: 20 sequences of 1,024 tokens and two sequences of 55,296 tokens,
compression ratio 4, Top-K 512.

| Selection form | Windows | Candidate scratch | Median time |
|---|---:|---:|---:|
| One packed launch | 1 | 2.8573 GiB | 5.3818 ms |
| 512 MiB memory policy | 7 | 512 MiB | 5.3246 ms |

The memory policy reduced candidate scratch by 5.715x. The measured time was
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

### 1M context, simulated CP64 worst rank

The simulation uses one CP64 rank's real shape: 16,384 query tokens at logical
rank 63, 262,144 globally visible compressed blocks, ratio 4 and Top-K 512.
Every entry executes the real cuDNN scorer/Top-K kernels; only the 64-rank
network is virtualized.

| Candidate budget | Windows | Median selection time | Versus full |
|---:|---:|---:|---:|
| 256 MiB | 64 | 78.895 ms | 2.81x |
| 512 MiB | 32 | 72.247 ms | 2.58x |
| 1 GiB | 16 | 48.121 ms | 1.72x |
| 2 GiB | 8 | 37.752 ms | 1.35x |
| **4 GiB (adaptive CP64 choice)** | **4** | **33.519 ms** | **1.20x** |
| 8 GiB | 2 | 29.464 ms | 1.05x |
| 15.875 GiB full | 1 | 28.039 ms | 1.00x |

Four GiB is the measured knee: it saves 11.875 GiB on the worst logical rank
while retaining about 84% of the full-scratch throughput. The 512 MiB policy
remains available when activation pressure matters more than selection latency.

### 1M THD, real CP4 KDA + DSA indexer E2E

Input: 20 sequences of 1,024 tokens plus two sequences of 514,048 tokens,
four physical CP ranks, KDA forward/backward, ratio-4 CSA pooling, compressed
block all-gather, and deterministic Top-K 512 selection. KDA and the independent
indexer branch run on separate CUDA streams.

| Candidate policy | Critical GPU median | Versus old default |
|---|---:|---:|
| Old reach-only adaptive policy | 228.220 ms | 1.00x |
| Fixed 4 GiB | 218.031 ms | 1.05x |
| **Row-aware adaptive, up to 8 GiB** | **209.504 ms** | **1.09x** |
| Fixed 16 GiB | 210.363 ms | 1.08x |

Eight GiB is the E2E knee. On the critical rank it uses 12 selection windows
and raises peak allocated memory from about 43.9 GiB to 49.7 GiB. Sixteen GiB
doubles candidate scratch without improving the critical path. The output and
all five KDA gradients remained finite; proxy loss, norms, selected-id sum,
and finite-logit sum were stable across seven repeats and matched the smaller
workspace policies.

Nsight Systems measured only 1.318 ms of kernel-idle time in the 212.454 ms
critical-rank hot path (0.62%). CSA pooling itself took 0.046 ms and compressed
block communication about 0.2--0.3 ms. The remaining critical work is dominated
by 12 cuDNN score kernels (88.5 ms total) and 12 deterministic Top-K kernels
(90.1 ms total), overlapped with KDA and NCCL. High-priority streams and
20--100 ms staggered launches produced no stable gain, so cuLA does not retain
those scheduling knobs.

An isolated NCU sample identifies the next optimization boundary inside the
cuDNN kernels. The score kernel used 112 registers per thread and 75.8 KiB of
dynamic shared memory per block; theoretical occupancy was 25% and achieved
occupancy 10.9%, with 45.2% compute and 35.3% maximum-memory throughput. The
Top-K kernel used 39 registers and 25.0 KiB of dynamic shared memory, reached
71.6% occupancy, but only 32.7% compute and 27.1% memory throughput. Further
material gains therefore require reducing score-kernel register/shared-memory
pressure and improving Top-K reduction/radix synchronization in cuDNN, rather
than adding more cuLA stream or caching policy.

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
first token. `cudnn_index_pool_cp` executes the aligned fast path: each rank
pools its native THD rows first and all-gathers only compressed blocks. Unequal
block counts use uneven compressed-axis collective splits; raw token tensors
are never padded or gathered.

Backward deliberately does not reduce BF16 block gradients before cuDNN. It
all-to-alls each loss rank's compressed slice to its input owner, invokes the
local cuDNN backward once per loss rank, accumulates KV/score gradients in
FP32, and reduce-scatters FP32 APE gradients. This has the same total token work
as the global reference and avoids an early BF16 CP merge. At 128K/CP4 the
forward, dKV, and dScore matched bitwise; dAPE differed only by FP32 reduction
order and was closer to the FP32 eager oracle than the reference.

Measured on the four-GPU validation host with ratio 4, head dimension 128, and
the 20x1K + 2-long THD mix:

| Global tokens | Raw / pooled forward payload | Forward | Forward + backward |
|---:|---:|---:|---:|
| 128K | 16 MiB / 2 MiB | 0.214 / 0.150 ms (1.43x) | 0.713 / 0.799 ms (0.89x) |
| 1M | 128 MiB / 16 MiB | 1.037 / 0.230 ms (4.51x) | 3.670 / 2.026 ms (1.81x) |

The 1M forward+backward incremental peak allocation fell from 1,280 MiB to
640 MiB (2.0x). At 128K the precision-preserving four-launch backward costs
about 11% end-to-end for this operator; at 1M, reduced communication and
working set make the complete path 1.81x faster. Timings are medians of
interleaved, order-reversed pairs using each iteration's slowest rank. These
are isolated index-pooling measurements, not whole-model throughput claims.

Misaligned shards still report a bounded neighbor halo (at most three raw
tokens for ratio 4), avoiding the need for a raw-token all-gather in the next
executor. The public function currently rejects those seams rather than
silently changing block membership. `coff=2` and ranks owning zero complete
blocks are also explicit unsupported boundaries.

For large CP, the next execution layer should keep tokens stationary and move
only compressed blocks or QSA partial statistics. It must preserve FP32 merge
semantics; reducing collective payload by lowering merge precision is outside
this runtime's contract.

## Current boundary

Implemented and tested:

- differentiable cuDNN index pooling;
- aligned CP pool-before-gather with precision-preserving backward;
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
