# Fully Fused Gated DeltaNet-2 Prefill SM90 Pipeline

> File: `cula/ops/gdn2/sm90/prefill_kernel.py`
> Class: `GDN2PrefillKernel`

## Recurrence

The public state uses `[sequence,Hv,V,K]`. For explanation, transpose one
head's state to the internal matrix `S_t` with shape `[K,V]`. At token `t`:

$$
\bar S_t = \operatorname{diag}(\exp(g_t)) S_{t-1}
$$

$$
u_t = w_t \odot v_t - (b_t \odot k_t)^T \bar S_t
$$

$$
S_t = \bar S_t + k_t u_t^T
$$

$$
o_t = \mathrm{scale}\; q_t^T S_t
$$

`q`, `k`, `b`, and `g` have 16 query/key heads. A GVA output head maps to its
owner query/key head while retaining its own `v`, `w`, recurrent state, and
output.

The kernel processes 64 tokens per chunk. It constructs the exact chunk-local
FP32 prefix from public raw `g`, builds the causal factor matrices for the
recurrence, and applies them without a global prefix workspace or a second
kernel launch. The intra-chunk factor matrices use the blockwise-rebased
factorization described in
[GDN2 SM90 stable factorization](gdn2_sm90_stable_factor.md): 16-token
sub-block operands with bounded exponents, warp-level MMA block pairs on the
producer warp group, and `n=16`-sliced state projections with per-block decay
deltas on the state warp groups.

## Launch and thread roles

Each CTA owns one `(sequence,value_head)` work unit and processes all of that
sequence's chunks in order.

| Resource | Product configuration |
|---|---:|
| Grid | `(N * Hv, 1, 1)` |
| Threads | 384 (12 warps, 3 warp groups) |
| Chunk size | 64 tokens |
| Dynamic shared memory | 232,192 B (V128 routes), 207,616 B (V64 route) |
| Register allocation | 168 registers per thread, all specializations |
| Spill | none: zero local load/store traffic |
| Residency | one CTA per SM |

The five V128 specializations sit 256 bytes under the 232,448-byte SM90a
per-CTA limit. That headroom is the binding constraint on the schedule: it
rules out a third input stage, a second factor-workspace stage, and any
de-aliasing of the FP16 Gram workspace on those routes. Only the V64 route
(`N=1`, `Hv=16`, initial and final state, `T>64`) has meaningful slack.

Spill is measured as SASS local load/store traffic, which is zero for every
specialization. The 1,024-byte `launch__stack_size` reported by the profiler
is the driver's fixed ABI reserve, not per-kernel spill.

The three warp groups have two coarse roles:

| Warp group | Role |
|---|---|
| WG0 | TMA input production, raw-G/factor preparation, shared factor publication, output consumption, and global stores |
| WG1 | FP32 recurrent-state slab for value rows `[0,64)` plus state/output WGMMA |
| WG2 | FP32 recurrent-state slab for value rows `[64,128)` plus state/output WGMMA |

WG1 and WG2 retain their FP32 state fragments in registers across chunks. They
exchange only staged operands and outputs through shared memory. CTA, named,
pipeline, and mbarrier synchronization provide explicit ownership transfers
between WG0 and the two state warp groups.

## Chunk pipeline

```text
 WG0 producer / factor / store                  WG1 + WG2 state math
          |                                                |
          |-- TMA raw Q,K,B,G,V,W ------------------------>|
          |                                                |
          |-- build FP32 G prefix and factor matrices      |
          |-- publish Qbar, erase, A_QK, A_KK^-1 --------->|
          |                                                |-- decay state
          |                                                |-- read old state
          |                                                |-- form write value
          |                                                |-- update state
          |<---------------- publish BF16 output tile -----|
          |-- TMA/store valid output tokens                |
          |                                                |
          +------------------- next chunk ------------------+
```

The input side is double-buffered by chunk stage. Value/write operands use
private producer stages, factor readiness and completion use explicit handoff
barriers, and output staging is pipelined independently. Invalid tail tokens
receive neutral values and are never written to global output.

## Stable LPT32 sequence schedule

The grid shape remains `N * Hv`, but the sequence dimension is remapped before
work starts:

1. Each sequence's cost is `ceil(length / 64)`.
2. Sequences are ranked by descending cost.
3. Equal costs retain ascending original sequence index.
4. Every value head uses the same remapping.

The rank is computed on-device from `cu_seqlens` for `N <= 32`. One warp
publishes the selected sequence through an existing shared slot, followed by a
single CTA synchronization. No host-side length sort, metadata copy, or extra
kernel is required.

This schedule keeps the longest sequence waves at the front of the grid. It
reduces tail under-utilization for imbalanced packed batches while preserving
exact recurrence order within every sequence.

## State modes and dynamic compilation

The host adapter caches one specialization per compile key:

```text
(device, Hv, has_initial_state, output_final_state,
 use_n1_hv16_v64, retain_final_tail)
```

The last two fields are shape-derived dispatch booleans introduced by the
[short-sequence / N=1 specialization](gdn2_sm90_short_n1_specialization.md):

- `retain_final_tail = output_final_state and not (N == 1 and T <= 64)`;
- `use_n1_hv16_v64 = N == 1 and Hv == 16 and has_initial_state and
  output_final_state and T > 64`.

`T` and `N` stay dynamic *within one dispatch route*, but crossing a route
boundary compiles a new specialization. Concretely:

- with `output_final_state=False`, both derived fields are constant `False`,
  so the key reduces to `(device, Hv, has_initial_state)` and `T`/`N` are
  fully dynamic;
- with `output_final_state=True`, there are up to three route
  specializations per `(device, Hv, has_initial_state)`: `N=1, T<=64`;
  `N=1, T>64` (a distinct V64 route only for `Hv=16` with an initial state);
  and every other supported shape.

Across `Hv={16,32,64}` and the four state modes this gives at most 19
specializations per device. Each first launch of a specialization pays a
one-time compilation on the order of tens of seconds; latency-sensitive
deployments should prewarm every route they will serve, including both sides
of the `N=1` / `T=64` boundaries when final states are requested.
Compilation and setup are excluded from steady-state benchmark timing.

## Fail-closed behavior

The public wrapper rejects unsupported tensor metadata before compilation.
The device kernel additionally checks sequence boundaries used by each CTA and
traps on invalid offsets. There is no alternate backend or silent fallback.

For shapes, dtypes, value preconditions, and the canonical benchmark, see
[GDN2 SM90 API](gdn2_sm90_api.md).
