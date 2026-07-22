# Lightning Attention SM90 Prefill Pipeline

> Dispatcher: `cula/ops/lightning/prefill.py`
> SM90 wrapper: `cula/ops/lightning/prefill_sm90.py`
> Kernel: `cula/ops/lightning/sm90/prefill_kernel.py`
> Schedule helpers: `cula/ops/lightning/sm90/schedule.py`
> Kernel class: `LightningSm90PrefillKernel`

## Recurrence

For each value head, Lightning Attention maintains an FP32 state in public
`[V, K]` orientation:

$$
S_t = \lambda_h S_{t-1} + v_t k_t^T,
\qquad
o_t = \mathrm{scale} \cdot S_t q_t,
\qquad
\lambda_h = e^{-d_h}.
$$

The kernel processes 64 tokens per chunk. It decomposes the recurrence into
four matrix products:

1. `QK`: query-key scores for the causal intra-chunk contribution.
2. `O1`: the incoming recurrent state multiplied by the current queries.
3. `O2`: values multiplied by the masked and decayed `QK` matrix.
4. `State`: decay the incoming state and add the weighted `V @ K` update.

The diagonal of the causal `QK` matrix is retained, so the output for token
`t` observes the state update from that same token.

## CTA and warp-group layout

Each work unit uses one 384-thread CTA split into three warp groups.

| Threads | Role | Main responsibility | Register target |
|---|---|---|---:|
| 0-127 | Load/store | Three input TMA producers and one output epilogue warp | 24 |
| 128-255 | Math0 | `QK` producer plus its share of `O1`, `O2`, and state fragments | 240 |
| 256-383 | Math1 | `QK` consumer plus its share of `O1`, `O2`, and state fragments | 240 |

The load/store warp group is divided further:

| Warp | Operation |
|---|---|
| 0 | TMA load Q |
| 1 | TMA load K |
| 2 | TMA load V |
| 3 | TMA store full output tiles or copy a packed tail |

The two math warp groups form a 256-thread collective for the register-source
matrix products. The FP32 recurrent state remains distributed in their
registers for the complete sequence.

## WGMMA operations

| Operation | Tile `(M, N, K)` | Operand source | Accumulator | Purpose |
|---|---:|---|---|---|
| QK | `(64, 64, 128)` | SMEM × SMEM | FP32 | Raw query-key scores |
| O1 | `(128, 64, 128)` | RMEM × SMEM | FP32 | Incoming state contribution |
| O2 | `(128, 64, 64)` | RMEM × SMEM | FP32 | Causal intra-chunk contribution |
| State | `(128, 128, 64)` | RMEM × SMEM | FP32 | Recurrent-state update |

`QK` is produced by Math0, transformed in registers with the causal mask and
per-distance decay, converted to BF16, and published through STMatrix. Both
math warp groups then consume the published tile for `O2`.

## Pipeline

```text
Q TMA warp ── Q[3 stages] ───────────────────────────────┐
K TMA warp ── K[3 stages] ───────┐                       │
V TMA warp ── V[2 stages] ───────┼──────────────┐        │
                                  │              │        │
Math0: QK WGMMA → mask + decay → BF16 publish   │        │
                                  │              │        │
Math0 + Math1:                    │              │        │
  O1 = decayed incoming state × Q ◄──────────────┼────────┘
  O2 = O1 + V × published QK      ◄──────────────┘
  publish scaled O ── O[3 stages] ────────────────► output warp
  state = chunk_decay × state + weighted V × K
```

The synchronization objects have separate ownership contracts:

| Object | Stages / participants | Direction |
|---|---:|---|
| Q TMA pipeline | 3 | Q warp → 256 math threads |
| K TMA pipeline | 3 | K warp → 256 math threads |
| V TMA pipeline | 2 | V warp → 256 math threads |
| Output pipeline | 3 | 256 math threads → output warp |
| QK published barrier | 256 threads | Math0 publication → Math1 consumption |
| QK consumed barrier | 256 threads | Consumers → next QK publication |
| TMA store pipeline | 3 | Output warp → global memory |

The producer and consumer handles are advanced independently. Q is released
after `O1`; K and V remain live until both the output and state update have
consumed them. The output pipeline is drained before the CTA completes or
reuses its storage for another persistent work unit.

## Per-chunk execution

For a chunk with `L <= 64` valid tokens:

1. The three producer warps issue TMA loads for Q, K, and V.
2. Math0 computes `QK`, zeros invalid and upper-triangular entries, and applies
   the distance-dependent decay.
3. Both math groups compute `O1` from the resident FP32 state and apply the
   token-position decay.
4. Both math groups accumulate `O2` from V and the published BF16 `QK` tile.
5. The scaled output is converted to BF16 and published to the output ring.
6. The state is multiplied by `lambda**L`, then updated with the valid,
   position-weighted V and K rows.
7. The output warp performs a TMA store for a full tile. A packed partial tail
   uses a bounded per-token copy and never crosses the sequence boundary.

The final partial chunk advances the recurrence by `L`, not by the padded tile
size of 64.

## Shared and register storage

The CTA allocates 189,440 bytes of dynamic shared memory, which limits Hopper
to one resident CTA per SM.

| Shared region | Shape / stages | Purpose |
|---|---|---|
| Q | `64 × 128 × 3` BF16 | Query TMA ring |
| K | `128 × 64 × 3` BF16 | Key TMA ring with state/QK dual views |
| V | `128 × 64 × 2` BF16 | Value TMA ring |
| QK | `64 × 64` BF16 | Masked and decayed publication tile |
| O | `128 × 64 × 3` BF16 | Output producer-consumer ring |
| Decay LUT | 65 FP32 values | `exp(-decay * distance)` for distances 0-64 |

The recurrent state is not stored in shared memory between chunks. Its FP32
fragments stay in the registers owned by Math0 and Math1. Optional initial and
final states are loaded from and written to public `[V, K]` tensors through an
explicit logical transpose.

## Fixed and packed scheduling

### Fixed length

The launch grid is `(1, value_heads, batch)`. Each CTA owns one
`(batch, value_head)` sequence and processes all of its chunks in order.

### Packed non-persistent

The launch grid is `(1, value_heads, num_sequences)`. Sequence boundaries come
from `cu_seqlens`, and `initial_state_indices` maps each sequence to its state
pool slot.

### Packed persistent

The launch uses at most one CTA per SM. CTA `c` processes the static-strided
work-unit sequence

```text
c, c + persistent_ctas, c + 2 * persistent_ctas, ...
```

over `num_sequences * value_heads` work units. There is no global atomic work
queue. A shared-memory proxy fence and CTA barrier separate consecutive work
units before pipeline storage and TensorMap views are reused.

## Supported specialization

- Target: `sm_90a` on compute capability 9.0.
- CuTe DSL: `nvidia-cutlass-dsl==4.5.1`.
- Q, K, V, and O: BF16.
- Decay and recurrent state: FP32.
- Key and value dimensions: 128.
- Chunk size: 64.
- Head mapping: MHA and GVA (`HV >= H` and `HV % H == 0`).
- Forward prefill only.

## Benchmark

Use the prefill-specific benchmark for fixed, stateful, and packed workloads:

```bash
python benchmarks/bench_lightning_attn_prefill.py \
  --modes no_state h0_ht varlen \
  --num-heads 64 \
  --iterations 100
```
