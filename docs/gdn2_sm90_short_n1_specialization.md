# GDN2 SM90 short-sequence / N=1 V64 specialization

## Design

A shape-driven dispatch extension to the production `GDN2PrefillKernel` that
improves the two highest-impact short/medium single-sequence routes without
changing the public API, numerical contract, or default behavior for any other
supported shape.

### Dispatch rules(automatic, no environment switch)

| Condition | Route |
|---|---|
| `N == 1 and T <= 64` | exact released preparation/commit schedule(V128, no retained tail) |
| `N == 1 and Hv == 16 and initial_state and final_state and T > 64` | V64 single State-WG with register-resident final-tail carry |
| all other supported shapes | unchanged V128 production path |

The dispatch is derived purely from input metadata inside `_compile`; the
compile-cache key is extended by the two derived booleans
(`use_n1_hv16_v64`, `retain_final_tail`) so each route compiles exactly one
specialization.

## Kernel changes(`GDN2PrefillKernel`)

Three new constructor parameters, each validated at construction:

- `value_tile: int = VALUE_SIZE` — `64` or `128`;
- `single_state_owner: bool = False` — must equal `(value_tile == 64)`;
- `retain_final_tail: bool = False` — register-resident final-tail carry for
  final-state routes.

All branches are compile-time (`cutlass.const_expr`), so the V128 production
path is instruction-identical to the released kernel when the new flags are
at their defaults.

## Validation(H20, source-bound)

- Correctness: `13 passed`(`9` matrix cases + `4` contract tests) including
  explicit `mha-t64-short-baseline` and `mha-t65-v64-boundary` boundary cases.
- Determinism: `100,000` launches / 6 rows PASS.
- Compute Sanitizer: memcheck / initcheck / synccheck / racecheck,
  `120` launches per tool, all PASS.
- Codegen/resources: no stack, local memory, or spill.
- Paired timing(S2/Q2 family): `0.9362x` vs released incumbent on the
  targeted rows(≈ 6.4% latency reduction); full S1-S5 all rows faster than
  pinned FLA.

## Scope boundary

This change is additive and does not alter the public `chunk_gdn2` signature,
the `[N,Hv,V,K]` state layout, supported-shape matrix, or any existing
behavior outside the two dispatch rules above.
