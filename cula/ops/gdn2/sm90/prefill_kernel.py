# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Hopper SM90a Gated DeltaNet-2 forward-prefill kernel.

WG1/WG2 retain distributed common transforms and resident FP32 state.
They construct the exact chunk-local FP32 prefix from public raw G in the
existing two-stage shared arena, then prepare Q/K/B/G-derived operands for
generation ``n + 1`` before executing recurrence generation ``n``. WG0
computes causal QK, erase, and collective inverse for ``n + 1`` concurrently
with that recurrence. No global G-prefix workspace or second launch is used.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, warp, warpgroup
from cutlass.cutlass_dsl import T

from cula.ops._mlir_compat import llvm

from .collective_inverse_hmma import CollectiveInverse
from .config import CHUNK_SIZE, HEAD_SIZE, VALUE_SIZE

_INV_LN2 = 1.4426950408889634
_WARP_GROUP_SIZE = 128
_THREADS_PER_CTA = 384
_WGMMA_K = 16
_RAW_KEY_TILES = HEAD_SIZE // _WGMMA_K
_VALUE_TILES = VALUE_SIZE // _WGMMA_K
_RAW_STAGES = 2
_VW_PRIVATE_STAGES = 1
_INPUT_STAGES = 2
_FACTOR_WORKSPACE_STAGES = 1
_WRITE_STAGES = 1
_OUTPUT_STAGES = 2
_PRODUCER_SIGNAL_WARPS = _WARP_GROUP_SIZE // 32
_PRODUCER_REGISTER_TARGET = 72
_STATE_REGISTER_TARGET = 216
_STATE_VALUE_TILE = 64
_QKBG_TRANSACTION_BYTES = 3 * CHUNK_SIZE * _WGMMA_K * 2 + CHUNK_SIZE * _WGMMA_K * 4
_VW_TRANSACTION_BYTES = 2 * CHUNK_SIZE * _WGMMA_K * 2
_STATE0_WRITE_BARRIER = 2
_STATE1_WRITE_BARRIER = 3
_STATE_COMMON_BARRIER = 4
_STATE_ITERATION_DONE_BARRIER = 5
_STATE0_PREFIX_BARRIER = 6
_STATE1_PREFIX_BARRIER = 7
_STORE_WG_BARRIER = 1
_INVERSE_BARRIER = 13
_QKB_STREAM_TILES = _RAW_KEY_TILES // 2
_MAX_SEQUENCE_SCHEDULE = 32


@cute.jit
def _device_fail_closed() -> None:
    llvm.inline_asm(
        None,
        [],
        "trap;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _convert_c_layout_to_a_layout(
    c_layout: cute.Layout,
    a_value_layout,
):
    """Convert a Hopper C-fragment layout to its matching RS-A layout."""

    return cute.make_layout(
        (
            a_value_layout,
            c_layout.shape[1],
            (
                c_layout.shape[2],
                cute.size(c_layout, mode=[0]) // cute.size(a_value_layout),
            ),
        ),
        stride=(
            c_layout.stride[0],
            c_layout.stride[1],
            (
                c_layout.stride[2],
                cute.size(a_value_layout, mode=[2]) * c_layout.stride[0][2],
            ),
        ),
    )


@cute.jit
def _make_acc_into_op(
    accumulator: cute.Tensor,
    tiled_mma: cute.TiledMma,
) -> cute.Tensor:
    operand = cute.make_rmem_tensor_like(
        _convert_c_layout_to_a_layout(
            accumulator.layout,
            tiled_mma.tv_layout_A.shape[1],
        ),
        cutlass.BFloat16,
    )
    operand_as_accumulator = cute.make_tensor(
        operand.iterator,
        accumulator.layout,
    )
    operand_as_accumulator.store(
        accumulator.load().to(cutlass.BFloat16),
    )
    return operand


@cute.jit
def _fence_f32_register(reg: cutlass.Float32) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [reg.ir_value()],
            "",
            "=f,0",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        ),
    )


@cute.jit
def _fence_u32_register(reg: cutlass.Uint32) -> cutlass.Uint32:
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [reg.ir_value()],
            "",
            "=r,0",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        ),
    )


@cute.jit
def _fence_register_fragment(fragment: cute.Tensor) -> None:
    if cutlass.const_expr(fragment.element_type is cutlass.Float32):
        values = cute.recast_tensor(fragment, cutlass.Float32)
        for item in cutlass.range_constexpr(cute.size(values)):
            values[item] = _fence_f32_register(values[item])
    else:
        values = cute.recast_tensor(fragment, cutlass.Uint32)
        for item in cutlass.range_constexpr(cute.size(values)):
            values[item] = _fence_u32_register(values[item])


@cute.jit
def _wgmma_gemm(
    tiled_mma: cute.TiledMma,
    accumulator: cute.Tensor,
    operand_a: cute.Tensor,
    operand_b: cute.Tensor,
    accumulate: bool,
) -> None:
    for k_block in cutlass.range(
        cute.size(operand_a, mode=[2]),
        unroll_full=True,
    ):
        tiled_mma.set(
            warpgroup.Field.ACCUMULATE,
            accumulate or k_block != 0,
        )
        cute.gemm(
            tiled_mma,
            accumulator,
            operand_a[(None, None, k_block)],
            operand_b[(None, None, k_block)],
            accumulator,
        )


@cute.jit
def _stable_lpt32_sequence(
    cu_seqlens: cute.Tensor,
    sequence_rank: cutlass.Int32,
    num_sequences: cutlass.Int32,
    lane: cutlass.Int32,
) -> cutlass.Int32:
    """Return the stable descending chunk-count sequence for one rank."""

    if num_sequences > cutlass.Int32(_MAX_SEQUENCE_SCHEDULE):
        _device_fail_closed()

    chunk_count = cutlass.Int32(-1)
    if lane < num_sequences:
        start = cutlass.Int64(cu_seqlens[lane])
        end = cutlass.Int64(
            cu_seqlens[lane + cutlass.Int32(1)],
        )
        if start < cutlass.Int64(0) or end <= start:
            _device_fail_closed()
        chunk_count = cutlass.Int32(
            (end - start + cutlass.Int64(CHUNK_SIZE - 1)) // cutlass.Int64(CHUNK_SIZE),
        )

    rank = cutlass.Int32(0)
    for source_lane in cutlass.range(num_sequences, unroll=0):
        other_chunk_count = cute.arch.shuffle_sync(
            chunk_count,
            source_lane,
        )
        if other_chunk_count > chunk_count or (other_chunk_count == chunk_count and source_lane < lane):
            rank = rank + cutlass.Int32(1)

    selected = cutlass.Int32(-1)
    if lane < num_sequences and rank == sequence_rank:
        selected = lane
    for delta in (16, 8, 4, 2, 1):
        other_selected = cute.arch.shuffle_sync_down(
            selected,
            delta,
        )
        if other_selected > selected:
            selected = other_selected
    sequence = cute.arch.shuffle_sync(selected, 0)
    if sequence < cutlass.Int32(0):
        _device_fail_closed()
    return sequence


_FACTOR_BLOCK = 16
_FACTOR_SUB_BLOCKS = CHUNK_SIZE // _FACTOR_BLOCK
# Three precomputed pair-product rows: the distance-two products
# (delta1*delta2, delta2*delta3) and the distance-three product
# (delta1*delta2*delta3), stored FP16 so the factor fold reads one row per
# pair without growing the near-capacity V128 shared-memory budget. The
# FP16 rounding (2^-11 relative, subnormal floor 6e-8) is subdominant to
# the BF16 fold-operand rounding (2^-8) and to every comparison tolerance.
_GS_PAIR_ROWS = 3
_FACTOR_LOWER_PAIRS = (
    # Ordered so the modulo-4 warp assignment balances pair count against
    # fold count: the three-pair warps carry the diagonal (fold-free) pairs.
    (0, 0),
    (1, 1),
    (2, 0),
    (3, 1),
    (2, 2),
    (3, 3),
    (3, 0),
    (1, 0),
    (2, 1),
    (3, 2),
)
_FACTOR_UPPER_PAIRS = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
)


def _factor_pair_scale(
    gs_delta: cute.Tensor,
    gs_pair: cute.Tensor,
    channel: cutlass.Int32,
    query_block: int,
    key_block: int,
) -> cutlass.Float32:
    """Per-channel exp(Gs(I) - Gs(J)) read as one precomputed row.

    Trace-time helper (deliberately not ``@cute.jit``): distance-one pairs
    read the plain FP32 delta row; longer distances read the FP16 product
    rows the gs_delta writer precomputes once per generation.
    """

    distance = query_block - key_block
    if distance == 1:
        return cutlass.Float32(
            gs_delta[
                channel,
                key_block + 1,
            ],
        )
    row = key_block if distance == 2 else 2
    return cutlass.Float32(
        gs_pair[
            channel,
            row,
        ],
    )


@cute.jit
def _factor_block_pair(
    thread_mma,
    a_tiled_copy,
    b_tiled_copy,
    a_thread_copy,
    b_thread_copy,
    key_coordinates: cute.Tensor,
    q_blocks: cute.Tensor,
    erase_blocks: cute.Tensor,
    k_blocks: cute.Tensor,
    gs_delta: cute.Tensor,
    gs_pair: cute.Tensor,
    qk_accumulator: cute.Tensor,
    erase_accumulator: cute.Tensor,
    query_block: cutlass.Constexpr,
    key_block: cutlass.Constexpr,
) -> None:
    """Accumulate one 16x16 sub-block pair for both factor matrices.

    Operand fragments stream one 16-channel k-tile at a time through
    LdMatrix tiled copies so the producer warp group stays inside its
    register budget without scalar shared-memory traffic; the shared
    ``k~'`` fragment is loaded once per tile and used by both matrices,
    and off-diagonal pairs fold the bounded per-channel correction
    ``exp(Gs(I) - Gs(J)) <= 1`` into the left fragments before each MMA.
    """

    q_tiles = cute.flat_divide(
        q_blocks[None, None, query_block, 0],
        (_FACTOR_BLOCK, _WGMMA_K),
    )
    erase_tiles = cute.flat_divide(
        erase_blocks[None, None, query_block, 0],
        (_FACTOR_BLOCK, _WGMMA_K),
    )
    key_tiles = cute.flat_divide(
        k_blocks[None, None, key_block, 0],
        (_FACTOR_BLOCK, _WGMMA_K),
    )
    for key_tile in cutlass.range_constexpr(HEAD_SIZE // _WGMMA_K):
        key_slice = key_tiles[None, None, 0, key_tile]
        key_fragment = thread_mma.make_fragment_B(
            thread_mma.partition_B(key_slice),
        )
        cute.copy(
            b_tiled_copy,
            b_thread_copy.partition_S(key_slice),
            b_thread_copy.retile(key_fragment),
        )
        if cutlass.const_expr(query_block != key_block):
            # Fold exp(Gs(I) - Gs(J)) into the shared right fragment once:
            # the folded operand is k * exp(Gs(I) - G_j) <= |k|, and both
            # matrices consume the same corrected fragment.
            for element in cutlass.range_constexpr(
                cute.size(key_fragment),
            ):
                _, tile_channel = key_coordinates[element]
                channel = cutlass.Int32(key_tile * _WGMMA_K) + tile_channel
                pair_scale = _factor_pair_scale(
                    gs_delta,
                    gs_pair,
                    channel,
                    query_block,
                    key_block,
                )
                key_fragment[element] = cutlass.BFloat16(
                    cutlass.Float32(key_fragment[element]) * pair_scale,
                )

        query_slice = q_tiles[None, None, 0, key_tile]
        query_fragment = thread_mma.make_fragment_A(
            thread_mma.partition_A(query_slice),
        )
        cute.copy(
            a_tiled_copy,
            a_thread_copy.partition_S(query_slice),
            a_thread_copy.retile(query_fragment),
        )
        erase_slice = erase_tiles[None, None, 0, key_tile]
        erase_fragment = thread_mma.make_fragment_A(
            thread_mma.partition_A(erase_slice),
        )
        cute.copy(
            a_tiled_copy,
            a_thread_copy.partition_S(erase_slice),
            a_thread_copy.retile(erase_fragment),
        )
        cute.gemm(
            thread_mma,
            qk_accumulator,
            query_fragment,
            key_fragment,
            qk_accumulator,
        )
        cute.gemm(
            thread_mma,
            erase_accumulator,
            erase_fragment,
            key_fragment,
            erase_accumulator,
        )


@cute.jit
def _store_factor_qk_block(
    qk_store_tiled,
    qk_store_thread,
    block_coordinates: cute.Tensor,
    aqk_blocks: cute.Tensor,
    qk_accumulator: cute.Tensor,
    valid_tokens: cutlass.Int32,
    scale: cutlass.Float32,
    query_block: cutlass.Constexpr,
    key_block: cutlass.Constexpr,
) -> None:
    masked = cute.make_fragment_like(qk_accumulator, cutlass.BFloat16)
    for element in cutlass.range_constexpr(cute.size(qk_accumulator)):
        local_row, local_column = block_coordinates[element]
        global_row = cutlass.Int32(query_block * _FACTOR_BLOCK) + local_row
        global_column = cutlass.Int32(key_block * _FACTOR_BLOCK) + local_column
        qk_value = cutlass.Float32(0.0)
        if cutlass.const_expr(query_block == key_block):
            # Diagonal blocks: the causal test is live; the column bound is
            # implied by column <= row < valid_tokens.
            if global_row < valid_tokens and global_row >= global_column:
                qk_value = qk_accumulator[element] * scale
        else:
            # Strictly lower blocks: row >= 16*I > 16*J + 15 >= column, so
            # causality and the column bound both follow from the row bound.
            if global_row < valid_tokens:
                qk_value = qk_accumulator[element] * scale
        masked[element] = cutlass.BFloat16(qk_value)
    cute.copy(
        qk_store_tiled,
        qk_store_thread.retile(masked),
        qk_store_thread.partition_D(
            aqk_blocks[None, None, query_block, key_block],
        ),
    )


@cute.jit
def _store_factor_erase_block(
    erase_store_tiled,
    erase_store_thread,
    block_coordinates: cute.Tensor,
    inverse_blocks: cute.Tensor,
    erase_accumulator: cute.Tensor,
    valid_tokens: cutlass.Int32,
    query_block: cutlass.Constexpr,
    key_block: cutlass.Constexpr,
) -> None:
    masked = cute.make_fragment_like(erase_accumulator, cutlass.Float16)
    for element in cutlass.range_constexpr(cute.size(erase_accumulator)):
        local_row, local_column = block_coordinates[element]
        global_row = cutlass.Int32(query_block * _FACTOR_BLOCK) + local_row
        global_column = cutlass.Int32(key_block * _FACTOR_BLOCK) + local_column
        erase_value = cutlass.Float32(0.0)
        if cutlass.const_expr(query_block == key_block):
            # Diagonal blocks: keep the strict-lower test; the column bound
            # is implied by column < row < valid_tokens.
            if global_row < valid_tokens and global_row > global_column:
                erase_value = erase_accumulator[element]
        else:
            # Strictly lower blocks: row > column and the column bound both
            # follow from the row bound.
            if global_row < valid_tokens:
                erase_value = erase_accumulator[element]
        masked[element] = cutlass.Float16(erase_value)
    cute.copy(
        erase_store_tiled,
        erase_store_thread.retile(masked),
        erase_store_thread.partition_D(
            inverse_blocks[None, None, query_block, key_block],
        ),
    )


@cute.jit
def _publish_factor_blocks(
    thread_in_group: cutlass.Int32,
    q_tilde: cute.Tensor,
    erase_tilde: cute.Tensor,
    k_prime: cute.Tensor,
    gs_delta: cute.Tensor,
    gs_pair: cute.Tensor,
    aqk: cute.Tensor,
    inverse: cute.Tensor,
    valid_tokens: cutlass.Int32,
    scale: cutlass.Float32,
    fill_static_upper: cutlass.Boolean,
) -> None:
    """Publish causal QK and the strict-lower erase Gram per sub-block pair.

    The operands are blockwise rebased: ``q~``/``e~`` carry
    ``exp(G_i - Gs(B(i)))`` and ``k~'`` carries ``exp(Gs(B(j)) - G_j)``, so
    every stored exponent spans at most 15 in-block token gaps. Ten lower
    16x16 blocks per matrix run on four warps with warp-level m16n8k16 MMAs
    sharing one ``k~'`` fragment per pair per k-tile; the six upper blocks
    are zero-filled. The FP16 Gram destination aliases the dead raw-G
    staging arena, so no factor operand overlaps it.
    """

    warp_index = thread_in_group // cutlass.Int32(32)
    lane_index = thread_in_group % cutlass.Int32(32)

    tiled_mma = cute.make_tiled_mma(
        warp.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            (_FACTOR_BLOCK, 8, _WGMMA_K),
        ),
        (1, 1, 1),
        permutation_mnk=(_FACTOR_BLOCK, _FACTOR_BLOCK, _WGMMA_K),
    )
    thread_mma = tiled_mma.get_slice(lane_index)

    q_blocks = cute.flat_divide(q_tilde, (_FACTOR_BLOCK, HEAD_SIZE))
    erase_blocks = cute.flat_divide(erase_tilde, (_FACTOR_BLOCK, HEAD_SIZE))
    k_blocks = cute.flat_divide(k_prime, (_FACTOR_BLOCK, HEAD_SIZE))
    aqk_blocks = cute.flat_divide(aqk, (_FACTOR_BLOCK, _FACTOR_BLOCK))
    inverse_blocks = cute.flat_divide(
        inverse,
        (_FACTOR_BLOCK, _FACTOR_BLOCK),
    )

    key_coordinates = thread_mma.partition_B(
        cute.make_identity_tensor((_FACTOR_BLOCK, _WGMMA_K)),
    )
    block_coordinates = thread_mma.partition_C(
        cute.make_identity_tensor((_FACTOR_BLOCK, _FACTOR_BLOCK)),
    )

    a_atom = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=2),
        cutlass.BFloat16,
    )
    b_atom = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=2),
        cutlass.BFloat16,
    )
    qk_store_atom = cute.make_copy_atom(
        warp.StMatrix8x8x16bOp(transpose=False, num_matrices=2),
        cutlass.BFloat16,
    )
    erase_store_atom = cute.make_copy_atom(
        warp.StMatrix8x8x16bOp(transpose=False, num_matrices=2),
        cutlass.Float16,
    )
    a_tiled_copy = cute.make_tiled_copy_A(a_atom, tiled_mma)
    b_tiled_copy = cute.make_tiled_copy_B(b_atom, tiled_mma)
    qk_store_tiled = cute.make_tiled_copy_C(qk_store_atom, tiled_mma)
    erase_store_tiled = cute.make_tiled_copy_C(erase_store_atom, tiled_mma)
    a_thread_copy = a_tiled_copy.get_slice(lane_index)
    b_thread_copy = b_tiled_copy.get_slice(lane_index)
    qk_store_thread = qk_store_tiled.get_slice(lane_index)
    erase_store_thread = erase_store_tiled.get_slice(lane_index)

    for pair in cutlass.range_constexpr(len(_FACTOR_LOWER_PAIRS)):
        query_block = _FACTOR_LOWER_PAIRS[pair][0]
        key_block = _FACTOR_LOWER_PAIRS[pair][1]
        if warp_index == cutlass.Int32(pair % 4):
            qk_accumulator = cute.make_rmem_tensor(
                thread_mma.partition_shape_C((_FACTOR_BLOCK, _FACTOR_BLOCK)),
                cutlass.Float32,
            )
            erase_accumulator = cute.make_rmem_tensor(
                thread_mma.partition_shape_C((_FACTOR_BLOCK, _FACTOR_BLOCK)),
                cutlass.Float32,
            )
            qk_accumulator.fill(0.0)
            erase_accumulator.fill(0.0)
            _factor_block_pair(
                thread_mma,
                a_tiled_copy,
                b_tiled_copy,
                a_thread_copy,
                b_thread_copy,
                key_coordinates,
                q_blocks,
                erase_blocks,
                k_blocks,
                gs_delta,
                gs_pair,
                qk_accumulator,
                erase_accumulator,
                query_block,
                key_block,
            )
            _store_factor_qk_block(
                qk_store_tiled,
                qk_store_thread,
                block_coordinates,
                aqk_blocks,
                qk_accumulator,
                valid_tokens,
                scale,
                query_block,
                key_block,
            )
            _store_factor_erase_block(
                erase_store_tiled,
                erase_store_thread,
                block_coordinates,
                inverse_blocks,
                erase_accumulator,
                valid_tokens,
                query_block,
                key_block,
            )

    for pair in cutlass.range_constexpr(len(_FACTOR_UPPER_PAIRS)):
        query_block = _FACTOR_UPPER_PAIRS[pair][0]
        key_block = _FACTOR_UPPER_PAIRS[pair][1]
        for linear in cutlass.range(
            thread_in_group,
            _FACTOR_BLOCK * _FACTOR_BLOCK,
            _WARP_GROUP_SIZE,
            unroll=1,
        ):
            local_row = linear // cutlass.Int32(_FACTOR_BLOCK)
            local_column = linear % cutlass.Int32(_FACTOR_BLOCK)
            # The inverse arena aliases the raw-G staging that every chunk's
            # G transaction rewrites, so its upper blocks are re-zeroed each
            # generation. The aqk arena is private and nothing ever writes
            # its upper blocks, so those zeros are published once per stage.
            if fill_static_upper:
                aqk_blocks[
                    local_row,
                    local_column,
                    query_block,
                    key_block,
                ] = cutlass.BFloat16(0.0)
            inverse_blocks[
                local_row,
                local_column,
                query_block,
                key_block,
            ] = cutlass.Float16(0.0)
    cute.arch.fence_proxy("async.shared", space="cta")


class GDN2PrefillKernel:
    """Production raw-G prefill kernel with stable LPT32 sequence scheduling."""

    value_tile = 128
    state_value_tile = _STATE_VALUE_TILE
    threads_per_cta = _THREADS_PER_CTA
    min_blocks_per_mp = 1

    def __init__(
        self,
        *,
        has_initial_state: bool,
        store_final_state: bool,
        value_tile: int = VALUE_SIZE,
        single_state_owner: bool = False,
        retain_final_tail: bool = False,
    ) -> None:
        if value_tile not in (64, VALUE_SIZE):
            raise ValueError(f"unsupported GDN2 value tile: {value_tile}")
        if single_state_owner != (value_tile == 64):
            raise ValueError("V64 requires exactly one recurrent State WG")
        self.has_initial_state = has_initial_state
        self.store_final_state = store_final_state
        self.value_tile = value_tile
        self.single_state_owner = single_state_owner
        self.retain_final_tail = retain_final_tail
        self.subgroup_prefix = True
        self.subgroup_exclusive_carry = True
        self.elide_state_common_barrier = True
        self.elide_state_iteration_done_barrier = True
        self.sequence_wave_rotation = 0
        self.length_ranked_sequence_order = True

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        b: cute.Tensor,
        w: cute.Tensor,
        cu_seqlens: cute.Tensor,
        g: cute.Tensor,
        capsule_aqk: cute.Tensor,
        capsule_akk: cute.Tensor,
        initial_state: cute.Tensor,
        output: cute.Tensor,
        final_state: cute.Tensor,
        num_sequences: cutlass.Int32,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        total_tokens: cutlass.Int32,
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ) -> None:
        state_op = warpgroup.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            (self.state_value_tile, HEAD_SIZE, _WGMMA_K),
            warpgroup.OperandSource.RMEM,
            OperandMajorMode.K,
            OperandMajorMode.K,
        )
        # Token-dimension MMAs use an n16 atom so the state-side projections
        # can issue per 16-token sub-block with the state fragments rescaled
        # by the bounded block deltas between slices.
        token_op = warpgroup.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            (self.state_value_tile, _FACTOR_BLOCK, _WGMMA_K),
            warpgroup.OperandSource.RMEM,
            OperandMajorMode.K,
            OperandMajorMode.K,
        )
        state_mma = cute.make_tiled_mma(
            cute.make_mma_atom(state_op),
            cute.make_layout((1, 1, 1)),
        )
        token_mma = cute.make_tiled_mma(
            cute.make_mma_atom(token_op),
            cute.make_layout((1, 1, 1)),
        )
        raw_layout = sm90_utils.make_smem_layout_a(
            cutlass.utils.LayoutEnum.ROW_MAJOR,
            (CHUNK_SIZE, CHUNK_SIZE, _WGMMA_K),
            cutlass.BFloat16,
            _RAW_STAGES,
        )
        g_staging_layout = cute.make_layout(
            (CHUNK_SIZE, _WGMMA_K, _RAW_STAGES),
            stride=(
                _WGMMA_K,
                1,
                CHUNK_SIZE * _WGMMA_K,
            ),
        )
        operand_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW32,
            cutlass.BFloat16,
        )
        key_operand_layout = cute.tile_to_shape(
            operand_layout_atom,
            (CHUNK_SIZE, HEAD_SIZE, _INPUT_STAGES),
            (0, 1, 2),
        )
        factor_workspace_layout = cute.tile_to_shape(
            operand_layout_atom,
            (CHUNK_SIZE, HEAD_SIZE, _FACTOR_WORKSPACE_STAGES),
            (0, 1, 2),
        )
        token_operand_layout = cute.tile_to_shape(
            operand_layout_atom,
            (CHUNK_SIZE, CHUNK_SIZE, _INPUT_STAGES),
            (0, 1, 2),
        )
        factor_inverse_layout = cute.make_layout(
            (CHUNK_SIZE, CHUNK_SIZE, _FACTOR_WORKSPACE_STAGES),
            stride=(
                CHUNK_SIZE,
                1,
                CHUNK_SIZE * CHUNK_SIZE,
            ),
        )
        state_update_layout = cute.tile_to_shape(
            operand_layout_atom,
            (HEAD_SIZE, CHUNK_SIZE, _INPUT_STAGES),
            (0, 1, 2),
        )
        write_layout = cute.make_layout(
            (CHUNK_SIZE, self.value_tile, _WRITE_STAGES),
            stride=(
                self.value_tile,
                1,
                CHUNK_SIZE * self.value_tile,
            ),
        )
        gamma_layout = cute.make_layout(
            (HEAD_SIZE, _INPUT_STAGES),
            stride=(1, HEAD_SIZE),
        )
        gs_delta_layout = cute.make_layout(
            (HEAD_SIZE, _FACTOR_SUB_BLOCKS, _INPUT_STAGES),
            stride=(
                _FACTOR_SUB_BLOCKS,
                1,
                HEAD_SIZE * _FACTOR_SUB_BLOCKS,
            ),
        )
        gs_pair_layout = cute.make_layout(
            (HEAD_SIZE, _GS_PAIR_ROWS, _INPUT_STAGES),
            stride=(
                _GS_PAIR_ROWS,
                1,
                HEAD_SIZE * _GS_PAIR_ROWS,
            ),
        )
        output_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW64,
            cutlass.BFloat16,
        )
        output_layout = cute.tile_to_shape(
            output_layout_atom,
            (CHUNK_SIZE, self.value_tile, _OUTPUT_STAGES),
            (0, 1, 2),
        )

        q_heads = cute.size(q, mode=[1])
        v_heads = cute.size(v, mode=[1])
        token_extent = cute.size(q, mode=[0])
        raw_q_layout = cute.make_layout(
            (token_extent, HEAD_SIZE, q_heads),
            stride=(q_heads * HEAD_SIZE, 1, HEAD_SIZE),
        )
        raw_v_layout = cute.make_layout(
            (token_extent, VALUE_SIZE, v_heads),
            stride=(v_heads * VALUE_SIZE, 1, VALUE_SIZE),
        )
        q_global = cute.make_tensor(q.iterator, raw_q_layout)
        k_global = cute.make_tensor(k.iterator, raw_q_layout)
        b_global = cute.make_tensor(b.iterator, raw_q_layout)
        g_global = cute.make_tensor(g.iterator, raw_q_layout)
        v_global = cute.make_tensor(v.iterator, raw_v_layout)
        w_global = cute.make_tensor(w.iterator, raw_v_layout)

        output_global_layout = cute.make_layout(
            (token_extent, VALUE_SIZE, v_heads),
            stride=(v_heads * VALUE_SIZE, 1, VALUE_SIZE),
        )
        tma_output_global = cute.make_tensor(
            output.iterator,
            output_global_layout,
        )

        q_atom, q_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            q_global,
            cute.slice_(raw_layout, (None, None, 0)),
            (CHUNK_SIZE, _WGMMA_K),
        )
        k_atom, k_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            k_global,
            cute.slice_(raw_layout, (None, None, 0)),
            (CHUNK_SIZE, _WGMMA_K),
        )
        b_atom, b_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            b_global,
            cute.slice_(raw_layout, (None, None, 0)),
            (CHUNK_SIZE, _WGMMA_K),
        )
        v_atom, v_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            v_global,
            cute.slice_(raw_layout, (None, None, 0)),
            (CHUNK_SIZE, _WGMMA_K),
        )
        w_atom, w_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            w_global,
            cute.slice_(raw_layout, (None, None, 0)),
            (CHUNK_SIZE, _WGMMA_K),
        )
        g_atom, g_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            g_global,
            cute.slice_(g_staging_layout, (None, None, 0)),
            (CHUNK_SIZE, _WGMMA_K),
        )
        output_atom, output_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            tma_output_global,
            cute.slice_(output_layout, (None, None, 0)),
            (CHUNK_SIZE, self.value_tile),
        )

        @cute.struct
        class SharedStorage:
            qkb0_barriers: cute.struct.MemRange[
                cutlass.Int64,
                2,
            ]
            qkb1_barriers: cute.struct.MemRange[
                cutlass.Int64,
                2,
            ]
            vw0_barriers: cute.struct.MemRange[
                cutlass.Int64,
                2 * _VW_PRIVATE_STAGES,
            ]
            vw1_barriers: cute.struct.MemRange[
                cutlass.Int64,
                2 * _VW_PRIVATE_STAGES,
            ]
            raw_handoff_barriers: cute.struct.MemRange[
                cutlass.Int64,
                2,
            ]
            factor_ready_barriers: cute.struct.MemRange[
                cutlass.Int64,
                2 * _INPUT_STAGES,
            ]
            factor_done_barriers: cute.struct.MemRange[
                cutlass.Int64,
                2 * _INPUT_STAGES,
            ]
            output_handoff_barriers: cute.struct.MemRange[
                cutlass.Int64,
                2 * _OUTPUT_STAGES,
            ]
            producer_value_work_by_warp: cute.struct.MemRange[
                cutlass.Int32,
                _PRODUCER_SIGNAL_WARPS,
            ]
            raw_q: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(raw_layout),
                ],
                128,
            ]
            raw_k: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(raw_layout),
                ],
                128,
            ]
            raw_b: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(raw_layout),
                ],
                128,
            ]
            raw_g: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    cute.cosize(g_staging_layout),
                ],
                128,
            ]
            raw_v: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(raw_layout),
                ],
                128,
            ]
            raw_w: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(raw_layout),
                ],
                128,
            ]
            q_bar: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(key_operand_layout),
                ],
                128,
            ]
            erase_bar: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(key_operand_layout),
                ],
                128,
            ]
            key_tail: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(state_update_layout),
                ],
                128,
            ]
            aqk_scaled: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(token_operand_layout),
                ],
                128,
            ]
            akk_inverse: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(token_operand_layout),
                ],
                128,
            ]
            write_value: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(write_layout),
                ],
                128,
            ]
            factor_workspace: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(factor_workspace_layout),
                ],
                128,
            ]
            gs_delta: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    cute.cosize(gs_delta_layout),
                ],
                128,
            ]
            gs_pair: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float16,
                    cute.cosize(gs_pair_layout),
                ],
                128,
            ]
            output: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(output_layout),
                ],
                128,
            ]
            gamma_end: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    cute.cosize(gamma_layout),
                ],
                128,
            ]

        self.shared_storage = SharedStorage
        self.dynamic_smem_bytes = SharedStorage.size_in_bytes()
        self.kernel(
            q_atom,
            q_tma,
            k_atom,
            k_tma,
            b_atom,
            b_tma,
            g_atom,
            g_tma,
            v_atom,
            v_tma,
            w_atom,
            w_tma,
            output_atom,
            output_tma,
            output,
            cu_seqlens,
            g,
            initial_state,
            final_state,
            num_sequences,
            num_q_heads,
            num_v_heads,
            total_tokens,
            scale,
            state_mma,
            token_mma,
            raw_layout,
            g_staging_layout,
            key_operand_layout,
            factor_workspace_layout,
            token_operand_layout,
            factor_inverse_layout,
            state_update_layout,
            write_layout,
            gamma_layout,
            gs_delta_layout,
            gs_pair_layout,
            output_layout,
        ).launch(
            grid=(
                num_sequences * num_v_heads * cutlass.Int32(VALUE_SIZE // self.value_tile),
                1,
                1,
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=(1, 1, 1),
            smem=self.dynamic_smem_bytes,
            stream=stream,
            min_blocks_per_mp=self.min_blocks_per_mp,
        )

    @cute.kernel
    def kernel(
        self,
        q_atom: cute.CopyAtom,
        q_tma: cute.Tensor,
        k_atom: cute.CopyAtom,
        k_tma: cute.Tensor,
        b_atom: cute.CopyAtom,
        b_tma: cute.Tensor,
        g_atom: cute.CopyAtom,
        g_tma: cute.Tensor,
        v_atom: cute.CopyAtom,
        v_tma: cute.Tensor,
        w_atom: cute.CopyAtom,
        w_tma: cute.Tensor,
        output_atom: cute.CopyAtom,
        output_tma: cute.Tensor,
        output: cute.Tensor,
        cu_seqlens: cute.Tensor,
        g: cute.Tensor,
        initial_state: cute.Tensor,
        final_state: cute.Tensor,
        num_sequences: cutlass.Int32,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        total_tokens: cutlass.Int32,
        scale: cutlass.Float32,
        state_mma: cute.TiledMma,
        token_mma: cute.TiledMma,
        raw_layout: cute.ComposedLayout,
        g_staging_layout: cute.Layout,
        key_operand_layout: cute.ComposedLayout,
        factor_workspace_layout: cute.ComposedLayout,
        token_operand_layout: cute.ComposedLayout,
        factor_inverse_layout: cute.Layout,
        state_update_layout: cute.ComposedLayout,
        write_layout: cute.Layout,
        gamma_layout: cute.Layout,
        gs_delta_layout: cute.Layout,
        gs_pair_layout: cute.Layout,
        output_layout: cute.ComposedLayout,
    ) -> None:
        thread, _, _ = cute.arch.thread_idx()
        work_index, _, _ = cute.arch.block_idx()
        warp_group = cute.arch.make_warp_uniform(
            thread // _WARP_GROUP_SIZE,
        )
        warp_index = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        thread_in_group = thread % _WARP_GROUP_SIZE
        allocator = cutlass.utils.SmemAllocator()
        storage = allocator.allocate(self.shared_storage)
        producer_value_work_by_warp = storage.producer_value_work_by_warp.get_tensor(
            cute.make_layout(
                (_PRODUCER_SIGNAL_WARPS,),
                stride=(1,),
            ),
        )

        value_tiles = cutlass.Int32(VALUE_SIZE // self.value_tile)
        sequence_stride = num_v_heads * value_tiles
        sequence_rank = work_index // sequence_stride
        value_work = work_index - sequence_rank * sequence_stride
        sequence = sequence_rank
        if cutlass.const_expr(self.length_ranked_sequence_order):
            if warp_index == cutlass.Int32(4):
                scheduled_sequence = _stable_lpt32_sequence(
                    cu_seqlens,
                    sequence_rank,
                    num_sequences,
                    thread % cutlass.Int32(32),
                )
                if thread % cutlass.Int32(32) == cutlass.Int32(0):
                    producer_value_work_by_warp[0] = scheduled_sequence
            cute.arch.sync_threads()
            sequence = cute.arch.make_warp_uniform(
                producer_value_work_by_warp[0],
            )
        elif cutlass.const_expr(self.sequence_wave_rotation > 0):
            if num_sequences > cutlass.Int32(self.sequence_wave_rotation):
                sequence = sequence + cutlass.Int32(
                    self.sequence_wave_rotation,
                )
                if sequence >= num_sequences:
                    sequence = sequence - num_sequences
        value_head = value_work // value_tiles
        value_tile_index = value_work - value_head * value_tiles
        value_start = value_tile_index * cutlass.Int32(self.value_tile)
        group_size = num_v_heads // num_q_heads
        q_head = value_head // group_size

        sequence_start_i64 = cutlass.Int64(cu_seqlens[sequence])
        sequence_end_i64 = cutlass.Int64(
            cu_seqlens[sequence + cutlass.Int32(1)],
        )
        if (
            sequence_start_i64 < cutlass.Int64(0)
            or sequence_end_i64 <= sequence_start_i64
            or sequence_end_i64 > cutlass.Int64(total_tokens)
        ):
            _device_fail_closed()
        if sequence == cutlass.Int32(0) and sequence_start_i64 != cutlass.Int64(0):
            _device_fail_closed()
        if sequence == num_sequences - cutlass.Int32(1) and sequence_end_i64 != cutlass.Int64(total_tokens):
            _device_fail_closed()
        raw_q = storage.raw_q.get_tensor(
            raw_layout.outer,
            swizzle=raw_layout.inner,
        )
        raw_k = storage.raw_k.get_tensor(
            raw_layout.outer,
            swizzle=raw_layout.inner,
        )
        raw_b = storage.raw_b.get_tensor(
            raw_layout.outer,
            swizzle=raw_layout.inner,
        )
        raw_g = storage.raw_g.get_tensor(g_staging_layout)
        raw_v = storage.raw_v.get_tensor(
            raw_layout.outer,
            swizzle=raw_layout.inner,
        )
        raw_w = storage.raw_w.get_tensor(
            raw_layout.outer,
            swizzle=raw_layout.inner,
        )
        shared_q = storage.q_bar.get_tensor(
            key_operand_layout.outer,
            swizzle=key_operand_layout.inner,
        )
        shared_erase = storage.erase_bar.get_tensor(
            key_operand_layout.outer,
            swizzle=key_operand_layout.inner,
        )
        shared_key_tail = storage.key_tail.get_tensor(
            state_update_layout.outer,
            swizzle=state_update_layout.inner,
        )
        shared_aqk = storage.aqk_scaled.get_tensor(
            token_operand_layout.outer,
            swizzle=token_operand_layout.inner,
        )
        shared_akk = storage.akk_inverse.get_tensor(
            token_operand_layout.outer,
            swizzle=token_operand_layout.inner,
        )
        shared_write = storage.write_value.get_tensor(write_layout)
        shared_output = storage.output.get_tensor(
            output_layout.outer,
            swizzle=output_layout.inner,
        )
        shared_factor_k = storage.factor_workspace.get_tensor(
            factor_workspace_layout.outer,
            swizzle=factor_workspace_layout.inner,
        )
        # The FP16 Gram/inverse aliases the raw-G staging arena. Raw G for
        # chunk c is dead once both state warp groups finish preparation
        # (which factor_ready orders before the factor stage), and WG0 issues
        # the chunk c+1 G loads only after the chunk c factor stage returns,
        # so the alias never overlaps a live read or an in-flight TMA write.
        raw_g_address = storage.raw_g.data_ptr().toint()
        shared_inverse = cute.make_tensor(
            cute.make_ptr(
                cutlass.Float16,
                raw_g_address,
                cute.AddressSpace.smem,
                assumed_align=128,
            ),
            factor_inverse_layout,
        )
        shared_gamma_end = storage.gamma_end.get_tensor(gamma_layout)
        shared_gs_delta = storage.gs_delta.get_tensor(gs_delta_layout)
        shared_gs_pair = storage.gs_pair.get_tensor(gs_pair_layout)
        qkb0_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.qkb0_barriers.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                _PRODUCER_SIGNAL_WARPS,
            ),
            tx_count=_QKBG_TRANSACTION_BYTES,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        qkb1_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.qkb1_barriers.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                _PRODUCER_SIGNAL_WARPS,
            ),
            tx_count=_QKBG_TRANSACTION_BYTES,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        vw0_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.vw0_barriers.data_ptr(),
            num_stages=_VW_PRIVATE_STAGES,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                _PRODUCER_SIGNAL_WARPS,
            ),
            tx_count=_VW_TRANSACTION_BYTES,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        vw1_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.vw1_barriers.data_ptr(),
            num_stages=_VW_PRIVATE_STAGES,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                _PRODUCER_SIGNAL_WARPS,
            ),
            tx_count=_VW_TRANSACTION_BYTES,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        raw_handoff = pipeline.PipelineAsync.create(
            barrier_storage=storage.raw_handoff_barriers.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                _WARP_GROUP_SIZE,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                2 * _WARP_GROUP_SIZE,
            ),
        )
        factor_ready_handoff = pipeline.PipelineAsync.create(
            barrier_storage=storage.factor_ready_barriers.data_ptr(),
            num_stages=_INPUT_STAGES,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                2 * _WARP_GROUP_SIZE,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                _WARP_GROUP_SIZE,
            ),
        )
        factor_done_handoff = pipeline.PipelineAsync.create(
            barrier_storage=storage.factor_done_barriers.data_ptr(),
            num_stages=_INPUT_STAGES,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                _WARP_GROUP_SIZE,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                2 * _WARP_GROUP_SIZE,
            ),
        )
        output_handoff = pipeline.PipelineAsync.create(
            barrier_storage=storage.output_handoff_barriers.data_ptr(),
            num_stages=_OUTPUT_STAGES,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                _WARP_GROUP_SIZE if self.single_state_owner else 2 * _WARP_GROUP_SIZE,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                _WARP_GROUP_SIZE,
            ),
        )
        store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=_OUTPUT_STAGES,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                _WARP_GROUP_SIZE,
            ),
        )

        if warp_index == cutlass.Int32(0):
            cpasync.prefetch_descriptor(q_atom)
            cpasync.prefetch_descriptor(k_atom)
            cpasync.prefetch_descriptor(b_atom)
            cpasync.prefetch_descriptor(g_atom)
            cpasync.prefetch_descriptor(v_atom)
            cpasync.prefetch_descriptor(w_atom)
            cpasync.prefetch_descriptor(output_atom)

        if warp_group == cutlass.Int32(0):
            cute.arch.warpgroup_reg_dealloc(
                _PRODUCER_REGISTER_TARGET,
            )
            producer_work_index, _, _ = cute.arch.block_idx()
            producer_sequence_rank = producer_work_index // sequence_stride
            producer_value_work = producer_work_index - producer_sequence_rank * sequence_stride
            producer_sequence = sequence
            if cutlass.const_expr(
                not self.length_ranked_sequence_order and self.sequence_wave_rotation > 0,
            ):
                producer_sequence = producer_sequence_rank
                if num_sequences > cutlass.Int32(
                    self.sequence_wave_rotation,
                ):
                    producer_sequence = producer_sequence + cutlass.Int32(
                        self.sequence_wave_rotation,
                    )
                    if producer_sequence >= num_sequences:
                        producer_sequence = producer_sequence - num_sequences
            if thread_in_group % cutlass.Int32(32) == cutlass.Int32(0):
                producer_value_work_by_warp[warp_index] = producer_value_work
            cute.arch.sync_warp()
            producer_sequence_start = cutlass.Int32(
                cu_seqlens[producer_sequence],
            )
            producer_sequence_end = cutlass.Int32(
                cu_seqlens[producer_sequence + cutlass.Int32(1)],
            )
            producer_sequence_chunks = (
                producer_sequence_end - producer_sequence_start + cutlass.Int32(CHUNK_SIZE - 1)
            ) // cutlass.Int32(
                CHUNK_SIZE,
            )
            output_wait = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                _OUTPUT_STAGES,
            )
            output_release = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                _OUTPUT_STAGES,
            )
            for pipeline_step in cutlass.range(
                producer_sequence_chunks + cutlass.Int32(1),
                unroll=1,
            ):
                factor_stage = cutlass.Int32(0)
                factor_valid_tokens = cutlass.Int32(0)
                if pipeline_step < producer_sequence_chunks:
                    qkb0_producer = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer,
                        1,
                    )
                    qkb1_producer = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer,
                        1,
                    )
                    local_chunk = pipeline_step
                    chunk_start = producer_sequence_start + local_chunk * cutlass.Int32(CHUNK_SIZE)
                    valid_tokens = cutlass.Int32(CHUNK_SIZE)
                    if local_chunk + cutlass.Int32(1) == producer_sequence_chunks:
                        valid_tokens = producer_sequence_end - chunk_start
                    factor_stage = pipeline_step % cutlass.Int32(_INPUT_STAGES)
                    factor_valid_tokens = valid_tokens
                    raw_handoff.producer_acquire(
                        pipeline.PipelineState(
                            1,
                            pipeline_step,
                            cutlass.Int32(0),
                            cutlass.Int32(1) - pipeline_step % cutlass.Int32(2),
                        ),
                    )

                    q_use = cute.domain_offset(
                        (chunk_start, cutlass.Int32(0), cutlass.Int32(0)),
                        q_tma,
                    )
                    k_use = cute.domain_offset(
                        (chunk_start, cutlass.Int32(0), cutlass.Int32(0)),
                        k_tma,
                    )
                    b_use = cute.domain_offset(
                        (chunk_start, cutlass.Int32(0), cutlass.Int32(0)),
                        b_tma,
                    )
                    g_use = cute.domain_offset(
                        (chunk_start, cutlass.Int32(0), cutlass.Int32(0)),
                        g_tma,
                    )
                    q_tiles = cute.local_tile(
                        q_use[None, None, q_head],
                        (CHUNK_SIZE, _WGMMA_K),
                        (None, None),
                    )
                    k_tiles = cute.local_tile(
                        k_use[None, None, q_head],
                        (CHUNK_SIZE, _WGMMA_K),
                        (None, None),
                    )
                    b_tiles = cute.local_tile(
                        b_use[None, None, q_head],
                        (CHUNK_SIZE, _WGMMA_K),
                        (None, None),
                    )
                    g_tiles = cute.local_tile(
                        g_use[None, None, q_head],
                        (CHUNK_SIZE, _WGMMA_K),
                        (None, None),
                    )
                    q_smem, q_gmem = cpasync.tma_partition(
                        q_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(raw_q, 0, 2),
                        cute.group_modes(q_tiles, 0, 2),
                    )
                    k_smem, k_gmem = cpasync.tma_partition(
                        k_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(raw_k, 0, 2),
                        cute.group_modes(k_tiles, 0, 2),
                    )
                    b_smem, b_gmem = cpasync.tma_partition(
                        b_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(raw_b, 0, 2),
                        cute.group_modes(b_tiles, 0, 2),
                    )
                    g_smem, g_gmem = cpasync.tma_partition(
                        g_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(raw_g, 0, 2),
                        cute.group_modes(g_tiles, 0, 2),
                    )

                    if warp_index == cutlass.Int32(0):
                        qkb0_pipeline.producer_acquire(qkb0_producer)
                        qkb0_barrier = qkb0_pipeline.producer_get_barrier(
                            qkb0_producer,
                        )
                        cute.copy(
                            q_atom,
                            q_gmem[(None, 0, cutlass.Int32(0))],
                            q_smem[(None, cutlass.Int32(0))],
                            tma_bar_ptr=qkb0_barrier,
                        )
                        cute.copy(
                            k_atom,
                            k_gmem[(None, 0, cutlass.Int32(0))],
                            k_smem[(None, cutlass.Int32(0))],
                            tma_bar_ptr=qkb0_barrier,
                        )
                        cute.copy(
                            b_atom,
                            b_gmem[(None, 0, cutlass.Int32(0))],
                            b_smem[(None, cutlass.Int32(0))],
                            tma_bar_ptr=qkb0_barrier,
                        )
                        cute.copy(
                            g_atom,
                            g_gmem[(None, 0, cutlass.Int32(0))],
                            g_smem[(None, cutlass.Int32(0))],
                            tma_bar_ptr=qkb0_barrier,
                        )
                        qkb0_pipeline.producer_commit(qkb0_producer)
                        qkb0_producer.advance()

                        qkb1_pipeline.producer_acquire(qkb1_producer)
                        qkb1_barrier = qkb1_pipeline.producer_get_barrier(
                            qkb1_producer,
                        )
                        cute.copy(
                            q_atom,
                            q_gmem[
                                (
                                    None,
                                    0,
                                    cutlass.Int32(_QKB_STREAM_TILES),
                                )
                            ],
                            q_smem[(None, cutlass.Int32(1))],
                            tma_bar_ptr=qkb1_barrier,
                        )
                        cute.copy(
                            k_atom,
                            k_gmem[
                                (
                                    None,
                                    0,
                                    cutlass.Int32(_QKB_STREAM_TILES),
                                )
                            ],
                            k_smem[(None, cutlass.Int32(1))],
                            tma_bar_ptr=qkb1_barrier,
                        )
                        cute.copy(
                            b_atom,
                            b_gmem[
                                (
                                    None,
                                    0,
                                    cutlass.Int32(_QKB_STREAM_TILES),
                                )
                            ],
                            b_smem[(None, cutlass.Int32(1))],
                            tma_bar_ptr=qkb1_barrier,
                        )
                        cute.copy(
                            g_atom,
                            g_gmem[
                                (
                                    None,
                                    0,
                                    cutlass.Int32(_QKB_STREAM_TILES),
                                )
                            ],
                            g_smem[(None, cutlass.Int32(1))],
                            tma_bar_ptr=qkb1_barrier,
                        )
                        qkb1_pipeline.producer_commit(qkb1_producer)
                        qkb1_producer.advance()

                    # State may begin consuming Q/K/B/G after each private
                    # stream has its first tile in flight. V/W is deliberately
                    # issued only after this generation's factor is complete,
                    # so its single shared write buffer cannot obstruct the
                    # next factor generation.
                    raw_handoff.producer_commit(
                        pipeline.PipelineState(
                            1,
                            pipeline_step,
                            cutlass.Int32(0),
                            cutlass.Int32(1) - pipeline_step % cutlass.Int32(2),
                        ),
                    )

                    for local_key_tile in cutlass.range(
                        1,
                        _QKB_STREAM_TILES,
                        unroll=1,
                    ):
                        if warp_index == cutlass.Int32(0):
                            qkb0_pipeline.producer_acquire(qkb0_producer)
                            qkb0_barrier = qkb0_pipeline.producer_get_barrier(
                                qkb0_producer,
                            )
                            cute.copy(
                                q_atom,
                                q_gmem[(None, 0, local_key_tile)],
                                q_smem[(None, cutlass.Int32(0))],
                                tma_bar_ptr=qkb0_barrier,
                            )
                            cute.copy(
                                k_atom,
                                k_gmem[(None, 0, local_key_tile)],
                                k_smem[(None, cutlass.Int32(0))],
                                tma_bar_ptr=qkb0_barrier,
                            )
                            cute.copy(
                                b_atom,
                                b_gmem[(None, 0, local_key_tile)],
                                b_smem[(None, cutlass.Int32(0))],
                                tma_bar_ptr=qkb0_barrier,
                            )
                            cute.copy(
                                g_atom,
                                g_gmem[(None, 0, local_key_tile)],
                                g_smem[(None, cutlass.Int32(0))],
                                tma_bar_ptr=qkb0_barrier,
                            )
                            qkb0_pipeline.producer_commit(qkb0_producer)
                            qkb0_producer.advance()

                            high_key_tile = local_key_tile + cutlass.Int32(_QKB_STREAM_TILES)
                            qkb1_pipeline.producer_acquire(qkb1_producer)
                            qkb1_barrier = qkb1_pipeline.producer_get_barrier(
                                qkb1_producer,
                            )
                            cute.copy(
                                q_atom,
                                q_gmem[(None, 0, high_key_tile)],
                                q_smem[(None, cutlass.Int32(1))],
                                tma_bar_ptr=qkb1_barrier,
                            )
                            cute.copy(
                                k_atom,
                                k_gmem[(None, 0, high_key_tile)],
                                k_smem[(None, cutlass.Int32(1))],
                                tma_bar_ptr=qkb1_barrier,
                            )
                            cute.copy(
                                b_atom,
                                b_gmem[(None, 0, high_key_tile)],
                                b_smem[(None, cutlass.Int32(1))],
                                tma_bar_ptr=qkb1_barrier,
                            )
                            cute.copy(
                                g_atom,
                                g_gmem[(None, 0, high_key_tile)],
                                g_smem[(None, cutlass.Int32(1))],
                                tma_bar_ptr=qkb1_barrier,
                            )
                            qkb1_pipeline.producer_commit(qkb1_producer)
                            qkb1_producer.advance()

                if pipeline_step > cutlass.Int32(0):
                    # Produce next-generation QKB above before draining the
                    # current V/W generation. This matches State's
                    # prepare-next-before-materialize-current order and
                    # removes the single-stage V/W/QKB circular wait in r22.
                    vw_chunk = pipeline_step - cutlass.Int32(1)
                    vw0_producer = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer,
                        _VW_PRIVATE_STAGES,
                    )
                    vw1_producer = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer,
                        _VW_PRIVATE_STAGES,
                    )
                    vw_chunk_start = producer_sequence_start + vw_chunk * cutlass.Int32(CHUNK_SIZE)
                    v_use = cute.domain_offset(
                        (
                            vw_chunk_start,
                            value_start,
                            cutlass.Int32(0),
                        ),
                        v_tma,
                    )
                    w_use = cute.domain_offset(
                        (
                            vw_chunk_start,
                            value_start,
                            cutlass.Int32(0),
                        ),
                        w_tma,
                    )
                    v_tiles = cute.local_tile(
                        v_use[None, None, value_head],
                        (CHUNK_SIZE, _WGMMA_K),
                        (None, None),
                    )
                    w_tiles = cute.local_tile(
                        w_use[None, None, value_head],
                        (CHUNK_SIZE, _WGMMA_K),
                        (None, None),
                    )
                    v_smem, v_gmem = cpasync.tma_partition(
                        v_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(raw_v, 0, 2),
                        cute.group_modes(v_tiles, 0, 2),
                    )
                    w_smem, w_gmem = cpasync.tma_partition(
                        w_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(raw_w, 0, 2),
                        cute.group_modes(w_tiles, 0, 2),
                    )
                    for local_value_tile in cutlass.range(
                        _VALUE_TILES // 2,
                        unroll=1,
                    ):
                        if warp_index == cutlass.Int32(0):
                            vw0_pipeline.producer_acquire(vw0_producer)
                            vw0_barrier = vw0_pipeline.producer_get_barrier(
                                vw0_producer,
                            )
                            cute.copy(
                                v_atom,
                                v_gmem[(None, 0, local_value_tile)],
                                v_smem[(None, 0)],
                                tma_bar_ptr=vw0_barrier,
                            )
                            cute.copy(
                                w_atom,
                                w_gmem[(None, 0, local_value_tile)],
                                w_smem[(None, 0)],
                                tma_bar_ptr=vw0_barrier,
                            )
                            vw0_pipeline.producer_commit(vw0_producer)
                            vw0_producer.advance()

                            if cutlass.const_expr(not self.single_state_owner):
                                high_value_tile = local_value_tile + cutlass.Int32(_VALUE_TILES // 2)
                                vw1_pipeline.producer_acquire(vw1_producer)
                                vw1_barrier = vw1_pipeline.producer_get_barrier(
                                    vw1_producer,
                                )
                                cute.copy(
                                    v_atom,
                                    v_gmem[(None, 0, high_value_tile)],
                                    v_smem[(None, 1)],
                                    tma_bar_ptr=vw1_barrier,
                                )
                                cute.copy(
                                    w_atom,
                                    w_gmem[(None, 0, high_value_tile)],
                                    w_smem[(None, 1)],
                                    tma_bar_ptr=vw1_barrier,
                                )
                                vw1_pipeline.producer_commit(vw1_producer)
                                vw1_producer.advance()

                if pipeline_step < producer_sequence_chunks:
                    factor_consumer_state = pipeline.PipelineState(
                        _INPUT_STAGES,
                        pipeline_step,
                        pipeline_step % cutlass.Int32(_INPUT_STAGES),
                        (pipeline_step // cutlass.Int32(_INPUT_STAGES)) % cutlass.Int32(2),
                    )
                    factor_producer_state = pipeline.PipelineState(
                        _INPUT_STAGES,
                        pipeline_step,
                        pipeline_step % cutlass.Int32(_INPUT_STAGES),
                        cutlass.Int32(1) - ((pipeline_step // cutlass.Int32(_INPUT_STAGES)) % cutlass.Int32(2)),
                    )
                    factor_ready_handoff.consumer_wait(
                        factor_consumer_state,
                    )
                    factor_done_handoff.producer_acquire(
                        factor_producer_state,
                    )

                    factor_q = shared_q[None, None, factor_stage]
                    factor_erase = shared_erase[None, None, factor_stage]
                    factor_k = shared_factor_k[
                        None,
                        None,
                        cutlass.Int32(0),
                    ]
                    factor_gs_delta = shared_gs_delta[
                        None,
                        None,
                        factor_stage,
                    ]
                    factor_gs_pair = shared_gs_pair[
                        None,
                        None,
                        factor_stage,
                    ]
                    factor_aqk = shared_aqk[None, None, factor_stage]
                    factor_akk = shared_akk[None, None, factor_stage]
                    factor_inverse = shared_inverse[
                        None,
                        None,
                        cutlass.Int32(0),
                    ]
                    fill_static_upper = pipeline_step < cutlass.Int32(
                        _INPUT_STAGES,
                    )
                    _publish_factor_blocks(
                        thread_in_group,
                        factor_q,
                        factor_erase,
                        factor_k,
                        factor_gs_delta,
                        factor_gs_pair,
                        factor_aqk,
                        factor_inverse,
                        factor_valid_tokens,
                        scale,
                        fill_static_upper,
                    )
                    if fill_static_upper:
                        # One-time zero publication of the akk upper blocks:
                        # the triangular convert below never rewrites them and
                        # the arena is private per stage.
                        for pair in cutlass.range_constexpr(
                            len(_FACTOR_UPPER_PAIRS),
                        ):
                            upper_query = _FACTOR_UPPER_PAIRS[pair][0]
                            upper_key = _FACTOR_UPPER_PAIRS[pair][1]
                            for linear in cutlass.range(
                                thread_in_group,
                                _FACTOR_BLOCK * _FACTOR_BLOCK,
                                _WARP_GROUP_SIZE,
                                unroll=1,
                            ):
                                local_row = linear // cutlass.Int32(_FACTOR_BLOCK)
                                local_column = linear % cutlass.Int32(_FACTOR_BLOCK)
                                factor_akk[
                                    upper_query * _FACTOR_BLOCK + local_row,
                                    upper_key * _FACTOR_BLOCK + local_column,
                                ] = cutlass.BFloat16(0.0)
                    cute.arch.barrier(
                        barrier_id=_INVERSE_BARRIER,
                        number_of_threads=_WARP_GROUP_SIZE,
                    )
                    CollectiveInverse().run(
                        factor_inverse,
                        _INVERSE_BARRIER,
                    )
                    cute.arch.barrier(
                        barrier_id=_INVERSE_BARRIER,
                        number_of_threads=_WARP_GROUP_SIZE,
                    )
                    for row_block in cutlass.range_constexpr(
                        _FACTOR_SUB_BLOCKS,
                    ):
                        # The inverse is unit-lower-triangular blockwise, so
                        # only the lower band through the diagonal block needs
                        # conversion; the upper zeros were published once.
                        band_columns = (row_block + 1) * _FACTOR_BLOCK
                        for linear in cutlass.range(
                            thread_in_group,
                            _FACTOR_BLOCK * band_columns,
                            _WARP_GROUP_SIZE,
                            unroll=1,
                        ):
                            row = row_block * _FACTOR_BLOCK + linear // band_columns
                            column = linear % band_columns
                            factor_akk[row, column] = cutlass.BFloat16(
                                factor_inverse[row, column],
                            )
                    # Last read of the inverse, which aliases the raw-G arena.
                    # Rendezvous before any warp can leave this iteration and
                    # issue the next chunk's raw-G TMA over it: raw_handoff for
                    # chunk c+1 is already released by the state warp groups at
                    # the end of prep(c), so nothing else holds warp 0 back.
                    cute.arch.barrier(
                        barrier_id=_INVERSE_BARRIER,
                        number_of_threads=_WARP_GROUP_SIZE,
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    factor_done_handoff.producer_commit(
                        factor_producer_state,
                    )
                    factor_ready_handoff.consumer_release(
                        factor_consumer_state,
                    )
                if pipeline_step > cutlass.Int32(0):
                    local_chunk = pipeline_step - cutlass.Int32(1)
                    chunk_start = producer_sequence_start + local_chunk * cutlass.Int32(CHUNK_SIZE)
                    valid_tokens = cutlass.Int32(CHUNK_SIZE)
                    if local_chunk + cutlass.Int32(1) == producer_sequence_chunks:
                        valid_tokens = producer_sequence_end - chunk_start

                    output_handoff.consumer_wait(output_wait)
                    if valid_tokens == cutlass.Int32(CHUNK_SIZE):
                        output_view = cute.domain_offset(
                            (
                                chunk_start,
                                value_start,
                                cutlass.Int32(0),
                            ),
                            output_tma,
                        )
                        output_tile = cute.zipped_divide(
                            output_view[None, None, value_head],
                            (CHUNK_SIZE, self.value_tile),
                        )[
                            (
                                (None, None),
                                (cutlass.Int32(0), cutlass.Int32(0)),
                            )
                        ]
                        output_stage = shared_output[
                            None,
                            None,
                            output_wait.index,
                        ]
                        output_smem, output_gmem = cpasync.tma_partition(
                            output_atom,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(output_stage, 0, 2),
                            cute.group_modes(output_tile, 0, 2),
                        )
                        if warp_index == cutlass.Int32(0):
                            cute.arch.fence_view_async_shared()
                            cute.copy(
                                output_atom,
                                output_smem,
                                output_gmem,
                            )
                            store_pipeline.producer_commit()
                            if local_chunk > cutlass.Int32(0):
                                store_pipeline.producer_acquire()
                        cute.arch.barrier(
                            barrier_id=_STORE_WG_BARRIER,
                            number_of_threads=_WARP_GROUP_SIZE,
                        )
                        if local_chunk > cutlass.Int32(0):
                            output_handoff.consumer_release(output_release)
                            output_release.advance()
                        if local_chunk + cutlass.Int32(1) == producer_sequence_chunks:
                            if warp_index == cutlass.Int32(0):
                                store_pipeline.producer_tail()
                            cute.arch.barrier(
                                barrier_id=_STORE_WG_BARRIER,
                                number_of_threads=_WARP_GROUP_SIZE,
                            )
                            output_handoff.consumer_release(output_release)
                            output_release.advance()
                    else:
                        if warp_index == cutlass.Int32(0):
                            store_pipeline.producer_tail()
                        cute.arch.barrier(
                            barrier_id=_STORE_WG_BARRIER,
                            number_of_threads=_WARP_GROUP_SIZE,
                        )
                        if local_chunk > cutlass.Int32(0):
                            output_handoff.consumer_release(output_release)
                            output_release.advance()
                        tail_value_work = producer_value_work_by_warp[warp_index]
                        tail_value_head = tail_value_work // value_tiles
                        tail_value_tile_index = tail_value_work - tail_value_head * value_tiles
                        tail_value_start = tail_value_tile_index * cutlass.Int32(self.value_tile)
                        for linear in cutlass.range(
                            thread_in_group,
                            valid_tokens * cutlass.Int32(self.value_tile),
                            _WARP_GROUP_SIZE,
                            unroll=1,
                        ):
                            local_token = linear // self.value_tile
                            value_index = linear % self.value_tile
                            output[
                                chunk_start + local_token,
                                tail_value_head,
                                tail_value_start + value_index,
                            ] = shared_output[
                                local_token,
                                value_index,
                                output_wait.index,
                            ]
                        cute.arch.barrier(
                            barrier_id=_STORE_WG_BARRIER,
                            number_of_threads=_WARP_GROUP_SIZE,
                        )
                        output_handoff.consumer_release(output_release)
                        output_release.advance()
                    output_wait.advance()

        else:
            cute.arch.warpgroup_reg_alloc(_STATE_REGISTER_TARGET)
            state_sequence_start = cutlass.Int32(cu_seqlens[sequence])
            state_sequence_end = cutlass.Int32(
                cu_seqlens[sequence + cutlass.Int32(1)],
            )
            state_sequence_chunks = (
                state_sequence_end - state_sequence_start + cutlass.Int32(CHUNK_SIZE - 1)
            ) // cutlass.Int32(
                CHUNK_SIZE,
            )
            state_slab = warp_group - cutlass.Int32(1)
            shared_value_start = state_slab * cutlass.Int32(self.state_value_tile)
            if cutlass.const_expr(self.single_state_owner):
                shared_value_start = cutlass.Int32(0)
            state_value_start = value_start + shared_value_start

            state_thread = state_mma.get_slice(thread_in_group)
            token_thread = token_mma.get_slice(thread_in_group)
            state_coordinates = state_thread.partition_C(
                cute.make_identity_tensor(
                    (self.state_value_tile, HEAD_SIZE),
                ),
            )
            token_coordinates = token_thread.partition_C(
                cute.make_identity_tensor(
                    (self.state_value_tile, CHUNK_SIZE),
                ),
            )
            state_accumulator = state_thread.make_fragment_C(
                state_thread.partition_shape_C(
                    (self.state_value_tile, HEAD_SIZE),
                ),
            )
            for element in cutlass.range_constexpr(
                cute.size(state_accumulator),
            ):
                value_index, key_index = state_coordinates[element]
                state_value = cutlass.Float32(0.0)
                if cutlass.const_expr(self.has_initial_state):
                    state_value = cutlass.Float32(
                        initial_state[
                            sequence,
                            value_head,
                            state_value_start + value_index,
                            key_index,
                        ],
                    )
                state_accumulator[element] = state_value

            output_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                _OUTPUT_STAGES,
            )
            for state_step in cutlass.range(
                state_sequence_chunks + cutlass.Int32(1),
                unroll=1,
            ):
                current_stage = cutlass.Int32(0)
                current_valid_tokens = cutlass.Int32(0)
                current_chunk = state_step - cutlass.Int32(1)
                factor_done_consumer_state = pipeline.PipelineState(
                    _INPUT_STAGES,
                    current_chunk,
                    current_chunk % cutlass.Int32(_INPUT_STAGES),
                    (current_chunk // cutlass.Int32(_INPUT_STAGES)) % cutlass.Int32(2),
                )
                if state_step > cutlass.Int32(0):
                    current_stage = current_chunk % cutlass.Int32(_INPUT_STAGES)
                    current_chunk_start = state_sequence_start + current_chunk * cutlass.Int32(CHUNK_SIZE)
                    current_valid_tokens = cutlass.Int32(CHUNK_SIZE)
                    if current_chunk + cutlass.Int32(1) == state_sequence_chunks:
                        current_valid_tokens = state_sequence_end - current_chunk_start

                    factor_done_handoff.consumer_wait(
                        factor_done_consumer_state,
                    )
                    factor_done_handoff.consumer_release(
                        factor_done_consumer_state,
                    )

                if state_step < state_sequence_chunks:
                    prepare_chunk = state_step
                    prepare_stage = prepare_chunk % cutlass.Int32(_INPUT_STAGES)
                    prepare_chunk_start = state_sequence_start + prepare_chunk * cutlass.Int32(CHUNK_SIZE)
                    prepare_valid_tokens = cutlass.Int32(CHUNK_SIZE)
                    if prepare_chunk + cutlass.Int32(1) == state_sequence_chunks:
                        prepare_valid_tokens = state_sequence_end - prepare_chunk_start

                    qkb_consumer = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer,
                        1,
                    )
                    raw_handoff.consumer_wait(
                        pipeline.PipelineState(
                            1,
                            prepare_chunk,
                            cutlass.Int32(0),
                            prepare_chunk % cutlass.Int32(2),
                        ),
                    )
                    factor_ready_handoff.producer_acquire(
                        pipeline.PipelineState(
                            _INPUT_STAGES,
                            prepare_chunk,
                            (prepare_chunk % cutlass.Int32(_INPUT_STAGES)),
                            cutlass.Int32(1) - ((prepare_chunk // cutlass.Int32(_INPUT_STAGES)) % cutlass.Int32(2)),
                        ),
                    )
                    for local_key_tile in cutlass.range(
                        _QKB_STREAM_TILES,
                        unroll=1,
                    ):
                        if state_slab == cutlass.Int32(0):
                            qkb_ready = qkb0_pipeline.consumer_try_wait(
                                qkb_consumer,
                            )
                            qkb0_pipeline.consumer_wait(
                                qkb_consumer,
                                qkb_ready,
                            )
                        else:
                            qkb_ready = qkb1_pipeline.consumer_try_wait(
                                qkb_consumer,
                            )
                            qkb1_pipeline.consumer_wait(
                                qkb_consumer,
                                qkb_ready,
                            )

                        raw_stage = state_slab
                        key_tile = state_slab * cutlass.Int32(_QKB_STREAM_TILES) + local_key_tile
                        if cutlass.const_expr(self.subgroup_prefix):
                            # Sixteen independent eight-lane subgroups cover
                            # one 16-channel raw-G tile. Each lane scans one
                            # consecutive eight-token segment, then a
                            # subgroup shuffle scan distributes the carry.
                            subgroup_lane = thread_in_group % cutlass.Int32(8)
                            tile_channel = thread_in_group // cutlass.Int32(8)
                            token_base = subgroup_lane * cutlass.Int32(8)
                            local_prefix = cute.make_rmem_tensor(
                                8,
                                cutlass.Float32,
                            )
                            segment_total = cutlass.Float32(0.0)
                            for local_index in cutlass.range_constexpr(8):
                                local_token = token_base + cutlass.Int32(local_index)
                                if local_token < prepare_valid_tokens:
                                    segment_total = segment_total + cutlass.Float32(
                                        raw_g[
                                            local_token,
                                            tile_channel,
                                            raw_stage,
                                        ],
                                    )
                                local_prefix[local_index] = segment_total

                            inclusive_segment_total = segment_total
                            for log_offset in cutlass.range_constexpr(3):
                                offset = 1 << log_offset
                                prior = cute.arch.shuffle_sync_up(
                                    inclusive_segment_total,
                                    offset,
                                    mask_and_clamp=0,
                                )
                                if subgroup_lane >= cutlass.Int32(offset):
                                    inclusive_segment_total = inclusive_segment_total + prior
                            if cutlass.const_expr(
                                self.subgroup_exclusive_carry,
                            ):
                                carry = inclusive_segment_total - segment_total
                            else:
                                carry = cutlass.Float32(0.0)
                                prior_segment_total = cute.arch.shuffle_sync_up(
                                    inclusive_segment_total,
                                    1,
                                    mask_and_clamp=0,
                                )
                                if subgroup_lane > cutlass.Int32(0):
                                    carry = prior_segment_total

                            for local_index in cutlass.range_constexpr(8):
                                local_token = token_base + cutlass.Int32(local_index)
                                if local_token < prepare_valid_tokens:
                                    raw_g[
                                        local_token,
                                        tile_channel,
                                        raw_stage,
                                    ] = local_prefix[local_index] + carry
                                else:
                                    raw_g[
                                        local_token,
                                        tile_channel,
                                        raw_stage,
                                    ] = cutlass.Float32(0.0)
                        elif thread_in_group < cutlass.Int32(_WGMMA_K):
                            prefix = cutlass.Float32(0.0)
                            for local_token in cutlass.range_constexpr(
                                CHUNK_SIZE,
                            ):
                                if cutlass.Int32(local_token) < prepare_valid_tokens:
                                    prefix = prefix + cutlass.Float32(
                                        raw_g[
                                            local_token,
                                            thread_in_group,
                                            raw_stage,
                                        ],
                                    )
                                    raw_g[
                                        local_token,
                                        thread_in_group,
                                        raw_stage,
                                    ] = prefix
                                else:
                                    raw_g[
                                        local_token,
                                        thread_in_group,
                                        raw_stage,
                                    ] = cutlass.Float32(0.0)
                        if state_slab == cutlass.Int32(0):
                            cute.arch.barrier(
                                barrier_id=_STATE0_PREFIX_BARRIER,
                                number_of_threads=_WARP_GROUP_SIZE,
                            )
                        else:
                            cute.arch.barrier(
                                barrier_id=_STATE1_PREFIX_BARRIER,
                                number_of_threads=_WARP_GROUP_SIZE,
                            )
                        # Block-boundary decay ratios for this 16-channel
                        # tile: delta[0] = exp(Gs(0)) and
                        # delta[m] = exp(Gs(m) - Gs(m-1)), all <= 1. Fully
                        # invalid tail blocks store 1 so downstream running
                        # products stay exact no-ops.
                        if thread_in_group < cutlass.Int32(_WGMMA_K * _FACTOR_SUB_BLOCKS):
                            tile_channel = thread_in_group % cutlass.Int32(_WGMMA_K)
                            sub_block = thread_in_group // cutlass.Int32(_WGMMA_K)
                            key_channel = key_tile * cutlass.Int32(_WGMMA_K) + tile_channel
                            block_start = sub_block * cutlass.Int32(_FACTOR_BLOCK)
                            delta_value = cutlass.Float32(1.0)
                            if block_start < prepare_valid_tokens:
                                block_start_g = cutlass.Float32(
                                    raw_g[
                                        block_start,
                                        tile_channel,
                                        raw_stage,
                                    ],
                                )
                                previous_g = cutlass.Float32(0.0)
                                if sub_block > cutlass.Int32(0):
                                    previous_g = cutlass.Float32(
                                        raw_g[
                                            block_start - cutlass.Int32(_FACTOR_BLOCK),
                                            tile_channel,
                                            raw_stage,
                                        ],
                                    )
                                delta_value = cute.math.exp2(
                                    (block_start_g - previous_g) * cutlass.Float32(_INV_LN2),
                                    fastmath=True,
                                )
                            shared_gs_delta[
                                key_channel,
                                sub_block,
                                prepare_stage,
                            ] = delta_value
                            if sub_block == cutlass.Int32(0):
                                # Precompute the distance-two and distance-three
                                # pair products with the same left-to-right
                                # multiply order the factor fold used inline, so
                                # the published values stay bitwise identical.
                                previous_g = cutlass.Float32(
                                    raw_g[
                                        cutlass.Int32(0),
                                        tile_channel,
                                        raw_stage,
                                    ],
                                )
                                block_deltas = cute.make_rmem_tensor(
                                    _FACTOR_SUB_BLOCKS - 1,
                                    cutlass.Float32,
                                )
                                for later_block in cutlass.range_constexpr(
                                    1,
                                    _FACTOR_SUB_BLOCKS,
                                ):
                                    later_start = cutlass.Int32(
                                        later_block * _FACTOR_BLOCK,
                                    )
                                    later_delta = cutlass.Float32(1.0)
                                    if later_start < prepare_valid_tokens:
                                        later_g = cutlass.Float32(
                                            raw_g[
                                                later_start,
                                                tile_channel,
                                                raw_stage,
                                            ],
                                        )
                                        later_delta = cute.math.exp2(
                                            (later_g - previous_g) * cutlass.Float32(_INV_LN2),
                                            fastmath=True,
                                        )
                                        previous_g = later_g
                                    block_deltas[later_block - 1] = later_delta
                                product_two_low = block_deltas[0] * block_deltas[1]
                                product_two_high = block_deltas[1] * block_deltas[2]
                                product_three = product_two_low * block_deltas[2]
                                shared_gs_pair[
                                    key_channel,
                                    cutlass.Int32(0),
                                    prepare_stage,
                                ] = cutlass.Float16(product_two_low)
                                shared_gs_pair[
                                    key_channel,
                                    cutlass.Int32(1),
                                    prepare_stage,
                                ] = cutlass.Float16(product_two_high)
                                shared_gs_pair[
                                    key_channel,
                                    cutlass.Int32(2),
                                    prepare_stage,
                                ] = cutlass.Float16(product_three)
                        # Unroll two token rows so the LDS -> FMUL -> EX2 ->
                        # convert chains of neighbouring iterations overlap;
                        # this prep latency sits on the steady-state critical
                        # ring ahead of every factor generation.
                        for linear in cutlass.range(
                            thread_in_group,
                            CHUNK_SIZE * _WGMMA_K,
                            _WARP_GROUP_SIZE,
                            unroll=2,
                        ):
                            local_token = linear // _WGMMA_K
                            tile_channel = linear % _WGMMA_K
                            key_channel = key_tile * cutlass.Int32(_WGMMA_K) + tile_channel
                            q_value = cutlass.BFloat16(0.0)
                            erase_value = cutlass.BFloat16(0.0)
                            if local_token < prepare_valid_tokens:
                                g_value = cutlass.Float32(
                                    raw_g[
                                        local_token,
                                        tile_channel,
                                        raw_stage,
                                    ],
                                )
                                block_start_g = cutlass.Float32(
                                    raw_g[
                                        (local_token // cutlass.Int32(_FACTOR_BLOCK)) * cutlass.Int32(_FACTOR_BLOCK),
                                        tile_channel,
                                        raw_stage,
                                    ],
                                )
                                gamma = cute.math.exp2(
                                    (g_value - block_start_g) * cutlass.Float32(_INV_LN2),
                                    fastmath=True,
                                )
                                raw_k_value = cutlass.Float32(
                                    raw_k[
                                        local_token,
                                        tile_channel,
                                        raw_stage,
                                    ],
                                )
                                q_value = cutlass.BFloat16(
                                    cutlass.Float32(
                                        raw_q[
                                            local_token,
                                            tile_channel,
                                            raw_stage,
                                        ],
                                    )
                                    * gamma,
                                )
                                erase_value = cutlass.BFloat16(
                                    cutlass.Float32(
                                        raw_b[
                                            local_token,
                                            tile_channel,
                                            raw_stage,
                                        ],
                                    )
                                    * raw_k_value
                                    * gamma,
                                )
                            shared_q[
                                local_token,
                                key_channel,
                                prepare_stage,
                            ] = q_value
                            shared_erase[
                                local_token,
                                key_channel,
                                prepare_stage,
                            ] = erase_value

                        if cutlass.const_expr(self.retain_final_tail) and (
                            local_key_tile + cutlass.Int32(1) == cutlass.Int32(_QKB_STREAM_TILES)
                        ):
                            retained_key_tail = cute.make_rmem_tensor(
                                8,
                                cutlass.BFloat16,
                            )
                            retained_gamma_end = cutlass.Float32(0.0)
                            for local_index in cutlass.range_constexpr(8):
                                linear = thread_in_group + cutlass.Int32(
                                    local_index * _WARP_GROUP_SIZE,
                                )
                                local_token = linear // _WGMMA_K
                                tile_channel = linear % _WGMMA_K
                                key_channel = key_tile * cutlass.Int32(_WGMMA_K) + tile_channel
                                key_value = cutlass.BFloat16(0.0)
                                factor_key_value = cutlass.BFloat16(0.0)
                                if local_token < prepare_valid_tokens:
                                    g_value = cutlass.Float32(
                                        raw_g[
                                            local_token,
                                            tile_channel,
                                            raw_stage,
                                        ],
                                    )
                                    g_end = cutlass.Float32(
                                        raw_g[
                                            prepare_valid_tokens - cutlass.Int32(1),
                                            tile_channel,
                                            raw_stage,
                                        ],
                                    )
                                    tail_gamma = cute.math.exp2(
                                        (g_end - g_value) * cutlass.Float32(_INV_LN2),
                                        fastmath=True,
                                    )
                                    block_start_g = cutlass.Float32(
                                        raw_g[
                                            (local_token // cutlass.Int32(_FACTOR_BLOCK)) * cutlass.Int32(_FACTOR_BLOCK),
                                            tile_channel,
                                            raw_stage,
                                        ],
                                    )
                                    key_value = cutlass.BFloat16(
                                        cutlass.Float32(
                                            raw_k[
                                                local_token,
                                                tile_channel,
                                                raw_stage,
                                            ],
                                        )
                                        * tail_gamma,
                                    )
                                    factor_key_value = cutlass.BFloat16(
                                        cutlass.Float32(
                                            raw_k[
                                                local_token,
                                                tile_channel,
                                                raw_stage,
                                            ],
                                        )
                                        * cute.math.exp2(
                                            (block_start_g - g_value) * cutlass.Float32(_INV_LN2),
                                            fastmath=True,
                                        ),
                                    )
                                retained_key_tail[local_index] = key_value
                                shared_factor_k[
                                    local_token,
                                    key_channel,
                                    cutlass.Int32(0),
                                ] = factor_key_value
                                if local_token == cutlass.Int32(0):
                                    last_block_start = (
                                        (prepare_valid_tokens - cutlass.Int32(1)) // cutlass.Int32(_FACTOR_BLOCK)
                                    ) * cutlass.Int32(_FACTOR_BLOCK)
                                    retained_gamma_end = cute.math.exp2(
                                        (
                                            cutlass.Float32(
                                                raw_g[
                                                    prepare_valid_tokens - cutlass.Int32(1),
                                                    tile_channel,
                                                    raw_stage,
                                                ],
                                            )
                                            - cutlass.Float32(
                                                raw_g[
                                                    last_block_start,
                                                    tile_channel,
                                                    raw_stage,
                                                ],
                                            )
                                        )
                                        * cutlass.Float32(_INV_LN2),
                                        fastmath=True,
                                    )

                            cute.arch.fence_proxy(
                                "async.shared",
                                space="cta",
                            )
                            factor_ready_handoff.producer_commit(
                                pipeline.PipelineState(
                                    _INPUT_STAGES,
                                    prepare_chunk,
                                    (prepare_chunk % cutlass.Int32(_INPUT_STAGES)),
                                    cutlass.Int32(1) - ((prepare_chunk // cutlass.Int32(_INPUT_STAGES)) % cutlass.Int32(2)),
                                ),
                            )

                            for local_index in cutlass.range_constexpr(8):
                                linear = thread_in_group + cutlass.Int32(
                                    local_index * _WARP_GROUP_SIZE,
                                )
                                local_token = linear // _WGMMA_K
                                tile_channel = linear % _WGMMA_K
                                key_channel = key_tile * cutlass.Int32(_WGMMA_K) + tile_channel
                                shared_key_tail[
                                    key_channel,
                                    local_token,
                                    prepare_stage,
                                ] = retained_key_tail[local_index]
                            if thread_in_group < cutlass.Int32(_WGMMA_K):
                                key_channel = key_tile * cutlass.Int32(_WGMMA_K) + thread_in_group
                                shared_gamma_end[
                                    key_channel,
                                    prepare_stage,
                                ] = retained_gamma_end
                        else:
                            # Same two-row unroll as the q~/e~ loop above: the
                            # k~ rebase chain is equally EX2-latency-bound.
                            for linear in cutlass.range(
                                thread_in_group,
                                CHUNK_SIZE * _WGMMA_K,
                                _WARP_GROUP_SIZE,
                                unroll=2,
                            ):
                                local_token = linear // _WGMMA_K
                                tile_channel = linear % _WGMMA_K
                                key_channel = key_tile * cutlass.Int32(_WGMMA_K) + tile_channel
                                key_value = cutlass.BFloat16(0.0)
                                factor_key_value = cutlass.BFloat16(0.0)
                                if local_token < prepare_valid_tokens:
                                    g_value = cutlass.Float32(
                                        raw_g[
                                            local_token,
                                            tile_channel,
                                            raw_stage,
                                        ],
                                    )
                                    g_end = cutlass.Float32(
                                        raw_g[
                                            prepare_valid_tokens - cutlass.Int32(1),
                                            tile_channel,
                                            raw_stage,
                                        ],
                                    )
                                    tail_gamma = cute.math.exp2(
                                        (g_end - g_value) * cutlass.Float32(_INV_LN2),
                                        fastmath=True,
                                    )
                                    block_start_g = cutlass.Float32(
                                        raw_g[
                                            (local_token // cutlass.Int32(_FACTOR_BLOCK)) * cutlass.Int32(_FACTOR_BLOCK),
                                            tile_channel,
                                            raw_stage,
                                        ],
                                    )
                                    key_value = cutlass.BFloat16(
                                        cutlass.Float32(
                                            raw_k[
                                                local_token,
                                                tile_channel,
                                                raw_stage,
                                            ],
                                        )
                                        * tail_gamma,
                                    )
                                    factor_key_value = cutlass.BFloat16(
                                        cutlass.Float32(
                                            raw_k[
                                                local_token,
                                                tile_channel,
                                                raw_stage,
                                            ],
                                        )
                                        * cute.math.exp2(
                                            (block_start_g - g_value) * cutlass.Float32(_INV_LN2),
                                            fastmath=True,
                                        ),
                                    )
                                shared_key_tail[
                                    key_channel,
                                    local_token,
                                    prepare_stage,
                                ] = key_value
                                shared_factor_k[
                                    local_token,
                                    key_channel,
                                    cutlass.Int32(0),
                                ] = factor_key_value
                                if local_token == cutlass.Int32(0):
                                    last_block_start = (
                                        (prepare_valid_tokens - cutlass.Int32(1)) // cutlass.Int32(_FACTOR_BLOCK)
                                    ) * cutlass.Int32(_FACTOR_BLOCK)
                                    shared_gamma_end[
                                        key_channel,
                                        prepare_stage,
                                    ] = cute.math.exp2(
                                        (
                                            cutlass.Float32(
                                                raw_g[
                                                    prepare_valid_tokens - cutlass.Int32(1),
                                                    tile_channel,
                                                    raw_stage,
                                                ],
                                            )
                                            - cutlass.Float32(
                                                raw_g[
                                                    last_block_start,
                                                    tile_channel,
                                                    raw_stage,
                                                ],
                                            )
                                        )
                                        * cutlass.Float32(_INV_LN2),
                                        fastmath=True,
                                    )
                        if state_slab == cutlass.Int32(0):
                            qkb0_pipeline.consumer_release(qkb_consumer)
                        else:
                            qkb1_pipeline.consumer_release(qkb_consumer)
                        qkb_consumer.advance()

                    raw_handoff.consumer_release(
                        pipeline.PipelineState(
                            1,
                            prepare_chunk,
                            cutlass.Int32(0),
                            prepare_chunk % cutlass.Int32(2),
                        ),
                    )

                    # Both State WGs publish disjoint halves of the common
                    # factor operands. The 256-producer factor-ready mbarrier
                    # is the only rendezvous: each producer fences its
                    # preceding stores before the release arrive, and Factor
                    # WG0 returns from the acquire wait only after all 256
                    # arrivals.
                    if cutlass.const_expr(
                        not self.elide_state_common_barrier,
                    ):
                        cute.arch.barrier(
                            barrier_id=_STATE_COMMON_BARRIER,
                            number_of_threads=2 * _WARP_GROUP_SIZE,
                        )
                    if cutlass.const_expr(not self.retain_final_tail):
                        cute.arch.fence_proxy("async.shared", space="cta")
                        factor_ready_handoff.producer_commit(
                            pipeline.PipelineState(
                                _INPUT_STAGES,
                                prepare_chunk,
                                (prepare_chunk % cutlass.Int32(_INPUT_STAGES)),
                                cutlass.Int32(1) - ((prepare_chunk // cutlass.Int32(_INPUT_STAGES)) % cutlass.Int32(2)),
                            ),
                        )

                if state_step > cutlass.Int32(0) and (
                    cutlass.const_expr(not self.single_state_owner) or state_slab == cutlass.Int32(0)
                ):
                    # Prioritize the next factor-ready publication above. V/W
                    # for the current generation was issued after its factor
                    # completed and can now materialize while WG0 starts the
                    # next factor.
                    vw_consumer = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer,
                        _VW_PRIVATE_STAGES,
                    )
                    for local_value_tile in cutlass.range(
                        _VALUE_TILES // 2,
                        unroll=1,
                    ):
                        if state_slab == cutlass.Int32(0):
                            vw_ready = vw0_pipeline.consumer_try_wait(
                                vw_consumer,
                            )
                            vw0_pipeline.consumer_wait(
                                vw_consumer,
                                vw_ready,
                            )
                        else:
                            vw_ready = vw1_pipeline.consumer_try_wait(
                                vw_consumer,
                            )
                            vw1_pipeline.consumer_wait(
                                vw_consumer,
                                vw_ready,
                            )

                        for linear in cutlass.range(
                            thread_in_group,
                            CHUNK_SIZE * _WGMMA_K,
                            _WARP_GROUP_SIZE,
                            unroll=1,
                        ):
                            local_token = linear // _WGMMA_K
                            tile_value = linear % _WGMMA_K
                            value_index = (
                                cutlass.Int32(
                                    local_value_tile * _WGMMA_K,
                                )
                                + tile_value
                            )
                            write_value = cutlass.BFloat16(0.0)
                            if local_token < current_valid_tokens:
                                write_value = cutlass.BFloat16(
                                    cutlass.Float32(
                                        raw_v[
                                            local_token,
                                            tile_value,
                                            state_slab,
                                        ],
                                    )
                                    * cutlass.Float32(
                                        raw_w[
                                            local_token,
                                            tile_value,
                                            state_slab,
                                        ],
                                    ),
                                )
                            shared_write[
                                local_token,
                                shared_value_start + value_index,
                                cutlass.Int32(0),
                            ] = write_value

                        if state_slab == cutlass.Int32(0):
                            vw0_pipeline.consumer_release(vw_consumer)
                        else:
                            vw1_pipeline.consumer_release(vw_consumer)
                        vw_consumer.advance()

                    if state_slab == cutlass.Int32(0):
                        cute.arch.barrier(
                            barrier_id=_STATE0_WRITE_BARRIER,
                            number_of_threads=_WARP_GROUP_SIZE,
                        )
                    else:
                        cute.arch.barrier(
                            barrier_id=_STATE1_WRITE_BARRIER,
                            number_of_threads=_WARP_GROUP_SIZE,
                        )

                    input_stage = current_stage

                    q_stage = shared_q[None, None, input_stage]
                    erase_stage = shared_erase[None, None, input_stage]
                    key_stage = shared_key_tail[None, None, input_stage]
                    aqk_stage = shared_aqk[None, None, input_stage]
                    akk_stage = shared_akk[None, None, input_stage]

                    q_token_blocks = cute.flat_divide(
                        q_stage,
                        (_FACTOR_BLOCK, HEAD_SIZE),
                    )
                    erase_token_blocks = cute.flat_divide(
                        erase_stage,
                        (_FACTOR_BLOCK, HEAD_SIZE),
                    )
                    aqk_operand = token_thread.make_fragment_B(
                        token_thread.partition_B(aqk_stage),
                    )
                    aqk_stages = aqk_operand
                    akk_operand = token_thread.make_fragment_B(
                        token_thread.partition_B(akk_stage),
                    )
                    akk_stages = akk_operand
                    key_operand = state_thread.make_fragment_B(
                        state_thread.partition_B(key_stage),
                    )
                    key_stages = key_operand

                    gs_delta_stage = shared_gs_delta[
                        None,
                        None,
                        input_stage,
                    ]

                    output_accumulator = token_thread.make_fragment_C(
                        token_thread.partition_shape_C(
                            (self.state_value_tile, CHUNK_SIZE),
                        ),
                    )
                    erase_projection = token_thread.make_fragment_C(
                        token_thread.partition_shape_C(
                            (self.state_value_tile, CHUNK_SIZE),
                        ),
                    )
                    # The projections consume blockwise-rebased q~/e~, so the
                    # state fragments advance by the bounded per-channel block
                    # delta before each 16-token slice. Each slice issues as
                    # its own WGMMA group against a private BF16 state copy —
                    # the FP32 state advances underneath without a hazard — so
                    # up to two slices stay in flight (depth-2 pipelining) and
                    # a completed slice is drained into the chunk accumulators
                    # while the next one runs. After the last slice the state
                    # carries exp(Gs(last)) and the end-of-chunk gamma
                    # completes the exact exp(G_end) recurrence scale.
                    slice_outputs = []
                    slice_erases = []
                    for token_block in cutlass.range_constexpr(
                        _FACTOR_SUB_BLOCKS,
                    ):
                        state_as_token_a = cute.make_rmem_tensor_like(
                            _convert_c_layout_to_a_layout(
                                state_accumulator.layout,
                                token_mma.tv_layout_A.shape[1],
                            ),
                            cutlass.BFloat16,
                        )
                        operand_view = cute.make_tensor(
                            state_as_token_a.iterator,
                            state_accumulator.layout,
                        )
                        for element in cutlass.range_constexpr(
                            cute.size(state_accumulator),
                        ):
                            _, key_index = state_coordinates[element]
                            advanced = state_accumulator[element] * cutlass.Float32(
                                gs_delta_stage[
                                    key_index,
                                    token_block,
                                ],
                            )
                            state_accumulator[element] = advanced
                            operand_view[element] = cutlass.BFloat16(advanced)
                        q_block_operand = token_thread.make_fragment_B(
                            token_thread.partition_B(
                                q_token_blocks[
                                    None,
                                    None,
                                    token_block,
                                    0,
                                ],
                            ),
                        )
                        erase_block_operand = token_thread.make_fragment_B(
                            token_thread.partition_B(
                                erase_token_blocks[
                                    None,
                                    None,
                                    token_block,
                                    0,
                                ],
                            ),
                        )
                        output_block = token_thread.make_fragment_C(
                            token_thread.partition_shape_C(
                                (self.state_value_tile, _FACTOR_BLOCK),
                            ),
                        )
                        erase_block = token_thread.make_fragment_C(
                            token_thread.partition_shape_C(
                                (self.state_value_tile, _FACTOR_BLOCK),
                            ),
                        )
                        slice_outputs.append(output_block)
                        slice_erases.append(erase_block)
                        _fence_register_fragment(state_as_token_a)
                        warpgroup.fence()
                        _wgmma_gemm(
                            token_mma,
                            output_block,
                            state_as_token_a,
                            q_block_operand,
                            False,
                        )
                        _wgmma_gemm(
                            token_mma,
                            erase_block,
                            state_as_token_a,
                            erase_block_operand,
                            False,
                        )
                        warpgroup.commit_group()
                        if token_block >= 1:
                            warpgroup.wait_group(1)
                            cute.autovec_copy(
                                slice_outputs[token_block - 1][(None, None, 0)],
                                output_accumulator[(None, None, token_block - 1)],
                            )
                            cute.autovec_copy(
                                slice_erases[token_block - 1][(None, None, 0)],
                                erase_projection[(None, None, token_block - 1)],
                            )
                    warpgroup.wait_group(0)
                    cute.autovec_copy(
                        slice_outputs[_FACTOR_SUB_BLOCKS - 1][(None, None, 0)],
                        output_accumulator[(None, None, _FACTOR_SUB_BLOCKS - 1)],
                    )
                    cute.autovec_copy(
                        slice_erases[_FACTOR_SUB_BLOCKS - 1][(None, None, 0)],
                        erase_projection[(None, None, _FACTOR_SUB_BLOCKS - 1)],
                    )
                    for element in cutlass.range_constexpr(
                        cute.size(output_accumulator),
                    ):
                        output_accumulator[element] = output_accumulator[element] * scale

                    for element in cutlass.range_constexpr(
                        cute.size(erase_projection),
                    ):
                        value_index, token_index = token_coordinates[element]
                        erase_projection[element] = (
                            cutlass.Float32(
                                shared_write[
                                    token_index,
                                    shared_value_start + value_index,
                                    cutlass.Int32(0),
                                ],
                            )
                            - erase_projection[element]
                        )

                    residual_a = _make_acc_into_op(
                        erase_projection,
                        token_mma,
                    )
                    value_new = token_thread.make_fragment_C(
                        token_thread.partition_shape_C(
                            (self.state_value_tile, CHUNK_SIZE),
                        ),
                    )
                    _fence_register_fragment(residual_a)
                    _fence_register_fragment(value_new)
                    warpgroup.fence()
                    _wgmma_gemm(
                        token_mma,
                        value_new,
                        residual_a,
                        akk_stages,
                        False,
                    )
                    warpgroup.commit_group()
                    warpgroup.wait_group(0)

                    value_new_a = _make_acc_into_op(
                        value_new,
                        token_mma,
                    )
                    _fence_register_fragment(value_new_a)
                    _fence_register_fragment(output_accumulator)
                    warpgroup.fence()
                    _wgmma_gemm(
                        token_mma,
                        output_accumulator,
                        value_new_a,
                        aqk_stages,
                        True,
                    )
                    warpgroup.commit_group()
                    warpgroup.wait_group(0)
                    output_handoff.producer_acquire(output_producer)
                    for element in cutlass.range_constexpr(
                        cute.size(output_accumulator),
                    ):
                        value_index, token_index = token_coordinates[element]
                        shared_output[
                            token_index,
                            shared_value_start + value_index,
                            output_producer.index,
                        ] = cutlass.BFloat16(output_accumulator[element])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    output_handoff.producer_commit(output_producer)
                    output_producer.advance()

                    for element in cutlass.range_constexpr(
                        cute.size(state_accumulator),
                    ):
                        _, key_index = state_coordinates[element]
                        state_accumulator[element] = state_accumulator[element] * shared_gamma_end[key_index, input_stage]

                    value_new_as_state_a = _make_acc_into_op(
                        value_new,
                        state_mma,
                    )
                    _fence_register_fragment(value_new_as_state_a)
                    _fence_register_fragment(state_accumulator)
                    warpgroup.fence()
                    _wgmma_gemm(
                        state_mma,
                        state_accumulator,
                        value_new_as_state_a,
                        key_stages,
                        True,
                    )
                    warpgroup.commit_group()
                    warpgroup.wait_group(0)
                    if cutlass.const_expr(
                        not self.elide_state_iteration_done_barrier,
                    ):
                        cute.arch.barrier(
                            barrier_id=_STATE_ITERATION_DONE_BARRIER,
                            number_of_threads=2 * _WARP_GROUP_SIZE,
                        )

                if state_step > cutlass.Int32(0):
                    if cutlass.const_expr(self.single_state_owner):
                        # WG2 remains a common-factor preparation helper in
                        # the V64/N=1 route.  It must not reuse gamma_end's
                        # two-stage arena until sole owner WG1 has consumed
                        # the current generation.
                        cute.arch.barrier(
                            barrier_id=_STATE_ITERATION_DONE_BARRIER,
                            number_of_threads=2 * _WARP_GROUP_SIZE,
                        )

            if cutlass.const_expr(self.store_final_state):
                if cutlass.const_expr(not self.single_state_owner) or state_slab == cutlass.Int32(0):
                    for element in cutlass.range_constexpr(
                        cute.size(state_accumulator),
                    ):
                        value_index, key_index = state_coordinates[element]
                        final_state[
                            sequence,
                            value_head,
                            state_value_start + value_index,
                            key_index,
                        ] = state_accumulator[element]
