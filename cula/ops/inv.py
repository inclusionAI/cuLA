# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Standalone CuTe kernel for 64x64 FP16 matrix inverse computation.

This kernel implements the block-wise Schur complement matrix inversion
for 64x64 lower triangular matrices using 4 progressive stages:
  Stage 1: Invert 8 diagonal 8x8 blocks
  Stage 2: Build 16x16 blocks from 8x8
  Stage 3: Build 32x32 blocks from 16x16
  Stage 4: Build full 64x64 inverse
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils


class MatrixInverse64x64:
    """
    64x64 FP16 lower triangular matrix inversion kernel.

    This kernel inverts a 64x64 lower triangular matrix using the
    block-wise Schur complement method. The matrix is divided into
    8x8 blocks and progressively inverted in 4 stages.

    Grid Configuration:
        - Grid size: 1 (single block per matrix)
        - Block size: 128 threads (4 warps × 32 lanes)
        - Shared memory: ~64 KB (64x64 FP16 matrix + synchronization overhead)
    """

    # Kernel configuration constants
    MATRIX_SIZE = 64
    MATRIX_DTYPE = cutlass.Float16  # Input/output data type
    THREADS_PER_CTA = 128  # 4 warps of 32 threads
    GRID_SIZE = 1  # Single CTA for entire matrix
    SMEM_ALIGN_BYTES = 1024

    def __init__(self, acc_dtype=cutlass.Float32, cuda_core_threads=128):
        """
        Initialize the matrix inverse kernel.

        Args:
            acc_dtype: Accumulator data type for intermediate computations (default: Float32)
            cuda_core_threads: Number of CUDA threads in the work-group (default: 128)
        """
        self.acc_dtype = acc_dtype
        self.cuda_core_threads = cuda_core_threads
        self.threads_per_cta = cuda_core_threads
        # Create a named barrier for synchronization across all threads
        self.cuda_wg_sync_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=cuda_core_threads,
        )

    @cute.jit
    def canonical_lane_id(self):
        """Get the canonical lane ID within the warp."""
        tidx, _, _ = cute.arch.thread_idx()
        lane_id = tidx % 32
        return lane_id

    @cute.jit
    def convert_layout_c_to_a(
        self,
        c_layout: cute.Layout,
        tiled_mma: cute.TiledMma,
    ):
        """Convert MMA accumulator layout to operand A layout."""
        cfrag_atom_size = cute.size(c_layout.shape[0])
        afrag_atom_size = cute.size(tiled_mma.tv_layout_A.shape[1])
        ratio = afrag_atom_size // cfrag_atom_size

        if cutlass.const_expr(ratio == 1):
            return c_layout

        divided = cute.logical_divide(c_layout, (None, None, ratio))
        a_layout = cute.make_layout(
            (cute.flatten((divided.shape[0], divided.shape[2][0])), divided.shape[1], divided.shape[2][1])
        )
        return a_layout

    @cute.jit
    def make_acc_as_a(self, acc: cute.Tensor, tiled_mma: cute.TiledMma, dtype: cute.Numeric):
        """Convert MMA accumulator to operand A format."""
        a_layout = self.convert_layout_c_to_a(acc.layout, tiled_mma)
        a_tensor = cute.make_rmem_tensor(a_layout, dtype=dtype)
        op_as_acc = cute.make_tensor(a_tensor.iterator, layout=acc.layout)
        op_as_acc.store(acc.load().to(dtype))
        return a_tensor

    @cute.jit
    def make_op_a_from_acc_rmem_16x8x8(
        self,
        acc_dtype: cute.Numeric,
        ab_dtype: cute.Numeric,
        acc_tensor: cute.Tensor,
    ):
        # For 16x8x8, ACC result 16x8 can directly serve as operand A
        a_tensor = cute.make_rmem_tensor_like(
            acc_tensor,
            dtype=ab_dtype,
        )
        a_tensor.store(acc_tensor.load().to(ab_dtype))
        return a_tensor

    @cute.jit
    def compute_diagonal_inverse_8x8(
        self,
        s_block: cute.Tensor,  # Input 8x8 block in smem
        lane_id: int,
        block_id: int = -1,  # Block ID for correct shuffle mask calculation
    ):
        """
        Compute inverse of a diagonal 8x8 lower triangular block.

        Each warp processes one 8x8 block. Each lane (0-7) computes one column.

        Args:
            s_block: 8x8 block in shared memory
            lane_id: local lane ID within the block (0-7)
            block_id: global block ID (0-7) for correct shuffle masking
        """
        # Every thread loads its row
        row = self.load_row_mat8x8(s_block, lane_id)

        # Calculate the actual physical starting position of this block in the warp
        # Blocks 0-3 are in Warp 0 (tidx 0-31), Blocks 4-7 are in Warp 1 (tidx 32-63)
        # Each block starts at: warp_base + block_index_in_warp * 8
        # where block_index_in_warp = block_id % 4
        actual_block_base = (block_id % 4) * 8  # 0, 8, 16, or 24 within the warp

        for src_row in cutlass.range_constexpr(8 - 1):
            row_scale = row[src_row] * cutlass.Float32(-1)
            for i in cutlass.range(src_row, unroll_full=True):
                # Correct target lane: within the block, we need threads at positions
                # actual_block_base + 0, actual_block_base + 1, ..., actual_block_base + 7
                # The src_row thread (0-7 within the block) maps to actual_block_base + src_row
                # in the warp
                target_lane = actual_block_base + src_row

                src_row_value = cute.arch.shuffle_sync_op(
                    value=row[i],
                    offset=target_lane,  # Use actual position, not just src_row
                    mask=0xFFFFFFFF,
                    mask_and_clamp=31,
                )
                if lane_id > src_row:
                    row[i] += row_scale * src_row_value
            if lane_id > src_row:
                row[src_row] = row_scale
        # Store row
        self.store_row_mat8x8(s_block, row, lane_id)

    @cute.jit
    def load_row_mat8x8(
        self,
        mat: cute.Tensor,
        idx: int,
    ) -> cute.Tensor:
        copy_atom_s2r_x = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mat.element_type,
            num_bits_per_copy=mat.element_type.width * 8,
        )
        row_tensor = cute.make_rmem_tensor(cute.make_layout(8), mat.element_type)
        cute.copy(copy_atom_s2r_x, mat[idx, None], row_tensor[None])
        row_f32_tensor = cute.make_rmem_tensor_like(row_tensor, cutlass.Float32)
        row_f32_tensor.store(row_tensor.load().to(cutlass.Float32))
        return row_f32_tensor

    @cute.jit
    def store_row_mat8x8(
        self,
        mat: cute.Tensor,
        row: cute.Tensor,
        idx: int,
    ):
        row_bf16 = cute.make_rmem_tensor_like(row, mat.element_type)
        row_bf16.store(row.load().to(mat.element_type))
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mat.element_type,
            num_bits_per_copy=mat.element_type.width * 8,
        )
        cute.copy(copy_atom, row_bf16[None], mat[idx, None])

    @cute.jit
    def compute_diagonal_inverse_8x8_to_16x16(
        self,
        mat: cute.Tensor,  # Input 16x16 block in smem
    ):
        """
        Compute inverse of a diagonal 16x16 lower triangular block using
        two 8x8 blocks and Schur complement.
        """
        dtype = mat.element_type
        mat8x8_2x2 = cute.flat_divide(mat, (8, 8))

        mma_atom_shape = (16, 8, 8)
        mma_atom = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=dtype,
            acc_dtype=self.acc_dtype,
            shape_mnk=mma_atom_shape,
        )
        tiled_mma = cute.make_tiled_mma(
            mma_atom,
            atom_layout_mnk=(1, 1, 1),
        )
        # TODO: check transpose for column major mat
        copy_op_s2r = cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=1)
        copy_op_s2r_t = cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=1)
        copy_op_r2s = cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=1)
        copy_atom_s2r = cute.make_copy_atom(copy_op_s2r, mat.element_type)
        copy_atom_s2r_t = cute.make_copy_atom(copy_op_s2r_t, mat.element_type)
        copy_atom_r2s = cute.make_copy_atom(copy_op_r2s, mat.element_type)

        tidx, _, _ = cute.arch.thread_idx()
        lane_id = tidx % 32

        thr_mma = tiled_mma.get_slice(lane_id)

        D_tiled_copy = cute.make_tiled_copy_A(copy_atom_s2r, tiled_mma)
        C_tiled_copy = cute.make_tiled_copy_B(copy_atom_s2r_t, tiled_mma)
        A_tiled_copy = cute.make_tiled_copy_B(copy_atom_s2r_t, tiled_mma)
        O_tiled_copy = cute.make_tiled_copy_C(copy_atom_r2s, tiled_mma)

        D_thr_copy = D_tiled_copy.get_slice(lane_id)
        C_thr_copy = C_tiled_copy.get_slice(lane_id)
        A_thr_copy = A_tiled_copy.get_slice(lane_id)
        O_thr_copy = O_tiled_copy.get_slice(lane_id)

        sDInv = mat8x8_2x2[None, None, 1, 1]
        sC = mat8x8_2x2[None, None, 1, 0]
        sAInv = mat8x8_2x2[None, None, 0, 0]
        sO = mat8x8_2x2[None, None, 1, 0]

        sC = cute.make_tensor(sC.iterator, layout=cute.select(sC.layout, mode=[1, 0]))
        sAInv = cute.make_tensor(sAInv.iterator, layout=cute.select(sAInv.layout, mode=[1, 0]))

        sDInv_bcast = cute.make_tensor(
            sDInv.iterator, cute.blocked_product(sDInv.layout, cute.make_layout((2, 1), stride=(0, 0)))
        )
        sO_bcast = cute.make_tensor(sO.iterator, cute.blocked_product(sO.layout, cute.make_layout((2, 1), stride=(0, 0))))

        a_shape = cute.dice(mma_atom_shape, (1, None, 1))
        c_shape = cute.dice(mma_atom_shape, (1, 1, None))
        tOrDInv = thr_mma.make_fragment_A(tiled_mma.partition_shape_A(a_shape))
        tOrC = thr_mma.make_fragment_B(thr_mma.partition_B(sC))
        tOrAInv = thr_mma.make_fragment_B(thr_mma.partition_B(sAInv))

        tDCrDC = thr_mma.make_fragment_C(tiled_mma.partition_shape_C(c_shape))
        tOrO = thr_mma.make_fragment_C(tiled_mma.partition_shape_C(c_shape))

        # ((vecsize, numvec), M, N)
        tOsDInv = D_thr_copy.partition_S(sDInv_bcast)
        tOrDInv_cv = D_thr_copy.retile(tOrDInv)
        tOsC = C_thr_copy.partition_S(sC)
        tOrC_cv = C_thr_copy.retile(tOrC)
        tOsAInv = A_thr_copy.partition_S(sAInv)
        tOrAInv_cv = A_thr_copy.retile(tOrAInv)
        tOsO = O_thr_copy.partition_D(sO_bcast)
        tOrO_cv = O_thr_copy.retile(tOrO)

        cute.copy(C_tiled_copy, tOsC, tOrC_cv)

        tDInv_src = tOsDInv
        tDInv_dst = tOrDInv_cv
        cute.copy(D_tiled_copy, tDInv_src, tDInv_dst)

        tDCrDC.fill(0.0)  # Clear C for D = A*B + C
        cute.gemm(tiled_mma, tDCrDC, tOrDInv, tOrC, tDCrDC)
        tDCrDC.store(tDCrDC.load() * cutlass.Float32(-1))

        tOrDC = self.make_op_a_from_acc_rmem_16x8x8(
            self.acc_dtype,
            mat.element_type,
            tDCrDC,
        )
        cute.copy(A_tiled_copy, tOsAInv, tOrAInv_cv)
        tOrO_cv.fill(0.0)  # Clear O for O = A*B + O
        cute.gemm(tiled_mma, tOrO_cv, tOrDC, tOrAInv, tOrO_cv)

        tOrO_cv_half = tOrO_cv[(None, 0), None, None]
        tOrO_f16 = cute.make_rmem_tensor_like(tOrO_cv_half, mat.element_type)
        tOrO_f16.store(tOrO_cv[(None, 0), None, None].load().to(mat.element_type))

        # NOTE: group here to make cutedsl happy
        src_shape = tOrO_f16.shape
        src_stride = tOrO_f16.layout.stride
        dst_shape = tOsO[(None, 0), None, None].shape
        dst_stride = tOsO[(None, 0), None, None].layout.stride
        tOrO_src = cute.make_tensor(
            tOrO_f16.iterator,
            layout=cute.make_layout(
                ((src_shape[0], 1), src_shape[1], src_shape[2]), stride=((src_stride[0], 0), src_stride[1], src_stride[2])
            ),
        )
        tOsO_dst = cute.make_tensor(
            tOsO.iterator,
            layout=cute.make_layout(
                ((dst_shape[0], 1), dst_shape[1], dst_shape[2]), stride=((dst_stride[0], 0), dst_stride[1], dst_stride[2])
            ),
        )
        cute.copy(O_tiled_copy, tOrO_src, tOsO_dst)

    @cute.jit
    def compute_diagonal_inverse_16x16_to_32x32(
        self,
        mat: cute.Tensor,  # Input 32x32 block in smem
    ):
        """
        Build 32x32 diagonal block inverse from two 16x16 blocks using Schur complement.

        Similar structure to 8->16 but operating on 16x16 blocks.

        Args:
            mat: 32x32 tensor in SMEM
        """
        dtype = mat.element_type
        lane_id = self.canonical_lane_id()

        # Divide 32x32 into 4 16x16 blocks
        mat16x16_2x2 = cute.flat_divide(mat, (16, 16))

        # MMA configuration for 16x8x16 operations
        mma_atom_shape = (16, 8, 16)
        mma_tiler = (16, 16, 16)

        mma_atom = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=dtype,
            acc_dtype=self.acc_dtype,
            shape_mnk=mma_atom_shape,
        )

        tiled_mma = cute.make_tiled_mma(
            mma_atom,
            atom_layout_mnk=(1, 1, 1),
            permutation_mnk=mma_tiler,
        )

        thr_mma = tiled_mma.get_slice(lane_id)

        # Copy atoms
        copy_atom_s2r = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=2),
            dtype,
        )
        copy_atom_s2r_t = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=2),
            dtype,
        )
        copy_atom_r2s = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=2),
            dtype,
        )

        # Tiled copies
        D_tiled_copy = cute.make_tiled_copy_A(copy_atom_s2r, tiled_mma)
        C_tiled_copy = cute.make_tiled_copy_B(copy_atom_s2r_t, tiled_mma)
        A_tiled_copy = cute.make_tiled_copy_B(copy_atom_s2r_t, tiled_mma)
        O_tiled_copy = cute.make_tiled_copy_C(copy_atom_r2s, tiled_mma)

        D_thr_copy = D_tiled_copy.get_slice(lane_id)
        C_thr_copy = C_tiled_copy.get_slice(lane_id)
        A_thr_copy = A_tiled_copy.get_slice(lane_id)
        O_thr_copy = O_tiled_copy.get_slice(lane_id)

        # Extract 16x16 blocks
        sDInv = mat16x16_2x2[None, None, 1, 1]
        sC = mat16x16_2x2[None, None, 1, 0]
        sAInv = mat16x16_2x2[None, None, 0, 0]
        sO = mat16x16_2x2[None, None, 1, 0]

        # Make column-major
        sC = cute.make_tensor(sC.iterator, layout=cute.select(sC.layout, mode=[1, 0]))
        sAInv = cute.make_tensor(sAInv.iterator, layout=cute.select(sAInv.layout, mode=[1, 0]))

        c_shape = cute.dice(mma_tiler, (1, 1, None))

        # Create fragments
        tOrDInv = thr_mma.make_fragment_A(thr_mma.partition_A(sDInv))
        tOrC = thr_mma.make_fragment_B(thr_mma.partition_B(sC))
        tOrAInv = thr_mma.make_fragment_B(thr_mma.partition_B(sAInv))
        tDCrDC = thr_mma.make_fragment_C(tiled_mma.partition_shape_C(c_shape))
        tOrO = thr_mma.make_fragment_C(tiled_mma.partition_shape_C(c_shape))

        # Partition
        tOsDInv = D_thr_copy.partition_S(sDInv)
        tOrDInv_cv = D_thr_copy.retile(tOrDInv)
        tOsC = C_thr_copy.partition_S(sC)
        tOrC_cv = C_thr_copy.retile(tOrC)
        tOsAInv = A_thr_copy.partition_S(sAInv)
        tOrAInv_cv = A_thr_copy.retile(tOrAInv)
        tOsO = O_thr_copy.partition_D(sO)
        tOrO_cv = O_thr_copy.retile(tOrO)

        # Copy and compute
        cute.copy(D_tiled_copy, tOsDInv, tOrDInv_cv)
        cute.copy(C_tiled_copy, tOsC, tOrC_cv)

        tDCrDC.fill(0.0)
        cute.gemm(tiled_mma, tDCrDC, tOrDInv, tOrC, tDCrDC)
        tDCrDC.store(tDCrDC.load() * cutlass.Float32(-1))

        tOrDC = self.make_acc_as_a(tDCrDC, tiled_mma, dtype)

        cute.copy(A_tiled_copy, tOsAInv, tOrAInv_cv)
        tOrO.fill(0.0)
        cute.gemm(tiled_mma, tOrO, tOrDC, tOrAInv, tOrO)

        # Convert and store
        tOrO_f16 = cute.make_rmem_tensor_like(tOrO_cv, dtype)
        tOrO_f16.store(tOrO_cv.load().to(dtype))
        cute.copy(O_tiled_copy, tOrO_f16, tOsO)

    @cute.jit
    def compute_diagonal_inverse_32x32_to_64x64(
        self,
        mat: cute.Tensor,  # Input 64x64 block in smem
    ):
        """
        Build full 64x64 matrix inverse from two 32x32 blocks using Schur complement.

        This is the final stage that computes the complete 64x64 inverse.

        Args:
            mat: 64x64 tensor in SMEM
        """
        # Divide 64x64 into 4 32x32 blocks
        mat32x32_2x2 = cute.flat_divide(mat, (32, 32))
        mat_16x2_2x2 = cute.logical_divide(mat32x32_2x2, (16, 16))

        warp_id_wg = cute.arch.warp_idx() % 4
        x = warp_id_wg // 2
        y = warp_id_wg % 2

        lane_id = self.canonical_lane_id()
        dtype = mat.element_type

        # MMA configurations
        mma_atom_shape = (16, 8, 16)
        mma_tiler1 = (16, 16, 32)
        mma_tiler2 = (16, 32, 16)

        mma_atom = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=dtype,
            acc_dtype=self.acc_dtype,
            shape_mnk=mma_atom_shape,
        )

        tiled_mma1 = cute.make_tiled_mma(
            mma_atom,
            atom_layout_mnk=(1, 1, 1),
            permutation_mnk=mma_tiler1,
        )
        tiled_mma2 = cute.make_tiled_mma(
            mma_atom,
            atom_layout_mnk=(1, 1, 1),
            permutation_mnk=mma_tiler2,
        )

        thr_mma1 = tiled_mma1.get_slice(lane_id)
        thr_mma2 = tiled_mma2.get_slice(lane_id)

        # Copy atoms
        copy_atom_s2r = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            dtype,
        )
        copy_atom_s2r_t = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            dtype,
        )
        copy_atom_r2s = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4),
            dtype,
        )

        # Tiled copies
        D_tiled_copy = cute.make_tiled_copy_A(copy_atom_s2r, tiled_mma1)
        C_tiled_copy = cute.make_tiled_copy_B(copy_atom_s2r_t, tiled_mma1)
        A_tiled_copy = cute.make_tiled_copy_B(copy_atom_s2r_t, tiled_mma2)
        O_tiled_s2r = cute.make_tiled_copy_C(copy_atom_s2r, tiled_mma2)
        O_tiled_r2s = cute.make_tiled_copy_C(copy_atom_r2s, tiled_mma2)

        D_thr_copy = D_tiled_copy.get_slice(lane_id)
        C_thr_copy = C_tiled_copy.get_slice(lane_id)
        A_thr_copy = A_tiled_copy.get_slice(lane_id)
        O_thr_s2r = O_tiled_s2r.get_slice(lane_id)
        O_thr_r2s = O_tiled_r2s.get_slice(lane_id)

        # Extract blocks with warp distribution
        sDInv = mat_16x2_2x2[(None, y), None, 1, 1]
        sC = mat_16x2_2x2[None, (None, x), 1, 0]
        sAInv = mat_16x2_2x2[(None, x), None, 0, 0]
        sO = mat_16x2_2x2[(None, y), None, 1, 0]

        # Make column-major
        sC = cute.make_tensor(sC.iterator, layout=cute.select(sC.layout, mode=[1, 0]))
        sAInv = cute.make_tensor(sAInv.iterator, layout=cute.select(sAInv.layout, mode=[1, 0]))

        # Create fragments
        tOrDInv = thr_mma1.make_fragment_A(thr_mma1.partition_A(sDInv))
        tOrC = thr_mma1.make_fragment_B(thr_mma1.partition_B(sC))
        tOrAInv = thr_mma2.make_fragment_B(thr_mma2.partition_B(sAInv))

        tDCrDC = thr_mma1.make_fragment_C(tiled_mma1.partition_shape_C((16, 16)))
        tOrO = thr_mma2.make_fragment_C(tiled_mma2.partition_shape_C((16, 32)))

        # Partition
        tOsDInv = D_thr_copy.partition_S(sDInv)
        tOrDInv_cv = D_thr_copy.retile(tOrDInv)
        tOsC = C_thr_copy.partition_S(sC)
        tOrC_cv = C_thr_copy.retile(tOrC)
        tOsAInv = A_thr_copy.partition_S(sAInv)
        tOrAInv_cv = A_thr_copy.retile(tOrAInv)

        # Copy and compute DC = -D*C
        cute.copy(D_tiled_copy, tOsDInv, tOrDInv_cv)
        cute.copy(C_tiled_copy, tOsC, tOrC_cv)

        tDCrDC.fill(0.0)
        cute.gemm(tiled_mma1, tDCrDC, tOrDInv, tOrC, tDCrDC)
        tDCrDC.store(tDCrDC.load() * cutlass.Float32(-1))

        tOrDC = self.make_acc_as_a(tDCrDC, tiled_mma2, dtype)

        # Compute O = -DC * A_inv
        cute.copy(A_tiled_copy, tOsAInv, tOrAInv_cv)
        tOrO.fill(0.0)
        cute.gemm(tiled_mma2, tOrO, tOrDC, tOrAInv, tOrO)

        # Convert and store
        tOrO_f16 = cute.make_rmem_tensor_like(tOrO, dtype)
        tOrO_f16.store(tOrO.load().to(dtype))

        # Synchronize all threads before storing
        self.cuda_wg_sync_barrier.arrive_and_wait()

        # Store result back to SMEM
        tOsO = O_thr_r2s.partition_D(sO)
        tOrO_cvt_cv = O_thr_r2s.retile(tOrO_f16)

        # Warp with x=0 writes the result
        if x == 0:
            cute.copy(O_tiled_r2s, tOrO_cvt_cv, tOsO)

        self.cuda_wg_sync_barrier.arrive_and_wait()

        # Warp with x=1 reads and accumulates (reduce operation)
        if x == 1:
            tOrO_red = cute.make_rmem_tensor_like(tOrO_f16)
            tOsO_s = O_thr_s2r.partition_S(sO)
            tOrO_red_cv = O_thr_s2r.retile(tOrO_red)
            cute.copy(O_tiled_s2r, tOsO_s, tOrO_red_cv)
            tOrO_f16.store(tOrO_f16.load() + tOrO_red.load())
            cute.copy(O_tiled_r2s, tOrO_cvt_cv, tOsO)

        # Final synchronization
        self.cuda_wg_sync_barrier.arrive_and_wait()

    @cute.jit
    def compute_matrix_inverse_64x64(
        self,
        s_mat: cute.Tensor,  # Input M matrix in smem, shape (64, 64)
        stage_limit: int = 99,  # Maximum stage to compute (1-4, default 99 for all)
    ):
        """
        Compute M^{-1} for 64x64 lower triangular matrix using block-wise Schur complement.

        Based on flat_collective_inverse.hpp algorithm:
        1. Divide into 8x8 blocks
        2. Compute 8x8 diagonal block inverses (lower triangular blocks)
        3. Use Schur complement for below-diagonal blocks
        4. Progressively combine: 8x8 -> 16x16 -> 32x32 -> 64x64

        For a lower triangular matrix L:
            inv([A  0 ]) = [inv(A)  0                    ]
               [C  D ]   [-inv(D)C*inv(A)  inv(D)        ]

        Args:
            s_mat: Input M matrix (lower triangular) in smem, shape (64, 64)
            stage_limit: Maximum stage to execute (1-4, default 99 for all)
        """
        tidx, _, _ = cute.arch.thread_idx()
        tidx = tidx % 128

        # Stage 1: Invert all 8 diagonal 8x8 blocks
        t8x8mat = cute.flat_divide(s_mat, (8, 8))

        # Block size progression: 8x8 -> 16x16 -> 32x32 -> 64x64
        # Process in stages

        # Stage 1: Invert all 8 diagonal 8x8 blocks
        t8x8mat = cute.flat_divide(s_mat, (8, 8))
        if tidx < 64:
            self.compute_diagonal_inverse_8x8(
                t8x8mat[None, None, tidx // 8, tidx // 8],
                tidx % 8,
                tidx // 8,  # Pass block_id for correct shuffle masking
            )
        self.cuda_wg_sync_barrier.arrive_and_wait()

        # Stage 2: Invert all 4 diagonal 16x16 blocks
        if stage_limit >= 2:
            t16x16mat = cute.flat_divide(s_mat, (16, 16))
            self.compute_diagonal_inverse_8x8_to_16x16(
                t16x16mat[None, None, tidx // 32, tidx // 32],
            )
            # Synchronize after stage 2
            self.cuda_wg_sync_barrier.arrive_and_wait()

        # Stage 3: Invert all 2 diagonal 32x32 blocks
        if stage_limit >= 3:
            t32x32mat = cute.flat_divide(s_mat, (32, 32))
            if tidx < 64:
                self.compute_diagonal_inverse_16x16_to_32x32(
                    t32x32mat[None, None, tidx // 32, tidx // 32],
                )
            self.cuda_wg_sync_barrier.arrive_and_wait()

        # Stage 4: Invert the full 64x64 matrix
        if stage_limit >= 4:
            self.compute_diagonal_inverse_32x32_to_64x64(s_mat)

    @cute.jit
    def __call__(self, torch_mat, stream):
        """
        Public interface that accepts PyTorch tensor and launches kernel.

        This method handles the conversion from PyTorch tensor to the JIT-compiled kernel.

        Args:
            torch_mat: 64x64 FP16 PyTorch tensor in GPU memory
            stream: CUDA stream for execution
        """
        # Call the JIT-compiled kernel with the PyTorch tensor
        # CuTe will handle the conversion to cute.Pointer internally
        self._jit_call(torch_mat, stream)

    @cute.jit
    def _jit_call(self, mat_iter, stream):
        """
        JIT-compiled entry point that accepts matrix pointer and launches kernel.

        This is decorated with @cute.jit so CuTe can handle CUDA operations properly.
        CuTe will automatically convert PyTorch tensors to cute.Pointer.

        Args:
            mat_iter: Pointer to 64x64 FP16 matrix in GPU memory
            stream: CUDA stream for execution
        """
        # Define SharedStorage structure for SMEM allocation
        smat_layout = cute.make_layout(
            (self.MATRIX_SIZE, self.MATRIX_SIZE),
            stride=(self.MATRIX_SIZE, 1),
        )

        @cute.struct
        class SharedStorage:
            smat: cute.struct.Align[
                cute.struct.MemRange[self.MATRIX_DTYPE, cute.cosize(smat_layout)],  # type: ignore
                self.SMEM_ALIGN_BYTES,
            ]

        self.shared_storage = SharedStorage

        # mat_iter is already a Tensor object from the PyTorch conversion
        # Use it directly without creating a new tensor
        mat = mat_iter

        # Launch kernel
        self.kernel(mat, smat_layout).launch(
            grid=(self.GRID_SIZE, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(self, mat: cute.Tensor, smat_layout: cute.Layout):
        """
        Core kernel that performs the 64x64 matrix inversion.

        This kernel implements the complete matrix inversion pipeline:
        1. Load 64x64 matrix from GMEM to SMEM (all 128 threads cooperatively)
        2. Compute 4-stage Schur complement inversion in SMEM
        3. Store result back to GMEM

        Thread Organization:
        - Total threads: 128 (4 warps × 32 lanes)
        - Each thread handles: (64×64)/128 = 32 elements
        - Linear indexing: linear_idx = tidx + i * 128
        - 2D mapping: m_idx = linear_idx / 64, n_idx = linear_idx % 64

        Memory Access Pattern:
        - GMEM load: All threads read cooperatively from global memory
        - SMEM store: All threads write to shared memory
        - Synchronization: NamedBarrier ensures all threads reach sync points

        Algorithm Stages:
        - Stage 0: GMEM→SMEM load with synchronization
        - Stage 1-4: 4-stage block-wise Schur complement inversion in SMEM
        - Stage Final: SMEM→GMEM store with synchronization

        Args:
            mat: 64x64 FP16 matrix tensor (from global memory, GMEM)
            smat_layout: Layout for the shared memory tensor
        """
        # Get thread indices for cooperative work distribution
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # Allocate shared memory using SmemAllocator
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Get the SMEM tensor from SharedStorage
        # Use recast_ptr to ensure proper FP16 type handling
        smat = cute.make_tensor(cute.recast_ptr(storage.smat.data_ptr(), dtype=self.MATRIX_DTYPE), layout=smat_layout)

        # Each thread will handle (64*64)/128 = 32 elements
        elements_per_thread = (self.MATRIX_SIZE * self.MATRIX_SIZE) // self.threads_per_cta

        # ========================================================================
        # Stage 0: Load 64x64 matrix from GMEM to SMEM
        # ========================================================================
        # All 128 threads cooperatively load the entire 64x64 matrix from global memory
        # to the shared memory buffer. Each thread loads its assigned elements.
        for i in range(elements_per_thread):
            # Calculate linear index for this thread's element
            linear_idx = tidx + i * self.threads_per_cta

            # Convert linear index to 2D coordinates
            m_idx = linear_idx // self.MATRIX_SIZE  # Row index (0-63)
            n_idx = linear_idx % self.MATRIX_SIZE  # Column index (0-63)

            # Bounds check and load from GMEM to SMEM
            if m_idx < self.MATRIX_SIZE and n_idx < self.MATRIX_SIZE:
                # Load element from global memory using 2D indexing
                # mat is a properly typed FP16 CuTe tensor
                val = mat[m_idx, n_idx]

                # Store element to shared memory (SMEM write)
                smat[m_idx, n_idx] = val

        # Synchronize all 128 threads after GMEM load
        # Ensures all threads have completed their loads before computation begins
        self.cuda_wg_sync_barrier.arrive_and_wait()

        # ========================================================================
        # Stage 1-4: Compute 4-stage block-wise Schur complement inversion
        # ========================================================================
        # Call the jit-compiled inversion function which operates on the SMEM buffer
        # This function assumes input is a 64x64 lower triangular matrix in SMEM
        # and performs 4 progressive stages:
        #   - Stage 1: Invert 8 diagonal 8×8 blocks
        #   - Stage 2: Build 16×16 blocks using Schur complement
        #   - Stage 3: Build 32×32 blocks using Schur complement
        #   - Stage 4: Build full 64×64 inverse using Schur complement
        # Get stage limit from environment variable (default: 99 for all stages)
        import os

        stage_limit = int(os.environ.get("CULA_STAGE_LIMIT", "99"))
        self.compute_matrix_inverse_64x64(smat, stage_limit)

        # Synchronize all threads after computation
        # Ensures all threads have completed inversion before store
        self.cuda_wg_sync_barrier.arrive_and_wait()

        # ========================================================================
        # Stage Final: Store result back to GMEM
        # ========================================================================
        # All 128 threads cooperatively store the computed inverse matrix from
        # the shared memory buffer back to global memory using the same distribution pattern as load.
        for i in range(elements_per_thread):
            # Calculate linear index for this thread's element
            linear_idx = tidx + i * self.threads_per_cta

            # Convert linear index to 2D coordinates
            m_idx = linear_idx // self.MATRIX_SIZE  # Row index (0-63)
            n_idx = linear_idx % self.MATRIX_SIZE  # Column index (0-63)

            # Bounds check and store from SMEM to GMEM
            if m_idx < self.MATRIX_SIZE and n_idx < self.MATRIX_SIZE:
                # Load element from shared memory
                val = smat[m_idx, n_idx]

                # Store element to global memory using 2D indexing
                mat[m_idx, n_idx] = val

        # Final synchronization of all threads
        # Ensures all threads have completed their stores before kernel exit
        self.cuda_wg_sync_barrier.arrive_and_wait()
