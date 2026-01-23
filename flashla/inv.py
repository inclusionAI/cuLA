"""
Standalone CuTe kernel for 64x64 FP16 matrix inverse computation.

This kernel implements the block-wise Schur complement matrix inversion
for 64x64 lower triangular matrices using 4 progressive stages:
  Stage 1: Invert 8 diagonal 8x8 blocks
  Stage 2: Build 16x16 blocks from 8x8
  Stage 3: Build 32x32 blocks from 16x16
  Stage 4: Build full 64x64 inverse
"""

import cutlass.cute as cute
import cutlass
import cutlass.pipeline as pipeline


class MatrixInverse64x64:
    """
    64x64 FP16 lower triangular matrix inversion kernel.
    
    This kernel inverts a 64x64 lower triangular matrix using the 
    block-wise Schur complement method. The matrix is divided into 
    8x8 blocks and progressively inverted in 4 stages.
    """
    
    def __init__(self, acc_dtype=cutlass.Float32, cuda_core_threads=128):
        """
        Initialize the matrix inverse kernel.
        
        Args:
            acc_dtype: Accumulator data type for intermediate computations (default: Float32)
            cuda_core_threads: Number of CUDA threads in the work-group (default: 128)
        """
        self.acc_dtype = acc_dtype
        self.cuda_core_threads = cuda_core_threads
        # Create a named barrier for synchronization across all threads
        self.cuda_wg_sync_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=cuda_core_threads,
        )
    
    def canonical_lane_id(self):
        """Get the canonical lane ID within the warp."""
        tidx, _, _ = cute.arch.thread_idx()
        lane_id = tidx % 32
        return lane_id
    
    def convert_layout_c_to_a(
        self,
        c_layout: cute.Layout,
        tiled_mma: cute.TiledMma,
    ):
        """Convert MMA accumulator layout to operand A layout."""
        cfrag_atom_size = cute.size(c_layout.shape[0])
        afrag_atom_size = cute.size(tiled_mma.tv_layout_A.shape[1])
        ratio = afrag_atom_size // cfrag_atom_size
        
        if ratio == 1:
            return c_layout
        
        divided = cute.logical_divide(c_layout, (None, None, ratio))
        a_layout = cute.make_layout((
            cute.flatten((divided.shape[0], divided.shape[2][0])),
            divided.shape[1],
            divided.shape[2][1]
        ))
        return a_layout
    
    def make_acc_as_a(self, acc: cute.Tensor, tiled_mma: cute.TiledMma, dtype: cute.Numeric):
        """Convert MMA accumulator to operand A format."""
        a_layout = self.convert_layout_c_to_a(acc.layout, tiled_mma)
        a_tensor = cute.make_rmem_tensor(a_layout, dtype=dtype)
        op_as_acc = cute.make_tensor(a_tensor.iterator, layout=acc.layout)
        op_as_acc.store(acc.load().to(dtype))
        return a_tensor
    
    def make_op_a_from_acc_rmem_16x8x8(
        self,
        acc_dtype: cute.Numeric,
        dst_dtype: cute.Numeric,
        acc: cute.Tensor,
    ):
        """Convert MMA accumulator to operand A format for 16x8x8 MMA."""
        # For 16x8x8 MMA, we need to reshape the accumulator
        a_layout = cute.make_layout((16, 8))
        a_tensor = cute.make_rmem_tensor(a_layout, dtype=dst_dtype)
        
        # Store accumulator values into the A operand tensor
        # This handles the conversion from accumulator format to A operand format
        for i in range(cute.size(a_layout)):
            idx_tuple = cute.unflatten(i, a_layout.shape)
            a_val = acc[idx_tuple]
            a_tensor[idx_tuple] = a_val.to(dst_dtype)
        
        return a_tensor
    
    def compute_diagonal_inverse_8x8(
        self,
        s_block: cute.Tensor,
        lane_id: int,
    ):
        """
        Compute inverse of an 8x8 block in SMEM using warp-level operations.
        
        This function inverts an 8x8 diagonal block using in-warp Gaussian 
        elimination and warp shuffle operations.
        
        Args:
            s_block: 8x8 tensor in SMEM
            lane_id: Lane ID within the warp (0-31)
        """
        # Load 8x8 block into registers with FP16->FP32 conversion
        s_row = s_block[lane_id // 8] if lane_id < 64 else None
        
        # In-warp Gaussian elimination for 8x8 matrix
        # This would involve warp shuffle operations for row operations
        # For now, this is a placeholder for the warp-level inversion logic
    
    def load_row_mat8x8(
        self,
        mat: cute.Tensor,
        idx: int,
    ) -> cute.Tensor:
        """
        Load a row from 8x8 matrix with FP16->FP32 conversion.
        
        Args:
            mat: 8x8 matrix tensor
            idx: Row index (0-7)
            
        Returns:
            Row tensor with FP32 dtype
        """
        row = mat[idx]
        return row.to(cutlass.Float32)
    
    def store_row_mat8x8(
        self,
        mat: cute.Tensor,
        row: cute.Tensor,
        idx: int,
    ):
        """
        Store a row to 8x8 matrix with FP32->FP16 conversion.
        
        Args:
            mat: 8x8 matrix tensor
            row: Row tensor with FP32 dtype
            idx: Row index (0-7)
        """
        mat[idx] = row.to(mat.element_type)
    
    def compute_diagonal_inverse_8x8_to_16x16(
        self,
        mat: cute.Tensor,  # Input 8x8 block in smem
    ):
        """
        Build 16x16 diagonal block inverse from two 8x8 blocks using Schur complement.
        
        Computes: inv([A 0; C D]) = [inv(A) 0; -inv(D)C*inv(A) inv(D)]
        
        Args:
            mat: 16x16 tensor in SMEM (divided into 4 8x8 blocks)
        """
        dtype = mat.element_type
        lane_id = self.canonical_lane_id()
        
        # Divide 16x16 into 4 8x8 blocks
        mat8x8_2x2 = cute.flat_divide(mat, (8, 8))
        
        # MMA configuration for 16x8x8 operations
        mma_atom_shape = (16, 8, 8)
        mma_tiler = (16, 16, 8)
        
        mma_atom = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=dtype,
            acc_dtype=self.acc_dtype,
            shape_mnk=mma_atom_shape,
        )
        
        tiled_mma = cute.make_tiled_mma(
            mma_atom,
            atom_layout_mnk=(1,1,1),
            permutation_mnk=mma_tiler,
        )
        
        thr_mma = tiled_mma.get_slice(lane_id)
        
        # Copy atoms for SMEM<->RMEM transfers
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
        
        # Tiled copy operations
        D_tiled_copy = cute.make_tiled_copy_A(copy_atom_s2r, tiled_mma)
        C_tiled_copy = cute.make_tiled_copy_B(copy_atom_s2r_t, tiled_mma)
        A_tiled_copy = cute.make_tiled_copy_B(copy_atom_s2r_t, tiled_mma)
        O_tiled_copy = cute.make_tiled_copy_C(copy_atom_r2s, tiled_mma)
        
        D_thr_copy = D_tiled_copy.get_slice(lane_id)
        C_thr_copy = C_tiled_copy.get_slice(lane_id)
        A_thr_copy = A_tiled_copy.get_slice(lane_id)
        O_thr_copy = O_tiled_copy.get_slice(lane_id)
        
        # Extract blocks: D=inv(D), C=C, A=inv(A), O=output
        sDInv = mat8x8_2x2[None, None, 1, 1]
        sC = mat8x8_2x2[None, None, 1, 0]
        sAInv = mat8x8_2x2[None, None, 0, 0]
        sO = mat8x8_2x2[None, None, 1, 0]
        
        # Make operand B column-major
        sC = cute.make_tensor(sC.iterator, layout=cute.select(sC.layout, mode=[1,0]))
        sAInv = cute.make_tensor(sAInv.iterator, layout=cute.select(sAInv.layout, mode=[1,0]))
        
        # Create MMA fragments
        a_shape = cute.dice(mma_tiler, (1,None,1))
        b_shape = cute.dice(mma_tiler, (None,1,1))
        c_shape = cute.dice(mma_tiler, (1,1,None))
        
        tOrDInv = thr_mma.make_fragment_A(tiled_mma.partition_shape_A(a_shape))
        tOrC = thr_mma.make_fragment_B(thr_mma.partition_B(sC))
        tOrAInv = thr_mma.make_fragment_B(thr_mma.partition_B(sAInv))
        tDCrDC = thr_mma.make_fragment_C(tiled_mma.partition_shape_C(c_shape))
        tOrO = thr_mma.make_fragment_C(tiled_mma.partition_shape_C(c_shape))
        
        # Partition shared memory
        tOsDInv = D_thr_copy.partition_S(sDInv)
        tOrDInv_cv = D_thr_copy.retile(tOrDInv)
        tOsC = C_thr_copy.partition_S(sC)
        tOrC_cv = C_thr_copy.retile(tOrC)
        tOsAInv = A_thr_copy.partition_S(sAInv)
        tOrAInv_cv = A_thr_copy.retile(tOrAInv)
        tOsO = O_thr_copy.partition_D(sO)
        tOrO_cv = O_thr_copy.retile(tOrO)
        
        # Copy D inverse and C from SMEM
        cute.copy(D_tiled_copy, tOsDInv, tOrDInv_cv)
        cute.copy(C_tiled_copy, tOsC, tOrC_cv)
        
        # Compute DC = -D*C
        tDCrDC.fill(0.0)
        cute.gemm(tiled_mma, tDCrDC, tOrDInv, tOrC, tDCrDC)
        tDCrDC.store(tDCrDC.load() * cutlass.Float32(-1))
        
        # Convert accumulator to A operand format
        tOrDC = self.make_op_a_from_acc_rmem_16x8x8(
            self.acc_dtype,
            dtype,
            tDCrDC,
        )
        
        # Copy A inverse and compute O = -DC * A_inv
        cute.copy(A_tiled_copy, tOsAInv, tOrAInv_cv)
        tOrO_cv.fill(0.0)
        cute.gemm(tiled_mma, tOrO_cv, tOrDC, tOrAInv, tOrO_cv)
        
        # Convert output back to FP16
        tOrO_f16 = cute.make_rmem_tensor_like(tOrO_cv[(None, 0), None, None], dtype)
        tOrO_f16.store(tOrO_cv[(None, 0), None, None].load().to(dtype))
        
        # Store result back to SMEM
        src_shape = tOrO_f16.shape
        src_stride = tOrO_f16.layout.stride
        dst_shape = tOsO[(None, 0), None, None].shape
        dst_stride = tOsO[(None, 0), None, None].layout.stride
        tOrO_src = cute.make_tensor(
            tOrO_f16.iterator,
            layout=cute.make_layout(
                ((src_shape[0], 1), src_shape[1], src_shape[2]),
                stride=((src_stride[0], 0), src_stride[1], src_stride[2])
            )
        )
        tOsO_dst = cute.make_tensor(
            tOsO.iterator,
            layout=cute.make_layout(
                ((dst_shape[0], 1), dst_shape[1], dst_shape[2]),
                stride=((dst_stride[0], 0), dst_stride[1], dst_stride[2])
            )
        )
        cute.copy(O_tiled_copy, tOrO_src, tOsO_dst)
    
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
            atom_layout_mnk=(1,1,1),
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
        sC = cute.make_tensor(sC.iterator, layout=cute.select(sC.layout, mode=[1,0]))
        sAInv = cute.make_tensor(sAInv.iterator, layout=cute.select(sAInv.layout, mode=[1,0]))
        
        a_shape = cute.dice(mma_tiler, (1,None,1))
        b_shape = cute.dice(mma_tiler, (None,1,1))
        c_shape = cute.dice(mma_tiler, (1,1,None))
        
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
            atom_layout_mnk=(1,1,1),
            permutation_mnk=mma_tiler1,
        )
        tiled_mma2 = cute.make_tiled_mma(
            mma_atom,
            atom_layout_mnk=(1,1,1),
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
        sC = cute.make_tensor(sC.iterator, layout=cute.select(sC.layout, mode=[1,0]))
        sAInv = cute.make_tensor(sAInv.iterator, layout=cute.select(sAInv.layout, mode=[1,0]))
        
        # Create fragments
        tOrDInv = thr_mma1.make_fragment_A(thr_mma1.partition_A(sDInv))
        tOrC = thr_mma1.make_fragment_B(thr_mma1.partition_B(sC))
        tOrAInv = thr_mma2.make_fragment_B(thr_mma2.partition_B(sAInv))
        
        tDCrDC = thr_mma1.make_fragment_C(tiled_mma1.partition_shape_C((16,16)))
        tOrO = thr_mma2.make_fragment_C(tiled_mma2.partition_shape_C((16,32)))
        
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
        
        if x == 0:
            cute.copy(O_tiled_r2s, tOrO_cvt_cv, tOsO)
        
        # Final synchronization
        self.cuda_wg_sync_barrier.arrive_and_wait()
    
    @cute.jit
    def compute_matrix_inverse_64x64(self, s_mat: cute.Tensor):
        """
        Compute 64x64 lower triangular matrix inverse using 4 progressive stages.
        
        Stage 1: Invert 8 diagonal 8x8 blocks
        Stage 2: Combine to form 16x16 diagonal blocks (2x2 of 8x8)
        Stage 3: Combine to form 32x32 diagonal blocks (2x2 of 16x16)
        Stage 4: Combine to form full 64x64 inverse (2x2 of 32x32)
        
        Args:
            s_mat: 64x64 tensor in SMEM (lower triangular matrix in FP16)
        """
        tidx, _, _ = cute.arch.thread_idx()
        
        # Stage 1: Invert 8 diagonal 8x8 blocks
        t8x8mat = cute.flat_divide(s_mat, (8, 8))
        if tidx < 64:
            # Each thread processes one 8x8 diagonal block
            block_idx = tidx // 8
            lane_id = tidx % 8
            if block_idx < 8:
                # This would call the 8x8 inversion for diagonal blocks
                self.compute_diagonal_inverse_8x8(t8x8mat[block_idx, block_idx], tidx % 32)
        
        self.cuda_wg_sync_barrier.arrive_and_wait()
        
        # Stage 2: Build 16x16 blocks from 8x8 (using Schur complement)
        t16x16mat = cute.flat_divide(s_mat, (16, 16))
        if tidx < 128:
            block_idx = (tidx // 32) % 4
            if block_idx < 4:
                self.compute_diagonal_inverse_8x8_to_16x16(t16x16mat[block_idx // 2, block_idx // 2])
        
        self.cuda_wg_sync_barrier.arrive_and_wait()
        
        # Stage 3: Build 32x32 blocks from 16x16
        t32x32mat = cute.flat_divide(s_mat, (32, 32))
        if tidx < 128:
            block_idx = (tidx // 32) % 2
            if block_idx < 2:
                self.compute_diagonal_inverse_16x16_to_32x32(t32x32mat[block_idx, block_idx])
        
        self.cuda_wg_sync_barrier.arrive_and_wait()
        
        # Stage 4: Build full 64x64 inverse
        self.compute_diagonal_inverse_32x32_to_64x64(s_mat)
