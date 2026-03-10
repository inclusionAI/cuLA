#include <torch/python.h>
#include "kda_bwd/kda_bwd_common.cuh"
#include "kda_bwd/kda_fwd_common.cuh"
#include "cutlass/cutlass.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

void ChunkKDAFwdIntra(
    at::Tensor q,
    at::Tensor k,
    at::Tensor g,
    at::Tensor beta,
    at::Tensor cu_seqlens,
    at::Tensor chunk_indices,
    at::Tensor Aqk_out,
    at::Tensor Akk_out,
    at::Tensor tile_counter,
    float scale,
    int chunk_size) {

    KDA_fwd_intra_params params;
    params.total_q_len = q.size(0) * q.size(1);
    params.b = cu_seqlens.size(0) - 1;
    params.h = q.size(2);
    params.d = q.size(3);
    params.chunk_size = chunk_size;
    params.scale = scale;
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.g_ptr = g.data_ptr();
    params.beta_ptr = beta.data_ptr();
    params.cu_seqlens_ptr = cu_seqlens.data_ptr();
    params.chunk_indices_ptr = chunk_indices.data_ptr();
    params.Aqk_out_ptr = Aqk_out.data_ptr();
    params.Akk_out_ptr = Akk_out.data_ptr();
    params.shape_Akk = cute::make_shape(params.total_q_len, params.chunk_size, params.h);
    params.stride_Akk = cute::make_stride(params.chunk_size * params.h, cute::_1{}, params.chunk_size);
    // printf("ChunkKDAFwdIntra, total_q_len: %d, b: %d, h: %d, d: %d, chunk_size: %d\n", params.total_q_len, params.b, params.h, params.d, chunk_size);
    int tile_num = chunk_indices.size(0);
    // printf("tile_num: %d, b: %d, h: %d\n", tile_num, params.b, params.h);
    auto device_prop = at::cuda::getCurrentDeviceProperties();
    params.num_sm = device_prop->multiProcessorCount;
    params.tile_scheduler_params = StaticPersistentTileScheduler::Params{tile_num, params.h, 4, params.num_sm, (int*)tile_counter.data_ptr()};

    sm100::run_kda_fwd_intra_sm100(params, at::cuda::getCurrentCUDAStream());
}

void ChunkKDABwdIntra(
    at::Tensor q,
    at::Tensor k,
    at::Tensor g,
    at::Tensor beta,
    at::Tensor dAqk,
    at::Tensor dAkk,
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor db,
    at::Tensor dg,
    at::Tensor cu_seqlens,
    at::Tensor chunk_indices,
    at::Tensor dq_out,
    at::Tensor dk_out,
    at::Tensor db_out,
    at::Tensor dg_out,
    at::Tensor tile_counter,
    int chunk_size) {
    //TODO: Implement the ChunkKDABwdIntra function

    KDA_bwd_intra_params params;
    params.total_q_len = q.size(0) * q.size(1);
    params.b = cu_seqlens.size(0) - 1;
    params.h = q.size(2);
    params.d = q.size(3);
    params.chunk_size = chunk_size;
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.g_ptr = g.data_ptr();
    params.beta_ptr = beta.data_ptr();
    params.dAqk_ptr = dAqk.data_ptr();
    params.dAkk_ptr = dAkk.data_ptr();
    params.dq_ptr = dq.data_ptr();
    params.dk_ptr = dk.data_ptr();
    params.db_ptr = db.data_ptr();
    params.dg_ptr = dg.data_ptr();
    params.cu_seqlens_ptr = cu_seqlens.data_ptr();
    params.chunk_indices_ptr = chunk_indices.data_ptr();
    params.dq_out_ptr = dq_out.data_ptr();
    params.dk_out_ptr = dk_out.data_ptr();
    params.db_out_ptr = db_out.data_ptr();
    params.dg_out_ptr = dg_out.data_ptr();
    // printf("ChunkKDABwdIntra, total_q_len: %d, b: %d, h: %d, d: %d, chunk_size: %d\n", params.total_q_len, params.b, params.h, params.d, chunk_size);
    int tile_num = chunk_indices.size(0);
    // printf("tile_num: %d, b: %d, h: %d\n", tile_num, params.b, params.h);
    auto device_prop = at::cuda::getCurrentDeviceProperties();
    params.num_sm = device_prop->multiProcessorCount;
    params.tile_scheduler_params = NaiveTileScheduler::Params{tile_num, params.h, 4, params.num_sm, (int*)tile_counter.data_ptr()};
    
    sm100::run_kda_bwd_intra_sm100(params, at::cuda::getCurrentCUDAStream());
}


void ChunkKDABwdWYDqkgFused(
    at::Tensor q,           // [B, T, H, K] bf16
    at::Tensor k,           // [B, T, H, K] bf16
    at::Tensor v,           // [B, T, H, V] bf16
    at::Tensor v_new,       // [B, T, H, V] bf16
    at::Tensor g,           // [B, T, H, K] fp32
    at::Tensor beta,        // [B, T, H] bf16
    at::Tensor A,           // [B, T, H, BT] bf16 (Akk matrix)
    at::Tensor h,           // [NT, H, K, V] fp32
    at::Tensor do_,         // [B, T, H, V] bf16
    at::Tensor dh,          // [NT, H, K, V] fp32
    at::Tensor dv,          // [B, T, H, V] bf16
    at::Tensor cu_seqlens,  // [B + 1]
    at::Tensor chunk_indices, // [NT * 2]
    at::Tensor dq_out,      // [B, T, H, K] fp32
    at::Tensor dk_out,      // [B, T, H, K] fp32
    at::Tensor dv_out,     // [B, T, H, V] bf16
    at::Tensor db_out,      // [B, T, H] fp32
    at::Tensor dg_out,      // [B, T, H, K] fp32
    at::Tensor dA_out,      // [B, T, H, BT] fp32
    float scale,
    int chunk_size) {
    
    KDA_bwd_wy_dqkg_fused_params params;
    params.total_q_len = q.size(0) * q.size(1);
    params.b = cu_seqlens.size(0) - 1;
    params.h = q.size(2);
    params.d = q.size(3);
    params.d_v = v.size(3);
    params.chunk_size = chunk_size;
    params.scale = scale;
    
    // Input pointers
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.v_new_ptr = v_new.data_ptr();
    params.g_ptr = g.data_ptr();
    params.beta_ptr = beta.data_ptr();
    params.A_ptr = A.data_ptr();
    params.h_ptr = h.data_ptr();
    params.do_ptr = do_.data_ptr();
    params.dh_ptr = dh.data_ptr();
    params.dv_ptr = dv.data_ptr();
    
    // Output pointers
    params.dq_out_ptr = dq_out.data_ptr();
    params.dk_out_ptr = dk_out.data_ptr();
    params.dv_out_ptr = dv_out.data_ptr();
    params.db_out_ptr = db_out.data_ptr();
    params.dg_out_ptr = dg_out.data_ptr();
    params.dA_out_ptr = dA_out.data_ptr();
    
    // Sequence info
    params.cu_seqlens_ptr = cu_seqlens.data_ptr();
    params.chunk_indices_ptr = chunk_indices.data_ptr();
    
    int tile_num = chunk_indices.size(0) / 2;
    auto device_prop = at::cuda::getCurrentDeviceProperties();
    params.num_sm = device_prop->multiProcessorCount;
    params.tile_scheduler_params = NaiveTileScheduler::Params{tile_num, params.h, 4, params.num_sm, nullptr};
    
    sm100::run_kda_bwd_wy_dqkg_fused_sm100(params, at::cuda::getCurrentCUDAStream());
}