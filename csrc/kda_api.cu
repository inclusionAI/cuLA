#include <torch/python.h>
#include "kda/kda_fwd_common.cuh"
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