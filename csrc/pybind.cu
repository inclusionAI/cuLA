#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/fast_math.h>

#include "lightning/sm100/prefill_fwd.h"
#include "kda_bwd/kda_bwd_common.cuh"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

struct Arch {
    int major;
    int minor;

    bool is_sm90() const {
        return major == 9 && minor == 0;
    }

    bool is_sm100() const {
        return major == 10;
    }

    void assert_is_supported() const {
        TORCH_CHECK(is_sm90() || is_sm100(), "Only SM90 and SM100 are supported");
    }
};

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
    int chunk_size);

void ChunkKDABwdWYDqkgFused(
    at::Tensor q,           // [B, T, H, K] bf16
    at::Tensor k,           // [B, T, H, K] bf16
    at::Tensor v,           // [B, T, H, V] bf16
    at::Tensor v_new,       // [B, T, H, V] bf16
    at::Tensor g,           // [B, T, H, K] fp32
    at::Tensor beta,        // [B, T, H] bf16
    at::Tensor A,           // [B, T, H, BT] fp32
    at::Tensor h,           // [NT, H, K, V] fp32
    at::Tensor do_,         // [B, T, H, V] bf16
    at::Tensor dh,          // [NT, H, K, V] fp32
    at::Tensor dv,          // [B, T, H, V] bf16
    at::Tensor cu_seqlens,  // [B + 1]
    at::Tensor chunk_indices, // [NT * 2]
    at::Tensor dq_out,      // [B, T, H, K] fp32
    at::Tensor dk_out,      // [B, T, H, K] fp32
    at::Tensor dv_out,      // [B, T, H, V] bf16
    at::Tensor db_out,      // [B, T, H] fp32
    at::Tensor dg_out,      // [B, T, H, K] fp32
    at::Tensor dA_out,      // [B, T, H, BT] fp32
    float scale,
    int chunk_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashLA";
    m.def("lightning_prefill_fwd", &FLACutlassSM100FwdRun);
    m.def("chunk_kda_bwd_intra_cuda", &ChunkKDABwdIntra);
    m.def("chunk_kda_bwd_wy_dqkg_fused_cuda", &ChunkKDABwdWYDqkgFused);
}
