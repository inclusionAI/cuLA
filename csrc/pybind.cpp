#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/fast_math.h>

#include "lightning/sm100/prefill_fwd.h"

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashLA";
    m.def("lightning_prefill_fwd", &FLACutlassSM100FwdRun);
}
