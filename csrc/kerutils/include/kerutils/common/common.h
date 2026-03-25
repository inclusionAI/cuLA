#pragma once

#include <cstdio>
#include <cstdlib>

namespace kerutils {}

#define KU_PRINTLN(fmt, ...) { cute::print(fmt, ##__VA_ARGS__); print("\n"); }

#define CHECK_CUDA(call)                                                                                  \
    do {                                                                                                  \
        cudaError_t status_ = call;                                                                       \
        if (status_ != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
            exit(1);                                                                              \
        }                                                                                                 \
    } while(0)

#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())

namespace ku = kerutils;
