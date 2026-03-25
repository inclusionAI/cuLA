#pragma once

#include <exception>
#include <string>
#include <sstream>

#include <cuda_runtime_api.h>

#include "kerutils/common/common.h"

namespace kerutils {

class KUException final : public std::exception {
    std::string message = {};

public:
    template<typename... Args>
    explicit KUException(const char *name, const char* file, const int line, Args&&... args) {
        std::ostringstream oss;
        
        oss << name << " error (" << file << ":" << line << "): ";
        (oss << ... << args);
        message = oss.str();
    }

    const char *what() const noexcept override {
        return message.c_str();
    }
};

#define THROW_KU_EXCEPTION(name, ...) \
    throw kerutils::KUException(name, __FILE__, __LINE__, __VA_ARGS__)

#define KU_CUDA_CHECK(call)                                                                                  \
do {                                                                                                  \
    cudaError_t status_ = call;                                                                       \
    if (status_ != cudaSuccess) {                                                                     \
        fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
        THROW_KU_EXCEPTION("CUDA", "CUDA error: ", cudaGetErrorString(status_));                       \
    }                                                                                                 \
} while(0)

// This `KU_ASSERT` is triggered no matter if the code is compiled with `-DNDEBUG` or not.
#define KU_ASSERT(cond, ...)                                                                                      \
    do {                                                                                                  \
        if (not (cond)) {                                                                                 \
            fprintf(stderr, "Assertion `%s` failed (%s:%d): ", #cond, __FILE__, __LINE__);          \
            if constexpr (sizeof(#__VA_ARGS__) > 1) {                                                \
                fprintf(stderr, ", " __VA_ARGS__);                                                        \
            }                                                                                             \
            fprintf(stderr, "\n");                                                                       \
            THROW_KU_EXCEPTION("Assertion", "Assertion `", #cond, "` failed.");                          \
        }                                                                                                 \
    } while(0)

#define KU_CHECK_KERNEL_LAUNCH() KU_CUDA_CHECK(cudaGetLastError())

template<typename T>
inline __host__ __device__ constexpr T ceil_div(const T &a, const T &b) {
    return (a + b - 1) / b;
}

template<typename T>
inline __host__ __device__ constexpr T ceil(const T &a, const T &b) {
    return (a + b - 1) / b * b;
}

}  // namespace kerutils
