import os
from pathlib import Path
import subprocess

from setuptools import setup, find_packages

from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    IS_WINDOWS,
    CUDA_HOME,
)


def is_flag_set(flag: str) -> bool:
    return os.getenv(flag, "FALSE").lower() in ["true", "1", "y", "yes"]


def get_features_args():
    features_args = []
    return features_args


def get_arch_flags():
    # Check NVCC Version
    # NOTE The "CUDA_HOME" here is not necessarily from the `CUDA_HOME` environment variable. For more details, see `torch/utils/cpp_extension.py`
    assert CUDA_HOME is not None, "PyTorch must be compiled with CUDA support"
    nvcc_version = subprocess.check_output(
        [os.path.join(CUDA_HOME, "bin", "nvcc"), "--version"], stderr=subprocess.STDOUT
    ).decode("utf-8")
    nvcc_version_number = nvcc_version.split("release ")[1].split(",")[0].strip()
    major, minor = map(int, nvcc_version_number.split("."))
    print(f"Compiling using NVCC {major}.{minor}")

    DISABLE_SM100 = is_flag_set("FLASHLA_DISABLE_SM100")
    DISABLE_SM90 = is_flag_set("FLASHLA_DISABLE_SM90")
    if major < 12 or (major == 12 and minor <= 8):
        assert (
            DISABLE_SM100
        ), "sm100 compilation requires NVCC 12.9 or higher. Please set FLASHLA_DISABLE_SM100=1 to disable sm100 compilation, or update your environment."

    arch_flags = []
    if not DISABLE_SM100:
        arch_flags.extend(["-gencode", "arch=compute_100a,code=sm_100a"])
    # if not DISABLE_SM90:
    #     arch_flags.extend(["-gencode", "arch=compute_90a,code=sm_90a"])
    return arch_flags


def get_nvcc_thread_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "32"
    return ["--threads", nvcc_threads]


subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"])

this_dir = os.path.dirname(os.path.abspath(__file__))

if IS_WINDOWS:
    cxx_args = ["/O2", "/std:c++17", "/DNDEBUG", "/W0"]
else:
    cxx_args = ["-O3", "-std=c++17", "-DNDEBUG", "-Wno-deprecated-declarations"]

ext_modules = []
ext_modules.append(
    CUDAExtension(
        name="flashla.cudac",
        sources=[
            "csrc/pybind.cu",
            "csrc/kda_api.cu",
            "csrc/kda/kda_fwd_intra_sm100.cu",
        ],
        extra_compile_args={
            "cxx": cxx_args + get_features_args(),
            "nvcc": [
                "-O3",
                "-std=c++17",
                # "-DNDEBUG",
                # "-D_USE_MATH_DEFINES",
                "-Wno-deprecated-declarations",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
                "-lineinfo",
                "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage",
            ]
            + get_features_args()
            + get_arch_flags()
            + get_nvcc_thread_args(),
        },
        include_dirs=[
            Path(this_dir) / "csrc",
            Path(this_dir) / "csrc" / "cutlass" / "include",
            Path(this_dir) / "csrc" / "cutlass" / "tools" / "util" / "include",
            "/usr/local/cuda/include/cccl",
        ],
    )
)

setup(
    name="flashla",
    use_scm_version={
        "write_to": "flashla/_version.py",
        "local_scheme": "node-and-date",
    },
    packages=find_packages(include=["flashla"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
