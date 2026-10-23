import os
import subprocess
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    IS_WINDOWS,
    BuildExtension,
    CUDAExtension,
)


def detect_gpu_archs() -> tuple[bool, bool, bool]:
    """
    Query all visible CUDA devices via torch and return three booleans:
      - has_sm100: major version numbers == 10 with minor == 0  (sm100)
      - has_sm103: major version numbers == 10 with minor == 3  (sm103)
      - has_sm90:  major version numbers == 9  with minor == 0  (sm90a)
    Returns (has_sm100, has_sm103, has_sm90)
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False, False, False
        has_sm100 = False
        has_sm103 = False
        has_sm90 = False
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            major, minor = prop.major, prop.minor
            print(f"  GPU {i}: {prop.name}, compute capability sm_{major}{minor}")
            if major == 10 and minor == 0:
                has_sm100 = True
            if major == 10 and minor == 3:
                has_sm103 = True
            if major == 9 and minor == 0:
                has_sm90 = True
        return has_sm100, has_sm103, has_sm90
    except Exception as e:
        print(f"Warning: failed to detect GPU architectures via torch: {e}")
        return False, False, False


def resolve_disable_flag(env_name: str, detected: bool) -> bool:
    """
    Resolve whether to disable a given SM target.
    - If CULA_BUILD_ALL_ARCHS is set, all targets are enabled unconditionally.
    - If the environment variable is explicitly set, honour it.
    - Otherwise, disable the target when no matching GPU is detected.
    """
    if os.getenv("CULA_BUILD_ALL_ARCHS", "0") == "1":
        return False
    env_val = os.getenv(env_name)
    if env_val is not None:
        return env_val.lower() in ["true", "1", "y", "yes"]
    disable = not detected
    if disable:
        print(f"  No matching GPU detected; auto-setting {env_name}=1 (disable). Set {env_name}=0 to override.")
    return disable


def get_features_args():
    features_args = []
    return features_args


USE_FAST_MATH = os.getenv("CULA_USE_FAST_MATH", "1") == "1"

if os.getenv("CULA_BUILD_ALL_ARCHS", "0") == "1":
    print("CULA_BUILD_ALL_ARCHS=1: enabling the SM90 native extension; SM100/SM103 use CuTeDSL JIT")
else:
    print("Detecting GPU architectures...")

_has_sm100, _has_sm103, _has_sm90 = detect_gpu_archs()
DISABLE_SM90 = resolve_disable_flag("CULA_DISABLE_SM90", _has_sm90)


def get_nvcc_version() -> tuple[int, int]:
    """Return the NVCC (major, minor) version tuple, e.g. ``(12, 9)``."""
    # NOTE The "CUDA_HOME" here is not necessarily from the `CUDA_HOME` environment variable.
    # For more details, see `torch/utils/cpp_extension.py`
    assert CUDA_HOME is not None, "PyTorch must be compiled with CUDA support"
    nvcc_version = subprocess.check_output(
        [os.path.join(CUDA_HOME, "bin", "nvcc"), "--version"], stderr=subprocess.STDOUT
    ).decode("utf-8")
    nvcc_version_number = nvcc_version.split("release ")[1].split(",")[0].strip()
    major, minor = map(int, nvcc_version_number.split("."))
    return major, minor


def get_nvcc_thread_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "32"
    return ["--threads", nvcc_threads]


this_dir = os.path.dirname(os.path.abspath(__file__))

if IS_WINDOWS:
    cxx_args = ["/O2", "/std:c++20", "/DNDEBUG", "/W0"]
else:
    cxx_args = ["-O3", "-std=c++20", "-DNDEBUG", "-Wno-deprecated-declarations"]

nvcc_common_args = [
    "-O3",
    "-std=c++20",
    "-DNDEBUG",
    # "-D_USE_MATH_DEFINES",
    "-Wno-deprecated-declarations",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-lineinfo",
    "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage",
    "-diag-suppress=3189",
]

include_dirs = [
    Path(this_dir) / "csrc",
    Path(this_dir) / "csrc" / "kerutils" / "include",
    Path(this_dir) / "csrc" / "cutlass" / "include",
    Path(this_dir) / "csrc" / "cutlass" / "tools" / "util" / "include",
]

ext_modules = []

if not DISABLE_SM90:
    subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"], check=True)
    major, minor = get_nvcc_version()
    print(f"Compiling the SM90 extension using NVCC {major}.{minor}")
    sm90_arch_flags = ["-gencode", "arch=compute_90a,code=sm_90a", "-DCULA_SM90A_ENABLED"]

    ext_modules.append(
        CUDAExtension(
            name="cula._cudac_sm90",
            sources=[
                "csrc/api/kda_sm90.cu",
                "csrc/api/pybind_sm90.cu",
                "csrc/kda/sm90/kda_fwd_sm90.cu",
                "csrc/kda/sm90/kda_fwd_sm90_safe_gate.cu",
            ],
            extra_compile_args={
                "cxx": cxx_args + get_features_args(),
                "nvcc": nvcc_common_args
                + get_features_args()
                + sm90_arch_flags
                + get_nvcc_thread_args()
                + (["--use_fast_math"] if USE_FAST_MATH else []),
            },
            include_dirs=include_dirs,
        )
    )

setup(
    name="cuda-linear-attention",
    packages=find_packages(include=["cula", "cula.*"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
