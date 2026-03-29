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


def detect_gpu_archs() -> tuple[set[int], set[int]]:
    """
    Query all visible CUDA devices via torch and return two sets:
      - sm100_majors: major version numbers == 10 with minor == 0  (sm100)
      - sm90_majors:  major version numbers == 9  with minor == 0  (sm90a)
    Returns (has_sm100, has_sm90) as booleans wrapped in a tuple.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False, False
        has_sm100 = False
        has_sm90 = False
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            major, minor = prop.major, prop.minor
            print(f"  GPU {i}: {prop.name}, compute capability sm_{major}{minor}")
            if major == 10 and minor == 0:
                has_sm100 = True
            if major == 9 and minor == 0:
                has_sm90 = True
        return has_sm100, has_sm90
    except Exception as e:
        print(f"Warning: failed to detect GPU architectures via torch: {e}")
        return False, False


def resolve_disable_flag(env_name: str, detected: bool) -> bool:
    """
    Resolve whether to disable a given SM target.
    - If the environment variable is explicitly set, honour it.
    - Otherwise, disable the target when no matching GPU is detected.
    """
    env_val = os.getenv(env_name)
    if env_val is not None:
        return env_val.lower() in ["true", "1", "y", "yes"]
    # Auto-detect: disable if no matching device found
    disable = not detected
    if disable:
        print(f"  No matching GPU detected; auto-setting {env_name}=1 (disable). Set {env_name}=0 to override.")
    return disable


def get_features_args():
    features_args = []
    return features_args


USE_FAST_MATH = os.getenv("CULA_USE_FAST_MATH", "1") == "1"

print("Detecting GPU architectures...")
_has_sm100, _has_sm90 = detect_gpu_archs()
DISABLE_SM100 = resolve_disable_flag("CULA_DISABLE_SM100", _has_sm100)
DISABLE_SM90 = resolve_disable_flag("CULA_DISABLE_SM90", _has_sm90)


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

    if major < 12 or (major == 12 and minor <= 8):
        assert DISABLE_SM100, (
            "sm100 compilation requires NVCC 12.9 or higher. Please set CULA_DISABLE_SM100=1 to disable sm100 compilation, or update your environment."
        )

    arch_flags = []
    if not DISABLE_SM100:
        arch_flags.extend(["-gencode", "arch=compute_100a,code=sm_100a"])
        arch_flags.extend(["-DCULA_SM100_ENABLED"])
    if not DISABLE_SM90:
        arch_flags.extend(["-gencode", "arch=compute_90a,code=sm_90a"])
        arch_flags.extend(["-DCULA_SM90A_ENABLED"])
    return arch_flags


def get_nvcc_thread_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "32"
    return ["--threads", nvcc_threads]


subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"])

this_dir = os.path.dirname(os.path.abspath(__file__))

if IS_WINDOWS:
    cxx_args = ["/O2", "/std:c++20", "/DNDEBUG", "/W0"]
else:
    cxx_args = ["-O3", "-std=c++20", "-DNDEBUG", "-Wno-deprecated-declarations"]

cuda_sources = [
    "csrc/api/pybind.cu",
]
if not DISABLE_SM100:
    cuda_sources.extend(
        [
            "csrc/api/kda_sm100.cu",
            "csrc/kda/sm100/kda_fwd_sm100.cu",
        ]
    )
if not DISABLE_SM90:
    cuda_sources.extend(
        [
            "csrc/api/kda_sm90.cu",
            "csrc/kda/sm90/kda_fwd_sm90.cu",
            "csrc/kda/sm90/kda_fwd_sm90_safe_gate.cu",
        ]
    )

ext_modules = []
ext_modules.append(
    CUDAExtension(
        name="cula.cudac",
        sources=cuda_sources,
        extra_compile_args={
            "cxx": cxx_args + get_features_args(),
            "nvcc": [
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
                "-diag-suppress=3189",  # suppress the warning of torch in C++ 20
            ]
            + get_features_args()
            + get_arch_flags()
            + get_nvcc_thread_args()
            + (["--use_fast_math"] if USE_FAST_MATH else []),
        },
        include_dirs=[
            Path(this_dir) / "csrc",
            Path(this_dir) / "csrc" / "kerutils" / "include",
            Path(this_dir) / "csrc" / "cutlass" / "include",
            Path(this_dir) / "csrc" / "cutlass" / "tools" / "util" / "include",
            "/usr/local/cuda/include/cccl",
        ],
    )
)

setup(
    name="cula",
    use_scm_version={
        "write_to": "cula/_version.py",
        "local_scheme": "node-and-date",
    },
    packages=find_packages(include=["cula", "cula.*"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
