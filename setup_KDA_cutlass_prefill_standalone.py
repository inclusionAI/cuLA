import os
from pathlib import Path

cuda_home = os.environ.get("CUDA_HOME", "")
nvcc = os.path.join(cuda_home, "bin", "nvcc") if cuda_home else ""
if not cuda_home or not os.path.isfile(nvcc):
    for candidate in ("/usr/local/cuda", "/usr/local/cuda-12.2", "/usr/local/cuda-12.8"):
        trial = os.path.join(candidate, "bin", "nvcc")
        if os.path.isfile(trial):
            os.environ["CUDA_HOME"] = candidate
            break

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
INCLUDE = ROOT / "include"

nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "-gencode=arch=compute_89,code=sm_89",
]

setup(
    name="kda_prefill_cuda",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="kda_prefill_cuda",
            sources=[
                str(ROOT / "src" / "kda_prefill_binding.cpp"),
                str(ROOT / "src" / "kda.cu"),
            ],
            include_dirs=[str(INCLUDE)],
            extra_compile_args={"cxx": ["-O3"], "nvcc": nvcc_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
