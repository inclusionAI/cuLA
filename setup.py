import os
import subprocess
import tarfile
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    IS_WINDOWS,
    BuildExtension,
    CUDAExtension,
)

# CUTLASS configuration
# Download URL can be customized via CULA_CUTLASS_URL environment variable
# Default: GitHub official repository
CUTLASS_VERSION = "v4.4.0"
CUTLASS_COMMIT = "c213bfdfc1f4ffb156c69d51d07efcf7f367f2fb"  # v4.4.0 tag
CUTLASS_DIR = Path(__file__).parent / "csrc" / "cutlass"

# Default download URL (GitHub) - supports commit hash
_DEFAULT_CUTLASS_URL = f"https://github.com/NVIDIA/cutlass/archive/{CUTLASS_COMMIT}.tar.gz"

# Customizable via environment variable
CUTLASS_URL = os.getenv("CULA_CUTLASS_URL", _DEFAULT_CUTLASS_URL)

# Optional: MD5 checksum for verification (set via env var)
CUTLASS_MD5 = os.getenv("CULA_CUTLASS_MD5")


def _verify_md5(file_path: Path, expected_md5: str) -> bool:
    """Verify file MD5 checksum."""
    import hashlib

    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5_hash.update(chunk)
    actual_md5 = md5_hash.hexdigest()
    if actual_md5 != expected_md5:
        print("  Warning: MD5 mismatch!")
        print(f"    Expected: {expected_md5}")
        print(f"    Actual:   {actual_md5}")
        return False
    return True


def download_cutlass():
    """Download and extract CUTLASS from archive if not present."""
    if CUTLASS_DIR.exists() and (CUTLASS_DIR / "include" / "cutlass").exists():
        return

    print("CUTLASS not found, downloading...")
    print(f"  URL: {CUTLASS_URL}")

    # Determine archive format from URL
    if CUTLASS_URL.endswith(".zip") or "gitee.com" in CUTLASS_URL:
        archive_path = CUTLASS_DIR.parent / "cutlass.zip"
        use_zip = True
    else:
        archive_path = CUTLASS_DIR.parent / "cutlass.tar.gz"
        use_zip = False

    # Download archive (follow redirects with curl)
    print("  Downloading...")
    subprocess.run(
        ["curl", "-L", "-s", "-o", str(archive_path), CUTLASS_URL],
        check=True,
        capture_output=True,
    )
    print(f"  Downloaded: {archive_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Verify MD5 if provided
    if CUTLASS_MD5:
        print("  Verifying MD5...")
        if _verify_md5(archive_path, CUTLASS_MD5):
            print("  MD5 verification passed")
        else:
            raise RuntimeError(f"MD5 verification failed for {CUTLASS_URL}")

    # Extract archive
    temp_extract_dir = CUTLASS_DIR.parent / "cutlass_extract"
    print("  Extracting...")
    if use_zip:
        import zipfile

        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(temp_extract_dir)
    else:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(temp_extract_dir)

    # Move extracted dir to target location
    # Try various possible directory names (GitHub: cutlass-{commit}, Gitee: cutlass-{version})
    possible_names = [
        temp_extract_dir / f"cutlass-{CUTLASS_COMMIT}",
        temp_extract_dir / f"cutlass-{CUTLASS_VERSION}",
        temp_extract_dir / "cutlass",
    ]
    extracted_dir = None
    for name in possible_names:
        if name.exists():
            extracted_dir = name
            break

    if extracted_dir is None:
        # Fallback: find any directory starting with 'cutlass'
        candidates = list(temp_extract_dir.glob("cutlass*"))
        if candidates:
            extracted_dir = candidates[0]
        else:
            raise RuntimeError(f"Cannot find extracted cutlass directory in {temp_extract_dir}")

    if CUTLASS_DIR.exists():
        import shutil

        shutil.rmtree(CUTLASS_DIR)
    extracted_dir.rename(CUTLASS_DIR)

    # Cleanup
    archive_path.unlink()
    temp_extract_dir.rmdir()

    print(f"CUTLASS {CUTLASS_VERSION} installed at {CUTLASS_DIR}")


def detect_gpu_archs() -> tuple[bool, bool, bool, bool]:
    """
    Query all visible CUDA devices via torch and return:
      (has_sm100, has_sm103, has_sm90, cuda_available)

    When cuda_available is False (no GPU visible), the caller should
    not auto-disable architectures — let the user control via env vars
    and the NVCC version check handle compatibility.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            print("  No CUDA devices visible — skipping GPU detection.")
            print("  Use CULA_DISABLE_SM100/SM103/SM90=0/1 to control targets.")
            return False, False, False, False
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
        return has_sm100, has_sm103, has_sm90, True
    except Exception as e:
        print(f"Warning: failed to detect GPU architectures via torch: {e}")
        return False, False, False, False


def _env_disable_flag(env_name: str) -> bool | None:
    """Parse a CULA_DISABLE_* env var. Returns None if not set."""
    val = os.getenv(env_name)
    if val is None:
        return None
    return val.lower() in ["true", "1", "y", "yes"]


def get_features_args():
    features_args = []
    return features_args


USE_FAST_MATH = os.getenv("CULA_USE_FAST_MATH", "1") == "1"

# ---------------------------------------------------------------------------
# Resolve SM disable flags: env vars take priority; fall back to GPU detection
# ---------------------------------------------------------------------------
_DISABLE_SM100_ENV = _env_disable_flag("CULA_DISABLE_SM100")
_DISABLE_SM103_ENV = _env_disable_flag("CULA_DISABLE_SM103")
_DISABLE_SM90_ENV = _env_disable_flag("CULA_DISABLE_SM90")

if _DISABLE_SM100_ENV is not None or _DISABLE_SM103_ENV is not None or _DISABLE_SM90_ENV is not None:
    # At least one env var is set — use env vars directly, skip GPU detection
    DISABLE_SM100 = bool(_DISABLE_SM100_ENV)
    DISABLE_SM103 = bool(_DISABLE_SM103_ENV)
    DISABLE_SM90 = bool(_DISABLE_SM90_ENV)
    print("GPU detection skipped (CULA_DISABLE_SM* env vars set).")
else:
    # No env vars — detect GPU and auto-disable unmatched archs
    print("Detecting GPU architectures...")
    _has_sm100, _has_sm103, _has_sm90, _cuda_available = detect_gpu_archs()
    if _cuda_available:
        DISABLE_SM100 = not _has_sm100
        DISABLE_SM103 = not _has_sm103
        DISABLE_SM90 = not _has_sm90
        if DISABLE_SM100:
            print("  No SM100 GPU detected; disabling sm100. Set CULA_DISABLE_SM100=0 to override.")
        if DISABLE_SM103:
            print("  No SM103 GPU detected; disabling sm103. Set CULA_DISABLE_SM103=0 to override.")
        if DISABLE_SM90:
            print("  No SM90 GPU detected; disabling sm90. Set CULA_DISABLE_SM90=0 to override.")
    else:
        # No GPU visible — enable all targets (cross-compilation mode)
        DISABLE_SM100 = False
        DISABLE_SM103 = False
        DISABLE_SM90 = False

# ---------------------------------------------------------------------------
# NVCC version check — auto-disable Blackwell targets if NVCC is too old
# ---------------------------------------------------------------------------
assert CUDA_HOME is not None, "PyTorch must be compiled with CUDA support"
_nvcc_out = subprocess.check_output([os.path.join(CUDA_HOME, "bin", "nvcc"), "--version"], stderr=subprocess.STDOUT).decode(
    "utf-8"
)
_nvcc_ver = _nvcc_out.split("release ")[1].split(",")[0].strip()
NVCC_MAJOR, NVCC_MINOR = map(int, _nvcc_ver.split("."))
print(f"Compiling using NVCC {NVCC_MAJOR}.{NVCC_MINOR}")

if NVCC_MAJOR < 12 or (NVCC_MAJOR == 12 and NVCC_MINOR <= 8):
    if not DISABLE_SM100:
        print(f"  NVCC {NVCC_MAJOR}.{NVCC_MINOR} does not support sm100 (requires 12.9+). Auto-disabling sm100.")
        DISABLE_SM100 = True
    if not DISABLE_SM103:
        print(f"  NVCC {NVCC_MAJOR}.{NVCC_MINOR} does not support sm103 (requires 12.9+). Auto-disabling sm103.")
        DISABLE_SM103 = True

# ---------------------------------------------------------------------------
# Patch PyTorch arch detection to avoid crash when no GPU is visible
# ---------------------------------------------------------------------------
if not os.getenv("TORCH_CUDA_ARCH_LIST"):
    import torch.utils.cpp_extension

    _orig_get_cuda_arch_flags = torch.utils.cpp_extension._get_cuda_arch_flags

    def _patched_get_cuda_arch_flags(cflags):
        try:
            return _orig_get_cuda_arch_flags(cflags)
        except IndexError:
            # No GPU visible and TORCH_CUDA_ARCH_LIST not set —
            # PyTorch's auto-detection returns an empty list and crashes.
            # Our own -gencode flags in extra_compile_args handle arch selection.
            return []

    torch.utils.cpp_extension._get_cuda_arch_flags = _patched_get_cuda_arch_flags


def get_arch_flags():
    arch_flags = []
    if not DISABLE_SM100:
        arch_flags.extend(["-gencode", "arch=compute_100a,code=sm_100a"])
        arch_flags.extend(["-DCULA_SM100_ENABLED"])
    if not DISABLE_SM103:
        arch_flags.extend(["-gencode", "arch=compute_103a,code=sm_103a"])
        arch_flags.extend(["-DCULA_SM103_ENABLED"])
    if not DISABLE_SM90:
        arch_flags.extend(["-gencode", "arch=compute_90a,code=sm_90a"])
        arch_flags.extend(["-DCULA_SM90A_ENABLED"])
    return arch_flags


def get_nvcc_thread_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "32"
    return ["--threads", nvcc_threads]


download_cutlass()

this_dir = os.path.dirname(os.path.abspath(__file__))

if IS_WINDOWS:
    cxx_args = ["/O2", "/std=c++20", "/DNDEBUG", "/W0"]
else:
    cxx_args = ["-O3", "-std=c++20", "-DNDEBUG", "-Wno-deprecated-declarations"]

cuda_sources = [
    "csrc/api/pybind.cu",
]
if not DISABLE_SM100 or not DISABLE_SM103:
    cuda_sources.extend(
        [
            "csrc/api/kda_sm100.cu",
            "csrc/api/kda_bwd_sm100.cu",
            "csrc/kda/sm100/kda_fwd_sm100.cu",
            "csrc/kda_bwd/sm100/kda_bwd_intra_sm100.cu",
            "csrc/kda_bwd/sm100/kda_bwd_wy_dqkg_fused_sm100.cu",
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
        ],
    )
)

CUDA_VERSION_TAG = f"cu{NVCC_MAJOR}{NVCC_MINOR}"

_long_description = ""
_readme_path = Path(this_dir) / "README.md"
if _readme_path.exists():
    _long_description = _readme_path.read_text(encoding="utf-8")

setup(
    name=f"ant-cula-{CUDA_VERSION_TAG}",
    description="cuLA CUDA extension",
    long_description=_long_description,
    long_description_content_type="text/markdown",
    author="cula contributors",
    license="Apache-2.0",
    python_requires=">=3.10",
    install_requires=[
        "nvidia-cutlass-dsl==4.4.2",
        "apache-tvm-ffi==0.1.9",
    ],
    extras_require={
        "dev": [
            "build",
            "twine",
        ],
    },
    packages=find_packages(include=["cula", "cula.*"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    use_scm_version={
        "write_to": "cula/_version.py",
        "local_scheme": "node-and-date",
        "fallback_version": "0.1.0",
    },
)
