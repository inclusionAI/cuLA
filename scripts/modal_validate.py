"""Modal validation harness for cuLA on H100 (modal client v1.4.2 API).

Clones the fork branch inside the container (this client ships no Mount),
builds cuLA, runs the compat + SM90 kernel JIT tests, returns a JSON summary.
"""

from __future__ import annotations

import json
import os
import subprocess
import time

import modal

FORK = "https://github.com/bikrammajhi/cuLA.git"
BRANCH = "mlir-compat-gateway"
CUDA_TAG = "cu129"
TORCH_VERSION = "2.9.1"
TESTS = [
    "tests/test_cutedsl_compat.py",
    "tests/test_lightning_attn_prefill_sm90.py",
    "tests/test_lightning_decode.py",
]

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .pip_install("wheel", "setuptools", "setuptools-scm")
    .pip_install(
        f"torch=={TORCH_VERSION}",
        index_url=f"https://download.pytorch.org/whl/{CUDA_TAG}",
    )
    .pip_install("nvidia-cutlass-dsl>=4.4.2,<4.7,!=4.5.0", "flash-linear-attention", "pytest")
)

app = modal.App("cula-validate-v4")


@app.function(image=image, gpu="h100", timeout=60 * 60)
def validate() -> str:
    start = time.time()
    summary = {"branch": BRANCH, "steps": {}}

    def step(name: str) -> None:
        summary["steps"][name] = "ok"

    subprocess.run(
        f"git clone --depth 1 --branch {BRANCH} {FORK} /work",
        shell=True,
        check=True,
    )
    step("clone")

    check = subprocess.run(
        "ls tests/ | head -25; echo ---; git log --oneline -3",
        shell=True,
        cwd="/work",
        capture_output=True,
        text=True,
    )
    summary["clone_check"] = check.stdout + check.stderr

    toolchain = {**os.environ, "CC": "gcc", "CXX": "g++", "CUDAHOSTCXX": "g++"}
    subprocess.run(
        "pip install -e /work --no-build-isolation",
        shell=True,
        env=toolchain,
        check=True,
    )
    step("build")

    probe = subprocess.run(
        "python -c 'import torch, cutlass, cula; "
        "print(torch.__version__, cutlass.__version__, torch.cuda.get_device_name(0))'",
        shell=True,
        capture_output=True,
        text=True,
    )
    summary["env"] = probe.stdout.strip()
    step("probe")

    run = subprocess.run(
        ["python", "-m", "pytest", *TESTS, "-v", "-m", "not sm100_only"],
        cwd="/work",
        capture_output=True,
        text=True,
    )
    summary["pytest_cmd"] = "pytest " + " ".join(TESTS) + " -v -m 'not sm100_only' (cwd=/work)"
    summary["pytest_rc"] = run.returncode
    summary["pytest_tail"] = (run.stdout + run.stderr)[-5000:]
    if run.returncode != 0:
        raise RuntimeError(
            f"pytest failed with exit code {run.returncode}; summary above captures "
            "the tail. See the 'pytest_tail' entry for the failing tests."
        )
    step("pytest")

    summary["elapsed_s"] = int(time.time() - start)
    return json.dumps(summary, indent=1)


@app.local_entrypoint()
def main() -> None:
    print(validate.remote())
