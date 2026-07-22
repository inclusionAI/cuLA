#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${PYTHON:-${repo_root}/.venv/bin/python}"
repeats="${REPEATS:-5}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-2}"
export NVSHMEM_DISABLE_CUDA_VMM="${NVSHMEM_DISABLE_CUDA_VMM:-1}"
export NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SYMMETRIC_SIZE:-256M}"
export NVSHMEM_IB_ENABLE="${NVSHMEM_IB_ENABLE:-0}"
export CULA_CP_ALLOW_FALLBACK=0
export CULA_CP_COMM_USE_CURRENT_STREAM=1
export CULA_CP_NVSHMEM_DIRECT_STORE_CONN1_ONLY=0
export CULA_CP_NVSHMEM_FUSED_REMOTE_MERGE=1
export CULA_CP_NVSHMEM_READY_WAIT=0

run_dist() {
    "${python}" -m torch.distributed.run --standalone --nproc_per_node=4 "$@"
}

cd "${repo_root}"

run_dist benchmarks/check_cp_predecessor_nvshmem.py \
    --world-size 4 --sequence-length 8192 --heads 16

run_benchmark() {
    local backend="$1"
    local sequence_length="$2"
    local heads="$3"
    run_dist benchmarks/bench_cp_predecessor_nvshmem.py \
        --world-size 4 \
        --backend "${backend}" \
        --sequence-length "${sequence_length}" \
        --heads "${heads}" \
        --warmup 20 \
        --iterations 60 \
        --repeats "${repeats}"
}

# Rotate backend order between shapes to reduce order bias.
run_benchmark nvshmem 4096 8
run_benchmark fla 4096 8
run_benchmark fla 8192 16
run_benchmark nvshmem 8192 16
