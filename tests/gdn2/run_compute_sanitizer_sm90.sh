#!/usr/bin/env bash
# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 OUTPUT_ROOT [TOTAL_PRODUCT_LAUNCHES_PER_TOOL]" >&2
  exit 2
fi

output_root=$1
iterations=${2:-120}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
python=${PYTHON:-python3}
source_manifest=${GDN2_SOURCE_MANIFEST:-}
required_gpu_uuid=${GDN2_REQUIRED_GPU_UUID:-}

if [[ ! $iterations =~ ^[1-9][0-9]*$ ]]; then
  echo "TOTAL_PRODUCT_LAUNCHES_PER_TOOL must be a positive integer" >&2
  exit 2
fi
if (( iterations < 120 )); then
  echo "formal sanitizer coverage requires at least 120 product launches per tool" >&2
  exit 2
fi
if [[ -e $output_root ]]; then
  echo "fresh OUTPUT_ROOT required: $output_root" >&2
  exit 2
fi
if ! command -v compute-sanitizer >/dev/null 2>&1; then
  echo "compute-sanitizer is required" >&2
  exit 2
fi

mkdir -p "$output_root"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUTE_DSL_ARCH=sm_90a
export CUTE_DSL_KEEP=1
export CUTE_DSL_NO_CACHE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTHONPATH=$repo_root
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

for tool in memcheck initcheck synccheck racecheck; do
  tool_root=$output_root/$tool
  cache_root=$output_root/cache/$tool
  mkdir -p \
    "$tool_root" \
    "$cache_root/cuda" \
    "$cache_root/cute" \
    "$cache_root/torchinductor" \
    "$cache_root/triton" \
    "$cache_root/xdg"

  export CUDA_CACHE_PATH=$cache_root/cuda
  export CUTE_DSL_CACHE_DIR=$cache_root/cute
  export TORCHINDUCTOR_CACHE_DIR=$cache_root/torchinductor
  export TRITON_CACHE_DIR=$cache_root/triton
  export XDG_CACHE_HOME=$cache_root/xdg

  command=(
    compute-sanitizer
    --tool "$tool"
    --target-processes all
    --error-exitcode 86
  )
  case $tool in
    memcheck)
      command+=(--leak-check full)
      ;;
    synccheck)
      command+=(--check-warpgroup-mma yes)
      ;;
    racecheck)
      command+=(
        --racecheck-report hazard
        --racecheck-memcpy-async yes
        --racecheck-trace-sync yes
      )
      ;;
  esac
  command+=(
    "$python"
    "$script_dir/stress_gdn2_sm90.py"
    --iterations "$iterations"
    --warmup 1
    --device 0
    --progress-every 0
    --source-root "$repo_root"
    --output "$tool_root/result.json"
  )
  if [[ -n $source_manifest ]]; then
    command+=(--source-manifest "$source_manifest")
  fi
  if [[ -n $required_gpu_uuid ]]; then
    command+=(--required-gpu-uuid "$required_gpu_uuid")
  fi

  printf '%q ' "${command[@]}" >"$tool_root/command.txt"
  printf '\n' >>"$tool_root/command.txt"
  set +e
  "${command[@]}" \
    >"$tool_root/stdout.log" \
    2>"$tool_root/stderr.log"
  return_code=$?
  set -e
  printf '%s\n' "$return_code" >"$tool_root/returncode.txt"
  if [[ $return_code -ne 0 ]]; then
    echo "$tool failed with return code $return_code" >&2
    exit "$return_code"
  fi

  if [[ $tool == racecheck ]]; then
    zero_summary="RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
  else
    zero_summary="ERROR SUMMARY: 0 errors"
  fi
  if ! grep -Fq "$zero_summary" "$tool_root/stdout.log"; then
    echo "$tool did not report the required zero-error summary" >&2
    exit 87
  fi
  "$python" - "$tool_root/result.json" "$iterations" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
minimum = int(sys.argv[2])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload["status"] != "PASS":
    raise SystemExit(f"stress receipt did not pass: {path}")
if payload["product_launches"] < minimum:
    raise SystemExit(
        f"incomplete product launch count: "
        f"{payload['product_launches']} < {minimum}",
    )
if payload["protocol"]["matrix_rows"] != 6:
    raise SystemExit("sanitizer stress matrix must contain six rows")
PY
done

printf '%s\n' PASS >"$output_root/DONE"
find "$output_root" -type f ! -name evidence-manifest.sha256 -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  >"$output_root/evidence-manifest.sha256"
printf 'GDN2_SM90_SANITIZERS_PASS tools=4 launches_per_tool=%s output=%s\n' \
  "$iterations" \
  "$output_root"
