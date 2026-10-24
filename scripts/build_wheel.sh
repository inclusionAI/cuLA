#!/usr/bin/env bash
# Build a wheel package for cuLA.
#
# Usage:
#   bash scripts/build_wheel.sh            # default: --no-isolation
#   bash scripts/build_wheel.sh --isolated # use isolated build environment
#
# Prerequisites:
#   pip install build wheel setuptools setuptools_scm
#
# The built wheel will be placed under dist/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Parse args
ISOLATION_FLAG="--no-isolation"
for arg in "$@"; do
    case "$arg" in
        --isolated)
            ISOLATION_FLAG=""
            echo "[build_wheel] Using isolated build environment"
            ;;
        *)
            echo "[build_wheel] Unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done
if [[ "$ISOLATION_FLAG" == "--no-isolation" ]]; then
    echo "[build_wheel] Using current environment (--no-isolation)"
fi

# Clean previous artifacts
echo "[build_wheel] Cleaning previous build artifacts..."
rm -rf dist build *.egg-info

# Show environment info
echo "[build_wheel] Python: $(python -V 2>&1)"
echo "[build_wheel] Packaging: pure Python (CuTeDSL kernels compile at runtime)"

# Build wheel
echo "[build_wheel] Building wheel..."
python -m build --wheel $ISOLATION_FLAG

# Show result
echo ""
echo "[build_wheel] Done. Wheel:"
ls -lh dist/*.whl
