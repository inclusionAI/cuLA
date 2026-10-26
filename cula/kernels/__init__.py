# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Framework-independent public surface for cuLA's CuTeDSL kernels.

The submodules use lazy exports so importing :mod:`cula.kernels` does not
compile kernels or import an external linear-attention framework.
"""

from cula.kernels.registry import KernelSpec, get_kernel, list_kernels

__all__ = ["KernelSpec", "get_kernel", "list_kernels"]
