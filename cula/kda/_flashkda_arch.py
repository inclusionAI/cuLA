# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Architecture checks for the SM90-derived FlashKDA implementation."""

from __future__ import annotations

import torch

_SUPPORTED_COMPUTE_CAPABILITIES = frozenset({(9, 0), (10, 0)})


def is_flashkda_supported(device: torch.device) -> bool:
    """Return whether FlashKDA is supported on *device*.

    The CuTeDSL kernels are implemented with the SM90 FlashKDA pipeline, which
    is also executable on SM100 Blackwell GPUs. SM103 is intentionally not
    enabled until it has equivalent hardware validation.
    """
    return device.type == "cuda" and torch.cuda.get_device_capability(device) in _SUPPORTED_COMPUTE_CAPABILITIES


def assert_flashkda_supported(device: torch.device) -> None:
    """Raise unless *device* is an SM90 or SM100 CUDA device."""
    if device.type != "cuda":
        raise RuntimeError(f"FlashKDA requires a CUDA device, got {device}.")
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) not in _SUPPORTED_COMPUTE_CAPABILITIES:
        raise RuntimeError(
            f"FlashKDA requires an SM90 (Hopper) or SM100 (Blackwell) device, got compute capability sm_{major}{minor}."
        )
