# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from cula.kda._flashkda_arch import assert_flashkda_supported, is_flashkda_supported


@pytest.mark.parametrize(
    ("capability", "supported"),
    [
        ((9, 0), True),
        ((10, 0), True),
        ((8, 0), False),
        ((10, 3), False),
    ],
)
def test_flashkda_supported_compute_capabilities(monkeypatch, capability, supported):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    device = torch.device("cuda")

    assert is_flashkda_supported(device) is supported
    if supported:
        assert_flashkda_supported(device)
    else:
        with pytest.raises(RuntimeError, match="SM90.*SM100"):
            assert_flashkda_supported(device)


def test_flashkda_rejects_cpu():
    device = torch.device("cpu")
    assert not is_flashkda_supported(device)
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        assert_flashkda_supported(device)
