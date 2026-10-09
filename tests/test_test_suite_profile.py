# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.conftest import _prepare_markexpr


@pytest.mark.parametrize(
    ("expression", "expanded", "include_cula_slow", "include_kda_slow"),
    [
        ("", "", False, False),
        ("cula_fast", "cula_fast", False, False),
        ("cula_slow", "(cula_slow or kda_slow)", True, True),
        (
            "cula_full",
            "((cula_slow or kda_slow) or not (cula_slow or kda_slow))",
            True,
            True,
        ),
        ("kda_full and not sanitizer", "(kda_fast or kda_slow) and not sanitizer", False, True),
    ],
)
def test_prepare_markexpr(expression, expanded, include_cula_slow, include_kda_slow):
    assert _prepare_markexpr(expression) == (expanded, include_cula_slow, include_kda_slow)
