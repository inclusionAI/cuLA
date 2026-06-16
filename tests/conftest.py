import re

import pytest
import torch


def _is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


def pytest_configure(config):
    config.addinivalue_line("markers", "sm100_only: only run on SM100 devices")
    config.addinivalue_line("markers", "sm90_only: skip on SM100 devices")
    config.addinivalue_line(
        "markers",
        "kda_fast: KDA test case included in fast (default) mode",
    )
    config.addinivalue_line(
        "markers",
        "kda_slow: KDA test case excluded from fast (default) mode; "
        "include via 'pytest -m kda_slow' or run the full sweep with '-m kda_full'",
    )
    config.addinivalue_line(
        "markers",
        "kda_fast_norecomp: fast-mode KDA config that also runs the disable_recompute=True "
        "variant in fast mode (other fast configs run disable_recompute=False only)",
    )

    markexpr = config.option.markexpr
    if markexpr and "kda_full" in markexpr:
        config.option.markexpr = re.sub(r"\bkda_full\b", "(kda_fast or kda_slow)", markexpr)


def pytest_collection_modifyitems(config, items):
    is_sm100 = _is_sm100()
    skip_non_sm100 = pytest.mark.skip(reason="SM100-only test: skip on non-SM100 devices")
    skip_on_sm100 = pytest.mark.skip(reason="SM90-only test: skip on SM100")

    marker_expr = config.option.markexpr or ""
    include_slow = "kda_slow" in marker_expr
    skip_slow = pytest.mark.skip(
        reason="kda_slow case: run 'pytest -m kda_slow' or the full sweep with '-m kda_full' to include"
    )
    skip_fast_norecomp = pytest.mark.skip(
        reason="disable_recompute=True runs in fast mode only for kda_fast_norecomp configs; "
        "include the rest via '-m kda_slow' or '-m kda_full'"
    )

    for item in items:
        if "sm100_only" in item.keywords and not is_sm100:
            item.add_marker(skip_non_sm100)
        if "sm90_only" in item.keywords and is_sm100:
            item.add_marker(skip_on_sm100)
        if include_slow:
            continue
        if "kda_slow" in item.keywords:
            item.add_marker(skip_slow)
            continue
        callspec = getattr(item, "callspec", None)
        if callspec is not None and callspec.params.get("disable_recompute") and "kda_fast_norecomp" not in item.keywords:
            item.add_marker(skip_fast_norecomp)
