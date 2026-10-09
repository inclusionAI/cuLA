import re

import pytest
import torch


def _is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


def _is_sm90() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


def _prepare_markexpr(markexpr: str) -> tuple[str, bool, bool]:
    """Expand suite aliases and report whether extended CUDA cases are requested."""
    include_cula_slow = bool(re.search(r"\b(cula_slow|cula_full)\b", markexpr))
    include_kda_slow = bool(re.search(r"\b(kda_slow|kda_full|cula_slow|cula_full)\b", markexpr))
    markexpr = re.sub(r"\bcula_full\b", "(cula_slow or not cula_slow)", markexpr)
    markexpr = re.sub(r"\bcula_slow\b", "(cula_slow or kda_slow)", markexpr)
    markexpr = re.sub(r"\bkda_full\b", "(kda_fast or kda_slow)", markexpr)
    return markexpr, include_cula_slow, include_kda_slow


def pytest_configure(config):
    config.addinivalue_line("markers", "sm100_only: only run on SM100 devices")
    config.addinivalue_line("markers", "sm90_only: only run on SM90 devices")
    config.addinivalue_line("markers", "benchmark: long-running benchmark-shaped coverage")
    config.addinivalue_line("markers", "sanitizer: tests intended to run under compute-sanitizer")
    config.addinivalue_line(
        "markers",
        "cula_fast: representative case kept in the default, time-bounded test suite",
    )
    config.addinivalue_line(
        "markers",
        "cula_slow: extended CUDA coverage excluded from the default suite; "
        "run with '-m cula_slow' or include everything with '-m cula_full'",
    )
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

    markexpr, config._cula_include_slow, config._kda_include_slow = _prepare_markexpr(config.option.markexpr or "")
    config.option.markexpr = markexpr


def pytest_collection_modifyitems(config, items):
    is_sm100 = _is_sm100()
    is_sm90 = _is_sm90()
    skip_non_sm100 = pytest.mark.skip(reason="SM100-only test: skip on non-SM100 devices")
    skip_non_sm90 = pytest.mark.skip(reason="SM90-only test: skip on non-SM90 devices")

    include_slow = config._kda_include_slow
    include_cula_slow = config._cula_include_slow
    skip_slow = pytest.mark.skip(
        reason="kda_slow case: run 'pytest -m kda_slow' or the full sweep with '-m kda_full' to include"
    )
    skip_fast_norecomp = pytest.mark.skip(
        reason="disable_recompute=True runs in fast mode only for kda_fast_norecomp configs; "
        "include the rest via '-m kda_slow' or '-m kda_full'"
    )
    skip_cula_slow = pytest.mark.skip(
        reason="extended CUDA case: run 'pytest -m cula_slow' or the full suite with '-m cula_full'"
    )

    for item in items:
        if "sm100_only" in item.keywords and not is_sm100:
            item.add_marker(skip_non_sm100)
        if "sm90_only" in item.keywords and not is_sm90:
            item.add_marker(skip_non_sm90)
        if "cula_slow" in item.keywords and "cula_fast" not in item.keywords and not include_cula_slow:
            item.add_marker(skip_cula_slow)
        if include_slow:
            continue
        if "kda_slow" in item.keywords:
            item.add_marker(skip_slow)
            continue
        callspec = getattr(item, "callspec", None)
        if callspec is not None and callspec.params.get("disable_recompute") and "kda_fast_norecomp" not in item.keywords:
            item.add_marker(skip_fast_norecomp)
