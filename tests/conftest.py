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
        "include via 'pytest -m kda_slow' or '-m \"kda_fast or kda_slow\"'",
    )
    config.addinivalue_line(
        "markers",
        "kda_backward: fast-mode KDA case that performs full backward validation "
        "(without this marker, fast-mode cases verify only forward output and final state)",
    )


def pytest_collection_modifyitems(config, items):
    is_sm100 = _is_sm100()
    skip_non_sm100 = pytest.mark.skip(reason="SM100-only test: skip on non-SM100 devices")
    skip_on_sm100 = pytest.mark.skip(reason="SM90-only test: skip on SM100")

    marker_expr = config.option.markexpr or ""
    include_slow = "kda_slow" in marker_expr
    skip_slow = pytest.mark.skip(
        reason="kda_slow case: run 'pytest -m kda_slow' or '-m \"kda_fast or kda_slow\"' to include"
    )

    for item in items:
        if "sm100_only" in item.keywords and not is_sm100:
            item.add_marker(skip_non_sm100)
        if "sm90_only" in item.keywords and is_sm100:
            item.add_marker(skip_on_sm100)
        if not include_slow and "kda_slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def needs_backward(request) -> bool:
    """Whether the current test should run full backward validation.

    Slow / full sweeps (``-m`` contains ``kda_slow``): every test that runs
    validates backward. Default fast run: only ``kda_backward`` cases validate
    backward; the remaining fast smoke cases verify forward output and final
    state only.
    """
    if "kda_slow" in (request.config.option.markexpr or ""):
        return True
    keywords = request.node.keywords
    return "kda_slow" in keywords or "kda_backward" in keywords
