import pytest
import torch


def _is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


def pytest_configure(config):
    config.addinivalue_line("markers", "sm100_only: only run on SM100 devices")
    config.addinivalue_line("markers", "sm90_only: skip on SM100 devices")


def pytest_collection_modifyitems(config, items):
    is_sm100 = _is_sm100()
    skip_non_sm100 = pytest.mark.skip(reason="SM100-only test: skip on non-SM100 devices")
    skip_on_sm100 = pytest.mark.skip(reason="SM90-only test: skip on SM100")

    for item in items:
        if "sm100_only" in item.keywords and not is_sm100:
            item.add_marker(skip_non_sm100)
        if "sm90_only" in item.keywords and is_sm100:
            item.add_marker(skip_on_sm100)
