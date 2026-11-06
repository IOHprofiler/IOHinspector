import os

import pytest

RUN_SLOW = "IOHINSPECTOR_RUN_SLOW"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        f"slow: long-running test, skipped unless the environment sets {RUN_SLOW}=1",
    )


def pytest_collection_modifyitems(config, items):
    if os.environ.get(RUN_SLOW) == "1":
        return
    skip = pytest.mark.skip(reason=f"slow; set {RUN_SLOW}=1 to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
