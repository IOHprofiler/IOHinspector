from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(params=["ioh040", "ioh0322"])
def ioh_logs(request) -> Path:
    """The seeded reader fixtures of ioh 0.4.0 and 0.3.22 (``tests/generate_test_data.py``)."""
    return FIXTURES / request.param
