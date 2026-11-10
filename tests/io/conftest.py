from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent / "data"
IOH040 = DATA / "ioh040"


@pytest.fixture
def ioh040() -> Path:
    """The seeded ioh 0.4.0 reader fixtures (``tests/generate_test_data.py``)."""
    return IOH040
