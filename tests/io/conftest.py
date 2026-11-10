from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(params=["ioh040", "ioh0322"])
def ioh_logs(request) -> Path:
    """The seeded reader fixtures of ioh 0.4.0 and 0.3.22 (``tests/generate_test_data.py``)."""
    return FIXTURES / request.param


@pytest.fixture(autouse=True)
def _restore_data_ids():
    """Keep the legacy process-global ``data_id`` counter as other tests expect it.

    ``DataManager`` numbers runs from a class-level counter, so ids depend on what
    was loaded before (a known defect outside this reader). Restoring it keeps the
    older tests, which select by ``data_id``, independent of these.
    """
    from iohinspector.data import Run

    lookup, current = dict(Run.__lookup__), Run.__current_id__
    yield
    Run.__lookup__.clear()
    Run.__lookup__.update(lookup)
    Run.__current_id__ = current
