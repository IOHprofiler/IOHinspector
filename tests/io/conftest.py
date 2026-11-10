from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent / "data"
IOH040 = DATA / "ioh040"


@pytest.fixture
def ioh040() -> Path:
    """The seeded ioh 0.4.0 reader fixtures (``tests/generate_test_data.py``)."""
    return IOH040


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
