import re
from pathlib import Path

import pytest

from iohinspector.format import check_dataset_id, check_source_path, run_id, run_ids

FORMAT_MD = Path(__file__).resolve().parents[2] / "docs" / "format.md"


def test_run_id_matches_the_test_vector_in_the_spec():
    text = FORMAT_MD.read_text(encoding="utf-8")
    m = re.search(
        r"dataset id `([^`]+)`, source file `([^`]+)` and run index `(\d+)` give run id `(\d+)`",
        text,
    )
    assert m, "test vector not found in docs/format.md"
    dataset_id, source_file, index, expected = m.groups()
    assert (
        run_id(dataset_id, source_file, int(index))
        == int(expected)
        == 7884767605179371010
    )


def test_run_id_is_stable_and_depends_on_every_part():
    base = run_id("ds", "a/b.dat", 0)
    assert base == run_id("ds", "a/b.dat", 0)
    others = {
        run_id("ds2", "a/b.dat", 0),
        run_id("ds", "a/c.dat", 0),
        run_id("ds", "a/b.dat", 1),
    }
    assert base not in others and len(others) == 3
    for value in [base, *others]:
        assert 0 <= value < 2**63


def test_run_ids_vectorised_matches_scalar():
    s = run_ids("ds", ["x.dat", "x.dat", "y.dat"], [0, 1, 0])
    assert s.dtype.is_integer() and s.to_list() == [
        run_id("ds", f, i) for f, i in [("x.dat", 0), ("x.dat", 1), ("y.dat", 0)]
    ]


def test_run_id_rejects_bad_index():
    with pytest.raises(ValueError):
        run_id("ds", "a.dat", -1)
    with pytest.raises(TypeError):
        run_id("ds", "a.dat", True)


@pytest.mark.parametrize("good", ["a", "A.b-c_1", "x" * 128])
def test_dataset_id_accepts(good):
    assert check_dataset_id(good) == good


@pytest.mark.parametrize("bad", ["", "-a", ".a", "a b", "a/b", "x" * 129, "é"])
def test_dataset_id_rejects(bad):
    with pytest.raises(ValueError):
        check_dataset_id(bad)


@pytest.mark.parametrize(
    "good", ["a.dat", "alg/data_f1/IOHprofiler_f1_DIM5.dat", "a.zip/x y/ü.dat", "..a/b"]
)
def test_source_path_accepts(good):
    assert check_source_path(good) == good


@pytest.mark.parametrize(
    "bad", ["", "/abs", "a//b", "a/", "./a", "a/../b", "..", "a\\b", "a\nb", "a\x00b"]
)
def test_source_path_rejects(bad):
    with pytest.raises(ValueError):
        check_source_path(bad)
