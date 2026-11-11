"""One test per known defect of the iohinspector 0.0.8 COCO loader.

The defects are listed in the plan (iohinspector.md, "Known defects to fix"):
instance ids are lost, and the ``.dat``/``.tdat`` merge is wrong. 0.0.8 set
every instance to 0, split runs on ``evaluations == 1``, and merged the two
files with ``unique(evaluations, keep="first")``, which drops a ``.tdat`` row
whenever the ``.dat`` file has a row with the same evaluation count, even with
other values. The reader's side is in ``test_coco.py`` (``test_fixture_runs``,
``test_conflicting_rows_kept_and_flagged``); these check ``DataManager``.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import polars as pl

from iohinspector import DataManager
from iohinspector.io.coco import read_coco

from .test_coco import DAT, TDAT, run_text, write_coco

FIXTURE = Path(__file__).resolve().parents[1] / "test_coco_data"


def load(folder: Path, **kwargs) -> tuple[DataManager, pl.DataFrame]:
    manager = DataManager()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        manager.add_folder(str(folder))
    return manager, manager.load(include_columns=["source_run_index"], **kwargs)


def info_instances(folder: Path) -> list[int]:
    line = next(folder.rglob("*.info")).read_text().splitlines()[2]
    return [int(i) for i in re.findall(r"(\d+):\d+\|", line)]


# --- Instance ids are lost --------------------------------------------------------


def test_manager_reads_instances():
    manager, _ = load(FIXTURE, monotonic=False)
    overview = manager.overview.sort("source_file", "source_run_index")
    assert overview["instance"].to_list() == info_instances(FIXTURE)


# --- The .dat/.tdat merge is wrong ------------------------------------------------

CONFLICT = TDAT.replace(
    "3 0 +1.0e-01 +1.01e+01 +1.01e+01 +3.0e-01 +4.0e-01",
    "3 0 +1.0e-01 +1.01e+01 +1.01e+01 +3.5e-01 +4.0e-01",
)


def test_manager_keeps_rows_the_files_disagree_on(tmp_path):
    _, df = load(write_coco(tmp_path, {"dat": DAT, "tdat": CONFLICT}), monotonic=False)
    rows = df.filter(pl.col("source_run_index") == 0, pl.col("evaluations") == 3)
    assert sorted(rows["x0"].to_list()) == [0.3, 0.35]
    # every row of both files, identical rows once: 4 + 2 rows
    assert df.height == 6


def test_manager_splits_runs_on_header_lines(tmp_path):
    # several evaluations counted as 1 (seen in the archive, bbob 2017 EvoSpace-PSO-GA)
    dat = run_text(
        1.0,
        [
            "1 0 +5.0e+00 +6.0e+00 +6.0e+00",
            "1 0 +4.0e+00 +5.0e+00 +5.0e+00",
            "2 0 +3.0e+00 +4.0e+00 +4.0e+00",
        ],
    ) + run_text(2.0, ["1 0 +1.0e+00 +3.0e+00 +3.0e+00"])
    _, df = load(
        write_coco(tmp_path, {"dat": dat}, entries="1:2|3.0e+00, 2:1|1.0e+00"),
        monotonic=False,
    )
    runs = df.group_by("source_run_index").len().sort("source_run_index")
    assert runs["len"].to_list() == [3, 1]


def test_manager_rows_equal_reader_trace():
    result = read_coco(FIXTURE, "defects")
    per_run = (
        result.trace.join(result.runs.select("run_id", "source_run_index"), on="run_id")
        .group_by("source_run_index")
        .agg(pl.col("evaluations"))
        .sort("source_run_index")["evaluations"]
        .to_list()
    )
    _, df = load(FIXTURE, monotonic=False)
    got = df.group_by("source_run_index").agg(pl.col("evaluations"))
    got = got.sort("source_run_index")
    assert [list(map(int, e)) for e in got["evaluations"].to_list()] == per_run
