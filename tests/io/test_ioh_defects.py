"""One test per known defect of the iohinspector 0.0.8 IOHexperimenter loader.

The defects are listed in the plan (iohinspector.md, "Known defects to fix"). Each
test reads a fixture made by ``tests/generate_test_data.py --reader-fixtures``
(ioh 0.4.0 and 0.3.22) and checks the corrected behaviour, both in the new reader
(``iohinspector.io.ioh``) and through ``DataManager``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import polars as pl
import pytest

from iohinspector import DataManager
from iohinspector.io import read_ioh


def read(folder: Path):
    return read_ioh(folder, dataset_id="defects")


def dat_runs(path: Path) -> list[list[list[str]]]:
    """The data lines of a ``.dat`` file as tokens, one list per run (split on headers)."""
    runs: list[list[list[str]]] = []
    for line in path.read_text().splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "evaluations":
            runs.append([])
        else:
            runs[-1].append(tokens)
    return runs


def load(folder: Path, **kwargs) -> pl.DataFrame:
    manager = DataManager()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        manager.add_folder(str(folder))
    return manager.load(include_columns=["source_run_index"], **kwargs)


def trace_by_run(result) -> list[list[tuple[int, float]]]:
    """The trace as ``(evaluations, raw_y)`` per run, in source order."""
    runs = result.runs.sort(["source_file", "source_run_index"])
    trace = result.trace
    out = []
    for run_id in runs["run_id"]:
        rows = trace.filter(pl.col("run_id") == run_id).sort("row")
        out.append(list(zip(rows["evaluations"], rows["raw_y"])))
    return out


def expected_by_run(dat: Path) -> list[list[tuple[int, float]]]:
    return [[(int(t[0]), float(t[1])) for t in run] for run in dat_runs(dat)]


# --- Runs are split on evaluations == 1 instead of on header lines ---------------


# The reader keeps every row of every run (split on headers, final rows, rows that
# repeat across runs, f = 0): one check for the four defects.
@pytest.mark.parametrize("case", ["every_third", "cross_run", "f_zero"])
def test_reader_keeps_every_row_of_each_run(ioh_logs, case):
    folder = ioh_logs / case
    dat = folder / "data_f1_Sphere" / "IOHprofiler_f1_DIM2.dat"
    result = read(folder)
    assert trace_by_run(result) == expected_by_run(dat)
    if case == "every_third":
        assert [len(r) for r in expected_by_run(dat)] == [4, 3, 3]
    if case == "f_zero":
        assert result.trace.filter(pl.col("raw_y") == 0.0).height == 2


def test_manager_splits_runs_on_header_lines(ioh_logs):
    folder = ioh_logs / "every_third"
    df = load(folder, monotonic=False)
    # evaluation 1 is never logged: 0.0.8 loads zero rows
    assert len(df) == 10
    runs = df.group_by("source_run_index").len().sort("source_run_index")
    assert runs["len"].to_list() == [4, 3, 3]


# --- A global raw_y.diff() != 0 filter removes rows across run boundaries --------


def test_manager_improvement_filter_is_per_run(ioh_logs):
    df = load(ioh_logs / "cross_run", monotonic=True)
    firsts = df.group_by("run_id").agg(pl.col("evaluations").min())
    # run 2 starts at the best value of run 1; 0.0.8 drops its first row
    assert firsts["evaluations"].to_list() == [1, 1]


# --- The final row of a run can be dropped ---------------------------------------


def test_manager_monotonic_load_keeps_final_rows(ioh_logs):
    df = load(ioh_logs / "cross_run", monotonic=True)
    lasts = df.group_by("source_run_index").agg(pl.col("evaluations").max())
    lasts = lasts.sort("source_run_index")
    assert lasts["evaluations"].to_list() == [9, 6]


# --- An assert fails when f = 0 --------------------------------------------------


def test_manager_loads_f_zero(ioh_logs):
    df = load(ioh_logs / "f_zero", monotonic=True)
    zeros = df.filter(pl.col("raw_y") == 0.0)
    # f = 0 at evaluation 2, kept as best-so-far until the final evaluation 4
    assert zeros["evaluations"].to_list() == [2, 4, 2, 4]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="metrics.utils.get_sequence asserts min > 0 under log scale; "
    "target grids are I10 (docs/definitions/targets.md, TG-3)",
)
def test_runtime_summary_on_f_zero(ioh_logs):
    from iohinspector.metrics import aggregate_running_time

    df = load(ioh_logs / "f_zero", monotonic=True, include_meta_data=True)
    aggregate_running_time(df)


# --- Run attributes and maximisation are not fully read --------------------------


def test_reader_reads_attributes_and_maximization(ioh_logs):
    result = read(ioh_logs / "attributes")
    assert result.runs["maximization"].to_list() == [True] * 4
    d = result.descriptors
    seeds = d.filter(pl.col("key") == "seed").sort("run_id")
    assert set(seeds["origin"]) == {"run"}
    assert set(seeds["type"]) == {"float"}
    assert sorted(seeds["value_float"]) == [2.0**40, 2.0**40, 2.0**40 + 1, 2.0**40 + 1]
    # "budget" and "instance" are runs columns, so the experiment attributes are renamed
    budget = d.filter(pl.col("key") == "budget_experiment")
    assert set(budget["origin"]) == {"experiment"}
    assert set(budget["value"]) == {"8"}
    renamed = {(r["from"], r["to"]) for r in result.manifest["renamed_columns"]}
    assert ("budget", "budget_experiment") in renamed
    assert ("instance", "instance_experiment") in renamed
    info = d.filter(pl.col("key") == "algorithm_info")
    assert set(info["value"]) == {'C:\\runs\\ea "v2"'}


def test_manager_reads_run_attributes(ioh_logs):
    manager = DataManager()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        manager.add_folder(str(ioh_logs / "attributes"))
    overview = manager.overview
    assert overview.height == 4
    assert sorted(overview["seed"].to_list()) == [
        2.0**40,
        2.0**40,
        2.0**40 + 1,
        2.0**40 + 1,
    ]
    assert manager.overview["maximization"].to_list() == [True] * 4
