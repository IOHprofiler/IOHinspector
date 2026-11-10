"""Tests of the IOHexperimenter reader (``iohinspector.io.ioh``)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import polars as pl
import pytest

from iohinspector.format import scan_dataset, validate_dataset
from iohinspector.io import READERS, ReaderError, detect_format, read
from iohinspector.io.ioh import infer_triggers, load_ioh_json, parse_dat, read_ioh

ROOT = Path(__file__).resolve().parents[1]
CORPUS = [
    ROOT / "test_data",
    ROOT / "io" / "data" / "ioh040",
    *sorted((ROOT / "golden" / "fixtures").glob("ioh040_*")),
]

HEADER = "evaluations raw_y"


def write_log(  # pylint: disable=too-many-arguments
    folder: Path,
    dat: str | None,
    runs: list[dict] | None = None,
    *,
    attributes=("evaluations", "raw_y"),
    extra_lines: str = "",
    dimension: int = 2,
    maximization: str = "false",
    algorithm: str = '{"name": "alg", "info": "info"}',
) -> Path:
    """A JSON file as IOHexperimenter writes it (no escaping) and its ``.dat`` file."""
    if runs is None:
        runs = [
            {"instance": 1, "evals": 10, "best": {"evals": 1, "y": 1.0, "x": [0, 0]}}
        ]
    folder.mkdir(parents=True, exist_ok=True)
    run_text = ",\n\t\t\t".join(
        r if isinstance(r, str) else json.dumps(r) for r in runs
    )
    text = (
        '{\n\t"version": "0.4.0", \n\t"suite": "BBOB", \n\t"function_id": 1, '
        '\n\t"function_name": "Sphere", '
        f'\n\t"maximization": {maximization}, '
        f'\n\t"algorithm": {algorithm},{extra_lines}'
        + (
            ""
            if attributes is None
            else f'\n\t"attributes": {json.dumps(list(attributes))},'
        )
        + '\n\t"scenarios": [\n\t\t{'
        f'"dimension": {dimension},\n\t\t"path": "{_dat_path(dimension)}",'
        f'\n\t\t"runs": [\n\t\t\t{run_text}\n\t\t]}}\n\t]\n}}\n'
    )
    path = folder / "IOHprofiler_f1_Sphere.json"
    path.write_text(text)
    if dat is not None:
        (folder / "data_f1_Sphere").mkdir(exist_ok=True)
        (folder / "data_f1_Sphere" / f"IOHprofiler_f1_DIM{dimension}.dat").write_text(
            dat
        )
    return path


def _dat_path(dimension: int) -> str:
    return f"data_f1_Sphere/IOHprofiler_f1_DIM{dimension}.dat"


def data_lines(folder: Path) -> int:
    """Data lines of every ``.dat`` file named by a JSON file under ``folder``."""
    count = 0
    for js in folder.rglob("*.json"):
        meta = load_ioh_json(js)
        for scen in meta.scenarios:
            dat = js.parent / scen.path
            for line in dat.read_text().splitlines():
                tokens = line.split()
                if tokens and tokens[0][0] not in "abcdefghijklmnopqrstuvwxyz":
                    count += 1
    return count


# --- whole corpus ---------------------------------------------------------------


@pytest.mark.parametrize("folder", CORPUS, ids=lambda p: p.name)
def test_rows_read_equal_data_lines(folder, tmp_path):
    result = read_ioh(folder, "corpus")
    assert result.trace.height == data_lines(folder)
    manifest = result.write(tmp_path / "ds")
    assert manifest["counts"]["trace_rows"] == data_lines(folder)
    assert validate_dataset(tmp_path / "ds").errors == []


@pytest.mark.parametrize("folder", CORPUS, ids=lambda p: p.name)
def test_values_equal_source_tokens(folder):
    result = read_ioh(folder, "corpus")
    runs = result.runs
    for source in runs["source_file"].unique().to_list():
        lines = (folder / source).read_text().splitlines()
        expected = [
            (int(t[0]), float(t[1]))
            for t in (line.split() for line in lines)
            if t and t[0] != "evaluations"
        ]
        ids = runs.filter(pl.col("source_file") == source).sort("source_run_index")
        got = (
            result.trace.join(ids.select("run_id", "source_run_index"), on="run_id")
            .sort("source_run_index", "row")
            .select("evaluations", "raw_y")
            .rows()
        )
        assert len(got) == len(expected)
        for (e1, y1), (e2, y2) in zip(got, expected):
            assert e1 == e2
            assert y1 == y2 or (math.isnan(y1) and math.isnan(y2))


def test_source_rows_sha256(tmp_path):
    dat = f"{HEADER}\n1  5.0\n2 4.0\n{HEADER}\n1 3.0\n"
    write_log(tmp_path, dat, runs=[_run(), _run()])
    result = read_ioh(tmp_path, "x")
    want = [
        hashlib.sha256(b"1 5.0\n2 4.0\n").hexdigest(),
        hashlib.sha256(b"1 3.0\n").hexdigest(),
    ]
    assert result.runs.sort("source_run_index")["source_rows_sha256"].to_list() == want


def test_reader_table_and_detection(tmp_path):
    write_log(tmp_path, f"{HEADER}\n1 1.0\n")
    assert "ioh" in READERS
    assert detect_format(tmp_path) == "ioh"
    assert detect_format(tmp_path / "IOHprofiler_f1_Sphere.json") == "ioh"
    assert read(tmp_path, "x").trace.height == 1
    (tmp_path / "other").mkdir()
    with pytest.raises(ReaderError):
        detect_format(tmp_path / "other")


def test_round_trip_through_storage(tmp_path):
    folder = ROOT / "io" / "data" / "ioh040" / "attributes"
    result = read_ioh(folder, "rt")
    result.write(tmp_path / "ds")
    stored = scan_dataset(tmp_path / "ds")
    trace = stored.trace.collect()
    assert trace.height == result.trace.height
    assert "x0_logged" in trace.columns
    positions = stored.positions.collect()
    assert positions.height == 12
    assert stored.run_columns(["algorithm_name", "seed"]).collect().height == 4


# --- runs and metadata ------------------------------------------------------------


def _run(instance=1, **attrs):
    return {
        "instance": instance,
        "evals": 2,
        "best": {"evals": 1, "y": 1.0, "x": [0, 0]},
        **attrs,
    }


def test_runs_columns(tmp_path):
    write_log(tmp_path, f"{HEADER}\n1 5.0\n2 4.0\n", runs=[_run(instance=3)])
    runs = read_ioh(tmp_path, "x").runs
    row = runs.row(0, named=True)
    assert row["suite"] == "BBOB"
    assert row["function_id"] == 1
    assert row["function_name"] == "Sphere"
    assert row["instance"] == 3
    assert row["dimension"] == 2
    assert row["maximization"] is False
    assert row["evals_used"] == 2
    assert row["best_y"] == 1.0
    assert row["source_file"] == "data_f1_Sphere/IOHprofiler_f1_DIM2.dat"


def test_run_ids_do_not_depend_on_load_order(tmp_path):
    for name in ("a", "b"):
        write_log(tmp_path / name, f"{HEADER}\n1 1.0\n")
    one = read_ioh(tmp_path, "x").runs.sort("source_file")
    two = read_ioh(tmp_path / "b", "x", root=tmp_path).runs
    assert one["run_id"][1] == two["run_id"][0]


def test_maximization(tmp_path):
    write_log(tmp_path, f"{HEADER}\n1 1.0\n2 3.0\n3 2.0\n", maximization="true")
    result = read_ioh(tmp_path, "x")
    assert result.runs["maximization"].to_list() == [True]
    triggers = result.manifest["sources"][1]["triggers"]
    assert triggers == [{"kind": "all", "params": {"inferred": True}}]


def test_more_dat_runs_than_json_runs(tmp_path):
    write_log(tmp_path, f"{HEADER}\n3 1.0\n{HEADER}\n3 2.0\n", runs=[_run()])
    result = read_ioh(tmp_path, "x")
    runs = result.runs.sort("source_run_index")
    assert runs["instance"].to_list() == [1, None]
    assert result.trace.height == 2
    codes = [w["code"] for w in result.manifest["warnings"]]
    assert "run_count_mismatch" in codes
    names = result.descriptors.filter(pl.col("key") == "algorithm_name")
    assert names.height == 2


def test_more_json_runs_than_dat_runs(tmp_path):
    write_log(tmp_path, f"{HEADER}\n1 1.0\n", runs=[_run(), _run(instance=2)])
    result = read_ioh(tmp_path, "x")
    assert result.runs.height == 2
    assert result.trace.height == 1
    assert result.runs.sort("source_run_index")["source_rows_sha256"][1] is None


def test_missing_dat_file(tmp_path):
    write_log(tmp_path, None)
    result = read_ioh(tmp_path, "x")
    assert result.runs.height == 1
    assert result.trace.height == 0
    dat = result.manifest["sources"][1]
    assert dat["sha256"] is None and dat["bytes"] is None
    assert result.manifest["warnings"][0]["code"] == "missing_data_file"
    result.write(tmp_path / "ds")
    assert validate_dataset(tmp_path / "ds").errors == []


def test_unreferenced_and_skipped_files(tmp_path):
    write_log(tmp_path, f"{HEADER}\n1 1.0\n")
    (tmp_path / "stray.dat").write_text(f"{HEADER}\n1 1.0\n")
    (tmp_path / "config.json").write_text('{"a": 1}')
    warnings = read_ioh(tmp_path, "x").manifest["warnings"]
    codes = {w["code"]: w.get("source") for w in warnings}
    assert codes["unreferenced_data_file"] == "stray.dat"
    assert codes["skipped_file"] == "config.json"


def test_precision(tmp_path):
    write_log(
        tmp_path, f"{HEADER}\n1 1e-12\n", extra_lines='\n\t"precision": "%.17g", '
    )
    result = read_ioh(tmp_path, "x")
    assert result.manifest["sources"][1]["precision"] == {
        "notation": "general",
        "digits": 17,
    }
    assert "warnings" not in result.manifest
    assert result.trace["raw_y"].to_list() == [1e-12]


def test_limited_precision_warning(tmp_path):
    write_log(tmp_path, f"{HEADER}\n1 0.0000000000\n")
    result = read_ioh(tmp_path, "x")
    assert result.manifest["sources"][1]["precision"] == {
        "notation": "fixed",
        "digits": 10,
    }
    assert [w["code"] for w in result.manifest["warnings"]] == ["limited_precision"]


# --- descriptors -----------------------------------------------------------------


def test_descriptors(tmp_path):
    extra = (
        '\n\t"experiment_attributes": [{"opt": "C:\\new"}, {"dimension": "x"}],'
        '\n\t"run_attributes": ["seed", "algorithm_name"],'
    )
    run = (
        '{"instance": 1, "evals": 2, "best": {"evals": 1, "y": 1, "x": [0, nan]}, '
        '"seed": 18446744073709551616, "algorithm_name": -inf}'
    )
    write_log(tmp_path, f"{HEADER}\n1 1.0\n", runs=[run], extra_lines=extra)
    result = read_ioh(tmp_path, "x")
    d = {r["key"]: r for r in result.descriptors.iter_rows(named=True)}
    assert d["algorithm_name"]["value"] == "alg"
    assert d["algorithm_info"]["value"] == "info"
    # the logger does not escape: "C:\new" is a backslash and an n, not a newline
    assert d["opt"]["value"] == "C:\\new"
    assert d["opt"]["origin"] == "experiment"
    assert d["dimension_experiment"]["value"] == "x"
    assert d["seed"]["type"] == "float"
    assert d["seed"]["value_float"] == 2.0**64
    assert d["algorithm_name_run"]["value_float"] == -math.inf
    assert d["best_evals"]["value_int"] == 1
    assert d["best_x"]["value"] == "[0, NaN]"
    codes = [w["code"] for w in result.manifest["warnings"]]
    assert "json_repaired" in codes


def test_algorithm_with_quotes(tmp_path):
    write_log(
        tmp_path, f"{HEADER}\n1 1.0\n", algorithm='{"name": "a"b", "info": "i\\"}'
    )
    d = read_ioh(tmp_path, "x").descriptors
    values = dict(zip(d["key"], d["value"]))
    assert values["algorithm_name"] == 'a"b'
    assert values["algorithm_info"] == "i\\"


def test_duplicate_run_attribute_names(tmp_path):
    run = '{"instance": 1, "evals": 2, "best": {"evals": 1, "y": 1, "x": [0]}, "instance": 4}'
    write_log(tmp_path, f"{HEADER}\n1 1.0\n", runs=[run])
    result = read_ioh(tmp_path, "x")
    assert result.runs["instance"].to_list() == [1]
    d = result.descriptors.filter(pl.col("key") == "instance_run")
    assert d["value_float"].to_list() == [4.0]


# --- .dat parsing ----------------------------------------------------------------


def test_extra_columns_none_and_text(tmp_path):
    dat = "evaluations raw_y sigma note x3\n1 1.0 None a 2.5\n2 0.5 0.25 None 1\n"
    write_log(tmp_path, dat, attributes=("evaluations", "raw_y", "sigma", "note", "x3"))
    result = read_ioh(tmp_path, "x")
    trace = result.trace.sort("row")
    assert trace["sigma"].to_list() == [None, 0.25]
    assert trace["note"].to_list() == ["a", None]
    assert trace["x3_logged"].to_list() == [2.5, 1.0]
    codes = [w["code"] for w in result.manifest.get("warnings", [])]
    assert "text_column" in codes


def test_text_column_is_text_in_every_file(tmp_path):
    for name, value in (("a", "0.5"), ("b", "high")):
        write_log(
            tmp_path / name,
            f"evaluations raw_y level\n1 1.0 {value}\n",
            attributes=("evaluations", "raw_y", "level"),
        )
    trace = read_ioh(tmp_path, "x").trace
    assert trace.schema["level"] == pl.String
    assert sorted(trace["level"].to_list()) == ["0.5", "high"]


def test_positions(tmp_path):
    dat = "evaluations raw_y x0 x0 x1\n1 1.0 7.0 0.5 -0.5\n"
    write_log(tmp_path, dat, attributes=("evaluations", "raw_y", "x0"))
    result = read_ioh(tmp_path, "x")
    assert result.trace["x0_logged"].to_list() == [7.0]
    assert result.positions is not None
    assert result.positions.select("x0", "x1").rows() == [(0.5, -0.5)]


def test_special_values(tmp_path):
    dat = f"{HEADER}\n1 inf\n2 -inf\n3 nan\n4 -nan\n5 NaN\n6 1e-300\n7 -0.0\n"
    write_log(tmp_path, dat)
    y = read_ioh(tmp_path, "x").trace.sort("row")["raw_y"].to_list()
    assert y[:2] == [math.inf, -math.inf]
    assert all(math.isnan(v) for v in y[2:5])
    assert y[5] == 1e-300
    assert math.copysign(1.0, y[6]) == -1.0


def test_windows_line_endings_and_blank_lines(tmp_path):
    write_log(tmp_path, f"{HEADER}\r\n1 1.0\r\n\r\n2 0.5\r\n")
    assert read_ioh(tmp_path, "x").trace.height == 2


@pytest.mark.parametrize(
    ("dat", "line", "message"),
    [
        (f"{HEADER}\n1 1.0\n2 1,5\n", 3, "'1,5' is not a number"),
        (f"{HEADER}\n1 1.0\n2.5 1.0\n", 3, "'2.5' is not an integer"),
        (f"{HEADER}\n1 None\n", 2, "'None' is not a number"),
        (f"{HEADER}\n1 1.0 2.0\n", 2, "3 fields, but the header has 2"),
        ("1 1.0\n", 1, "data line before the first header"),
        (f"{HEADER}\n1 1.0\nevaluations raw_y z\n", 3, "differs from the first header"),
        ("evaluations y\n1 1.0\n", 1, "no 'raw_y' column"),
        ("evaluations raw_y raw_y\n1 1.0 1.0\n", 1, "more than once"),
        (f"{HEADER}\n1 1.0\n2 infinity\n", 3, "'infinity' is not a number"),
    ],
)
def test_malformed_dat(tmp_path, dat, line, message):
    write_log(tmp_path, dat, attributes=None)
    with pytest.raises(ReaderError, match=message) as info:
        read_ioh(tmp_path, "x")
    assert info.value.line == line
    assert str(info.value).startswith(
        f"data_f1_Sphere/IOHprofiler_f1_DIM2.dat:{line}: "
    )


def test_flatfile_separator_bug_is_named(tmp_path):
    dat = "evaluations raw_y sigma\n11.0 0.5\n"
    write_log(tmp_path, dat, attributes=("evaluations", "raw_y", "sigma"))
    with pytest.raises(ReaderError, match="FlatFile separator bug"):
        read_ioh(tmp_path, "x")


def test_flatfile_output_is_refused(tmp_path):
    header = (
        "suite_name problem_name problem_id problem_instance optimization_type "
        "dimension run evaluations raw_y"
    )
    path = tmp_path / "f.dat"
    path.write_text(header + "\n")
    with pytest.raises(ReaderError, match="FlatFile output"):
        parse_dat(path, dimension=2)


def test_malformed_json(tmp_path):
    path = write_log(tmp_path, f"{HEADER}\n1 1.0\n")
    path.write_text(path.read_text().replace('"function_id": 1', '"function_id": one'))
    with pytest.raises(ReaderError, match="invalid JSON") as info:
        read_ioh(tmp_path, "x")
    assert info.value.line == 4


def test_json_missing_key(tmp_path):
    path = write_log(tmp_path, f"{HEADER}\n1 1.0\n")
    path.write_text(path.read_text().replace('"maximization": false, ', ""))
    with pytest.raises(ReaderError, match="missing key 'maximization'"):
        read_ioh(tmp_path, "x")


def test_empty_input(tmp_path):
    with pytest.raises(ReaderError, match="no IOHexperimenter JSON"):
        read_ioh(tmp_path, "x")
    with pytest.raises(FileNotFoundError):
        read_ioh(tmp_path / "missing", "x")


# --- trigger inference -------------------------------------------------------------


def _trace(runs):
    rows = [(i, r, e, y) for i, run in enumerate(runs) for r, (e, y) in enumerate(run)]
    return pl.DataFrame(
        rows, schema=["run_index", "row", "evaluations", "raw_y"], orient="row"
    )


@pytest.mark.parametrize(
    ("runs", "maximization", "kind"),
    [
        ([[(1, 5.0), (2, 6.0), (3, 1.0), (3, 1.0)]], False, "all"),
        ([[(1, 5.0), (3, 4.0), (9, 4.0), (10, 8.0)]], False, "improvement"),
        ([[(1, 5.0), (3, 6.0), (9, 7.0)]], True, "improvement"),
        ([[(1, 5.0), (3, 6.0), (9, 7.0)]], False, None),
        ([[(3, 5.0), (6, 4.0), (9, 3.0)]], False, "improvement"),
        ([[(1, 5.0), (2, 4.0)]], False, None),
        ([], False, None),
    ],
)
def test_infer_triggers(runs, maximization, kind):
    got = infer_triggers(_trace(runs), maximization)
    assert (got[0]["kind"] if got else None) == kind


# --- DataManager on the new reader ---------------------------------------------------


def test_manager_columns_with_positions():
    from iohinspector import DataManager

    manager = DataManager()
    with pytest.warns(UserWarning, match="metadata schema"):
        manager.add_folder(str(ROOT / "io" / "data" / "ioh040" / "attributes"))
    df = manager.load(monotonic=False)
    assert df.height == 12
    assert df.columns[:3] == ["evaluations", "raw_y", "sigma"]
    assert {"x0", "position_x0", "position_x1", "position_x5"} <= set(df.columns)
    assert df.group_by("data_id").len()["len"].sort().to_list() == [2, 3, 3, 4]
    assert df.schema["evaluations"] == pl.UInt64
    overview = manager.overview
    assert overview["instance_exp_attr"].to_list() == ["seven"] * 4
    assert overview["unset"].is_nan().all()
