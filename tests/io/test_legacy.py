"""The legacy IOHprofiler reader: data files, merging and variants (WP I06)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import polars as pl
import pytest

from iohinspector.format import validate_dataset
from iohinspector.io import ReaderError, detect_format, read_legacy
from iohinspector.io.legacy import detect_legacy

from .legacy_writer import LegacySet

ROOT = Path(__file__).resolve().parents[1]
EXTDATA = ROOT / "golden" / "fixtures" / "extdata"
COCO = ROOT / "test_coco_data"
DATA_SUFFIXES = (".dat", ".cdat", ".idat", ".tdat")
NUMBER = re.compile(r"^[+-]?(?:[0-9]|\.[0-9]|inf|nan)", re.IGNORECASE)

RUNS = [
    [5.0, 4.0, 4.0, 6.0, 3.0, 3.0, 2.5, 2.7, 1.0, 1.0, 1.5],
    [9.0, 9.0, 8.0, 7.5, 7.5],
    [3.0],
]


def data_lines(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    return [
        " ".join(s.split()) for s in lines if s.strip() and NUMBER.match(s.split()[0])
    ]


def write_and_validate(result, tmp_path: Path) -> None:
    result.write(tmp_path / "dataset")
    validate_dataset(tmp_path / "dataset").raise_for_errors()


# --------------------------------------------------------------------------- extdata


def test_extdata_reads_every_line(tmp_path):
    result = read_legacy(EXTDATA, "extdata")
    files = sorted(p for p in EXTDATA.rglob("*") if p.suffix in DATA_SUFFIXES)
    counts = {
        s["path"]: s["data_rows"]
        for s in result.manifest["sources"]
        if "data_rows" in s
    }
    assert counts == {
        p.relative_to(EXTDATA).as_posix(): len(data_lines(p)) for p in files
    }
    assert result.trace.height == sum(counts.values()) == 1386
    assert result.runs.height == 30
    assert sorted(result.runs["function_id"].unique()) == [1, 2, 3]
    assert result.runs["maximization"].all()
    assert result.runs["evals_used"].unique().to_list() == [5000]
    write_and_validate(result, tmp_path)


def test_extdata_values_and_hashes():
    result = read_legacy(EXTDATA / "ONE_PLUS_LAMDA_EA", "x")
    path = EXTDATA / "ONE_PLUS_LAMDA_EA/data_f1/IOHprofiler_f1_DIM100_i1.dat"
    lines = data_lines(path)
    tokens = [line.split(" ") for line in lines]
    assert result.trace["evaluations"].to_list() == [int(t[0]) for t in tokens]
    assert result.trace["raw_y"].to_list() == [float(t[1]) for t in tokens]
    assert result.trace["l"].to_list() == [float(t[7]) for t in tokens]
    assert result.trace.columns[4:] == [
        "best-so-far f(x)",
        "current af(x)+b",
        "best af(x)+b",
        "mutation_rate",
        "lambda",
        "l",
    ]
    # runs split on header lines; hashes over the normalised data lines
    text = path.read_text().splitlines()
    starts = [i for i, s in enumerate(text) if s.startswith('"')] + [len(text)]
    for k, run in enumerate(result.runs.iter_rows(named=True)):
        first, end = starts[k] + 1, starts[k + 1]
        part = [" ".join(s.split()) for s in text[first:end]]
        digest = hashlib.sha256(("\n".join(part) + "\n").encode()).hexdigest()
        assert run["source_rows_sha256"] == digest
        assert run["source_run_index"] == k and run["instance"] == 1


def test_extdata_descriptors_and_names():
    result = read_legacy(EXTDATA / "ONE_PLUS_LAMDA_EA_ws", "x")
    assert result.runs["function_name"].unique().sort().to_list() == ["bla", "blubb"]
    assert result.runs["suite"].unique().to_list() == [""]
    keys = result.descriptors["key"].unique().sort().to_list()
    assert keys == ["algorithm_info", "algorithm_name"]
    codes = {w["code"] for w in result.manifest["warnings"]}
    assert codes == {"function_ids_assigned", "maximization_inferred"}


def test_detection():
    assert detect_format(EXTDATA) == "ioh-legacy"
    assert detect_legacy(EXTDATA / "ONE_PLUS_LAMDA_EA" / "IOHprofiler_f1_i1.info")
    assert not detect_legacy(COCO)
    with pytest.raises(ReaderError, match="no legacy IOHprofiler .info file"):
        read_legacy(COCO, "x")


def test_run_ids_are_stable():
    a = read_legacy(EXTDATA, "extdata").runs["run_id"]
    b = read_legacy(EXTDATA, "extdata").runs["run_id"]
    assert a.to_list() == b.to_list() and a.n_unique() == 30


# --------------------------------------------------------------------- data-file kinds


def test_kinds_merge_identical_rows(tmp_path):
    info = LegacySet(RUNS, kinds=("dat", "cdat", "idat", "tdat")).write(tmp_path)
    result = read_legacy(info, "x")
    trace = result.trace
    # .cdat holds every evaluation, and every other row is identical to one of it
    assert trace.height == sum(len(r) for r in RUNS)
    assert not trace["merge_conflict"].any()
    first = trace.filter(pl.col("row") == 0)["logged_in"].to_list()
    assert first == ["dat+tdat+cdat", "dat+tdat+cdat", "dat+tdat+cdat+idat"]
    run0 = trace.head(len(RUNS[0]))
    assert run0["logged_in"].to_list()[2:4] == ["cdat+idat", "cdat"]
    assert run0["raw_y"].to_list() == RUNS[0]
    rows = {
        s["path"].rsplit(".", 1)[1]: s["data_rows"]
        for s in result.manifest["sources"][1:]
    }
    for kind, n in rows.items():
        path = tmp_path / "data_f1" / f"IOHprofiler_f1_DIM2.{kind}"
        assert n == len(data_lines(path))
    assert result.runs["source_file"].unique().to_list() == [
        "data_f1/IOHprofiler_f1_DIM2.dat"
    ]
    triggers = {s["kind"]: s.get("triggers") for s in result.manifest["sources"]}
    assert triggers["ioh-legacy-dat"] == [
        {"kind": "improvement", "params": {"inferred": True}}
    ]
    assert triggers["ioh-legacy-cdat"] == [{"kind": "all"}]
    write_and_validate(result, tmp_path)


def test_conflicting_rows_are_kept_and_flagged(tmp_path):
    info = LegacySet(RUNS, kinds=("dat", "tdat")).write(tmp_path)
    tdat = tmp_path / "data_f1" / "IOHprofiler_f1_DIM2.tdat"
    tdat.write_text(tdat.read_text().replace("2 +4.00000e+00", "2 +4.50000e+00", 1))
    result = read_legacy(info, "x")
    clash = result.trace.filter("merge_conflict")
    assert clash["evaluations"].to_list() == [2, 2]
    assert sorted(clash["logged_in"].to_list()) == ["dat", "tdat"]
    assert sorted(clash["raw_y"].to_list()) == [4.0, 4.5]
    assert "merge_conflict" in {w["code"] for w in result.manifest["warnings"]}


def test_repeated_rows_within_a_file_stay(tmp_path):
    info = LegacySet(RUNS[:1], kinds=("dat", "cdat")).write(tmp_path)
    cdat = tmp_path / "data_f1" / "IOHprofiler_f1_DIM2.cdat"
    lines = cdat.read_text().splitlines()
    cdat.write_text("\n".join(lines + [lines[-1]]) + "\n")
    trace = read_legacy(info, "x").trace
    assert trace.height == len(RUNS[0]) + 1
    assert trace.tail(2)["logged_in"].to_list() == ["dat+cdat", "cdat"]
    assert not trace["merge_conflict"].any()


def test_unsorted_evaluations_are_not_merged(tmp_path):
    info = LegacySet(RUNS[:1], kinds=("dat", "cdat")).write(tmp_path)
    cdat = tmp_path / "data_f1" / "IOHprofiler_f1_DIM2.cdat"
    lines = cdat.read_text().splitlines()
    lines[2], lines[3] = lines[3], lines[2]
    cdat.write_text("\n".join(lines) + "\n")
    result = read_legacy(info, "x")
    n_dat = len(data_lines(tmp_path / "data_f1" / "IOHprofiler_f1_DIM2.dat"))
    assert result.trace.height == n_dat + len(RUNS[0])
    assert result.trace["logged_in"].to_list()[:n_dat] == ["dat"] * n_dat
    assert "unsorted_evaluations" in {w["code"] for w in result.manifest["warnings"]}


def test_primary_file_when_dat_is_missing(tmp_path):
    info = LegacySet(RUNS, kinds=("cdat", "tdat")).write(tmp_path)
    result = read_legacy(info, "x")
    assert result.runs["source_file"].unique().to_list() == [
        "data_f1/IOHprofiler_f1_DIM2.tdat"
    ]
    assert result.trace.height == sum(len(r) for r in RUNS)


def test_best_so_far_warning_without_improvements(tmp_path):
    runs = [[5.0, 1.0, 4.0, 4.0, 3.0, 3.0]]
    info = LegacySet(runs, kinds=("idat",), interval=3).write(tmp_path)
    result = read_legacy(info, "x")
    assert result.trace["raw_y"].to_list() == [4.0, 3.0]
    assert "best_so_far_differs" in {w["code"] for w in result.manifest["warnings"]}


# --------------------------------------------------------------------------- variants


def test_parameters_positions_and_run_attributes(tmp_path):
    n = len(RUNS[0])
    data = LegacySet(
        RUNS[:1],
        params={"sigma": [[0.5] * n], "instance": [[1.0] * n]},
        positions=[[[i, -i] for i in range(n)]],
        header_extra=', seed = 12, note = "x", dynamicAttribute = "s|t"',
    )
    info = data.write(tmp_path)
    info.write_text(
        info.read_text().replace(f"|{min(RUNS[0]):g}", f"|{min(RUNS[0]):g};7;0.5")
    )
    result = read_legacy(info, "x")
    assert (
        "x0" not in result.trace.columns and "instance_logged" in result.trace.columns
    )
    assert result.positions is not None
    assert result.positions["x1"].to_list()[:3] == [0.0, -1.0, -4.0]
    assert result.manifest["renamed_columns"] == [
        {
            "from": "instance",
            "to": "instance_logged",
            "source": "data_f1/IOHprofiler_f1_DIM2.dat",
            "table": "trace",
        }
    ]
    d = {
        r["key"]: (r["type"], r["origin"], r["value"])
        for r in result.descriptors.iter_rows(named=True)
    }
    assert d["seed"] == ("int", "experiment", "12")
    assert d["note"] == ("string", "experiment", "x")
    assert d["s"] == ("float", "run", "7.0") and d["t"] == ("float", "run", "0.5")
    write_and_validate(result, tmp_path)


def test_two_column_files(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "f.dat").write_text(
        '"function evaluation" "best-so-far f(x)"\n1 3\n4 2\n'
        '"function evaluation" "best-so-far f(x)"\n1 5\r\n'
    )
    (tmp_path / "a.info").write_text(
        "funcId = 3, DIM = 4, algId = 'A'\n%\nd/f.dat, 1:4|2, 2:1|5\n"
    )
    result = read_legacy(tmp_path, "x")
    assert result.trace.columns == ["run_id", "row", "evaluations", "raw_y"]
    assert result.trace["raw_y"].to_list() == [3.0, 2.0, 5.0]
    assert result.runs["best_y"].to_list() == [2.0, 5.0]
    assert result.manifest["sources"][1]["precision"] == {"notation": "unknown"}


def test_blocks_sharing_a_data_file(tmp_path):
    (tmp_path / "d").mkdir()
    header = '"function evaluation" "current f(x)"\n'
    (tmp_path / "d" / "f.dat").write_text(
        header + "1 1.0\n" + header + "1 2.0\n" + header + "1 3.0\n"
    )
    (tmp_path / "a.info").write_text(
        "funcId = 1, DIM = 5, algId = 'A', p = 1\n%\nd/f.dat, 1:1|1\n"
        "funcId = 1, DIM = 5, algId = 'A', p = 2\n%\nd/f.dat, 2:1|2, 3:1|3\n"
    )
    result = read_legacy(tmp_path, "x")
    assert result.runs["instance"].to_list() == [1, 2, 3]
    p = result.descriptors.filter(pl.col("key") == "p").sort("run_id")
    by_run = dict(zip(p["run_id"], p["value"], strict=True))
    assert [by_run[r] for r in result.runs["run_id"]] == ["1", "2", "2"]


def test_missing_data_file_and_run_count_mismatch(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "f.dat").write_text('"evaluations" "raw y"\n1 1.0\n')
    (tmp_path / "a.info").write_text(
        "funcId = 1, DIM = 5, algId = 'A'\n%\nd/f.dat, 1:1|1, 2:1|1\n"
        "funcId = 2, DIM = 5, algId = 'A'\n%\nd/g.dat, 1:1|1\n"
    )
    (tmp_path / "d" / "stray.cdat").write_text('"e" "f"\n1 1\n')
    result = read_legacy(tmp_path, "x")
    assert result.runs.height == 3 and result.trace.height == 1
    assert result.runs["source_rows_sha256"].null_count() == 2
    codes = {w["code"] for w in result.manifest["warnings"]}
    assert {
        "run_count_mismatch",
        "missing_data_file",
        "unreferenced_data_file",
    } <= codes
    missing = [s for s in result.manifest["sources"] if s["path"] == "d/g.dat"]
    assert missing[0]["sha256"] is None
    write_and_validate(result, tmp_path)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('"e" "f"\n1 1.0\n2\n', r"d/f.dat:3: 1 values, but the header has 2 columns"),
        ('"e" "f"\n1 1.0\n2 abc\n', r"d/f.dat:3: column 'f': 'abc' is not a number"),
        ('"e" "f"\n1.5 1.0\n', r"d/f.dat:2: column 'e': '1.5' is not an integer"),
        ('1 1.0\n"e" "f"\n', r"d/f.dat:1: data line before the first header"),
        ('"e" "e"\n', r"d/f.dat:1: the header names \['e'\] more than once"),
    ],
)
def test_malformed_data_files(tmp_path, content, message):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "f.dat").write_text(content)
    (tmp_path / "a.info").write_text("funcId = 1, DIM = 5\n%\nd/f.dat, 1:2|1\n")
    with pytest.raises(ReaderError, match=message):
        read_legacy(tmp_path, "x")


def test_files_with_different_run_counts_fail(tmp_path):
    info = LegacySet(RUNS, kinds=("dat", "cdat")).write(tmp_path)
    cdat = tmp_path / "data_f1" / "IOHprofiler_f1_DIM2.cdat"
    text = cdat.read_text()
    cdat.write_text(text[: text.index('"function evaluation"', 1)])
    with pytest.raises(
        ReaderError, match=r"IOHprofiler_f1_DIM2.cdat: 1 run\(s\), but .* has 3"
    ):
        read_legacy(info, "x")
