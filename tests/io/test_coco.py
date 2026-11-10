"""Tests of the COCO single-objective reader (``iohinspector.io.coco``).

The checks of values against the source parse the files here, independently of
the reader, so that a shared bug cannot hide.
"""

from __future__ import annotations

import hashlib
import io
import re
import tarfile
import zipfile
from pathlib import Path

import polars as pl
import pytest

from iohinspector.format import scan_dataset, validate_dataset
from iohinspector.io import READERS, ReaderError, detect_format, read
from iohinspector.io.coco import detect_coco, read_coco

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "test_coco_data"
INFO = next(FIXTURE.rglob("*.info"))
NEW = (
    "% f evaluations | g evaluations | best noise-free fitness - Fopt ({fopt}) + "
    "sum g_i+ | measured fitness | best measured fitness or single-digit g-values | "
    "x1 | x2..."
)
OLD = (
    "% function evaluation | noise-free fitness - Fopt ({fopt}) | best noise-free "
    "fitness - Fopt | measured fitness | best measured fitness | x1 | x2..."
)
HEADER = (
    "suite = 'bbob', funcId = 1, DIM = 2, Precision = 1.000e-08, algId = 'alg', "
    "coco_version = '2.6', logger = 'bbob', data_format = 'bbob-new2'"
)


# ----------------------------------------------------------------- independent parsing


def file_runs(path: Path) -> list[list[list[str]]]:
    """Data lines of a COCO data file as tokens, one list per run."""
    runs: list[list[list[str]]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        if line.startswith("%"):
            if re.match(r"%\s*(f evaluations|function evaluation)", line):
                runs.append([])
            continue
        runs[-1].append(line.split())
    return runs


def as_values(tokens: list[str]) -> tuple:
    return (int(tokens[0]), int(tokens[1]) if "." not in tokens[1] else float(tokens[1]),
            *(float(t) for t in tokens[2:]))


def trace_rows(result, kind: str | None = None) -> list[list[tuple]]:
    """The trace (with positions) per run in source order, as value tuples."""
    trace = result.trace
    if result.positions is not None:
        trace = trace.join(result.positions, on=["run_id", "row"], how="left")
    if kind is not None and "logged_in" in trace.columns:
        trace = trace.filter(pl.col("logged_in").str.split("+").list.contains(kind))
    second = "g_evaluations" if "g_evaluations" in trace.columns else "current_delta_f"
    xs = sorted((c for c in trace.columns if re.fullmatch(r"x\d+", c)), key=lambda c: int(c[1:]))
    out = []
    for rid in result.runs.sort("source_file", "source_run_index")["run_id"]:
        rows = trace.filter(pl.col("run_id") == rid).sort("row")
        cols = ["evaluations", second, "raw_y", "measured_f", "best_measured_f", *xs]
        out.append([tuple(v for v in r if v is not None) for r in rows.select(cols).iter_rows()])
    return out


def write_coco(  # pylint: disable=too-many-arguments
    folder: Path,
    files: dict[str, str],
    info: str | None = None,
    *,
    entries: str = "1:3|1.0e-01, 2:2|5.0e-01",
    header: str = HEADER,
    name: str = "bbobexp_f1.info",
) -> Path:
    """An ``.info`` file and data files ``data_f1/bbobexp_f1_DIM2.<kind>``."""
    (folder / "data_f1").mkdir(parents=True, exist_ok=True)
    if info is None:
        info = f"{header}\n% alg: some text\ndata_f1/bbobexp_f1_DIM2.dat, {entries}\n"
    (folder / name).write_text(info)
    for kind, text in files.items():
        (folder / "data_f1" / f"bbobexp_f1_DIM2.{kind}").write_text(text)
    return folder


def run_text(fopt: float, rows: list[str], style: str = NEW) -> str:
    return "\n".join([style.format(fopt=f"{fopt:.12e}"), *rows]) + "\n"


DAT = run_text(10.0, ["1 0 +1.0e+00 +1.1e+01 +1.1e+01 +1.0e-01 +2.0e-01",
                      "3 0 +1.0e-01 +1.01e+01 +1.01e+01 +3.0e-01 +4.0e-01"]) + run_text(
    -5.0, ["1 0 +5.0e-01 -4.5e+00 -4.5e+00 +5.0e-01 +6.0e-01"])
TDAT = run_text(10.0, ["1 0 +1.0e+00 +1.1e+01 +1.1e+01 +1.0e-01 +2.0e-01",
                       "2 0 +1.0e+00 +1.2e+01 +1.1e+01 +7.0e-01 +8.0e-01",
                       "3 0 +1.0e-01 +1.01e+01 +1.01e+01 +3.0e-01 +4.0e-01"]) + run_text(
    -5.0, ["1 0 +5.0e-01 -4.5e+00 -4.5e+00 +5.0e-01 +6.0e-01",
           "2 0 +5.0e-01 -4.4e+00 -4.5e+00 +9.0e-01 +1.0e+00"])


def codes(result) -> list[str]:
    return [w["code"] for w in result.manifest.get("warnings", [])]


# ------------------------------------------------------------------------- the fixture


def test_reader_table_and_detection(tmp_path):
    assert "coco" in READERS
    assert detect_format(FIXTURE) == "coco"
    assert detect_format(INFO) == "coco"
    assert detect_coco(FIXTURE)
    assert not detect_coco(ROOT / "io" / "fixtures" / "ioh040")
    assert not detect_coco(ROOT / "golden" / "fixtures" / "extdata")
    assert not detect_coco(tmp_path)
    assert read(FIXTURE, "x").runs.height == 15


def test_fixture_runs():
    result = read_coco(FIXTURE, "fixture")
    runs = result.runs.sort("source_run_index")
    info = INFO.read_text().splitlines()[2]
    entries = re.findall(r"(\d+):(\d+)\|", info)
    assert runs["instance"].to_list() == [int(i) for i, _ in entries]
    assert runs["evals_used"].to_list() == [int(e) for _, e in entries]
    dat = INFO.parent / "data_f1" / "bbobexp_f1_DIM2_i1.dat"
    fopts = re.findall(r"Fopt \(([^)]*)\)", dat.read_text())
    assert runs["optimum"].to_list() == [float(f) for f in fopts]
    assert set(runs["suite"]) == {"bbob"}
    assert set(runs["function_id"]) == {1}
    assert set(runs["dimension"]) == {2}
    assert not any(runs["maximization"])
    assert runs["best_y"].is_null().all()
    assert set(runs["source_file"]) == {
        dat.relative_to(FIXTURE).as_posix()
    }


def test_fixture_descriptors():
    result = read_coco(FIXTURE, "fixture")
    d = result.descriptors
    one = d.filter(pl.col("run_id") == result.runs["run_id"][0])
    values = dict(zip(one["key"], one["value"]))
    assert values["algorithm_name"] == "BFGS-scipy-2019"
    assert values["algorithm_info"].startswith("BFGS-scipy-2019: BFGS from SciPy")
    assert values["Precision"] == "1e-08"
    assert values["coco_version"] == "2.3.6"
    assert values["data_format"] == "bbob-new2"
    assert values["final_delta_f"] == "2.7e-13"
    origins = dict(zip(one["key"], one["origin"]))
    assert origins["Precision"] == "experiment"
    assert origins["final_delta_f"] == "run"


@pytest.mark.parametrize("kind", ["dat", "tdat", "rdat"])
def test_fixture_lossless_per_file(kind):
    """Every row of every data file is in the trace, flagged with its file."""
    result = read_coco(FIXTURE, "fixture")
    path = next(FIXTURE.rglob(f"*.{kind}"))
    expected = [[as_values(t) for t in run] for run in file_runs(path)]
    assert trace_rows(result, kind) == expected
    source = next(s for s in result.manifest["sources"] if s["path"].endswith(kind))
    assert source["data_rows"] == sum(len(r) for r in expected)
    assert source["kind"] == f"coco-{kind}"


def test_fixture_source_rows_sha256():
    result = read_coco(FIXTURE, "fixture")
    dat = next(FIXTURE.rglob("*.dat"))
    hashes = [
        hashlib.sha256("".join(" ".join(t) + "\n" for t in run).encode()).hexdigest()
        for run in file_runs(dat)
    ]
    assert result.runs.sort("source_run_index")["source_rows_sha256"].to_list() == hashes


def test_fixture_round_trip(tmp_path):
    result = read_coco(FIXTURE, "fixture")
    result.write(tmp_path / "ds")
    assert validate_dataset(tmp_path / "ds").ok
    stored = scan_dataset(tmp_path / "ds")
    assert stored.trace.collect().height == result.trace.height
    assert stored.runs.collect()["instance"].sort().to_list() == list(range(1, 16))
    manifest = stored.manifest
    kinds = sorted(s.get("kind", "") for s in manifest["sources"])
    assert kinds == ["coco-dat", "coco-info", "coco-rdat", "coco-tdat"]


# ---------------------------------------------------------------------- merging files


def test_identical_rows_merged_and_others_kept(tmp_path):
    result = read_coco(write_coco(tmp_path, {"dat": DAT, "tdat": TDAT}), "m")
    trace = result.trace.join(result.runs.select("run_id", "source_run_index"), on="run_id")
    first = trace.filter(pl.col("source_run_index") == 0).sort("row")
    assert first["evaluations"].to_list() == [1, 2, 3]
    assert first["logged_in"].to_list() == ["dat+tdat", "tdat", "dat+tdat"]
    assert not first["merge_conflict"].any()
    assert "merge_conflict" not in codes(result)


def test_conflicting_rows_kept_and_flagged(tmp_path):
    tdat = TDAT.replace("3 0 +1.0e-01 +1.01e+01 +1.01e+01 +3.0e-01 +4.0e-01",
                        "3 0 +1.0e-01 +1.01e+01 +1.01e+01 +3.5e-01 +4.0e-01")
    result = read_coco(write_coco(tmp_path, {"dat": DAT, "tdat": tdat}), "m")
    rows = result.trace.filter(pl.col("evaluations") == 3)
    assert rows["logged_in"].to_list() == ["dat", "tdat"]
    assert rows["merge_conflict"].to_list() == [True, True]
    assert "merge_conflict" in codes(result)
    positions = result.positions.join(rows.select("run_id", "row"), on=["run_id", "row"])
    assert sorted(positions["x0"].to_list()) == [0.3, 0.35]


def test_rows_repeated_in_one_file_stay(tmp_path):
    dat = run_text(1.0, ["1 0 +1.0e+00 +2.0e+00 +2.0e+00", "1 0 +1.0e+00 +2.0e+00 +2.0e+00"])
    result = read_coco(write_coco(tmp_path, {"dat": dat}, entries="1:1|1.0e+00"), "m")
    assert result.trace.height == 2
    assert result.positions is None


def test_different_run_counts_fail(tmp_path):
    with pytest.raises(ReaderError, match="different numbers of runs"):
        read_coco(write_coco(tmp_path, {"dat": DAT, "tdat": run_text(1.0, [])}), "m")


def test_restart_lines_do_not_split_runs(tmp_path):
    rdat = run_text(10.0, ["% restart: independent restart",
                           "2 0 +1.0e+00 +1.2e+01 +1.1e+01 +7.0e-01 +8.0e-01",
                           "% restart: independent restart"]) + run_text(-5.0, [])
    result = read_coco(write_coco(tmp_path, {"dat": DAT, "tdat": TDAT, "rdat": rdat}), "m")
    assert result.runs.height == 2
    row = result.trace.filter(pl.col("evaluations") == 2).sort("row")
    assert row["logged_in"].to_list() == ["tdat+rdat", "tdat"]


def test_dat_missing_tdat_is_source_file(tmp_path):
    result = read_coco(write_coco(tmp_path, {"tdat": TDAT}), "m")
    assert set(result.runs["source_file"]) == {"data_f1/bbobexp_f1_DIM2.tdat"}
    assert result.trace.height == 5
    assert "logged_in" not in result.trace.columns


# ---------------------------------------------------------------------- variants


def test_old_header_style(tmp_path):
    dat = run_text(79.48, ["1 +5.8e-01 +5.8e-01 +8.0e+01 +8.0e+01 -3.2e-01 -6.6e-01",
                           "4 +2.7e-13 +2.7e-13 +7.9e+01 +7.9e+01"], style=OLD)
    info = ("funcId = 1, DIM = 2, Precision = 1.000e-08, algId = 'ALPS'\n"
            "% 0001 ALPS\ndata_f1/bbobexp_f1_DIM2.dat, 1:4|-1.0e-08\n")
    result = read_coco(write_coco(tmp_path, {"dat": dat}, info), "o")
    assert result.trace["current_delta_f"].to_list() == [0.58, 2.7e-13]
    assert result.positions.height == 1  # only the first row wrote x
    assert result.runs["optimum"].to_list() == [79.48]
    assert result.runs["suite"].to_list() == ["bbob"]
    assert "suite_inferred" in codes(result)
    values = dict(zip(result.descriptors["key"], result.descriptors["value"]))
    assert values["algorithm_info"] == "0001 ALPS"
    assert values["final_delta_f"] == "-1e-08"
    assert "coco_version" not in values


def test_noisy_suite_inferred(tmp_path):
    info = HEADER.replace("suite = 'bbob', ", "").replace("funcId = 1", "funcId = 101")
    result = read_coco(write_coco(tmp_path, {"dat": DAT}, header=info), "n")
    assert set(result.runs["suite"]) == {"bbob-noisy"}
    assert set(result.runs["function_id"]) == {101}


def test_unfinalized_runs_and_messy_info(tmp_path):
    info = (f"\n{HEADER}\r\n% alg\r\n"
            "data_f1/bbobexp_f1_DIM2.dat, 1:3|1.0e-01, 2\r\n"
            ":2001991|1.4e-03, 5\n\n")
    result = read_coco(write_coco(tmp_path, {"dat": DAT}, info), "u")
    runs = result.runs.sort("source_run_index")
    assert runs["instance"].to_list() == [1, 2]
    assert runs["evals_used"].to_list() == [3, None]
    assert {"unfinalized_run", "info_line_skipped"} <= set(codes(result))


def test_run_count_mismatch(tmp_path):
    result = read_coco(write_coco(tmp_path, {"dat": DAT}, entries="7:3|1.0e-01"), "c")
    runs = result.runs.sort("source_run_index")
    assert runs["instance"].to_list() == [7, None]
    assert "run_count_mismatch" in codes(result)
    assert result.descriptors.filter(pl.col("key") == "algorithm_name").height == 2


def test_missing_data_files(tmp_path):
    result = read_coco(write_coco(tmp_path, {}), "missing")
    assert result.runs.height == 2
    assert result.trace.height == 0
    assert "missing_data_file" in codes(result)
    entry = next(s for s in result.manifest["sources"] if s["path"].endswith(".dat"))
    assert entry["sha256"] is None


def test_reserved_header_key_renamed(tmp_path):
    header = HEADER + ", instance = 3, budget = 'large'"
    result = read_coco(write_coco(tmp_path, {"dat": DAT}, header=header), "r")
    keys = set(result.descriptors["key"])
    assert {"instance_experiment", "budget_experiment"} <= keys
    renamed = {r["from"]: r["to"] for r in result.manifest["renamed_columns"]}
    assert renamed == {"instance": "instance_experiment", "budget": "budget_experiment"}


def test_folder_header_and_unreferenced_files(tmp_path):
    write_coco(tmp_path / "sub", {"dat": DAT, "mdat": run_text(1.0, [])})
    info = HEADER + ", folder = 'sub'\n% alg\ndata_f1/bbobexp_f1_DIM2.dat, 1:3|1.0e-01, 2:1|5.0e-01\n"
    (tmp_path / "x.info").write_text(info)
    (tmp_path / "sub" / "bbobexp_f1.info").unlink()
    (tmp_path / "sub" / "data_f1" / "other.dat").write_text(DAT)
    result = read_coco(tmp_path, "f")
    assert set(result.runs["source_file"]) == {"sub/data_f1/bbobexp_f1_DIM2.dat"}
    unreferenced = [w["source"] for w in result.manifest["warnings"]
                    if w["code"] == "unreferenced_data_file"]
    assert unreferenced == ["sub/data_f1/bbobexp_f1_DIM2.mdat", "sub/data_f1/other.dat"]


def test_other_info_files_are_skipped(tmp_path):
    write_coco(tmp_path, {"dat": DAT})
    (tmp_path / "legacy.info").write_text(
        "suite = 'PBO', funcId = 1, DIM = 16, algId = 'ea'\n%\n"
        "data_f1/IOHprofiler_f1_DIM16.dat, 1:10|16\n"
    )
    (tmp_path / "data_f1" / "IOHprofiler_f1_DIM16.dat").write_text(
        '"function evaluation" "current f(x)"\n1 16\n'
    )
    result = read_coco(tmp_path, "s")
    assert result.runs.height == 2
    assert "skipped_file" in codes(result)
    with pytest.raises(ReaderError, match="not a COCO"):
        read_coco(tmp_path / "legacy.info", "s")


@pytest.mark.parametrize(
    ("dat", "line", "message"),
    [
        ("1 0 1 2 3\n" + DAT, 1, "before the first header"),
        (run_text(1.0, ["1 0 +1.0e+00 +2.0e+00"]), 2, "4 fields"),
        (run_text(1.0, ["1.5 0 +1.0e+00 +2.0e+00 +2.0e+00"]), 2, "not an integer"),
        (run_text(1.0, ["1 0 +1.0e+00 abc +2.0e+00"]), 2, "not a number"),
        (run_text(1.0, ["1 0 1 2 2"]) + run_text(1.0, [], style=OLD), 3, "header style"),
    ],
)
def test_malformed_data_file(tmp_path, dat, line, message):
    write_coco(tmp_path, {"dat": dat}, entries="1:1|1.0e+00")
    with pytest.raises(ReaderError, match=message) as err:
        read_coco(tmp_path, "bad")
    assert err.value.line == line
    assert err.value.path == "data_f1/bbobexp_f1_DIM2.dat"


@pytest.mark.parametrize(
    ("info", "message"),
    [
        ("funcId = 1, Precision = 1e-8, algId = 'a'\n%\nd.dat, 1:1|1\n", "no DIM"),
        (f"{HEADER}\n%\ndata_f1/bbobexp_f1_DIM2.dat, 1:x|1\n", "run entry"),
        (f"{HEADER}\n%\ndata_f1/a.dat, 1:1|1, data_f1/b.dat, 2:1|1\n", "one is supported"),
    ],
)
def test_malformed_info_file(tmp_path, info, message):
    write_coco(tmp_path, {"dat": DAT}, info)
    with pytest.raises(ReaderError, match=message) as err:
        read_coco(tmp_path / "bbobexp_f1.info", "bad")
    assert err.value.path == "bbobexp_f1.info"


def test_special_values(tmp_path):
    dat = run_text(1.0, ["1 0 +inf nan -Inf", "2 0 +1.0e+00 NaN +2.0e+00"])
    result = read_coco(write_coco(tmp_path, {"dat": dat}, entries="1:2|1.0e+00"), "s")
    raw = result.trace["raw_y"].to_list()
    assert raw[0] == float("inf") and raw[1] == 1.0


def test_run_ids_stable(tmp_path):
    a = read_coco(FIXTURE, "same")
    b = read_coco(FIXTURE, "same")
    assert a.runs["run_id"].to_list() == b.runs["run_id"].to_list()


# ------------------------------------------------------------------------- archives


def _tgz(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo("alg/evil")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tar.addfile(link)
    return buf.getvalue()


def _folder_files(folder: Path, prefix: str) -> dict[str, bytes]:
    return {
        f"{prefix}{p.relative_to(folder).as_posix()}": p.read_bytes()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def test_nested_archives(tmp_path):
    src = write_coco(tmp_path / "src", {"dat": DAT, "tdat": TDAT})
    inner = _tgz(_folder_files(src, "alg/"))
    outer = tmp_path / "in" / "outer.zip"
    outer.parent.mkdir()
    with zipfile.ZipFile(outer, "w") as zf:
        zf.writestr("inner.tgz", inner)
        zf.writestr("../escape.info", b"x")
        zf.writestr("README.txt", b"hello")
    assert detect_coco(outer.parent)
    result = read_coco(outer.parent, "arch")
    assert set(result.runs["source_file"]) == {
        "outer.zip/inner.tgz/alg/data_f1/bbobexp_f1_DIM2.dat"
    }
    archives = {a["path"]: a.get("archive") for a in result.manifest["archives"]}
    assert archives == {"outer.zip": None, "outer.zip/inner.tgz": "outer.zip"}
    sources = {s["path"]: s.get("archive") for s in result.manifest["sources"]}
    assert sources["outer.zip/inner.tgz/alg/bbobexp_f1.info"] == "outer.zip/inner.tgz"
    skipped = [w for w in result.manifest["warnings"] if w["code"] == "skipped_archive_member"]
    assert len(skipped) == 2
    archive = result.descriptors.filter(pl.col("key") == "archive")["value"].unique()
    assert archive.to_list() == ["outer.zip"]
    result.write(tmp_path / "ds")
    assert validate_dataset(tmp_path / "ds").ok


def test_archive_as_input(tmp_path):
    src = write_coco(tmp_path / "src", {"dat": DAT})
    path = tmp_path / "alg.tgz"
    path.write_bytes(_tgz(_folder_files(src, "alg/")))
    assert detect_format(path) == "coco"
    result = read_coco(path, "a")
    assert set(result.runs["source_file"]) == {"alg.tgz/alg/data_f1/bbobexp_f1_DIM2.dat"}
    assert result.manifest["archives"][0]["sha256"] == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_broken_archive_is_skipped(tmp_path):
    write_coco(tmp_path, {"dat": DAT})
    (tmp_path / "broken.zip").write_bytes(b"not a zip")
    result = read_coco(tmp_path, "b")
    assert "skipped_archive" in codes(result)


def test_no_info_file(tmp_path):
    (tmp_path / "x.dat").write_text(DAT)
    with pytest.raises(ReaderError, match="no COCO"):
        read_coco(tmp_path, "none")
