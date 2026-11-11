"""Tests of ``iohinspector.verify`` and ``iohinspector verify``, including the
mutation tests: a corrupted source or dataset must never pass."""

from __future__ import annotations

import json
import re
import runpy
import shutil
import tarfile
import zipfile
from pathlib import Path

import polars as pl
import pytest

from iohinspector.cli import main
from iohinspector.convert import convert
from iohinspector.verify import verify
from iohinspector.verify._scan import KINDS, METADATA_KINDS

TESTS = Path(__file__).resolve().parent
GOLDEN = TESTS / "golden" / "fixtures"
CUSTOM_MAPPING = {
    "evaluations": "evals",
    "value": "f",
    "run_by": ["alg", "seed"],
    "fields": {
        "algorithm_name": "alg",
        "dimension": {"constant": 2},
        "maximization": {"constant": False},
    },
    "descriptors": ["seed"],
}

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _custom_csv(folder: Path) -> Path:
    folder.mkdir(parents=True)
    rows = ["alg,seed,evals,f"]
    for i in range(1, 6):  # two runs, interleaved
        rows += [f"A,1,{i},{10 - i}.5", f"B,2,{i},{20 - 2 * i}.25"]
    (folder / "data.csv").write_text("\n".join(rows) + "\n")
    return folder / "data.csv"


#: name -> a function that puts the input under ``tmp`` and returns its path.
CORPUS = {
    "ioh040": lambda tmp: _copy(TESTS / "io" / "fixtures" / "ioh040", tmp),
    "ioh0322": lambda tmp: _copy(TESTS / "io" / "fixtures" / "ioh0322", tmp),
    "ioh040_max": lambda tmp: _copy(GOLDEN / "ioh040_max", tmp),
    "coco": lambda tmp: _copy(TESTS / "test_coco_data", tmp),
    "coco_old": lambda tmp: _copy(GOLDEN / "coco_old", tmp),
    "nevergrad": lambda tmp: _copy(TESTS / "io" / "data" / "nevergrad_sample.csv", tmp),
    "custom_csv": lambda tmp: _custom_csv(tmp / "custom"),
}


def _copy(source: Path, tmp: Path) -> Path:
    target = tmp / source.name
    tmp.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy(source, target)
    return target


def _convert(name: str, tmp: Path) -> tuple[Path, Path]:
    source = CORPUS[name](tmp / "in")
    mapping = CUSTOM_MAPPING if name == "custom_csv" else None
    convert(source, tmp / "ds", mapping=mapping)
    return source, tmp / "ds"


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_corpus_verifies(tmp_path, name):
    source, ds = _convert(name, tmp_path)
    report = verify(source, ds)
    assert report.ok, report.summary()
    assert report.files >= 1 and report.runs >= 1 and not report.not_verified
    trace = pl.read_parquet(ds / "trace.parquet")
    if "logged_in" not in trace.columns:
        assert report.rows == trace.height


def test_coco_family_compares_every_file(tmp_path):
    source, ds = _convert("coco_old", tmp_path)
    report = verify(source, ds)
    assert report.files == 4  # .dat and .tdat of two dimensions
    assert "logged_in" in pl.read_parquet(ds / "trace.parquet").columns


def test_archives_verify(tmp_path):
    folder = TESTS / "io" / "fixtures" / "ioh040"
    inner = tmp_path / "inner.zip"
    with zipfile.ZipFile(inner, "w") as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(folder).as_posix())
    outer = tmp_path / "in" / "outer.tar.gz"
    outer.parent.mkdir()
    with tarfile.open(outer, "w:gz") as tar:
        tar.add(inner, arcname="logs/inner.zip")
    convert(outer, tmp_path / "ds")
    report = verify(outer, tmp_path / "ds")
    assert report.ok, report.summary()
    assert report.files == 5


def test_every_reader_kind_has_a_scanner():
    """Each data kind the readers write must be verifiable."""
    kinds: set[str] = set()
    for path in (TESTS.parent / "src" / "iohinspector" / "io").glob("*.py"):
        kinds |= set(re.findall(r'"((?:ioh|coco|nevergrad)[a-z-]*)"', path.read_text()))
    data_like = {k for k in kinds if re.search(r"(dat|csv)$", k)}
    known = set(KINDS) | set(METADATA_KINDS) | {"custom-csv"}
    assert data_like <= known


# --------------------------------------------------------------------------- mutations


def _data_file(ds: Path, source: Path, runs: int = 1) -> Path:
    """The first primary data file of the dataset with at least ``runs`` runs."""
    table = pl.read_parquet(ds / "runs.parquet")
    counts = table.group_by("source_file").len().sort("source_file")
    for path, n in counts.iter_rows():
        if n >= runs:
            base = source if source.is_dir() else source.parent
            return base / path
    raise AssertionError("no data file with enough runs")


def _is_data(line: str, csv: bool) -> bool:
    tokens = line.split()
    if csv:
        return bool(line.strip())
    return (
        bool(tokens)
        and re.fullmatch(r"[+-]?[0-9.]+(?:[eE][+-]?[0-9]+)?", tokens[0]) is not None
    )


def _blocks(lines: list[str]) -> list[tuple[int, int]]:
    """``(start, end)`` of each run: from its header line to the next one."""
    starts = [
        i
        for i, line in enumerate(lines)
        if line.strip()
        and not _is_data(line, False)
        and not re.match(r"\s*%(?!\s*(?:f evaluations|function evaluation))", line)
    ]
    return list(zip(starts, starts[1:] + [len(lines)]))


def drop_row(lines: list[str], csv: bool) -> list[str]:
    rows = [i for i, line in enumerate(lines) if _is_data(line, csv)]
    rows = rows[1:] if csv else rows
    del lines[rows[len(rows) // 2]]
    return lines


def change_digit(lines: list[str], csv: bool) -> list[str]:
    rows = [i for i, line in enumerate(lines) if _is_data(line, csv)]
    rows = rows[1:] if csv else rows
    i = rows[len(rows) // 2]
    sep = "," if csv else " "
    tokens = lines[i].split(sep) if csv else lines[i].split()
    digit = r"\.[0-9]*[1-9]" if csv else r"[1-9]"  # CSV: a value, not a key column
    k = next(j for j in range(1, len(tokens)) if re.search(digit, tokens[j]))
    m = re.search(r"[1-9]", tokens[k])
    assert m is not None
    digit = "1" if m[0] != "1" else "2"
    tokens[k] = tokens[k].replace(m[0], digit, 1)  # the first 1-9 is at m
    lines[i] = sep.join(tokens)
    return lines


def swap_runs(lines: list[str], csv: bool) -> list[str]:
    if csv:  # swap the first rows of two different runs
        first = 1
        second = next(
            i
            for i in range(2, len(lines))
            if lines[i].split(",")[0] != lines[1].split(",")[0]
        )
        lines[first], lines[second] = lines[second], lines[first]
        return lines
    (a0, a1), (b0, b1) = _blocks(lines)[:2]
    return lines[:a0] + lines[b0:b1] + lines[a1:b0] + lines[a0:a1] + lines[b1:]


MUTATIONS = {"drop_row": drop_row, "change_digit": change_digit, "swap_runs": swap_runs}
#: Each mutation must be caught by a per-run check, not only by the file sha256.
RUN_CHECKS = {
    "rows",
    "first_evaluation",
    "last_evaluation",
    "best_value",
    "values",
    "source_rows_sha256",
}


@pytest.mark.parametrize("mutation", sorted(MUTATIONS))
@pytest.mark.parametrize("name", sorted(CORPUS))
def test_source_mutation_is_detected(tmp_path, name, mutation):
    source, ds = _convert(name, tmp_path)
    csv = name in ("nevergrad", "custom_csv")
    target = _data_file(ds, source, runs=2 if mutation == "swap_runs" else 1)
    text = target.read_text()
    lines = MUTATIONS[mutation](text.split("\n"), csv)
    target.write_text("\n".join(lines))
    assert target.read_text() != text
    report = verify(source, ds)
    assert not report.ok
    checks = {m.check for m in report.mismatches}
    assert "source_sha256" in checks
    assert checks & RUN_CHECKS, checks


def test_changed_trace_value_is_detected(tmp_path):
    source, ds = _convert("ioh040", tmp_path)
    trace = pl.read_parquet(ds / "trace.parquet")
    trace = trace.with_columns(
        raw_y=pl.when(pl.int_range(pl.len()) == 5)
        .then(pl.col("raw_y") + 1)
        .otherwise(pl.col("raw_y"))
    )
    trace.write_parquet(ds / "trace.parquet", compression="zstd", statistics=True)
    report = verify(source, ds, validate=False)
    assert {m.check for m in report.mismatches} == {"values", "best_value"} or {
        m.check for m in report.mismatches
    } == {"values"}


def test_changed_stored_hash_is_detected(tmp_path):
    source, ds = _convert("custom_csv", tmp_path)
    runs = pl.read_parquet(ds / "runs.parquet")
    runs = runs.with_columns(source_rows_sha256=pl.lit("0" * 64))
    runs.write_parquet(ds / "runs.parquet")
    report = verify(source, ds, validate=False)
    assert {m.check for m in report.mismatches} == {"source_rows_sha256"}


def test_missing_and_extra_files(tmp_path):
    source, ds = _convert("ioh040", tmp_path)
    dat = _data_file(ds, source)
    dat.unlink()
    (source / "extra.txt").write_text("x\n")
    report = verify(source, ds)
    checks = {m.check for m in report.mismatches}
    assert "source_missing" in checks and "rows_not_in_source" in checks
    assert "extra.txt" in report.unlisted_files


def test_unknown_kind_is_not_verified(tmp_path):
    source, ds = _convert("nevergrad", tmp_path)
    manifest = json.loads((ds / "manifest.json").read_text())
    manifest["sources"][0]["kind"] = "future-format"
    (ds / "manifest.json").write_text(json.dumps(manifest))
    report = verify(source, ds, validate=False)
    assert report.not_verified == ["nevergrad_sample.csv (future-format)"]
    assert {m.check for m in report.mismatches} == {"rows_not_in_source"}


def test_cli_verify(tmp_path, capsys):
    source, ds = _convert("ioh040", tmp_path)
    assert main(["verify", str(source), str(ds)]) == 0
    report = json.loads(Path(f"{ds}.verify.json").read_text())
    assert report["ok"] is True and report["runs"] == 11
    assert "OK." in capsys.readouterr().out
    change_digit_file = _data_file(ds, source)
    change_digit_file.write_text(
        "\n".join(change_digit(change_digit_file.read_text().split("\n"), False))
    )
    out = tmp_path / "r.json"
    assert main(["verify", str(source), str(ds), "--report", str(out)]) == 1
    assert json.loads(out.read_text())["ok"] is False
    assert main(["verify", str(tmp_path / "nothing"), str(ds)]) == 2


def test_verify_corpus_script(tmp_path, capsys):
    script = runpy.run_path(str(TESTS.parent / "scripts" / "verify_corpus.py"))
    corpus = tmp_path / "corpus"
    _copy(TESTS / "io" / "fixtures" / "ioh040", tmp_path / "x")
    (tmp_path / "x" / "ioh040").rename(corpus)
    (corpus / "junk.txt").write_text("not a log\n")
    assert script["main"]([str(corpus), "--out", str(tmp_path / "out")]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert sum(line.startswith("OK ") for line in lines) == 4
    assert any(line.startswith("ERROR") and "junk.txt" in line for line in lines)
    assert lines[-1] == "1 dataset(s) failed"
