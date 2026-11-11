"""Tests of ``iohinspector.convert`` and ``iohinspector convert``."""

from __future__ import annotations

import json
import shutil
import tarfile
import zipfile
from pathlib import Path

import polars as pl
import pytest

from iohinspector.cli import main
from iohinspector.convert import ConvertError, PathPattern, convert, parse_tag
from iohinspector.format import scan_dataset, validate_dataset
from iohinspector.io import ReaderError, read_ioh

TESTS = Path(__file__).resolve().parent
IOH = TESTS / "io" / "fixtures" / "ioh040"
NEVERGRAD = TESTS / "io" / "data" / "nevergrad_sample.csv"
COCO = TESTS / "test_coco_data"

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _descriptors(path: Path, key: str) -> pl.DataFrame:
    return scan_dataset(path).descriptors.filter(pl.col("key") == key).collect()


def _zip(folder: Path, target: Path, prefix: str = "") -> Path:
    with zipfile.ZipFile(target, "w") as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                archive.write(path, prefix + path.relative_to(folder).as_posix())
    return target


@pytest.mark.parametrize(
    ("source", "reader"),
    [
        (IOH, "ioh"),
        (COCO, "coco"),
        (NEVERGRAD, "nevergrad-csv"),
    ],
)
def test_detects_and_writes_a_valid_dataset(tmp_path, source, reader):
    manifest = convert(source, tmp_path / "ds")
    assert manifest["reader"]["name"] == reader
    assert manifest["extra"]["convert"]["format"] == reader
    assert validate_dataset(tmp_path / "ds").ok
    assert "updated" not in manifest


def test_legacy_ioh_profiler_is_refused(tmp_path):
    (tmp_path / "data_f1").mkdir()
    (tmp_path / "IOHprofiler_f1.info").write_text(
        "suite = 'PBO', funcId = 1, DIM = 16, algId = 'ea'\n%\n"
        "data_f1/IOHprofiler_f1_DIM16.dat, 1:2|16\n"
    )
    (tmp_path / "data_f1" / "IOHprofiler_f1_DIM16.dat").write_text(
        '"function evaluation" "current f(x)"\n1 16\n2 16\n'
    )
    with pytest.raises(ValueError, match="legacy IOHprofiler format is not supported"):
        convert(tmp_path, tmp_path / "ds")


def test_same_runs_as_the_reader(tmp_path):
    convert(IOH, tmp_path / "ds", dataset_id="x")
    direct = read_ioh(IOH, "x")
    runs = pl.read_parquet(tmp_path / "ds" / "runs.parquet")
    assert sorted(runs["run_id"]) == sorted(direct.runs["run_id"])
    assert (
        pl.read_parquet(tmp_path / "ds" / "trace.parquet").height == direct.trace.height
    )


def test_default_dataset_id_comes_from_the_name(tmp_path):
    assert convert(NEVERGRAD, tmp_path / "ds")["dataset_id"] == "nevergrad_sample"


def test_zip_paths_archives_and_original_untouched(tmp_path):
    archive = _zip(IOH, tmp_path / "logs.zip", "run1/")
    before = archive.read_bytes()
    manifest = convert(archive, tmp_path / "ds")
    assert archive.read_bytes() == before
    assert [a["path"] for a in manifest["archives"]] == ["logs.zip"]
    paths = [s["path"] for s in manifest["sources"]]
    assert all(p.startswith("logs.zip/run1/") for p in paths)
    assert {s.get("archive") for s in manifest["sources"]} == {"logs.zip"}
    assert validate_dataset(tmp_path / "ds").ok


def test_nested_archives_in_a_folder(tmp_path):
    inner = _zip(IOH / "f_zero", tmp_path / "inner.zip")
    folder = tmp_path / "input"
    shutil.copytree(IOH / "attributes", folder / "plain")
    with tarfile.open(folder / "outer.tar.gz", "w:gz") as tar:
        tar.add(inner, arcname="sub/inner.zip")
    manifest = convert(folder, tmp_path / "ds")
    archives = {a["path"]: a.get("archive") for a in manifest["archives"]}
    assert archives == {
        "outer.tar.gz": None,
        "outer.tar.gz/sub/inner.zip": "outer.tar.gz",
    }
    sources = {s["path"]: s.get("archive") for s in manifest["sources"]}
    assert any(p.startswith("plain/") and a is None for p, a in sources.items())
    nested = [p for p, a in sources.items() if a == "outer.tar.gz/sub/inner.zip"]
    assert nested and all(p.startswith("outer.tar.gz/sub/inner.zip/") for p in nested)


def test_tags_and_path_tags(tmp_path):
    manifest = convert(
        IOH,
        tmp_path / "ds",
        tags=["campaign=pilot", "seed:int=7", "ok:bool=true", ("scale", 0.5)],
        tags_from_path="{case}",
    )
    seed = _descriptors(tmp_path / "ds", "seed")
    assert seed["value_int"].unique().to_list() == [7]
    assert seed["origin"].unique().to_list() == ["user"]
    assert _descriptors(tmp_path / "ds", "ok")["value_bool"].unique().to_list() == [
        True
    ]
    assert _descriptors(tmp_path / "ds", "scale")["value_float"].unique().to_list() == [
        0.5
    ]
    case = _descriptors(tmp_path / "ds", "case")
    assert case["origin"].unique().to_list() == ["path"]
    assert set(case["value"]) == {"attributes", "cross_run", "every_third", "f_zero"}
    assert manifest["descriptors"]["case"]["origins"] == ["path"]
    assert manifest["extra"]["convert"]["tags_from_path"] == "{case}"
    assert "updated" not in manifest


def test_typed_path_tags_and_algorithm_name(tmp_path):
    folder = tmp_path / "in" / "myalg_100"
    shutil.copytree(IOH / "f_zero", folder)
    convert(
        tmp_path / "in",
        tmp_path / "ds",
        tags_from_path="{algorithm_name}_{maxevals:int}",
    )
    name = _descriptors(tmp_path / "ds", "algorithm_name")
    assert name["value"].unique().to_list() == ["myalg"]
    assert name["origin"].unique().to_list() == ["user"]
    assert _descriptors(tmp_path / "ds", "maxevals")[
        "value_int"
    ].unique().to_list() == [100]


def test_path_pattern_matching():
    pattern = PathPattern("*/{alg}/{dim:int}")
    assert pattern.values("x.zip/cma/5/data.dat") == {"alg": "cma", "dim": 5}
    assert pattern.values("x.zip/cma") is None
    assert PathPattern("{a}").values("only") == {"a": "only"}


@pytest.mark.parametrize("bad", ["*", "{a}/{a}", "{a", "{a:complex}", "{}"])
def test_bad_path_patterns(bad):
    with pytest.raises(ConvertError):
        PathPattern(bad)


def test_unmatched_path_pattern_writes_nothing(tmp_path):
    with pytest.raises(ConvertError, match="does not match"):
        convert(IOH, tmp_path / "ds", tags_from_path="nothere/{x}")
    assert not (tmp_path / "ds").exists()


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("a=b", ("a", "b")),
        ("a=", ("a", "")),
        ("n:int=-3", ("n", -3)),
        ("x:float=inf", ("x", float("inf"))),
        ("k=v=w", ("k", "v=w")),
    ],
)
def test_parse_tag(tag, expected):
    assert parse_tag(tag) == expected


@pytest.mark.parametrize(
    "tag", ["novalue", "=x", "n:int=1.5", "b:bool=maybe", "t:date=1", "budget=3"]
)
def test_bad_tags(tag):
    with pytest.raises(ConvertError):
        parse_tag(tag)


def test_custom_csv_needs_its_mapping(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("alg,seed,evals,f\nA,1,1,3.0\nA,1,2,2.0\nB,1,1,5\n")
    mapping = {
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
    with pytest.raises(ConvertError):
        convert(csv, tmp_path / "ds", format="custom-csv")
    with pytest.raises(ConvertError):
        convert(NEVERGRAD, tmp_path / "ds", format="nevergrad-csv", mapping=mapping)
    manifest = convert(csv, tmp_path / "ds", mapping=mapping)
    assert manifest["reader"]["name"] == "custom-csv"
    assert manifest["counts"]["runs"] == 2


def test_unknown_input(tmp_path):
    (tmp_path / "notes.txt").write_text("hello\n")
    with pytest.raises(ReaderError, match="no reader recognises"):
        convert(tmp_path / "notes.txt", tmp_path / "ds")
    with pytest.raises(ConvertError, match="unknown format"):
        convert(IOH, tmp_path / "ds", format="sos")


def test_cli_convert(tmp_path, capsys):
    code = main(["convert", str(IOH), str(tmp_path / "ds"), "--tag", "seed:int=1"])
    assert code == 0
    out = capsys.readouterr().out
    assert "ioh, 11 run(s)" in out
    assert main(["convert", str(IOH), str(tmp_path / "ds")]) == 2  # exists
    assert main(["convert", str(IOH), str(tmp_path / "ds"), "--overwrite"]) == 0
    assert main(["convert", str(tmp_path / "missing"), str(tmp_path / "x")]) == 2


def test_cli_convert_with_mapping_file(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("alg,evals,f\nA,1,3.0\n")
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps(
            {
                "evaluations": "evals",
                "value": "f",
                "fields": {
                    "algorithm_name": "alg",
                    "dimension": {"constant": 1},
                    "maximization": {"constant": True},
                },
            }
        )
    )
    assert (
        main(["convert", str(csv), str(tmp_path / "ds"), "--mapping", str(mapping)])
        == 0
    )
