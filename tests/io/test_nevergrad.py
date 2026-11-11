import csv
import gzip
import hashlib
from pathlib import Path

import polars as pl
import pytest

from iohinspector.convert import convert
from iohinspector.format import scan_dataset, validate_dataset
from iohinspector.io import READERS, ReaderError, read_nevergrad
from iohinspector.io._csv import MappingError
from iohinspector.io.nevergrad import is_nevergrad_csv

DATA = Path(__file__).parent / "data"
SAMPLE = DATA / "nevergrad_sample.csv"
DOWNLOADED = sorted((DATA / "downloaded").glob("*.csv.gz"))

HEADER = "loss,seed,name,rescale,dimension,budget,optimizer_name,flag,note\n"


def convert_nevergrad(source, output, **kwargs):
    return convert(source, output, format="nevergrad-csv", **kwargs)


def read_source(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def descriptors(ds, key):
    d = ds.descriptors.filter(pl.col("key") == key).collect()
    return dict(zip(d["run_id"], d["value"], strict=True))


def check_against_source(path, out):
    """Row counts and every value of the dataset against the source file."""
    rows = read_source(path)
    report = validate_dataset(out)
    report.raise_for_errors()
    ds = scan_dataset(out, warn=False)
    runs = ds.runs.collect().sort("run_index")
    trace = ds.trace.collect()
    assert runs.height == len(rows) == trace.height
    assert ds.manifest["sources"][0]["data_rows"] == len(rows)
    by_run = trace.join(runs.select("run_id", "run_index"), on="run_id")
    by_run = by_run.sort("run_index")
    assert by_run["raw_y"].to_list() == [float(r["loss"]) for r in rows]
    assert by_run["evaluations"].to_list() == [int(r["budget"]) for r in rows]
    assert runs["dimension"].to_list() == [int(r["dimension"]) for r in rows]
    assert runs["budget"].to_list() == [int(r["budget"]) for r in rows]
    algs = descriptors(ds, "algorithm_name")
    assert [algs[i] for i in runs["run_id"]] == [r["optimizer_name"] for r in rows]
    # the hash of each run's data line, computed from the source text
    for rec, row in zip(runs.iter_rows(named=True), rows, strict=True):
        line = " ".join(row.values()) + "\n"
        assert rec["source_rows_sha256"] == hashlib.sha256(line.encode()).hexdigest()
    return ds, runs, rows


def test_registered():
    assert READERS["nevergrad-csv"].read is read_nevergrad
    assert is_nevergrad_csv(SAMPLE)


def test_sample_values_and_counts(tmp_path):
    out = tmp_path / "ds"
    manifest = convert_nevergrad(SAMPLE, out)
    ds, runs, rows = check_against_source(SAMPLE, out)
    assert manifest["reader"]["name"] == "nevergrad-csv"
    assert manifest["sources"][0]["kind"] == "nevergrad-csv"
    assert (runs["suite"] == "NEVERGRAD").all()
    assert not runs["maximization"].any()
    # every column except those stored as run, trace or algorithm is a descriptor
    keys = set(ds.descriptors.select("key").unique().collect()["key"])
    stored = {"loss", "budget", "dimension", "optimizer_name"}
    empty = {c for c in rows[0] if all(r[c] == "" for r in rows)}
    assert keys == (set(rows[0]) - stored - empty) | {
        "algorithm_name",
        "algorithm_info",
    }
    seeds = ds.descriptors.filter(pl.col("key") == "seed").collect()
    assert set(seeds["type"]) == {"int"}
    assert sorted(seeds["value_int"].to_list()) == sorted(int(r["seed"]) for r in rows)
    split = ds.descriptors.filter(pl.col("key") == "split").collect()
    assert set(split["type"]) == {"bool"}
    names = ds.runs.select("function_id", "function_name").unique().collect()
    ids = manifest["extra"]["function_ids"]
    assert dict(zip(names["function_name"], names["function_id"], strict=True)) == ids
    assert list(ids) == sorted(ids)


def test_gzip(tmp_path):
    gz = tmp_path / "x.csv.gz"
    gz.write_bytes(gzip.compress(SAMPLE.read_bytes()))
    m = convert_nevergrad(gz, tmp_path / "ds", dataset_id="ng-1")
    assert m["dataset_id"] == "ng-1"
    assert m["sources"][0]["path"] == "x.csv.gz"
    assert scan_dataset(tmp_path / "ds", warn=False).runs.collect().height == 120


def test_same_input_same_run_ids(tmp_path):
    convert_nevergrad(SAMPLE, tmp_path / "a")
    convert_nevergrad(SAMPLE, tmp_path / "b")
    a = scan_dataset(tmp_path / "a", warn=False).runs.collect()
    b = scan_dataset(tmp_path / "b", warn=False).runs.collect()
    assert a["run_id"].to_list() == b["run_id"].to_list()


@pytest.mark.skipif(not DOWNLOADED, reason="run scripts/download_nevergrad.py")
@pytest.mark.parametrize("path", DOWNLOADED, ids=lambda p: p.name)
def test_downloaded_corpus(path, tmp_path):
    convert_nevergrad(path, tmp_path / "ds")
    check_against_source(path, tmp_path / "ds")


def write(tmp_path, rows, header=HEADER, name="ng.csv"):
    p = tmp_path / name
    p.write_text(header + "".join(r + "\n" for r in rows))
    return p


def test_rescale_and_function_ids(tmp_path):
    src = write(
        tmp_path,
        [
            "1.5,1,sphere,False,10,100,CMA,True,",
            "2.5,2,sphere,True,10,100,CMA,False,x",
            "inf,3,ackley,,10,200,DE,True,y",
        ],
    )
    m = convert_nevergrad(src, tmp_path / "ds")
    ds = scan_dataset(tmp_path / "ds", warn=False)
    runs = ds.runs.collect().sort("run_index")
    assert runs["function_name"].to_list() == ["sphere", "sphere_rescaled", "ackley"]
    assert runs["function_id"].to_list() == [2, 3, 1]
    assert m["extra"]["function_ids"] == {
        "ackley": 1,
        "sphere": 2,
        "sphere_rescaled": 3,
    }
    trace = ds.trace.collect()
    assert sorted(trace["raw_y"].to_list()) == [1.5, 2.5, float("inf")]
    rescale = ds.descriptors.filter(pl.col("key") == "rescale").collect()
    assert rescale["type"].to_list() == ["bool", "bool"]
    note = ds.descriptors.filter(pl.col("key") == "note").collect()
    assert sorted(note["value"]) == ["x", "y"]


def test_older_columns(tmp_path):
    header = "loss,function_class,transform,dimension,budget,optimizer_name\n"
    src = write(tmp_path, ["3.0,Sphere,True,5,50,A"], header)
    convert_nevergrad(src, tmp_path / "ds")
    runs = scan_dataset(tmp_path / "ds", warn=False).runs.collect()
    assert runs["function_name"].to_list() == ["Sphere_rescaled"]


def test_reserved_names_are_renamed(tmp_path):
    header = "loss,name,dimension,budget,optimizer_name,instance,row\n"
    src = write(tmp_path, ["3.0,f,5,50,A,7,r"], header)
    m = convert_nevergrad(src, tmp_path / "ds")
    assert {
        "from": "instance",
        "to": "instance_src",
        "source": "ng.csv",
        "table": "descriptors",
    } in m["renamed_columns"]
    ds = scan_dataset(tmp_path / "ds", warn=False)
    assert descriptors(ds, "instance_src")


@pytest.mark.parametrize(
    "row, message",
    [
        (
            "abc,1,f,False,5,10,A,True,",
            "ng.csv:2: column 'loss': 'abc' is not a number",
        ),
        (
            "1.0,1,f,False,5,10.5,A,True,",
            "ng.csv:2: budget: '10.5' is not an integer",
        ),
        ("1.0,1,f,False,5,10,,True,", "ng.csv:2: algorithm_name is empty"),
        ("1.0,1,f,False,5,10,A", "ng.csv:2: expected 9 fields"),
        ("1_0,1,f,False,5,10,A,True,", "'1_0' is not a number"),
    ],
)
def test_errors_name_file_and_line(tmp_path, row, message):
    src = write(tmp_path, [row])
    with pytest.raises(ReaderError, match=message):
        convert_nevergrad(src, tmp_path / "ds")
    assert not (tmp_path / "ds").exists()


def test_not_nevergrad(tmp_path):
    src = write(tmp_path, ["1,2"], "a,b\n")
    assert not is_nevergrad_csv(src)
    with pytest.raises(MappingError, match="not a Nevergrad result file"):
        convert_nevergrad(src, tmp_path / "ds")
    empty = write(tmp_path, [], HEADER, "empty.csv")
    with pytest.raises(ReaderError, match="no data rows"):
        convert_nevergrad(empty, tmp_path / "ds")
