import json
import random

import polars as pl
import pytest

from iohinspector.convert import convert
from iohinspector.format import scan_dataset, validate_dataset
from iohinspector.io import READERS, ReaderError, read_custom_csv
from iohinspector.io._csv import MappingError
from iohinspector.io.custom_csv import load_mapping, validate_mapping

MAPPING = {
    "version": 1,
    "evaluations": "evals",
    "value": "f",
    "run_by": ["alg", "fid", "dim", "seed"],
    "fields": {
        "algorithm_name": "alg",
        "function_id": "fid",
        "dimension": "dim",
        "maximization": {"constant": False},
        "suite": {"constant": "mine"},
    },
    "descriptors": ["seed", "popsize"],
    "trace_columns": ["sigma", "label"],
}
HEADER = "alg,fid,dim,seed,popsize,evals,f,sigma,label\n"


def convert_custom_csv(source, output, mapping):
    return convert(source, output, mapping=mapping)


def make_rows(seed=0, n_runs=6):
    """Runs of two algorithms, written interleaved, as ``(key, rows)``."""
    rng = random.Random(seed)  # nosec B311 - test data, not security
    runs = []
    for k in range(n_runs):
        alg, fid, dim = ("A", "B")[k % 2], 1 + k % 3, 5
        n = rng.randint(1, 8)
        rows = [
            (
                alg,
                fid,
                dim,
                k,
                20 + k,
                e * 10,
                round(rng.random(), 6),
                rng.random(),
                f"l{e}",
            )
            for e in range(1, n + 1)
        ]
        runs.append(rows)
    lines = []
    for i in range(max(len(r) for r in runs)):  # interleave the runs
        for rows in runs:
            if i < len(rows):
                lines.append(rows[i])
    return runs, lines


def write_csv(path, lines, header=HEADER):
    path.write_text(header + "".join(",".join(map(str, r)) + "\n" for r in lines))
    return path


@pytest.fixture
def source(tmp_path):
    runs, lines = make_rows()
    return write_csv(tmp_path / "data.csv", lines), runs, lines


def test_registered():
    assert READERS["custom-csv"].read is read_custom_csv


def test_long_format_values_and_counts(tmp_path, source):
    path, runs, lines = source
    out = tmp_path / "ds"
    manifest = convert_custom_csv(path, out, MAPPING)
    validate_dataset(out).raise_for_errors()
    ds = scan_dataset(out, warn=False)
    assert manifest["counts"]["runs"] == len(runs)
    assert manifest["counts"]["trace_rows"] == len(lines)
    assert manifest["sources"][0]["data_rows"] == len(lines)
    assert manifest["reader"]["options"]["mapping"] == MAPPING
    r = ds.runs.collect().sort("source_run_index")
    assert (r["suite"] == "mine").all()
    # runs are numbered in order of first appearance, which is the order of seeds here
    assert r["function_id"].to_list() == [rows[0][1] for rows in runs]
    trace = ds.trace.collect().join(r.select("run_id", "source_run_index"), on="run_id")
    trace = trace.sort("source_run_index", "row")
    expected = [row for rows in runs for row in rows]
    assert trace["evaluations"].to_list() == [row[5] for row in expected]
    assert trace["raw_y"].to_list() == [row[6] for row in expected]
    assert trace["sigma"].to_list() == [row[7] for row in expected]
    assert trace["label"].to_list() == [row[8] for row in expected]
    assert trace["row"].to_list() == [i for rows in runs for i in range(len(rows))]
    d = ds.descriptors.collect()
    algs = d.filter(pl.col("key") == "algorithm_name")
    assert sorted(algs["value"]) == sorted(rows[0][0] for rows in runs)
    seed = d.filter(pl.col("key") == "seed")
    assert set(seed["type"]) == {"int"} and set(seed["origin"]) == {"run"}
    assert sorted(seed["value_int"]) == list(range(len(runs)))


def test_run_hash_follows_source_order(tmp_path, source):
    import hashlib

    path, runs, _lines = source
    out = tmp_path / "ds"
    convert_custom_csv(path, out, MAPPING)
    r = scan_dataset(out, warn=False).runs.collect().sort("source_run_index")
    for rec, rows in zip(r.iter_rows(named=True), runs, strict=True):
        text = "".join(" ".join(map(str, row)) + "\n" for row in rows)
        assert rec["source_rows_sha256"] == hashlib.sha256(text.encode()).hexdigest()


def test_one_row_per_run_with_constants_and_function_names(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text(
        "name,dim,evals,f,opt\nsphere,3,100,0.5,0\n"
        "ackley,3,100,inf,0\nsphere,3,100,nan,0\n"
    )
    mapping = {
        "evaluations": "evals",
        "value": "f",
        "fields": {
            "algorithm_name": {"constant": "RS"},
            "algorithm_info": {"constant": "v1"},
            "function_name": "name",
            "dimension": "dim",
            "maximization": {"constant": True},
            "optimum": "opt",
            "budget": "evals",
        },
        "descriptors": "rest",
    }
    out = tmp_path / "ds"
    manifest = convert_custom_csv(path, out, mapping)
    ds = scan_dataset(out, warn=False)
    runs = ds.runs.collect().sort("source_run_index")
    assert runs["function_name"].to_list() == ["sphere", "ackley", "sphere"]
    assert runs["function_id"].to_list() == [2, 1, 2]
    assert manifest["extra"]["function_ids"] == {"ackley": 1, "sphere": 2}
    assert runs["maximization"].all() and runs["optimum"].to_list() == [0.0] * 3
    y = ds.trace.collect().join(runs.select("run_id", "source_run_index"), on="run_id")
    y = y.sort("source_run_index")["raw_y"].to_list()
    assert y[0] == 0.5 and y[1] == float("inf") and y[2] != y[2]
    d = ds.descriptors.collect()
    assert set(d.filter(pl.col("key") == "algorithm_info")["value"]) == {"v1"}
    # "rest" keeps the columns not used elsewhere: none here
    assert set(d["key"]) == {"algorithm_name", "algorithm_info"}


def test_positions_and_rest(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text(
        "a,d,e,y,x1,x2,note,dimension\nA,2,1,3.5,0.1,0.2,hi,7\nA,2,2,2.5,0.3,0.4,hi,7\n"
    )
    mapping = {
        "evaluations": "e",
        "value": "y",
        "run_by": ["a"],
        "fields": {
            "algorithm_name": "a",
            "dimension": "d",
            "maximization": {"constant": False},
        },
        "positions": ["x1", "x2"],
        "descriptors": "rest",
    }
    out = tmp_path / "ds"
    manifest = convert_custom_csv(path, out, mapping)
    validate_dataset(out).raise_for_errors()
    ds = scan_dataset(out, warn=False)
    pos = ds.positions.collect().sort("row")
    assert pos["x0"].to_list() == [0.1, 0.3] and pos["x1"].to_list() == [0.2, 0.4]
    d = ds.descriptors.collect()
    # the column "dimension" has a reserved name and is renamed
    assert {"dimension_src", "note"} <= set(d["key"])
    assert manifest["renamed_columns"] == [
        {
            "from": "dimension",
            "to": "dimension_src",
            "source": "p.csv",
            "table": "descriptors",
        }
    ]


def test_tab_delimiter(tmp_path):
    path = tmp_path / "t.tsv"
    path.write_text("e\ty\td\n1\t2.5\t3\n")
    mapping = {
        "delimiter": "\t",
        "evaluations": "e",
        "value": "y",
        "fields": {
            "algorithm_name": {"constant": "X"},
            "dimension": "d",
            "maximization": {"constant": False},
        },
    }
    convert_custom_csv(path, tmp_path / "ds", mapping)
    assert scan_dataset(tmp_path / "ds", warn=False).trace.collect()[
        "raw_y"
    ].to_list() == [2.5]


def test_load_mapping_from_text_and_file(tmp_path):
    text = json.dumps(MAPPING)
    assert load_mapping(text) == load_mapping(MAPPING)
    f = tmp_path / "m.json"
    f.write_text(text)
    assert load_mapping(f) == load_mapping(str(f))
    assert validate_mapping(MAPPING, HEADER.strip().split(",")).run_by == [
        "alg",
        "fid",
        "dim",
        "seed",
    ]


def mapping_with(**change):
    m = json.loads(json.dumps(MAPPING))
    for key, value in change.items():
        if value is None:
            m.pop(key, None)
        else:
            m[key] = value
    return m


@pytest.mark.parametrize(
    "mapping, message",
    [
        ([], "must be a JSON object"),
        (mapping_with(evalutions="x"), "unknown key 'evalutions'"),
        (mapping_with(evaluations=None), "'evaluations' is required"),
        (mapping_with(evaluations=3), "evaluations: expected the name of a column"),
        (mapping_with(version=2), "version must be 1"),
        (mapping_with(delimiter=";;"), "single character"),
        (
            mapping_with(fields={"dimension": "dim"}),
            "fields.algorithm_name is required",
        ),
        (
            mapping_with(fields={**MAPPING["fields"], "dimention": "dim"}),
            "unknown field 'dimention'. Did you mean 'dimension'",
        ),
        (
            mapping_with(fields={**MAPPING["fields"], "dimension": {"constant": "5"}}),
            "fields.dimension: the constant must be of type int",
        ),
        (
            mapping_with(fields={**MAPPING["fields"], "maximization": {"const": True}}),
            "expected a column name or",
        ),
        (mapping_with(run_by="alg"), "run_by: expected a list"),
        (mapping_with(descriptors=["seed", "seed"]), "listed twice"),
    ],
)
def test_invalid_mappings(tmp_path, mapping, message):
    with pytest.raises(MappingError, match=message):
        validate_mapping(mapping)


def test_invalid_json_text():
    with pytest.raises(MappingError, match="not valid JSON"):
        load_mapping('{"evaluations": ')


def test_mapping_against_columns(tmp_path, source):
    path, _runs, _lines = source
    with pytest.raises(
        MappingError, match="evaluations: column 'evls' is not in the file"
    ):
        convert_custom_csv(path, tmp_path / "ds", mapping_with(evaluations="evls"))
    with pytest.raises(MappingError, match="Did you mean 'popsize'"):
        convert_custom_csv(path, tmp_path / "ds", mapping_with(descriptors=["popsiz"]))
    with pytest.raises(MappingError, match="same column"):
        convert_custom_csv(path, tmp_path / "ds", mapping_with(value="evals"))
    with pytest.raises(MappingError, match="is in trace_columns and positions"):
        convert_custom_csv(
            path,
            tmp_path / "ds",
            mapping_with(positions=["sigma"], fields={**MAPPING["fields"]}),
        )
    assert not (tmp_path / "ds").exists()


def test_duplicate_header(tmp_path):
    path = write_csv(tmp_path / "d.csv", [], "a,a,b\n")
    with pytest.raises(MappingError, match="duplicate column names"):
        convert_custom_csv(
            path,
            tmp_path / "ds",
            {
                "evaluations": "a",
                "value": "b",
                "fields": {
                    "algorithm_name": {"constant": "x"},
                    "dimension": {"constant": 1},
                    "maximization": {"constant": False},
                },
            },
        )


def bad(tmp_path, lines, mapping=None, header=HEADER):
    mapping = MAPPING if mapping is None else mapping
    path = write_csv(tmp_path / "bad.csv", lines, header)
    return path, mapping


@pytest.mark.parametrize(
    "line, message",
    [
        (
            ("A", 1, 5, 0, 20, 10, "x", 0.1, "l"),
            r"bad.csv:2: column 'f': 'x' is not a number",
        ),
        (
            ("A", 1, 5, 0, 20, 1.5, 1.0, 0.1, "l"),
            r"bad.csv:2: column 'evals': '1.5' is not an integer",
        ),
        (
            ("A", 1, "five", 0, 20, 10, 1.0, 0.1, "l"),
            r"bad.csv:2: dimension: 'five' is not an integer",
        ),
        (("", 1, 5, 0, 20, 10, 1.0, 0.1, "l"), r"bad.csv:2: algorithm_name is empty"),
        (("A", 1, 5, 0, 20, 10, 1.0, "s", "l"), None),  # a string column is fine
    ],
)
def test_data_errors_name_file_and_line(tmp_path, line, message):
    path, mapping = bad(tmp_path, [line])
    if message is None:
        convert_custom_csv(path, tmp_path / "ds", mapping)
        return
    with pytest.raises(ReaderError, match=message):
        convert_custom_csv(path, tmp_path / "ds", mapping)


def test_run_level_column_must_be_constant(tmp_path):
    lines = [
        ("A", 1, 5, 0, 20, 10, 1.0, 0.1, "l"),
        ("A", 1, 5, 0, 21, 20, 0.5, 0.1, "l"),
    ]
    path, mapping = bad(tmp_path, lines)
    with pytest.raises(
        ReaderError, match=r"bad.csv:3: column 'popsize' is '21', but '20' in line 2"
    ):
        convert_custom_csv(path, tmp_path / "ds", mapping)


def test_ragged_row_and_empty(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text(HEADER + "A,1,5\n")
    with pytest.raises(ReaderError, match="r.csv:2: expected 9 fields"):
        convert_custom_csv(path, tmp_path / "ds", MAPPING)
    path.write_text(HEADER)
    with pytest.raises(ReaderError, match="no data rows"):
        convert_custom_csv(path, tmp_path / "ds", MAPPING)
    path.write_text("")
    with pytest.raises(ReaderError, match="file is empty"):
        convert_custom_csv(path, tmp_path / "ds", MAPPING)


def test_positions_must_match_dimension(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text("a,d,e,y,x1\nA,2,1,3.5,0.1\n")
    mapping = {
        "evaluations": "e",
        "value": "y",
        "positions": ["x1"],
        "fields": {
            "algorithm_name": "a",
            "dimension": "d",
            "maximization": {"constant": False},
        },
    }
    with pytest.raises(ReaderError, match="dimension 2, but positions lists 1"):
        convert_custom_csv(path, tmp_path / "ds", mapping)


def test_mixed_types_keep_exact_values(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text(
        "e,y,d,big,flt,mix\n1,1.0,2,12345678901234567890,1,1\n2,1.0,2,1,2.5,a\n"
    )
    mapping = {
        "evaluations": "e",
        "value": "y",
        "descriptors": ["big", "flt", "mix"],
        "fields": {
            "algorithm_name": {"constant": "x"},
            "dimension": "d",
            "maximization": {"constant": False},
        },
    }
    convert_custom_csv(path, tmp_path / "ds", mapping)
    d = scan_dataset(tmp_path / "ds", warn=False).descriptors.collect()
    got = {(r["key"], r["value"]): r["type"] for r in d.iter_rows(named=True)}
    assert got[("big", "12345678901234567890")] == "int"
    assert got[("flt", "1.0")] == "float" and got[("flt", "2.5")] == "float"
    assert got[("mix", "a")] == "string"
    big = d.filter(pl.col("key") == "big").sort("value")
    assert big["value_int"].to_list() == [1, None]


def test_extra_trace_column_types(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text(
        "e,y,d,i,f,b,s,huge\n1,1.0,2,1,1,true,a,12345678901234567890\n"
        "2,1.0,2,,2.5,false,,2\n"
    )
    mapping = {
        "evaluations": "e",
        "value": "y",
        "trace_columns": ["i", "f", "b", "s", "huge"],
        "run_by": ["d"],
        "fields": {
            "algorithm_name": {"constant": "x"},
            "dimension": "d",
            "maximization": {"constant": False},
        },
    }
    convert_custom_csv(path, tmp_path / "ds", mapping)
    t = scan_dataset(tmp_path / "ds", warn=False).trace.collect().sort("row")
    assert t.schema["i"] == pl.Int64 and t["i"].to_list() == [1, None]
    assert t.schema["f"] == pl.Float64 and t["f"].to_list() == [1.0, 2.5]
    assert t.schema["b"] == pl.Boolean and t["b"].to_list() == [True, False]
    assert t.schema["s"] == pl.String and t["s"].to_list() == ["a", None]
    assert t["huge"].to_list() == ["12345678901234567890", "2"]
