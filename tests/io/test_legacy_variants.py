"""Old variants that IOHanalyzer's ``readFiles.R`` accepts, one test each (WP I06).

The list is in ``docs/readers-legacy.md``, section "Variants".
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from iohinspector.io import read_legacy
from iohinspector.io._legacy_info import parse_header
from iohinspector.io._legacy_meta import header_value

HEAD = '"function evaluation" "current f(x)"\n'


def make(tmp_path: Path, info: str, files: dict[str, str]) -> Path:
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(text.encode())
    (tmp_path / "a.info").write_bytes(info.encode())
    return tmp_path


def codes(result) -> set[str]:
    return {w["code"] for w in result.manifest.get("warnings", [])}


def descriptor(result, key: str) -> list[tuple[str, str]]:
    d = result.descriptors.filter(pl.col("key") == key)
    return list(zip(d["type"], d["value"], strict=True))


def test_v1_constrained_is_logical():
    items = parse_header("constrained = TRUE, other = TRUE", "a.info", 1)
    assert [header_value(i) for i in items] == [True, "TRUE"]


def test_v2_record_without_runs_still_reads_the_data(tmp_path):
    # IOHanalyzer skips the block (`next`); its data rows are kept here
    make(
        tmp_path,
        "funcId = 1, DIM = 2, algId = 'A'\n%\nd/f.dat\n",
        {"d/f.dat": HEAD + "1 1\n"},
    )
    result = read_legacy(tmp_path, "x")
    assert result.trace.height == 1 and result.runs["instance"].to_list() == [None]
    assert "run_count_mismatch" in codes(result)


def test_v3_cut_off_record_and_zero_evaluation_entries(tmp_path):
    make(
        tmp_path,
        "funcId = 1, DIM = 2, algId = 'A'\n%\nd/f.dat, 1:0|0, 2:3|1, \n",
        {"d/f.dat": HEAD + "1 2\n3 1\n"},
    )
    result = read_legacy(tmp_path, "x")
    assert result.runs["instance"].to_list() == [2]
    assert {"incomplete_info_record", "info_run_without_evaluations"} <= codes(result)


def test_v4_unquoted_header_names(tmp_path):
    make(
        tmp_path,
        "funcId = 1, DIM = 2, algId = 'A'\n%\nd/f.dat, 1:2|1\n",
        {"d/f.dat": "evaluations raw_y sigma\n1 2 0.5\n2 1 0.25\n"},
    )
    result = read_legacy(tmp_path, "x")
    assert result.trace.columns == ["run_id", "row", "evaluations", "raw_y", "sigma"]


def test_v5_scientific_and_special_values_merge(tmp_path):
    make(
        tmp_path,
        "funcId = 1, DIM = 2, algId = 'A'\n%\nd/f.dat, 1:3|-inf\n",
        {
            "d/f.dat": HEAD + "1 +NaN\n2 -1.5E+02\n3 -inf\n",
            "d/f.cdat": HEAD + "1 nan\n2 -1.50000e+02\n3 -Inf\n",
        },
    )
    result = read_legacy(tmp_path, "x")
    trace = result.trace
    assert trace.height == 3 and not trace["merge_conflict"].any()
    assert trace["raw_y"].is_nan().to_list() == [True, False, False]
    assert trace["raw_y"].to_list()[1:] == [-150.0, float("-inf")]
    assert result.runs["best_y"].to_list() == [float("-inf")]


def test_v6_fixed_precision_is_recorded(tmp_path):
    make(
        tmp_path,
        "funcId = 1, DIM = 2, algId = 'A'\n%\nd/f.dat, 1:2|0\n",
        {"d/f.dat": HEAD + "1 0.000100\n2 0.000000\n"},
    )
    result = read_legacy(tmp_path, "x")
    dat = result.manifest["sources"][1]
    assert dat["precision"] == {"notation": "fixed", "digits": 6}
    assert "limited_precision" in codes(result)


def test_v7_empty_info_file_is_skipped(tmp_path):
    make(
        tmp_path,
        "funcId = 1, DIM = 2\n%\nd/f.dat, 1:1|1\n",
        {"d/f.dat": HEAD + "1 1\n"},
    )
    (tmp_path / "empty.info").write_text("")
    result = read_legacy(tmp_path, "x")
    assert "skipped_file" in codes(result) and result.trace.height == 1


def test_v8_reserved_descriptor_keys_are_renamed(tmp_path):
    make(
        tmp_path,
        "funcId = 1, DIM = 2, algId = 'A', instance = 3, budget = 'x'\n%\n"
        "d/f.dat, 1:1|1\n",
        {"d/f.dat": HEAD + "1 1\n"},
    )
    result = read_legacy(tmp_path, "x")
    assert descriptor(result, "instance_experiment") == [("int", "3")]
    assert descriptor(result, "budget_experiment") == [("string", "x")]
    assert {r["table"] for r in result.manifest["renamed_columns"]} == {"descriptors"}


def test_v9_utf8_names_and_non_utf8_failure(tmp_path):
    make(
        tmp_path,
        "funcId = 1, DIM = 2, algId = 'Ä-algorithm'\n%\nd/f.dat, 1:1|1\n",
        {"d/f.dat": HEAD + "1 1\n"},
    )
    assert descriptor(read_legacy(tmp_path, "x"), "algorithm_name") == [
        ("string", "Ä-algorithm")
    ]
    (tmp_path / "d" / "f.dat").write_bytes(b'"e" "f\xff"\n1 1\n')
    with pytest.raises(ValueError, match="not UTF-8"):
        read_legacy(tmp_path, "x")
