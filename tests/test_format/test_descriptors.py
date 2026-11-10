"""Descriptors (decision D17): storage, exactness, grouping and editing."""

import hashlib
import json
import math
import shutil

import numpy as np
import polars as pl
import pytest

from iohinspector.format import (
    DESCRIPTORS_SCHEMA,
    FormatError,
    decode_descriptor,
    descriptor_columns,
    descriptor_frame,
    encode_descriptor,
    new_manifest,
    run_ids,
    scan_dataset,
    update_descriptors,
    validate_dataset,
    validate_manifest,
    write_dataset,
)

from .synthetic import make_dataset

BIG_INTS = [2**63 - 1, 2**63, 2**64 - 1, -(2**63), -(2**63) - 1, 10**30]


# -- encoding -------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, kind, text",
    [
        (True, "bool", "true"),
        (np.bool_(False), "bool", "false"),
        (0, "int", "0"),
        (-17, "int", "-17"),
        (np.int64(5), "int", "5"),
        (2**64 - 1, "int", "18446744073709551615"),
        (0.1, "float", "0.1"),
        (-0.0, "float", "-0.0"),
        (float("inf"), "float", "inf"),
        (float("nan"), "float", "nan"),
        (5e-324, "float", "5e-324"),
        (np.float32(0.5), "float", "0.5"),
        ("CMA-ES", "string", "CMA-ES"),
        ("", "string", ""),
    ],
)
def test_encode_and_decode(value, kind, text):
    encoded = encode_descriptor(value)
    assert encoded[:2] == (kind, text)
    decoded = decode_descriptor(kind, text)
    if kind == "float" and math.isnan(float(value)):
        assert math.isnan(decoded)
    else:
        assert decoded == value and type(decoded) in (bool, int, float, str)


def test_encode_typed_columns():
    assert encode_descriptor(3) == ("int", "3", 3, None, None)
    assert encode_descriptor(2**63) == ("int", str(2**63), None, None, None)
    assert encode_descriptor(1.5) == ("float", "1.5", None, 1.5, None)
    assert encode_descriptor(True) == ("bool", "true", None, None, True)


def test_encode_rejects():
    with pytest.raises(TypeError):
        encode_descriptor(object())


@pytest.mark.parametrize("value", [None, [1, 2], (1, 2), {"a": 1}, []])
def test_encode_rejects_non_scalars(value):
    with pytest.raises(FormatError, match="scalars"):
        encode_descriptor(value)


def test_descriptor_frame_skips_none_and_refuses_containers():
    df = descriptor_frame(
        [(1, "a", 3, "run"), (1, "b", None, "run"), (2, "b", "x", "run")]
    )
    assert df.select("run_id", "key", "type").rows() == [
        (1, "a", "int"),
        (2, "b", "string"),
    ]
    assert descriptor_frame([(1, "b", None, "run")]).height == 0
    with pytest.raises(FormatError):
        descriptor_frame([(1, "a", [1], "run")])


def test_descriptor_frame_accepts_tuples_and_mappings():
    df = descriptor_frame(
        [(1, "a", 1, "run"), {"run_id": 2, "key": "a", "value": "x", "origin": "user"}]
    )
    assert dict(df.schema) == DESCRIPTORS_SCHEMA
    assert df["type"].to_list() == ["int", "string"]


# -- datasets -------------------------------------------------------------------


def parameter_study(tmp_path, *, files=2, runs_per_file=6):
    """One algorithm, one function, one dimension and one instance, many settings.

    Runs differ only in their descriptors: algorithm × function × dimension ×
    instance is far from unique.
    """
    paths = [f"study/part{i}.dat" for i in range(files)]
    manifest = new_manifest(
        "study",
        "test",
        "0",
        sources=[{"path": p, "sha256": None, "bytes": None} for p in paths],
    )
    runs = pl.DataFrame(
        {
            "algorithm_name": ["DE"] * (files * runs_per_file),
            "function_id": [1] * (files * runs_per_file),
            "dimension": [5] * (files * runs_per_file),
            "instance": [1] * (files * runs_per_file),
            "maximization": [False] * (files * runs_per_file),
            "source_file": [p for p in paths for _ in range(runs_per_file)],
            "source_run_index": [i for _ in paths for i in range(runs_per_file)],
        }
    )
    ids = run_ids("study", runs["source_file"], runs["source_run_index"]).to_list()
    records, trace = [], []
    for n, rid in enumerate(ids):
        records += [
            (rid, "popsize", [10, 20, 40][n % 3], "algorithm"),
            (rid, "mutation", [0.5, 0.9][n % 2], "algorithm"),
            (rid, "seed", BIG_INTS[n % len(BIG_INTS)], "run"),
            (rid, "local_seed", n, "run"),
        ]
        rows = n + 1
        trace.append(
            pl.DataFrame(
                {
                    "run_id": [rid] * rows,
                    "row": list(range(rows)),
                    "evaluations": list(range(1, rows + 1)),
                    "raw_y": [float(rows - r) for r in range(rows)],
                }
            )
        )
    path = tmp_path / "study"
    write_dataset(path, manifest, runs, trace, descriptors=descriptor_frame(records))
    return path, ids


def test_one_algorithm_with_several_parameter_settings(tmp_path):
    path, ids = parameter_study(tmp_path)
    assert validate_dataset(path).ok
    ds = scan_dataset(path)
    runs = ds.runs.collect()
    key = ["algorithm_name", "function_id", "dimension", "instance"]
    assert runs.select(key).unique().height == 1 and runs.height == len(ids)
    groups = ds.run_columns(["popsize", "mutation"]).collect()
    assert (
        groups.schema["popsize"] == pl.Int64 and groups.schema["mutation"] == pl.Float64
    )
    assert groups.select("popsize", "mutation").unique().height == 6


def test_grouping_by_a_descriptor(tmp_path):
    path, ids = parameter_study(tmp_path)
    ds = scan_dataset(path)
    by = ["popsize"]
    per_group = (
        ds.trace.join(ds.run_columns(by), on="run_id")
        .group_by(by)
        .agg(pl.len().alias("rows"), pl.col("run_id").n_unique().alias("runs"))
        .sort(by)
        .collect()
    )
    expected = {10: 0, 20: 0, 40: 0}
    for n in range(len(ids)):
        expected[[10, 20, 40][n % 3]] += n + 1
    assert per_group["popsize"].to_list() == [10, 20, 40]
    assert per_group["rows"].to_list() == [expected[10], expected[20], expected[40]]
    assert per_group["runs"].to_list() == [4, 4, 4]

    mixed_by = ["function_id", "dimension", "mutation", "algorithm_name"]
    table = ds.run_columns(mixed_by).collect()
    assert table.columns == ["run_id", *mixed_by] and table.height == len(ids)
    assert ds.run_columns().collect().columns == ["run_id", "algorithm_name"]
    with pytest.raises(ValueError, match="no runs column or descriptor"):
        ds.run_columns(["popsiz"])


def test_exact_large_integers(tmp_path):
    path, ids = parameter_study(tmp_path)
    ds = scan_dataset(path)
    seeds = ds.descriptors.filter(pl.col("key") == "seed").collect()
    by_run = dict(zip(seeds["run_id"], zip(seeds["value"], seeds["value_int"])))
    for n, rid in enumerate(ids):
        exact = BIG_INTS[n % len(BIG_INTS)]
        text, as_int = by_run[rid]
        assert decode_descriptor("int", text) == exact
        assert as_int == (exact if -(2**63) <= exact < 2**63 else None)
    column = ds.run_columns(["seed", "local_seed"]).collect()
    assert column.schema["seed"] == pl.String  # some seeds do not fit in int64
    assert set(column["seed"]) == {str(v) for v in BIG_INTS}
    assert column.schema["local_seed"] == pl.Int64
    assert sorted(ds.manifest["descriptors"]["seed"]["examples"]) == sorted(
        sorted(str(v) for v in BIG_INTS)[:5]
    )


def test_mixed_value_types_under_one_key(tmp_path):
    syn = make_dataset(functions=(1,), dimensions=(2,), runs_per_file=6, seed=40)
    values = [100, "unlimited", 1e3, True, "[1, 2]", "None"]
    ids = syn.runs["run_id"].to_list()
    records = [(rid, "budget_note", v, "experiment") for rid, v in zip(ids, values * 2)]
    path = tmp_path / "mixed"
    write_dataset(
        path, syn.manifest, syn.runs, syn.trace, descriptors=descriptor_frame(records)
    )
    assert validate_dataset(path).ok
    ds = scan_dataset(path)
    stored = ds.descriptors.filter(pl.col("key") == "budget_note").collect()
    for rid, value in zip(ids, values * 2):
        row = stored.filter(pl.col("run_id") == rid).row(0, named=True)
        assert decode_descriptor(row["type"], row["value"]) == value
    summary = ds.manifest["descriptors"]["budget_note"]
    assert summary["types"] == ["bool", "float", "int", "string"]
    assert summary["origins"] == ["experiment"] and summary["distinct"] == 6

    col = ds.run_columns(["budget_note"]).collect()
    assert col.schema["budget_note"] == pl.String
    texts = dict(zip(col["run_id"], col["budget_note"]))
    assert [texts[r] for r in ids[:6]] == [
        "100",
        "unlimited",
        "1000.0",
        "true",
        "[1, 2]",
        "None",
    ]
    with pytest.raises(ValueError, match="several types"):
        ds.run_columns(["budget_note"], mixed="error")
    standalone = descriptor_columns(ds.descriptors, ["budget_note"]).collect()
    assert standalone.height == len(ids)


def _file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_adding_a_user_tag_after_conversion(tmp_path):
    syn = make_dataset(positions=True, seed=41)
    path = tmp_path / "ds"
    write_dataset(
        path,
        syn.manifest,
        syn.runs,
        syn.trace,
        positions=syn.positions,
        descriptors=syn.descriptors,
    )
    untouched = {
        name: _file_digest(path / name)
        for name in ("trace.parquet", "positions.parquet")
    }
    stat = {name: (path / name).stat().st_mtime_ns for name in untouched}
    ids = syn.runs["run_id"].to_list()

    tagged = ids[: len(ids) // 2]
    manifest = update_descriptors(path, [(rid, "batch", "A", "run") for rid in tagged])
    assert manifest["descriptors"]["batch"] == {
        "types": ["string"],
        "origins": ["user"],
        "runs": len(tagged),
        "distinct": 1,
        "examples": ["A"],
    }
    assert "updated" in manifest
    report = validate_dataset(path)
    assert report.ok, report.errors

    update_descriptors(
        path, [(ids[0], "batch", "B", "user"), (ids[-1], "batch", 7, "user")]
    )
    ds = scan_dataset(path)
    col = dict(ds.run_columns(["batch"]).collect().iter_rows())
    assert col[ids[0]] == "B" and col[ids[1]] == "A" and col[ids[-1]] == "7"
    assert ds.manifest["descriptors"]["batch"]["types"] == ["int", "string"]

    update_descriptors(path, [(ids[0], "algorithm_name", "renamed", "user")])
    runs = scan_dataset(path).runs.collect()
    assert runs.filter(pl.col("run_id") == ids[0])["algorithm_name"].item() == "renamed"
    assert validate_dataset(path).ok

    update_descriptors(path, remove_keys=["batch"])
    assert "batch" not in scan_dataset(path).manifest["descriptors"]
    assert validate_dataset(path).ok
    for name, digest in untouched.items():
        assert _file_digest(path / name) == digest
        assert (path / name).stat().st_mtime_ns == stat[name]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"remove_keys": ["algorithm_name"]}, "cannot be removed"),
        ({"remove_keys": ["nope"]}, "do not exist"),
        ({"upsert": [(12345, "batch", "A", "user")]}, "not in runs"),
        ({"upsert": [(None, "dimension", 3, "user")]}, "reserved|null"),
        ({"upsert": [(None, "algorithm_info", 3, "user")]}, "disagree|string"),
        ({"upsert": [(None, "x", 1, "user"), (None, "x", 2, "user")]}, "not unique"),
    ],
)
def test_bad_edits_are_refused_and_change_nothing(tmp_path, kwargs, message):
    syn = make_dataset(seed=42)
    path = tmp_path / "ds"
    write_dataset(path, syn.manifest, syn.runs, syn.trace, descriptors=syn.descriptors)
    rid = syn.runs["run_id"][0]
    if "upsert" in kwargs:
        kwargs = {
            "upsert": [
                (rid if r is None else r, k, v, o) for r, k, v, o in kwargs["upsert"]
            ]
        }
    before = {p.name: _file_digest(p) for p in path.iterdir()}
    with pytest.raises(FormatError, match=message):
        update_descriptors(path, **kwargs)
    assert {p.name: _file_digest(p) for p in path.iterdir()} == before


def test_writer_refuses_bad_descriptors(tmp_path):
    syn = make_dataset(seed=43)
    rid = syn.runs["run_id"][0]
    cases = [
        (descriptor_frame([(rid, "algorithm_name", "other", "algorithm")]), "differ"),
        (descriptor_frame([(rid, "instance", 3, "run")]), "reserved"),
        (descriptor_frame([(rid, "k", 1, "somewhere")]), "unknown origin"),
        (descriptor_frame([(rid, "k", 1, "run"), (rid, "k", 2, "run")]), "not unique"),
        (descriptor_frame([(rid, "k\n", 1, "run")]), "control"),
        (
            pl.DataFrame({"run_id": [rid], "key": ["k"], "value": ["1"]}),
            "descriptor_frame",
        ),
    ]
    for frame, message in cases:
        with pytest.raises(FormatError, match=message):
            write_dataset(
                tmp_path / "ds", syn.manifest, syn.runs, syn.trace, descriptors=frame
            )
        assert not (tmp_path / "ds").exists()


def test_given_algorithm_descriptors_must_agree_and_may_be_user_edited(tmp_path):
    syn = make_dataset(seed=44)
    rows = syn.runs.select("run_id", "algorithm_name").rows()
    given = descriptor_frame([(r, "algorithm_name", name, "user") for r, name in rows])
    write_dataset(tmp_path / "ds", syn.manifest, syn.runs, syn.trace, descriptors=given)
    desc = scan_dataset(tmp_path / "ds").descriptors.collect()
    origins = (
        desc.filter(pl.col("key") == "algorithm_name")["origin"].unique().to_list()
    )
    assert origins == ["user"]


# -- validation of descriptors ----------------------------------------------------


@pytest.fixture(scope="module")
def described(tmp_path_factory):
    syn = make_dataset(seed=45)
    path = tmp_path_factory.mktemp("described") / "ds"
    write_dataset(path, syn.manifest, syn.runs, syn.trace, descriptors=syn.descriptors)
    return path


@pytest.fixture
def ds(described, tmp_path):
    shutil.copytree(described, tmp_path / "ds")
    return tmp_path / "ds"


def rewrite_descriptors(path, change):
    file = path / "descriptors.parquet"
    change(pl.read_parquet(file)).write_parquet(file, compression="zstd")


def first(df, key):
    return (pl.col("key") == key) & (pl.col("run_id") == df["run_id"].min())


@pytest.mark.parametrize(
    "change, fragment",
    [
        (
            lambda d: d.with_columns(
                pl.when(first(d, "algorithm_name"))
                .then(pl.lit("x"))
                .otherwise("value")
                .alias("value")
            ),
            "disagree with runs.parquet",
        ),
        (
            lambda d: d.filter(~first(d, "algorithm_info")),
            "lack a required algorithm_info",
        ),
        (lambda d: d.with_columns(pl.lit("folder").alias("origin")), "unknown origin"),
        (
            lambda d: d.with_columns(
                pl.when(first(d, "seed"))
                .then(pl.lit("integer"))
                .otherwise("type")
                .alias("type")
            ),
            "unknown type",
        ),
        (
            lambda d: d.with_columns(
                pl.when(first(d, "seed"))
                .then(pl.lit("007"))
                .otherwise("value")
                .alias("value")
            ),
            "not canonical",
        ),
        (
            lambda d: d.with_columns(
                pl.when(first(d, "seed"))
                .then(pl.lit(99, pl.Int64))
                .otherwise("value_int")
                .alias("value_int")
            ),
            "not canonical",
        ),
        (
            lambda d: d.with_columns(
                pl.when(first(d, "seed"))
                .then(pl.lit(1.0))
                .otherwise("value_float")
                .alias("value_float")
            ),
            "typed columns of other types",
        ),
        (lambda d: pl.concat([d, d.head(1)]).sort("key", "run_id"), "not unique"),
        (
            lambda d: d.with_columns(
                pl.when(first(d, "seed"))
                .then(pl.lit(1, pl.Int64))
                .otherwise("run_id")
                .alias("run_id")
            ).sort("key", "run_id"),
            "not in runs",
        ),
        (lambda d: d.reverse(), "not sorted"),
        (
            lambda d: d.with_columns(
                pl.when(pl.col("key") == "folder")
                .then(pl.lit("instance"))
                .otherwise("key")
                .alias("key")
            ).sort("key", "run_id"),
            "reserved",
        ),
        (lambda d: d.drop("value_bool"), "column 'value_bool' is missing"),
    ],
)
def test_descriptor_problems_are_reported(ds, change, fragment):
    rewrite_descriptors(ds, change)
    report = validate_dataset(ds)
    assert any(fragment in e for e in report.errors), report.errors


def test_float_text_must_round_trip(ds):
    syn_rows = descriptor_frame([(1, "f", 0.1, "run")])
    assert syn_rows["value"][0] == "0.1"
    rows = pl.read_parquet(ds / "descriptors.parquet")
    rid = rows["run_id"][0]
    extra = pl.DataFrame(
        {
            "run_id": [rid],
            "key": ["zz_float"],
            "origin": ["run"],
            "type": ["float"],
            "value": ["0.1"],
            "value_int": [None],
            "value_float": [0.10000000000000002],
            "value_bool": [None],
        },
        schema=DESCRIPTORS_SCHEMA,
    )
    rewrite_descriptors(ds, lambda d: pl.concat([d, extra]))
    report = validate_dataset(ds)
    assert any(
        "float row(s) have a value that is not canonical" in e for e in report.errors
    )


def test_manifest_summary_must_match(ds):
    file = ds / "manifest.json"
    m = json.loads(file.read_text())
    m["descriptors"]["seed"]["distinct"] += 1
    m["descriptors"]["popsize"]["examples"] = ["12345"]
    m["descriptors"]["ghost"] = dict(m["descriptors"]["seed"])
    file.write_text(json.dumps(m))
    errors = "\n".join(validate_dataset(ds).errors)
    assert "seed'].distinct" in errors and "popsize'].examples" in errors
    assert "'ghost' is in the manifest" in errors


def test_missing_descriptors_file_and_counts(ds):
    file = ds / "manifest.json"
    m = json.loads(file.read_text())
    m["counts"]["descriptors_rows"] += 1
    file.write_text(json.dumps(m))
    assert any("counts.descriptors_rows" in e for e in validate_dataset(ds).errors)
    (ds / "descriptors.parquet").unlink()
    assert any(
        "descriptors.parquet is missing" in e for e in validate_dataset(ds).errors
    )


@pytest.mark.parametrize(
    "summary",
    [
        {
            "types": ["integer"],
            "origins": ["run"],
            "runs": 1,
            "distinct": 1,
            "examples": ["1"],
        },
        {
            "types": ["int"],
            "origins": ["run"],
            "runs": 1,
            "distinct": 1,
            "examples": list("123456"),
        },
        {
            "types": ["int"],
            "origins": ["disk"],
            "runs": 1,
            "distinct": 1,
            "examples": ["1"],
        },
        {
            "types": ["int"],
            "origins": ["run"],
            "runs": 0,
            "distinct": 1,
            "examples": ["1"],
        },
    ],
)
def test_manifest_schema_checks_the_summary(summary):
    m = new_manifest("x", "test", "0")
    m["descriptors"] = {"seed": summary}
    assert any("descriptors" in p for p in validate_manifest(m))


def test_upsert_skips_none_and_refuses_containers(tmp_path):
    syn = make_dataset(seed=43)
    path = tmp_path / "ds"
    write_dataset(path, syn.manifest, syn.runs, syn.trace, descriptors=syn.descriptors)
    ids = syn.runs["run_id"].to_list()
    update_descriptors(
        path, [(ids[0], "tag", "a", "user"), (ids[1], "tag", None, "user")]
    )
    ds = scan_dataset(path)
    assert ds.manifest["descriptors"]["tag"]["runs"] == 1
    before = {p.name: _file_digest(p) for p in path.iterdir()}
    with pytest.raises(FormatError, match="scalars"):
        update_descriptors(path, [(ids[0], "tag", ["a"], "user")])
    assert {p.name: _file_digest(p) for p in path.iterdir()} == before


def test_grouping_with_a_key_missing_in_some_runs(tmp_path):
    syn = make_dataset(functions=(1,), dimensions=(2,), runs_per_file=6, seed=44)
    ids = syn.runs["run_id"].to_list()
    records = [
        (rid, "popsize", 10 * (n % 2 + 1), "algorithm") for n, rid in enumerate(ids)
    ]
    records += [(ids[0], "note", None, "run"), (ids[1], "note", "x", "run")]
    path = tmp_path / "missing"
    write_dataset(
        path, syn.manifest, syn.runs, syn.trace, descriptors=descriptor_frame(records)
    )
    assert validate_dataset(path).ok
    ds = scan_dataset(path)
    assert ds.manifest["descriptors"]["note"]["runs"] == 1
    col = ds.run_columns(["note"]).collect()
    assert col.schema["note"] == pl.String
    by_run = dict(zip(col["run_id"], col["note"]))
    assert by_run[ids[1]] == "x"
    assert all(by_run.get(r) is None for r in ids if r != ids[1])


@pytest.mark.parametrize("kind", ["json", "null"])
def test_validation_rejects_dropped_types(ds, kind):
    rewrite_descriptors(
        ds,
        lambda d: d.with_columns(
            pl.when(pl.col("key") == "seed")
            .then(pl.lit(kind))
            .otherwise("type")
            .alias("type")
        ),
    )
    report = validate_dataset(ds)
    assert any("unknown type" in e for e in report.errors), report.errors
