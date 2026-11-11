"""Descriptors (decision D17): named, typed values that describe runs.

``descriptors.parquet`` has one row per run and key. The value is kept exactly: its
type in ``type``, canonical text in ``value``, and a typed copy in ``value_int``,
``value_float`` or ``value_bool``. See ``docs/format.md``, "descriptors.parquet".
"""

from __future__ import annotations

import math
import numbers
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl

from . import _frames as fr
from ._io import replace_table
from .manifest import read_manifest, utc_now, write_manifest
from .schema import (
    ALGORITHM_DESCRIPTOR_ORIGINS,
    ALGORITHM_DESCRIPTORS,
    DESCRIPTOR_COLUMNS,
    DESCRIPTOR_ORIGINS,
    DESCRIPTOR_TYPES,
    DESCRIPTORS_FILE,
    DESCRIPTORS_SCHEMA,
    DESCRIPTORS_SORT_KEYS,
    RUNS_FILE,
    descriptor_key_problem,
)

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_INT_TEXT = r"^(0|-?[1-9][0-9]*)$"
MAX_EXAMPLES = 5

Encoded = tuple[str, str, "int | None", "float | None", "bool | None"]


def encode_descriptor(value: Any) -> Encoded:
    """``(type, value, value_int, value_float, value_bool)`` for a Python value.

    Descriptor values are scalars: bool, int (any size), float and str map to
    ``bool``, ``int``, ``float`` and ``string``. numpy scalars are accepted. None
    has no descriptor (a key without a value is not stored), and lists, tuples and
    dicts are not converted: both raise ``FormatError``. A reader that meets a
    container converts it to text itself and stores a ``string``. Raises
    ``TypeError`` for other values.
    """
    if isinstance(value, (bool, np.bool_)):
        b = bool(value)
        return "bool", "true" if b else "false", None, None, b
    if isinstance(value, numbers.Integral):
        i = int(value)
        return "int", str(i), i if _INT64_MIN <= i <= _INT64_MAX else None, None, None
    if isinstance(value, numbers.Real):
        f = float(value)
        return "float", repr(f), None, f, None
    if isinstance(value, str):
        return "string", value, None, None, None
    if value is None or isinstance(value, (Mapping, list, tuple)):
        what = "None" if value is None else type(value).__name__
        raise fr.FormatError(
            f"descriptor values are scalars (string, int, float, bool); got {what}. "
            "Leave out a key without a value, and convert a list or dict to text"
        )
    raise TypeError(f"descriptor values cannot be of type {type(value).__name__}")


def decode_descriptor(kind: str, value: str) -> Any:
    """The Python value of a stored descriptor, from its ``type`` and ``value``."""
    if kind == "string":
        return value
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    if kind == "bool":
        return value == "true"
    raise ValueError(f"unknown descriptor type {kind!r}")


def descriptor_frame(
    records: Iterable[Sequence[Any] | Mapping[str, Any]],
) -> pl.DataFrame:
    """A descriptors table from ``(run_id, key, value, origin)`` records.

    Each record is a 4-tuple or a mapping with those names; ``value`` is a scalar
    Python value, encoded with ``encode_descriptor``. A record whose value is None
    (a key without a value) gets no row. A list, tuple or dict value raises
    ``FormatError``.
    """
    cols: dict[str, list[Any]] = {c.name: [] for c in DESCRIPTOR_COLUMNS}
    for record in records:
        if isinstance(record, Mapping):
            run, key, value, origin = (
                record["run_id"],
                record["key"],
                record["value"],
                record["origin"],
            )
        else:
            run, key, value, origin = record
        if value is None:
            continue
        kind, text, vi, vf, vb = encode_descriptor(value)
        for name, item in zip(
            DESCRIPTORS_SCHEMA,
            (run, key, origin, kind, text, vi, vf, vb),
            strict=True,
        ):
            cols[name].append(item)
    return pl.DataFrame(cols, schema=DESCRIPTORS_SCHEMA)


def empty_descriptors() -> pl.DataFrame:
    return pl.DataFrame(schema=DESCRIPTORS_SCHEMA)


def conform_descriptors(data: Any) -> pl.DataFrame:
    """Descriptors given to the writer, with the format's columns and types."""
    if data is None:
        return empty_descriptors()
    frames = [_conform_frame(df) for df in fr.iter_frames(data, 2**20, "descriptors")]
    return pl.concat(frames, how="vertical") if frames else empty_descriptors()


def _conform_frame(df: pl.DataFrame) -> pl.DataFrame:
    missing = [c for c in DESCRIPTORS_SCHEMA if c not in df.columns]
    unknown = [c for c in df.columns if c not in DESCRIPTORS_SCHEMA]
    if missing or unknown:
        raise fr.FormatError(
            f"descriptors: expected the columns {list(DESCRIPTORS_SCHEMA)}, "
            f"missing {missing}, unknown {unknown}; build them with descriptor_frame"
        )
    return pl.DataFrame(
        [
            fr.conform(df[c.name], c.dtype, f"descriptors column {c.name!r}")
            for c in DESCRIPTOR_COLUMNS
        ]
    )


def _missing_info_rows(given: pl.DataFrame, runs: pl.DataFrame) -> pl.DataFrame:
    """``algorithm_info`` descriptors with value ``""`` for runs that lack one."""
    has_info = given.filter(pl.col("key") == "algorithm_info").select("run_id")
    return (
        runs.select("run_id")
        .join(has_info, on="run_id", how="anti")
        .select(
            "run_id",
            pl.lit("algorithm_info").alias("key"),
            pl.lit("algorithm").alias("origin"),
            pl.lit("string").alias("type"),
            pl.lit("").alias("value"),
            pl.lit(None, pl.Int64).alias("value_int"),
            pl.lit(None, pl.Float64).alias("value_float"),
            pl.lit(None, pl.Boolean).alias("value_bool"),
        )
    )


def prepare_descriptors(data: Any, runs: pl.DataFrame) -> pl.DataFrame:
    """The full descriptors table for ``runs``, sorted and checked.

    Every run must have an ``algorithm_name`` descriptor; runs without
    ``algorithm_info`` get ``""`` (origin ``algorithm``). Raises ``FormatError`` on
    problems.
    """
    given = conform_descriptors(data)
    df = pl.concat([given, _missing_info_rows(given, runs)], how="vertical")
    df = sort_descriptors(df)
    problems = descriptor_problems(df, runs)
    if problems:
        raise fr.FormatError(fr.first_problem(problems))
    return df


def sort_descriptors(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(list(DESCRIPTORS_SORT_KEYS), maintain_order=True)


def descriptor_problems(df: pl.DataFrame, runs: pl.DataFrame | None) -> list[str]:
    """Problems with a descriptors table that has the format's columns and types.

    With ``runs`` (a table with ``run_id``), also checks referential integrity and
    that every run has the algorithm descriptors.
    """
    not_null = [c.name for c in DESCRIPTOR_COLUMNS if not c.nullable]
    problems = fr.null_problems(df, not_null, "descriptors")
    if problems:
        return problems
    problems = [*_row_problems(df), *_value_problems(df)]
    if runs is not None:
        problems += _run_problems(df, runs)
    return problems


def _row_problems(df: pl.DataFrame) -> list[str]:
    """Known origins and types, valid keys, one value per run and key, sort order."""
    problems = []
    for column, allowed in (("origin", DESCRIPTOR_ORIGINS), ("type", DESCRIPTOR_TYPES)):
        bad = df.filter(~pl.col(column).is_in(list(allowed)))
        if bad.height:
            problems.append(
                f"descriptors: {bad.height} row(s) have an unknown {column}, for example "
                f"{bad[column][0]!r}; allowed are {', '.join(allowed)}"
            )
    for key in df["key"].unique().sort().to_list():
        problem = descriptor_key_problem(key)
        if problem is not None:
            problems.append(f"descriptors: invalid key {key!r}: {problem}")
    pair = pl.struct("run_id", "key")
    if df.select(pair.n_unique()).item() != df.height:
        dup = df.filter(pair.is_duplicated()).row(0, named=True)
        problems.append(
            f"descriptors: (run_id, key) is not unique, for example run {dup['run_id']} "
            f"key {dup['key']!r}"
        )
    ordered = sort_descriptors(df)
    if not ordered.select("key", "run_id").equals(df.select("key", "run_id")):
        problems.append("descriptors: rows are not sorted by key and run_id")
    return problems


def _run_problems(df: pl.DataFrame, runs: pl.DataFrame) -> list[str]:
    """Every run id exists; every run has the algorithm descriptors, as strings."""
    problems = []
    unknown = df.join(runs.select("run_id"), on="run_id", how="anti")
    if unknown.height:
        problems.append(
            f"descriptors: {unknown.height} row(s) refer to run ids that are not in "
            f"runs, for example {unknown['run_id'][0]}"
        )
    for key in ALGORITHM_DESCRIPTORS:
        rows = df.filter(pl.col("key") == key)
        absent = runs.select("run_id").join(rows, on="run_id", how="anti")
        if absent.height:
            problems.append(
                f"descriptors: {absent.height} run(s) lack a required {key} "
                f"descriptor, for example run {absent['run_id'][0]}"
            )
        bad = rows.filter(
            (pl.col("type") != "string")
            | ~pl.col("origin").is_in(list(ALGORITHM_DESCRIPTOR_ORIGINS))
        )
        if bad.height:
            r = bad.row(0, named=True)
            problems.append(
                f"descriptors: {bad.height} {key} descriptor(s) are not of type string "
                f"with origin {' or '.join(ALGORITHM_DESCRIPTOR_ORIGINS)}, for example "
                f"run {r['run_id']}: type {r['type']!r}, origin {r['origin']!r}"
            )
    return problems


def _value_problems(df: pl.DataFrame) -> list[str]:
    """Each row's value is canonical for its type and matches its typed column."""
    problems = []
    typed = {"int": "value_int", "float": "value_float", "bool": "value_bool"}
    for kind in DESCRIPTOR_TYPES:
        rows = df.filter(pl.col("type") == kind)
        if not rows.height:
            continue
        unused = [c for t, c in typed.items() if t != kind]
        if kind not in typed:
            unused = list(typed.values())
        extra = rows.filter(
            pl.any_horizontal([pl.col(c).is_not_null() for c in unused])
        )
        if extra.height:
            problems.append(
                f"descriptors: {extra.height} {kind} row(s) fill typed columns of other types"
            )
        bad = _bad_values(rows, kind)
        if bad is not None and bad.height:
            r = bad.row(0, named=True)
            problems.append(
                f"descriptors: {bad.height} {kind} row(s) have a value that is not canonical "
                f"or disagrees with the typed column, for example key {r['key']!r} "
                f"value {r['value'][:60]!r}"
            )
    return problems


def _bad_values(rows: pl.DataFrame, kind: str) -> pl.DataFrame | None:
    value = pl.col("value")
    if kind == "int":
        parsed = value.cast(pl.Int64, strict=False)
        return rows.filter(
            ~value.str.contains(_INT_TEXT)
            | (parsed.is_null() != pl.col("value_int").is_null())
            | (parsed != pl.col("value_int")).fill_null(False)
        )
    if kind == "bool":
        return rows.filter(
            ~value.is_in(["true", "false"])
            | pl.col("value_bool").is_null()
            | ((value == "true") != pl.col("value_bool"))
        )
    if kind == "float":
        pairs = rows.select("value", "value_float").unique()
        bad = [
            text
            for text, number in pairs.iter_rows()
            if number is None or not _same_float(text, number)
        ]
        return rows.filter(value.is_in(bad))
    return None


def _same_float(text: str, number: float) -> bool:
    try:
        parsed = float(text)
    except ValueError:
        return False
    if math.isnan(parsed) or math.isnan(number):
        return math.isnan(parsed) and math.isnan(number)
    return parsed == number and math.copysign(1.0, parsed) == math.copysign(1.0, number)


# -- manifest summary -------------------------------------------------------------


def summarize(df: pl.DataFrame) -> dict[str, dict[str, Any]]:
    """The manifest's ``descriptors`` summary of a descriptors table."""
    if not df.height:
        return {}
    stats = (
        df.group_by("key")
        .agg(
            pl.col("type").unique().sort().alias("types"),
            pl.col("origin").unique().sort().alias("origins"),
            pl.col("run_id").n_unique().alias("runs"),
            pl.struct("type", "value").n_unique().alias("distinct"),
            pl.col("value").unique().sort().head(MAX_EXAMPLES).alias("examples"),
        )
        .sort("key")
    )
    return {
        row["key"]: {
            "types": list(row["types"]),
            "origins": list(row["origins"]),
            "runs": int(row["runs"]),
            "distinct": int(row["distinct"]),
            "examples": list(row["examples"]),
        }
        for row in stats.iter_rows(named=True)
    }


def summary_problems(summary: Any, df: pl.DataFrame) -> list[str]:
    """Differences between a manifest summary and the descriptors table."""
    if not isinstance(summary, dict):
        return ["manifest descriptors: must be an object"]
    actual = summarize(df)
    problems = []
    for key in sorted(set(summary) ^ set(actual)):
        where = "the manifest" if key in summary else "descriptors.parquet only"
        problems.append(f"manifest descriptors: key {key!r} is in {where}")
    known: dict[str, set[str]] = {}
    if df.height:
        values = df.group_by("key").agg(pl.col("value").unique())
        known = {k: set(v) for k, v in values.iter_rows()}
    for key in sorted(set(summary) & set(actual)):
        problems += _entry_problems(key, summary[key], actual[key], known[key])
    return problems


def _entry_problems(
    key: str, given: Any, expected: dict[str, Any], values: set[str]
) -> list[str]:
    """Differences between one key's summary and the table's values of that key."""
    if not isinstance(given, dict):
        return [f"manifest descriptors[{key!r}] must be an object"]
    problems = [
        f"manifest descriptors[{key!r}].{field} is {given.get(field)!r}, "
        f"but the table has {expected[field]!r}"
        for field in ("types", "origins", "runs", "distinct")
        if given.get(field) != expected[field]
    ]
    examples = given.get("examples")
    texts = [str(e) for e in examples] if isinstance(examples, list) else []
    if not (
        1 <= len(texts) <= MAX_EXAMPLES
        and len(set(texts)) == len(texts)
        and set(texts) <= values
    ):
        problems.append(
            f"manifest descriptors[{key!r}].examples must be 1 to {MAX_EXAMPLES} "
            "distinct values of the key"
        )
    return problems


# -- columns for grouping ----------------------------------------------------------


def descriptor_columns(
    descriptors: pl.LazyFrame | pl.DataFrame,
    keys: Sequence[str],
    *,
    mixed: Literal["string", "error"] = "string",
) -> pl.LazyFrame:
    """``run_id`` and one column per descriptor key, one row per run that has any.

    A key with one type gets that type (String for ``string``, Int64
    for ``int`` unless a value is outside int64, then String with the exact decimal
    text, Float64, Boolean). A key with several types gets the ``value`` text, or
    raises ``ValueError`` with ``mixed="error"``. Runs without the key are null.
    Raises ``ValueError`` for a key the table does not have.
    """
    lf = descriptors.lazy()
    keys = list(dict.fromkeys(keys))
    base = lf.filter(pl.col("key").is_in(keys)).select("run_id").unique()
    for key in keys:
        base = base.join(_key_column(lf, key, mixed), on="run_id", how="left")
    return base.select("run_id", *keys)


def _key_column(lf: pl.LazyFrame, key: str, mixed: str) -> pl.LazyFrame:
    """``run_id`` and the values of one key, typed as ``descriptor_columns`` says."""
    rows = lf.filter(pl.col("key") == key)
    stats = (
        rows.group_by("type")
        .agg(pl.col("value_int").null_count().alias("nulls"))
        .collect()
    )
    if not stats.height:
        raise ValueError(f"no descriptor or runs column is named {key!r}")
    types = set(stats["type"].to_list())
    big = stats.filter(pl.col("type") == "int")["nulls"].sum() > 0
    if types == {"int"} and not big:
        expr = pl.col("value_int")
    elif types == {"float"}:
        expr = pl.col("value_float")
    elif types == {"bool"}:
        expr = pl.col("value_bool")
    elif len(types) > 1 and mixed == "error":
        raise ValueError(f"descriptor {key!r} has several types: {sorted(types)}")
    else:  # string, int with values outside int64, or several types
        expr = pl.col("value")
    return rows.select("run_id", expr.alias(key))


# -- editing after conversion -------------------------------------------------------


def update_descriptors(
    path: str | os.PathLike[str],
    upsert: pl.DataFrame | Iterable[Sequence[Any] | Mapping[str, Any]] | None = None,
    remove_keys: Iterable[str] = (),
    *,
    origin: str | None = "user",
) -> dict[str, Any]:
    """Add, change or remove descriptors of the stored dataset ``path``.

    ``upsert`` is a descriptors table (see ``descriptor_frame``) or records
    ``(run_id, key, value, origin)``; its rows replace those with the same
    ``(run_id, key)`` and are stored with origin ``origin`` (default ``user``), or
    with their own origins when ``origin`` is None (as the converter does for
    descriptors taken from paths). ``remove_keys`` drops
    keys from every run; ``algorithm_name`` and ``algorithm_info`` can be changed
    (to strings) but not removed. Only descriptors.parquet and the manifest are
    rewritten; runs, trace and positions are not touched. Nothing is written when
    a check fails. Returns the new manifest.
    """
    path = Path(path)
    manifest = read_manifest(path, warn=False)
    runs = pl.read_parquet(path / RUNS_FILE, columns=["run_id"])
    table = _without_keys(pl.read_parquet(path / DESCRIPTORS_FILE), remove_keys)
    if upsert is not None:
        if not isinstance(upsert, pl.DataFrame):
            upsert = descriptor_frame(upsert)
        new = conform_descriptors(upsert)
        if origin is not None:
            new = new.with_columns(pl.lit(origin).alias("origin"))
        pairs = new.select("run_id", "key")
        table = pl.concat(
            [table.join(pairs, on=["run_id", "key"], how="anti"), new], how="vertical"
        )

    table = sort_descriptors(table)
    problems = descriptor_problems(table, runs)
    if problems:
        raise fr.FormatError(fr.first_problem(problems))

    manifest["descriptors"] = summarize(table)
    manifest["counts"]["descriptors_rows"] = table.height
    manifest["updated"] = utc_now()
    replace_table(
        table, list(DESCRIPTOR_COLUMNS), "descriptors", path / DESCRIPTORS_FILE
    )
    write_manifest(path, manifest)
    return manifest


def _without_keys(table: pl.DataFrame, remove_keys: Iterable[str]) -> pl.DataFrame:
    """``table`` without the keys to remove, which must exist and not be required."""
    remove = list(dict.fromkeys(remove_keys))
    protected = [k for k in remove if k in ALGORITHM_DESCRIPTORS]
    if protected:
        raise fr.FormatError(
            f"descriptors {protected} are required and cannot be removed"
        )
    existing = set(table["key"].unique().to_list())
    absent = [k for k in remove if k not in existing]
    if absent:
        raise fr.FormatError(f"descriptors {absent} do not exist")
    return table.filter(~pl.col("key").is_in(remove))
