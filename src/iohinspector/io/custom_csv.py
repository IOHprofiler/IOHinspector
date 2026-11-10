"""Reader for custom CSV files with a declared column mapping.

The mapping is a JSON object. Names of columns are strings; a constant is written
``{"constant": value}``. Example for a file with one row per logged evaluation::

    {
      "version": 1,
      "delimiter": ",",
      "evaluations": "evals",
      "value": "f",
      "run_by": ["alg", "fid", "dim", "seed"],
      "fields": {
        "algorithm_name": "alg",
        "function_id": "fid",
        "dimension": "dim",
        "maximization": {"constant": false}
      },
      "descriptors": ["seed"],
      "trace_columns": ["sigma"]
    }

Keys:

``evaluations``, ``value`` (required)
    Columns with the evaluation count (an integer) and the objective value (a
    number, ``inf`` and ``nan`` included). They become ``evaluations`` and
    ``raw_y`` of the trace.
``fields`` (``algorithm_name``, ``dimension`` and ``maximization`` are required)
    Per-run values: ``algorithm_name``, ``algorithm_info``, ``suite``,
    ``function_id`` (integer), ``function_name``, ``instance`` (integer),
    ``dimension`` (integer), ``maximization`` (true or false), ``optimum``
    (number), ``budget`` and ``evals_used`` (integers). Without ``function_id``,
    the functions named in ``function_name`` get ids 1, 2, ... in the order of
    their sorted names; without both, the function id is 1.
``run_by``
    Columns that identify a run. Rows with the same values form one run, in the
    order of the file, and may be interleaved with other runs. Without ``run_by``
    every row is a run of its own. Columns that describe a run (those in
    ``fields`` and ``descriptors``) must not change within it.
``descriptors``
    A list of columns kept as descriptors (origin ``run``), or ``"rest"`` for every
    column not used for anything else. Default: none. The type of a descriptor is
    that of its whole column: bool, int, float or string; an empty cell gives no
    descriptor.
``trace_columns``
    Columns kept per evaluation as extra trace columns, of the same inferred types;
    an empty cell is null.
``positions``
    Columns ``x0``, ``x1``, ... of the evaluated point, one per dimension; the
    number must equal the dimension of every run.
``delimiter``
    One character, default ``,``.

A column name that the format reserves (``budget``, ``dimension``, ``row``, ...)
is renamed with the suffix ``_src`` when it is kept as a descriptor or trace column,
and recorded in ``manifest.renamed_columns``.

Every data line of the file is kept. Source-run hashes (``source_rows_sha256``) are
computed from the fields of each line as the ``csv`` module reads them (quotes
removed), joined by one space.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._csv import (
    FIELDS,
    CsvMapping,
    MappingError,
    Source,
    check_mapping,
    convert,
)

READER_NAME = "custom-csv"
MAPPING_VERSION = 1
_KEYS = (
    "version",
    "delimiter",
    "evaluations",
    "value",
    "fields",
    "run_by",
    "descriptors",
    "trace_columns",
    "positions",
)
_CONSTANT_TYPES = {
    "string": (str,),
    "int": (int,),
    "float": (int, float),
    "bool": (bool,),
}


def _suggest(name: str, known: Sequence[str]) -> str:
    close = difflib.get_close_matches(name, list(known), n=1)
    return f" Did you mean {close[0]!r}?" if close else ""


def _column(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise MappingError(f"{where}: expected the name of a column (a string)")
    return value


def _columns(value: Any, where: str) -> list[str]:
    if not isinstance(value, list):
        raise MappingError(f"{where}: expected a list of column names")
    cols = [_column(v, f"{where}[{i}]") for i, v in enumerate(value)]
    if len(set(cols)) != len(cols):
        raise MappingError(f"{where}: a column is listed twice")
    return cols


def _source(name: str, kind: str, value: Any) -> Source:
    where = f"fields.{name}"
    if isinstance(value, str):
        return Source(column=_column(value, where))
    if isinstance(value, dict) and set(value) == {"constant"}:
        const = value["constant"]
        ok = isinstance(const, _CONSTANT_TYPES[kind]) and (
            kind == "bool" or not isinstance(const, bool)
        )
        if not ok:
            raise MappingError(f"{where}: the constant must be of type {kind}")
        if name == "algorithm_name" and const == "":
            raise MappingError(f"{where}: the algorithm name must not be empty")
        return Source(constant=const)
    raise MappingError(
        f'{where}: expected a column name or {{"constant": value}}, got {value!r}'
    )


def parse_mapping(obj: Mapping[str, Any]) -> CsvMapping:
    """A ``CsvMapping`` from the JSON object; raises ``MappingError`` with the reason."""
    if not isinstance(obj, Mapping):
        raise MappingError("the mapping must be a JSON object")
    for key in obj:
        if key not in _KEYS:
            raise MappingError(
                f"unknown key {key!r}.{_suggest(key, _KEYS)} Allowed keys: {', '.join(_KEYS)}"
            )
    if obj.get("version", MAPPING_VERSION) != MAPPING_VERSION:
        raise MappingError(f"version must be {MAPPING_VERSION}")
    for key in ("evaluations", "value", "fields"):
        if key not in obj:
            raise MappingError(f"{key!r} is required")
    delimiter = obj.get("delimiter", ",")
    if not isinstance(delimiter, str) or len(delimiter) != 1 or delimiter in '\r\n"':
        raise MappingError("delimiter must be a single character")
    raw_fields = obj["fields"]
    if not isinstance(raw_fields, dict):
        raise MappingError("fields: expected an object")
    fields: dict[str, Source] = {}
    for name, value in raw_fields.items():
        if name not in FIELDS:
            raise MappingError(
                f"fields: unknown field {name!r}.{_suggest(name, list(FIELDS))} "
                f"Allowed fields: {', '.join(FIELDS)}"
            )
        fields[name] = _source(name, FIELDS[name][0], value)
    for name, (_kind, required) in FIELDS.items():
        if required and name not in fields:
            raise MappingError(f"fields.{name} is required")
    descriptors: Sequence[str] | str = ()
    if "descriptors" in obj:
        descriptors = (
            "rest"
            if obj["descriptors"] == "rest"
            else _columns(obj["descriptors"], "descriptors")
        )
    return CsvMapping(
        evaluations=_column(obj["evaluations"], "evaluations"),
        value=_column(obj["value"], "value"),
        fields=fields,
        delimiter=delimiter,
        run_by=_columns(obj.get("run_by", []), "run_by"),
        descriptors=descriptors,
        trace_columns=_columns(obj.get("trace_columns", []), "trace_columns"),
        positions=_columns(obj.get("positions", []), "positions"),
    )


def load_mapping(mapping: CsvMapping | Mapping[str, Any] | str | Path) -> CsvMapping:
    """A mapping from an object, a JSON text or a path to a JSON file."""
    if isinstance(mapping, CsvMapping):
        return mapping
    if isinstance(mapping, Path) or (
        isinstance(mapping, str) and not mapping.lstrip().startswith("{")
    ):
        mapping = Path(mapping).read_text(encoding="utf-8")
    if isinstance(mapping, str):
        try:
            mapping = json.loads(mapping)
        except json.JSONDecodeError as exc:
            raise MappingError(f"the mapping is not valid JSON: {exc}") from exc
    return parse_mapping(mapping)  # type: ignore[arg-type]


def validate_mapping(
    mapping: CsvMapping | Mapping[str, Any] | str | Path,
    columns: Sequence[str] | None = None,
) -> CsvMapping:
    """Check a mapping, and when ``columns`` are given, that the file has them all."""
    parsed = load_mapping(mapping)
    if columns is not None:
        check_mapping(parsed, columns)
    return parsed


def convert_custom_csv(
    source: str | Path,
    output: str | Path,
    mapping: CsvMapping | Mapping[str, Any] | str | Path,
    *,
    dataset_id: str | None = None,
    source_name: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert a CSV file with a declared mapping to a dataset in ``output``.

    ``mapping`` is a JSON object, JSON text or a path to a JSON file; see the module
    documentation. Raises ``MappingError`` for a mapping that is not valid or does
    not fit the file's columns, and ``FormatError`` for data that breaks the
    format, with the file name and line. Returns the manifest.
    """
    parsed = load_mapping(mapping)
    return convert(
        source,
        output,
        parsed,
        reader_name=READER_NAME,
        kind="custom-csv",
        origin="run",
        dataset_id=dataset_id,
        source_name=source_name,
        overwrite=overwrite,
        reader_options={"mapping": _mapping_json(mapping, parsed)},
        delimiter=parsed.delimiter,
    )


def _mapping_json(given: Any, parsed: CsvMapping) -> dict[str, Any]:
    """The mapping as JSON, to store in the manifest."""
    if isinstance(given, Mapping):
        return json.loads(json.dumps(given))
    fields = {
        k: s.column if s.column is not None else {"constant": s.constant}
        for k, s in parsed.fields.items()
    }
    return {
        "version": MAPPING_VERSION,
        "delimiter": parsed.delimiter,
        "evaluations": parsed.evaluations,
        "value": parsed.value,
        "fields": fields,
        "run_by": list(parsed.run_by),
        "descriptors": parsed.descriptors
        if isinstance(parsed.descriptors, str)
        else list(parsed.descriptors),
        "trace_columns": list(parsed.trace_columns),
        "positions": list(parsed.positions),
    }
