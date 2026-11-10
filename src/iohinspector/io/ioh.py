"""Reader for IOHexperimenter Analyzer output: JSON metadata plus ``.dat`` files.

Covers the JSON Analyzer of IOHexperimenter 0.3 and 0.4. One JSON file describes
one function of one algorithm; each of its ``scenarios`` names the ``.dat`` file of
one dimension and lists that file's runs in order. A ``.dat`` file starts every run
with a header line (``evaluations raw_y ...``) followed by one line per logged
evaluation.

The reader is lossless (``docs/format.md``): every data line becomes a trace row,
in source order, including repeated values and the final row the logger writes at
the end of each run. Nothing is deduplicated or made monotone. Runs are split on
header lines, never on ``evaluations == 1``. Malformed input raises
``ReaderError`` with the file name and line number.

Quirks of the logger, each handled and recorded in the manifest:

- Strings in the JSON are written without escaping, so a backslash or quote in an
  algorithm name or info or in an experiment attribute breaks the JSON or changes
  its value. The reader re-escapes those fixed-format lines before parsing, and
  warns (``json_repaired``) when that changed the text.
- Non-finite numbers are written as bare ``nan`` and ``inf``, which JSON does not
  allow. They are read as IEEE values (also ``json_repaired``).
- Values are written with ``%.10f`` unless the JSON says ``"precision"``
  (``%.17g`` since the full-precision option); positions with ``%f``. The source
  entries record the precision, and a ``limited_precision`` warning says that
  values below 5e-11 were written as 0. Nothing is corrected.
- A ``.dat`` file named in the JSON may be missing. Its runs are kept without trace
  rows, the file is listed as a source with a null sha256 and size, and a
  ``missing_data_file`` warning is added.
- The JSON and the ``.dat`` file can disagree on the number of runs (the JSON is
  rewritten after every run, and some triggers leave runs out of it). Runs are
  matched in order; runs only in the ``.dat`` file have no instance, evaluation
  count or best value, and a ``run_count_mismatch`` warning is added.
- The JSON does not say which triggers the logger used. The reader infers ``all``
  (every evaluation logged) or ``improvement`` from the data, and records that
  with ``params: {"inferred": true}``; otherwise the triggers stay unknown.
- IOHexperimenter's ``FlatFile`` logger wrote data rows with one field fewer than
  the header (the separator bug fixed after 0.4.0). The Analyzer writes no meta
  data columns and was not affected; a ``.dat`` file with that shape is refused
  with a message naming the bug, because the glued values cannot be split
  reliably.
- Missing values of logged properties are written as ``None``; they are stored as
  null in extra trace columns. ``evaluations`` and ``raw_y`` must have values.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from ..format import descriptor_frame, file_entry, new_manifest, run_id
from .base import (
    NUMBER,
    ReaderError,
    ReadResult,
    Renamer,
    check_floats,
    check_integers,
    parse_floats,
    source_path,
)

READER_NAME = "ioh"
LOGGER_NAME = "IOHexperimenter"
JSON_KIND = "ioh-json"
DAT_KIND = "ioh-dat"
#: Written for logged properties that have no value.
NO_VALUE = "None"
#: The Analyzer's default number format, ``%.10f``.
DEFAULT_PRECISION = {"notation": "fixed", "digits": 10}
#: Keys every run object of the JSON has; the others are run attributes.
RUN_KEYS = ("instance", "evals", "best")
#: Descriptor keys this reader derives from a run's ``best`` entry.
BEST_KEYS = ("best_evals", "best_x")
ALGORITHM_KEYS = ("algorithm_name", "algorithm_info")
#: Header of the FlatFile meta data columns (they precede the properties).
FLATFILE_META = (
    "suite_name",
    "problem_name",
    "problem_id",
    "problem_instance",
    "optimization_type",
    "dimension",
    "run",
)


# --------------------------------------------------------------------------- JSON


@dataclass
class IohRun:
    """One entry of ``scenarios[].runs[]``."""

    instance: int | None
    evals: int | None
    best_evals: int | None
    best_y: float | None
    best_x: list[Any] | None
    #: Run attributes in file order, as ``(name, value)``; values are floats.
    attributes: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class IohScenario:
    dimension: int
    path: str
    runs: list[IohRun]


@dataclass
class IohJson:  # pylint: disable=too-many-instance-attributes
    """The parsed metadata file."""

    path: Path
    version: str
    suite: str
    function_id: int
    function_name: str | None
    maximization: bool
    algorithm_name: str
    algorithm_info: str
    experiment_attributes: list[tuple[str, str]]
    run_attribute_names: list[str]
    attributes: list[str] | None
    precision: str | None
    scenarios: list[IohScenario]
    #: What was repaired before parsing (empty when the text was valid as written).
    repairs: list[str]


class _Object(dict):
    """A JSON object that also keeps its ``(key, value)`` pairs, duplicates included."""

    pairs: list[tuple[str, Any]]


def _object_hook(pairs: list[tuple[str, Any]]) -> _Object:
    obj = _Object(pairs)
    obj.pairs = pairs
    return obj


_ALGORITHM_LINE = re.compile(
    r'^(?P<pre>\s*"algorithm":\s*\{"name":\s*")(?P<name>.*?)'
    r'(?P<mid>",\s*"info":\s*")(?P<info>.*)(?P<post>"\},?\s*)$'
)
_STRING_LINE = re.compile(
    r'^(?P<pre>\s*"(?:version|suite|function_name|precision)":\s*")(?P<value>.*)'
    r'(?P<post>",\s*)$'
)
_EXPERIMENT_LINE = re.compile(
    r'^(?P<pre>\s*"experiment_attributes":\s*\[\{)(?P<items>.*)(?P<post>\}\],?\s*)$'
)
_BARE_CONSTANT = re.compile(r"(?<=[:\[,])(\s*)(-?)(nan|inf)(?=\s*[,\]}])")


def _escape(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)[1:-1]


def _repair_line(line: str) -> str:
    """Re-escape the strings of a line the Analyzer writes without escaping."""
    m = _ALGORITHM_LINE.match(line)
    if m:
        return m["pre"] + _escape(m["name"]) + m["mid"] + _escape(m["info"]) + m["post"]
    m = _STRING_LINE.match(line)
    if m:
        return m["pre"] + _escape(m["value"]) + m["post"]
    m = _EXPERIMENT_LINE.match(line)
    if m:
        body = m["items"]
        if not (len(body) >= 2 and body[0] == '"' and body[-1] == '"'):
            return line
        items = []
        for item in body[1:-1].split('"}, {"'):
            key, sep, value = item.partition('": "')
            if not sep:
                return line
            items.append(f'"{_escape(key)}": "{_escape(value)}"')
        return m["pre"] + "}, {".join(items) + m["post"]
    return line


def _constant(m: re.Match[str]) -> str:
    if m[3] == "nan":
        return m[1] + "NaN"
    return m[1] + ("-Infinity" if m[2] else "Infinity")


def _parse_json_text(text: str, path: Path) -> tuple[Any, list[str]]:
    repairs = []
    lines = text.split("\n")
    fixed = [_repair_line(line) for line in lines]
    if fixed != lines:
        repairs.append("re-escaped strings the logger wrote without escaping")
        text = "\n".join(fixed)
    try:
        return json.loads(text, object_pairs_hook=_object_hook), repairs
    except json.JSONDecodeError as first:
        repaired = _BARE_CONSTANT.sub(_constant, text)
        if repaired == text:
            raise ReaderError(
                f"invalid JSON: {first.msg}", path, first.lineno
            ) from first
        try:
            data = json.loads(repaired, object_pairs_hook=_object_hook)
        except json.JSONDecodeError as exc:
            raise ReaderError(f"invalid JSON: {exc.msg}", path, exc.lineno) from exc
        repairs.append("read bare nan and inf as IEEE values")
        return data, repairs


def _get(obj: Any, key: str, kind: type, path: Path, where: str) -> Any:
    if not isinstance(obj, dict) or key not in obj:
        raise ReaderError(f"{where}: missing key {key!r}", path)
    value = obj[key]
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise ReaderError(f"{where}: {key!r} has type {type(value).__name__}", path)
    return value


def _optional(obj: dict, key: str, kind: type, path: Path, where: str) -> Any:
    if key not in obj or obj[key] is None:
        return None
    return _get(obj, key, kind, path, where)


def _number(value: Any, path: Path, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReaderError(f"{where}: {value!r} is not a number", path)
    return float(value)


def _parse_run(obj: Any, path: Path, where: str) -> IohRun:
    """A run object. The first ``instance``, ``evals`` and ``best`` are the run's own
    keys; every other pair is a run attribute, also one that repeats those names."""
    if not isinstance(obj, _Object):
        raise ReaderError(f"{where}: not an object", path)
    fixed: dict[str, Any] = {}
    attributes = []
    for key, value in obj.pairs:
        if key in RUN_KEYS and key not in fixed:
            fixed[key] = value
        else:
            attributes.append(
                (key, _number(value, path, f"{where}, run attribute {key!r}"))
            )
    best = _optional(fixed, "best", dict, path, where) or {}
    best_y = best.get("y")
    best_x = best.get("x")
    if best_x is not None and not isinstance(best_x, list):
        raise ReaderError(f"{where}: best.x is not a list", path)
    return IohRun(
        instance=_optional(fixed, "instance", int, path, where),
        evals=_optional(fixed, "evals", int, path, where),
        best_evals=_optional(best, "evals", int, path, where + ", best"),
        best_y=None if best_y is None else _number(best_y, path, where + ", best.y"),
        best_x=best_x,
        attributes=attributes,
    )


def _experiment_attributes(data: dict, path: Path) -> list[tuple[str, str]]:
    items = _optional(data, "experiment_attributes", list, path, "top level") or []
    out = []
    for item in items:
        if not isinstance(item, _Object):
            raise ReaderError("experiment_attributes: entries must be objects", path)
        for key, value in item.pairs:
            if not isinstance(value, str):
                value = json.dumps(value)
            out.append((key, value))
    return out


def _string_list(data: dict, key: str, path: Path) -> list[str] | None:
    items = _optional(data, key, list, path, "top level")
    if items is not None and not all(isinstance(x, str) for x in items):
        raise ReaderError(f"{key}: entries must be strings", path)
    return items


def load_ioh_json(path: str | Path) -> IohJson:
    """Parse an IOHexperimenter JSON metadata file, repairing the logger's quirks."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ReaderError(f"not UTF-8: {exc}", path) from exc
    data, repairs = _parse_json_text(text, path)
    if not isinstance(data, dict):
        raise ReaderError("the top level is not an object", path)
    algorithm = _get(data, "algorithm", dict, path, "top level")
    scenarios = []
    for si, scen in enumerate(_get(data, "scenarios", list, path, "top level")):
        where = f"scenarios[{si}]"
        runs = _get(scen, "runs", list, path, where)
        scenarios.append(
            IohScenario(
                dimension=_get(scen, "dimension", int, path, where),
                path=_get(scen, "path", str, path, where),
                runs=[
                    _parse_run(r, path, f"{where}.runs[{i}]")
                    for i, r in enumerate(runs)
                ],
            )
        )
    return IohJson(
        path=path,
        version=str(data.get("version", "")),
        suite=_optional(data, "suite", str, path, "top level") or "",
        function_id=_get(data, "function_id", int, path, "top level"),
        function_name=_optional(data, "function_name", str, path, "top level"),
        maximization=_get(data, "maximization", bool, path, "top level"),
        algorithm_name=_get(algorithm, "name", str, path, "algorithm"),
        algorithm_info=_optional(algorithm, "info", str, path, "algorithm") or "",
        experiment_attributes=_experiment_attributes(data, path),
        run_attribute_names=_string_list(data, "run_attributes", path) or [],
        attributes=_string_list(data, "attributes", path),
        precision=_optional(data, "precision", str, path, "top level"),
        scenarios=scenarios,
        repairs=repairs,
    )


def is_ioh_json(path: Path) -> bool:
    """A cheap check: a ``.json`` file that names ``scenarios`` and ``function_id``."""
    if path.suffix.lower() != ".json" or not path.is_file():
        return False
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            head = fh.read(1 << 16)
    except OSError:
        return False
    return '"scenarios"' in head and '"function_id"' in head


def precision_entry(precision: str | None) -> dict[str, Any]:
    """The manifest ``precision`` of a JSON ``precision`` value (absent: ``%.10f``)."""
    if precision is None:
        return dict(DEFAULT_PRECISION)
    m = re.fullmatch(r"%\.?(\d*)([fFeEgG])", precision.strip())
    if not m:
        return {"notation": "unknown"}
    notation = {"f": "fixed", "e": "scientific", "g": "general"}[m[2].lower()]
    return {"notation": notation, "digits": int(m[1]) if m[1] else 6}


# --------------------------------------------------------------------------- .dat


@dataclass
class DatRun:
    """One run of a ``.dat`` file."""

    header_line: int
    rows: int
    sha256: str


@dataclass
class DatFile:
    """A parsed ``.dat`` file. ``trace`` has ``run_index``, ``row``, ``evaluations``,
    ``raw_y`` and the extra columns under their source names."""

    path: str
    runs: list[DatRun]
    trace: pl.DataFrame
    positions: pl.DataFrame | None
    #: Extra columns kept as text, with the first line that is not a number.
    text_columns: dict[str, int]
    header_mismatch: bool = False


@dataclass
class _Layout:
    header: list[str]
    properties: list[str]
    positions: int
    mismatch: bool


def _layout(
    header: list[str],
    attributes: list[str] | None,
    dimension: int,
    path: str,
    line: int,
) -> _Layout:
    if tuple(header[: len(FLATFILE_META)]) == FLATFILE_META:
        raise ReaderError(
            "this is FlatFile output with meta data columns, not Analyzer output",
            path,
            line,
        )
    xs = [f"x{i}" for i in range(dimension)]
    mismatch = False
    if attributes is not None and header[: len(attributes)] == attributes:
        props, rest = header[: len(attributes)], header[len(attributes) :]
        if rest and rest != xs:
            raise ReaderError(
                f"header columns {rest} are neither logged properties nor x0..x{dimension - 1}",
                path,
                line,
            )
    else:
        mismatch = attributes is not None
        if dimension and len(header) >= dimension + 2 and header[-dimension:] == xs:
            props, rest = header[:-dimension], xs
        else:
            props, rest = header, []
    for required in ("evaluations", "raw_y"):
        if required not in props:
            raise ReaderError(f"the header has no {required!r} column", path, line)
    duplicates = sorted({p for p in props if props.count(p) > 1})
    if duplicates:
        raise ReaderError(f"the header names {duplicates} more than once", path, line)
    return _Layout(header, props, len(rest), mismatch)


@dataclass
class _Lines:
    """Data lines of a ``.dat`` file, normalised to single spaces."""

    text: list[str] = field(default_factory=list)
    numbers: list[int] = field(default_factory=list)
    run_index: list[int] = field(default_factory=list)
    row: list[int] = field(default_factory=list)


def _split_lines(
    path: Path, display: str, attributes: list[str] | None, dimension: int
):
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReaderError(f"not UTF-8: {exc}", display) from exc
    layout: _Layout | None = None
    runs: list[DatRun] = []
    hashes: list[Any] = []
    lines = _Lines()
    for number, raw in enumerate(text.split("\n"), 1):
        tokens = raw.split()
        if not tokens:
            continue
        if NUMBER.fullmatch(tokens[0]) is None:
            if layout is None:
                layout = _layout(tokens, attributes, dimension, display, number)
            elif tokens != layout.header:
                raise ReaderError(
                    f"header {' '.join(tokens)!r} differs from the first header "
                    f"{' '.join(layout.header)!r}",
                    display,
                    number,
                )
            runs.append(DatRun(number, 0, ""))
            hashes.append(hashlib.sha256())
            continue
        if layout is None:
            raise ReaderError("data line before the first header line", display, number)
        if len(tokens) != len(layout.header):
            raise ReaderError(
                _width_message(len(tokens), len(layout.header)), display, number
            )
        line = " ".join(tokens)
        hashes[-1].update(line.encode("utf-8") + b"\n")
        lines.text.append(line)
        lines.numbers.append(number)
        lines.run_index.append(len(runs) - 1)
        lines.row.append(runs[-1].rows)
        runs[-1].rows += 1
    for run, digest in zip(runs, hashes):
        run.sha256 = digest.hexdigest()
    return layout, runs, lines


def _width_message(got: int, want: int) -> str:
    message = f"{got} fields, but the header has {want}"
    if got == want - 1:
        message += (
            ". This is the shape of IOHexperimenter's FlatFile separator bug (a run "
            "number glued to the first value), which cannot be undone reliably; log "
            "the data again with a fixed IOHexperimenter"
        )
    return message


def parse_dat(
    path: str | Path,
    *,
    display: str | None = None,
    dimension: int = 0,
    attributes: list[str] | None = None,
    text_columns: Iterable[str] = (),
) -> DatFile:
    """Parse an Analyzer ``.dat`` file.

    ``attributes`` is the JSON's list of logged properties; header columns after
    them named ``x0`` ... ``x(dimension-1)`` are positions. ``display`` names the
    file in errors (default: ``path``). Extra columns are float64, with ``None``
    as null; a column with other tokens is kept as text, as are the columns named
    in ``text_columns``.
    """
    path = Path(path)
    display = display or str(path)
    layout, runs, lines = _split_lines(path, display, attributes, dimension)
    if layout is None:
        return DatFile(display, [], _empty_trace(), None, {})
    names = layout.properties + [f"x{i}" for i in range(layout.positions)]
    width = len(names)
    if lines.text:
        split = (
            pl.Series("l", lines.text)
            .str.split_exact(" ", width - 1)
            .struct.rename_fields([f"c{i}" for i in range(width)])
            .struct.unnest()
        )
    else:
        split = pl.DataFrame(
            {f"c{i}": [] for i in range(width)},
            schema={f"c{i}": pl.String for i in range(width)},
        )
    keys = {
        "run_index": pl.Series("run_index", lines.run_index, dtype=pl.Int64),
        "row": pl.Series("row", lines.row, dtype=pl.Int64),
    }
    columns, text = _property_columns(
        split, layout.properties, lines.numbers, display, set(text_columns)
    )
    trace = pl.DataFrame({**keys, **columns})
    positions = None
    if layout.positions:
        offset = len(layout.properties)
        positions = pl.DataFrame(
            {
                **keys,
                **{
                    f"x{i}": check_floats(
                        split[f"c{offset + i}"], lines.numbers, display, f"x{i}"
                    )
                    for i in range(layout.positions)
                },
            }
        )
    return DatFile(display, runs, trace, positions, text, layout.mismatch)


def _property_columns(
    split: pl.DataFrame,
    properties: list[str],
    numbers: Sequence[int],
    display: str,
    force_text: set[str],
) -> tuple[dict[str, pl.Series], dict[str, int]]:
    """``evaluations``, ``raw_y`` and the extra columns, in that order."""
    index = {name: i for i, name in enumerate(properties)}
    columns = {
        "evaluations": check_integers(
            split[f"c{index['evaluations']}"], numbers, display, "evaluations"
        ),
        "raw_y": check_floats(split[f"c{index['raw_y']}"], numbers, display, "raw_y"),
    }
    text: dict[str, int] = {}
    for i, name in enumerate(properties):
        if name not in columns:
            columns[name] = _extra_column(
                split[f"c{i}"], name, numbers, force_text, text
            )
    return columns, text


def _extra_column(
    tokens: pl.Series,
    name: str,
    numbers: Sequence[int],
    force_text: set[str],
    text: dict[str, int],
) -> pl.Series:
    """float64 with ``None`` as null, or the tokens as text when some token is not
    a number (its line goes into ``text``) or the column is in ``force_text``."""
    values = None if name in force_text else parse_floats(tokens, name, NO_VALUE)
    if values is not None:
        return values
    numeric = tokens.str.contains(f"^(?:{NUMBER.pattern})$") | (tokens == NO_VALUE)
    first = (~numeric).arg_true()
    if not first.is_empty():
        text[name] = numbers[int(first[0])]
    return (
        pl.select(pl.when(tokens == NO_VALUE).then(None).otherwise(tokens))
        .to_series()
        .alias(name)
    )


def _empty_trace() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "run_index": pl.Int64,
            "row": pl.Int64,
            "evaluations": pl.Int64,
            "raw_y": pl.Float64,
        }
    )


def infer_triggers(
    trace: pl.DataFrame, maximization: bool
) -> list[dict[str, Any]] | None:
    """``all`` or ``improvement`` when every run fits it, else None (unknown).

    ``all``: each run logs evaluations 1, 2, 3, ...; the final row may repeat the
    last evaluation. ``improvement``: no row is worse than the best value before
    it (ties are allowed, because logged values are rounded); the last row of a
    run is left out, because the logger always writes the final evaluation. The
    result is a guess from the data, so it is marked ``inferred``; data with no
    run of three or more rows gives None.
    """
    if trace.is_empty():
        return None
    best = pl.col("raw_y").cum_max() if maximization else pl.col("raw_y").cum_min()
    df = trace.select(
        "row",
        "evaluations",
        "raw_y",
        prev_eval=pl.col("evaluations").shift(1).over("run_index"),
        best=best.shift(1).over("run_index"),
        last=pl.col("row") == pl.col("row").max().over("run_index"),
    )
    if df.filter(pl.col("row") > 0, ~pl.col("last")).is_empty():
        return None
    step = pl.col("evaluations") - pl.col("prev_eval")
    consecutive = (
        pl.when(pl.col("row") == 0)
        .then(pl.col("evaluations") == 1)
        .when(pl.col("last"))
        .then(step.is_in([0, 1]))
        .otherwise(step == 1)
    )
    if df.select(consecutive.all()).item():
        return [{"kind": "all", "params": {"inferred": True}}]
    better = (
        pl.col("raw_y") >= pl.col("best")
        if maximization
        else pl.col("raw_y") <= pl.col("best")
    )
    improving = (
        pl.when((pl.col("row") == 0) | pl.col("last")).then(True).otherwise(better)
    )
    if df.select(improving.fill_null(False).all()).item():
        return [{"kind": "improvement", "params": {"inferred": True}}]
    return None


# --------------------------------------------------------------------------- reader


def detect_ioh(path: str | Path) -> bool:
    """True for an IOHexperimenter JSON file, or a folder that holds one."""
    path = Path(path)
    if path.is_dir():
        return any(is_ioh_json(p) for p in sorted(path.rglob("*.json")))
    return is_ioh_json(path)


@dataclass
class _Builder:  # pylint: disable=too-many-instance-attributes
    """Collects the tables while the files are read."""

    dataset_id: str
    root: Path
    runs: list[dict[str, Any]] = field(default_factory=list)
    descriptors: list[tuple[int, str, Any, str]] = field(default_factory=list)
    traces: list[pl.DataFrame] = field(default_factory=list)
    positions: list[pl.DataFrame] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)
    renamer: Renamer = field(default_factory=Renamer)
    limited_precision: int = 0

    def warn(self, code: str, message: str, source: str | None = None) -> None:
        entry = {"code": code, "message": message}
        if source is not None:
            entry["source"] = source
        self.warnings.append(entry)


def _ioh_jsons(source: Path) -> list[Path]:
    if source.is_dir():
        return [p for p in sorted(source.rglob("*.json")) if p.is_file()]
    return [source]


def read_ioh(
    source: str | Path,
    dataset_id: str,
    *,
    root: str | Path | None = None,
) -> ReadResult:
    """Read IOHexperimenter JSON + ``.dat`` output into the tables of format v1.

    ``source`` is a JSON file or a folder, searched recursively for JSON files;
    JSON files that are not IOHexperimenter metadata are skipped with a
    ``skipped_file`` warning. Source paths are relative to ``root``, by default
    the folder itself or the JSON file's folder.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"{source} not found")
    root_path = (
        Path(root)
        if root is not None
        else (source if source.is_dir() else source.parent)
    )
    builder = _Builder(dataset_id, root_path)
    metas: list[IohJson] = []
    for path in _ioh_jsons(source):
        if not is_ioh_json(path):
            if source.is_dir():
                builder.warn(
                    "skipped_file",
                    "not IOHexperimenter metadata",
                    source_path(path, root_path),
                )
                continue
            raise ReaderError("not an IOHexperimenter JSON metadata file", path)
        metas.append(load_ioh_json(path))
    if not metas:
        raise ReaderError("no IOHexperimenter JSON metadata found", source)
    dats = _read_dats(metas, builder)
    for meta in metas:
        _add_json(meta, dats, builder)
    if source.is_dir():
        _unreferenced(source, dats, builder)
    if builder.limited_precision:
        builder.warn(
            "limited_precision",
            f"{builder.limited_precision} data file(s) were written with %.10f (positions "
            "with %f): values below 5e-11 are stored as 0, and nothing was corrected",
        )
    return _result(builder)


def _read_dats(metas: list[IohJson], builder: _Builder) -> dict[str, DatFile | None]:
    """Parse every ``.dat`` file once; a column kept as text anywhere is text everywhere."""
    targets: dict[str, tuple[Path, IohJson, IohScenario]] = {}
    for meta in metas:
        for scen in meta.scenarios:
            local = meta.path.parent / scen.path
            display = source_path(local, builder.root)
            if display in targets:
                raise ReaderError(f"{display} is named twice", meta.path)
            targets[display] = (local, meta, scen)
    dats: dict[str, DatFile | None] = {}
    for display, (local, meta, scen) in targets.items():
        dats[display] = (
            parse_dat(
                local,
                display=display,
                dimension=scen.dimension,
                attributes=meta.attributes,
            )
            if local.is_file()
            else None
        )
    text = {name for d in dats.values() if d is not None for name in d.text_columns}
    for display, dat in dats.items():
        if dat is None:
            continue
        floats = [
            n for n in text if n in dat.trace.columns and n not in dat.text_columns
        ]
        if floats:
            local, meta, scen = targets[display]
            dats[display] = parse_dat(
                local,
                display=display,
                dimension=scen.dimension,
                attributes=meta.attributes,
                text_columns=text,
            )
    return dats


def _add_json(
    meta: IohJson, dats: dict[str, DatFile | None], builder: _Builder
) -> None:
    json_path = source_path(meta.path, builder.root)
    logger = {"name": LOGGER_NAME, "version": meta.version}
    builder.sources.append(
        file_entry(meta.path, json_path, kind=JSON_KIND, logger=logger)
    )
    if meta.repairs:
        builder.warn("json_repaired", "; ".join(meta.repairs), json_path)
    precision = precision_entry(meta.precision)
    keys = _key_names(meta, json_path, builder.renamer)
    for scen in meta.scenarios:
        display = source_path(meta.path.parent / scen.path, builder.root)
        dat = dats[display]
        entry: dict[str, Any] = {
            "kind": DAT_KIND,
            "logger": logger,
            "precision": precision,
        }
        if dat is None:
            builder.sources.append(
                {"path": display, "sha256": None, "bytes": None, **entry}
            )
            builder.warn(
                "missing_data_file", f"named in {json_path} but not found", display
            )
            dat_runs = 0
        else:
            entry["data_rows"] = dat.trace.height
            triggers = infer_triggers(dat.trace, meta.maximization)
            if triggers is not None:
                entry["triggers"] = triggers
            builder.sources.append(
                file_entry(meta.path.parent / scen.path, display, **entry)
            )
            if precision == DEFAULT_PRECISION:
                builder.limited_precision += 1
            if dat.header_mismatch:
                builder.warn(
                    "header_mismatch",
                    f"the header differs from the attributes in {json_path}",
                    display,
                )
            for name, line in sorted(dat.text_columns.items()):
                builder.warn(
                    "text_column",
                    f"column {name!r} is kept as text: line {line} is not a number",
                    display,
                )
            dat_runs = len(dat.runs)
        if dat_runs != len(scen.runs) and dat is not None:
            builder.warn(
                "run_count_mismatch",
                f"{dat_runs} run(s) in the data file, {len(scen.runs)} in {json_path}; "
                "runs are matched in order",
                display,
            )
        _add_runs(meta, scen, dat, display, keys, builder)


def _key_names(
    meta: IohJson, json_path: str, renamer: Renamer
) -> dict[tuple[str, str], str]:
    """Descriptor keys per ``(origin, source name)``, renamed where they clash."""
    taken: set[str] = set(ALGORITHM_KEYS) | set(BEST_KEYS)
    names: dict[tuple[str, str], str] = {}
    for origin, keys in (
        ("experiment", [k for k, _ in meta.experiment_attributes]),
        (
            "run",
            meta.run_attribute_names
            + [k for r in _all_runs(meta) for k, _ in r.attributes],
        ),
    ):
        for key in keys:
            if (origin, key) in names:
                continue
            new = renamer.descriptor_key(key, taken, origin, json_path)
            names[(origin, key)] = new
            taken.add(new)
    return names


def _all_runs(meta: IohJson) -> Iterable[IohRun]:
    for scen in meta.scenarios:
        yield from scen.runs


def _add_runs(  # pylint: disable=too-many-arguments
    meta: IohJson,
    scen: IohScenario,
    dat: DatFile | None,
    display: str,
    keys: dict[tuple[str, str], str],
    builder: _Builder,
) -> None:
    n_dat = 0 if dat is None else len(dat.runs)
    ids: dict[int, int] = {}
    for index in range(max(n_dat, len(scen.runs))):
        rid = run_id(builder.dataset_id, display, index)
        ids[index] = rid
        info = scen.runs[index] if index < len(scen.runs) else None
        builder.runs.append(
            {
                "run_id": rid,
                "suite": meta.suite,
                "function_id": meta.function_id,
                "function_name": meta.function_name,
                "instance": None if info is None else info.instance,
                "dimension": scen.dimension,
                "maximization": meta.maximization,
                "evals_used": None if info is None else info.evals,
                "best_y": None if info is None else info.best_y,
                "n_objectives": 1,
                "source_file": display,
                "source_run_index": index,
                "source_rows_sha256": dat.runs[index].sha256
                if index < n_dat and dat
                else None,
            }
        )
        builder.descriptors.extend(_run_descriptors(rid, meta, info, keys))
    if dat is not None and dat.trace.height:
        builder.traces.append(_with_run_ids(dat.trace, ids, display, builder.renamer))
        if dat.positions is not None:
            builder.positions.append(_with_run_ids(dat.positions, ids, display, None))


def _run_descriptors(
    rid: int, meta: IohJson, info: IohRun | None, keys: dict[tuple[str, str], str]
) -> list[tuple[int, str, Any, str]]:
    out: list[tuple[int, str, Any, str]] = [
        (rid, "algorithm_name", meta.algorithm_name, "algorithm"),
        (rid, "algorithm_info", meta.algorithm_info, "algorithm"),
    ]
    out += [
        (rid, keys[("experiment", k)], v, "experiment")
        for k, v in meta.experiment_attributes
    ]
    if info is not None:
        out.append((rid, "best_evals", info.best_evals, "run"))
        if info.best_x is not None:
            out.append((rid, "best_x", json.dumps(info.best_x), "run"))
        seen: set[str] = set()
        for k, v in info.attributes:
            key = keys[("run", k)]
            if key not in seen:
                seen.add(key)
                out.append((rid, key, v, "run"))
    return out


def _with_run_ids(
    frame: pl.DataFrame, ids: dict[int, int], display: str, renamer: Renamer | None
) -> pl.DataFrame:
    frame = frame.with_columns(
        run_id=pl.col("run_index").replace_strict(ids, return_dtype=pl.Int64)
    ).drop("run_index")
    if renamer is None:
        return frame.select("run_id", pl.exclude("run_id"))
    fixed = ["run_id", "row", "evaluations", "raw_y"]
    renames = {}
    taken = set(fixed)
    for name in frame.columns:
        if name in fixed:
            continue
        new = renamer.trace_column(name, taken, display)
        taken.add(new)
        if new != name:
            renames[name] = new
    return frame.rename(renames).select(
        fixed + [c for c in frame.rename(renames).columns if c not in fixed]
    )


def _unreferenced(
    source: Path, dats: dict[str, DatFile | None], builder: _Builder
) -> None:
    for path in sorted(source.rglob("*.dat")):
        display = source_path(path, builder.root)
        if display not in dats:
            builder.warn(
                "unreferenced_data_file",
                "no JSON metadata names this file; not read",
                display,
            )


def _result(builder: _Builder) -> ReadResult:
    manifest = new_manifest(
        builder.dataset_id,
        READER_NAME,
        sources=builder.sources,
        warnings=builder.warnings,
        renamed_columns=builder.renamer.records,
    )
    runs = (
        pl.DataFrame(builder.runs, schema=_RUNS_INPUT, orient="row")
        if builder.runs
        else pl.DataFrame(schema=_RUNS_INPUT)
    )
    trace = (
        pl.concat(builder.traces, how="diagonal")
        if builder.traces
        else _empty_trace()
        .drop("run_index")
        .insert_column(0, pl.Series("run_id", [], pl.Int64))
    )
    positions = (
        pl.concat(builder.positions, how="diagonal") if builder.positions else None
    )
    return ReadResult(
        manifest=manifest,
        runs=runs,
        trace=trace,
        descriptors=descriptor_frame(builder.descriptors),
        positions=positions,
    )


_RUNS_INPUT = {
    "run_id": pl.Int64,
    "suite": pl.String,
    "function_id": pl.Int32,
    "function_name": pl.String,
    "instance": pl.Int32,
    "dimension": pl.Int32,
    "maximization": pl.Boolean,
    "evals_used": pl.Int64,
    "best_y": pl.Float64,
    "n_objectives": pl.Int32,
    "source_file": pl.String,
    "source_run_index": pl.Int64,
    "source_rows_sha256": pl.String,
}
