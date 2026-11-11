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

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from ..format import file_entry, run_id
from .base import (
    NUMBER,
    DataLines,
    ReaderError,
    ReadResult,
    Tables,
    check_floats,
    check_integers,
    first_false,
    input_root,
    is_number,
    read_text,
    source_path,
    split_lines,
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
    data, repairs = _parse_json_text(read_text(path, str(path)), path)
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
class DatFile:
    """A parsed ``.dat`` file. ``trace`` has ``run_index``, ``row``, ``evaluations``,
    ``raw_y`` and the extra columns under their source names, as text with
    ``None`` as null (``typed`` makes the numeric ones float64)."""

    path: str
    lines: DataLines
    trace: pl.DataFrame
    positions: pl.DataFrame | None
    #: Extra columns with a token that is not a number, with its line.
    text_columns: dict[str, int]
    header_mismatch: bool = False

    def typed(self) -> pl.DataFrame:
        """``trace`` with every extra column that holds only numbers as float64."""
        extra = self.trace.columns[4:]
        return self.trace.with_columns(
            to_float(c) for c in extra if c not in self.text_columns
        )


def to_float(column: str) -> pl.Expr:
    """A text column of number tokens (and nulls) as float64."""
    return pl.col(column).str.to_lowercase().cast(pl.Float64)


def _layout(
    header: list[str], attributes: list[str] | None, dimension: int
) -> tuple[list[str], int]:
    """The logged properties and the number of position columns of a header.
    Raises ``ValueError`` for a bad header."""
    if tuple(header[: len(FLATFILE_META)]) == FLATFILE_META:
        raise ValueError(
            "this is FlatFile output with meta data columns, not Analyzer output"
        )
    props, rest = _positions(header, attributes, dimension)
    missing = [c for c in ("evaluations", "raw_y") if c not in props]
    duplicates = sorted({p for p in props if props.count(p) > 1})
    if missing:
        raise ValueError(f"the header has no {missing[0]!r} column")
    if duplicates:
        raise ValueError(f"the header names {duplicates} more than once")
    return props, len(rest)


def _positions(
    header: list[str], attributes: list[str] | None, dimension: int
) -> tuple[list[str], list[str]]:
    """The header split into logged properties and position columns."""
    xs = [f"x{i}" for i in range(dimension)]
    n = 0 if attributes is None else len(attributes)
    if attributes is not None and header[:n] == attributes:
        rest = header[n:]
        if rest and rest != xs:
            raise ValueError(
                f"header columns {rest} are neither logged properties nor "
                f"x0..x{dimension - 1}"
            )
        return header[: len(attributes)], rest
    if dimension and len(header) >= dimension + 2 and header[-dimension:] == xs:
        return header[:-dimension], xs
    return header, []


def _check_lines(lines: DataLines, header: list[str], display: str) -> None:
    """Every header repeats the first, and every data line has its width."""
    for number, text in lines.headers[1:]:
        if text.split() != header:
            raise ReaderError(
                f"header {' '.join(text.split())!r} differs from the first header "
                f"{' '.join(header)!r}",
                display,
                number,
            )
    width = len(header)
    bad = lines.frame.filter(pl.col("width") != width)
    if bad.is_empty():
        return
    got = bad["width"][0]
    message = f"{got} fields, but the header has {width}"
    if got == width - 1:
        message += (
            ". This is the shape of IOHexperimenter's FlatFile separator bug (a run "
            "number glued to the first value), which cannot be undone reliably; log "
            "the data again with a fixed IOHexperimenter"
        )
    raise ReaderError(message, display, bad["line"][0])


def _is_header(_line: str, tokens: list[str]) -> bool:
    return NUMBER.fullmatch(tokens[0]) is None


def parse_dat(
    path: str | Path,
    *,
    display: str | None = None,
    dimension: int = 0,
    attributes: list[str] | None = None,
) -> DatFile:
    """Parse an Analyzer ``.dat`` file.

    ``attributes`` is the JSON's list of logged properties; header columns after
    them named ``x0`` ... ``x(dimension-1)`` are positions. ``display`` names the
    file in errors (default: ``path``). Every run must repeat the first header.
    """
    display = display or str(path)
    lines = split_lines(read_text(Path(path), display), display, _is_header)
    if not lines.headers:
        return DatFile(display, lines, _EMPTY, None, {})
    number, text = lines.headers[0]
    header = text.split()
    try:
        props, n_x = _layout(header, attributes, dimension)
    except ValueError as exc:
        raise ReaderError(str(exc), display, number) from exc
    _check_lines(lines, header, display)
    dat = _typed(lines, props, n_x, display)
    dat.header_mismatch = (
        attributes is not None and header[: len(attributes)] != attributes
    )
    return dat


def _typed(lines: DataLines, props: list[str], n_x: int, display: str) -> DatFile:
    """The trace, text columns and positions of checked data lines."""
    columns = lines.tokens(len(props) + n_x)
    tokens = dict(zip(props, columns))
    numbers = lines.frame["line"].to_list()
    keys = lines.frame.select("run_index", "row")
    extra = [c for c in props if c not in ("evaluations", "raw_y")]
    trace = keys.with_columns(
        check_integers(tokens["evaluations"], numbers, display, "evaluations"),
        check_floats(tokens["raw_y"], numbers, display, "raw_y"),
        *(tokens[c].replace(NO_VALUE, None).alias(c) for c in extra),
    )
    text_columns = {}
    for name in extra:
        i = first_false(is_number(tokens[name]) | (tokens[name] == NO_VALUE))
        if i is not None:
            text_columns[name] = numbers[i]
    positions = None
    if n_x:
        n_props = len(props)
        positions = keys.with_columns(
            check_floats(x, numbers, display, f"x{i}")
            for i, x in enumerate(columns[n_props:])
        )
    return DatFile(display, lines, trace, positions, text_columns)


_EMPTY = pl.DataFrame(
    schema={"run_index": pl.Int64, "row": pl.Int64}
    | {"evaluations": pl.Int64, "raw_y": pl.Float64}
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
class _Reader:
    """Reads the JSON files and their ``.dat`` files into ``tables``."""

    tables: Tables
    root: Path
    #: Extra trace columns and whether every file holds only numbers in them.
    numeric: dict[str, bool] = field(default_factory=dict)
    limited_precision: int = 0


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
    the folder itself or the JSON file's folder. A column kept as text in one
    ``.dat`` file is text in all.
    """
    source, root_path = input_root(source, root)
    reader = _Reader(Tables(dataset_id), root_path)
    metas = _metas(source, reader)
    named = _named_dats(metas, root_path)
    for meta in metas:
        _add_json(meta, reader)
    for path in sorted(source.rglob("*.dat")) if source.is_dir() else []:
        display = source_path(path, root_path)
        if display not in named:
            reader.tables.warn(
                "unreferenced_data_file",
                "no JSON metadata names this file; not read",
                display,
            )
    if reader.limited_precision:
        reader.tables.warn(
            "limited_precision",
            f"{reader.limited_precision} data file(s) were written with %.10f "
            "(positions with %f): values below 5e-11 are stored as 0, and nothing "
            "was corrected",
        )
    result = reader.tables.result(READER_NAME)
    numeric = [c for c, ok in reader.numeric.items() if ok]
    result.lazy_trace = result.lazy_trace.with_columns(map(to_float, numeric))
    return result


def _metas(source: Path, reader: _Reader) -> list[IohJson]:
    if not source.is_dir() and not is_ioh_json(source):
        raise ReaderError("not an IOHexperimenter JSON metadata file", source)
    paths = sorted(source.rglob("*.json")) if source.is_dir() else [source]
    metas = []
    for path in (p for p in paths if p.is_file()):
        if is_ioh_json(path):
            metas.append(load_ioh_json(path))
        else:
            display = source_path(path, reader.root)
            reader.tables.warn("skipped_file", "not IOHexperimenter metadata", display)
    if not metas:
        raise ReaderError("no IOHexperimenter JSON metadata found", source)
    return metas


def _named_dats(metas: list[IohJson], root: Path) -> set[str]:
    named: set[str] = set()
    for meta in metas:
        for scen in meta.scenarios:
            display = source_path(meta.path.parent / scen.path, root)
            if display in named:
                raise ReaderError(f"{display} is named twice", meta.path)
            named.add(display)
    return named


def _add_json(meta: IohJson, reader: _Reader) -> None:
    tables = reader.tables
    json_path = source_path(meta.path, reader.root)
    logger = {"name": LOGGER_NAME, "version": meta.version}
    tables.sources.append(
        file_entry(meta.path, json_path, kind=JSON_KIND, logger=logger)
    )
    if meta.repairs:
        tables.warn("json_repaired", "; ".join(meta.repairs), json_path)
    keys = _key_names(meta, json_path, tables)
    for scen in meta.scenarios:
        local = meta.path.parent / scen.path
        display = source_path(local, reader.root)
        dat = None
        if local.is_file():
            dat = parse_dat(
                local,
                display=display,
                dimension=scen.dimension,
                attributes=meta.attributes,
            )
            _dat_source(dat, local, meta, json_path, reader)
            if len(dat.lines.rows) != len(scen.runs):
                tables.warn(
                    "run_count_mismatch",
                    f"{len(dat.lines.rows)} run(s) in the data file, "
                    f"{len(scen.runs)} in {json_path}; runs are matched in order",
                    display,
                )
        else:
            tables.sources.append(
                {"path": display, "sha256": None, "bytes": None, **_dat_entry(meta)}
            )
            tables.warn(
                "missing_data_file", f"named in {json_path} but not found", display
            )
        ids = _add_runs(meta, scen, dat, display, keys, tables)
        if dat is not None and dat.trace.height:
            _add_frames(dat, ids, reader)


def _dat_entry(meta: IohJson) -> dict[str, Any]:
    """The source entry fields every ``.dat`` file of ``meta`` has."""
    return {
        "kind": DAT_KIND,
        "logger": {"name": LOGGER_NAME, "version": meta.version},
        "precision": precision_entry(meta.precision),
    }


def _dat_source(
    dat: DatFile, local: Path, meta: IohJson, json_path: str, reader: _Reader
) -> None:
    """The source entry and warnings of a parsed ``.dat`` file."""
    tables, display = reader.tables, dat.path
    fields = dict(_dat_entry(meta), data_rows=dat.trace.height)
    triggers = infer_triggers(dat.trace, meta.maximization)
    if triggers is not None:
        fields["triggers"] = triggers
    tables.sources.append(file_entry(local, display, **fields))
    if fields["precision"] == DEFAULT_PRECISION:
        reader.limited_precision += 1
    if dat.header_mismatch:
        tables.warn(
            "header_mismatch",
            f"the header differs from the attributes in {json_path}",
            display,
        )
    for name, line in sorted(dat.text_columns.items()):
        tables.warn(
            "text_column",
            f"column {name!r} is kept as text: line {line} is not a number",
            display,
        )


def _key_names(
    meta: IohJson, json_path: str, tables: Tables
) -> dict[tuple[str, str], str]:
    """Descriptor keys per ``(origin, source name)``, renamed where they clash."""
    taken: set[str] = set(ALGORITHM_KEYS) | set(BEST_KEYS)
    names: dict[tuple[str, str], str] = {}
    runs = [r for scen in meta.scenarios for r in scen.runs]
    for origin, keys in (
        ("experiment", [k for k, _ in meta.experiment_attributes]),
        ("run", meta.run_attribute_names + [k for r in runs for k, _ in r.attributes]),
    ):
        for key in keys:
            if (origin, key) not in names:
                new = tables.renamer.descriptor_key(key, taken, origin, json_path)
                names[(origin, key)] = new
                taken.add(new)
    return names


def _add_runs(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    meta: IohJson,
    scen: IohScenario,
    dat: DatFile | None,
    display: str,
    keys: dict[tuple[str, str], str],
    tables: Tables,
) -> dict[int, int]:
    """Add the runs of one scenario; returns the run id of each run index."""
    hashes = [] if dat is None else dat.lines.sha256
    ids: dict[int, int] = {}
    for index in range(max(len(hashes), len(scen.runs))):
        rid = ids[index] = run_id(tables.dataset_id, display, index)
        info = scen.runs[index] if index < len(scen.runs) else None
        tables.runs.append(
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
                "source_rows_sha256": hashes[index] if index < len(hashes) else None,
            }
        )
        tables.descriptors.extend(_run_descriptors(rid, meta, info, keys))
    return ids


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


def _add_frames(dat: DatFile, ids: dict[int, int], reader: _Reader) -> None:
    """Add the trace (extra columns renamed where reserved) and positions."""
    run = pl.col("run_index").replace_strict(ids, return_dtype=pl.Int64)
    trace = dat.trace.select(run.alias("run_id"), pl.exclude("run_index"))
    taken = set(trace.columns[:4])
    renames = {}
    for name in trace.columns[4:]:
        new = reader.tables.renamer.trace_column(name, taken, dat.path)
        taken.add(new)
        renames[name] = new
        numeric = name not in dat.text_columns
        reader.numeric[new] = reader.numeric.get(new, True) and numeric
    reader.tables.add("trace", trace.rename(renames))
    if dat.positions is not None:
        reader.tables.add(
            "positions",
            dat.positions.select(run.alias("run_id"), pl.exclude("run_index")),
        )
