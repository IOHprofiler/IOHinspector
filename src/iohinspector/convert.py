"""Convert benchmark logs to dataset format v1: ``iohinspector convert``.

``convert`` reads the input with a reader of ``iohinspector.io.READERS`` (detected,
or named by ``format``; custom CSV needs a column mapping) and writes the dataset.
The input is a file, a folder, or an archive (zip, tar and compressed tar), and
archives inside folders and archives are opened too (``iohinspector.io.archive``).
A file inside an archive gets the source path ``<archive path>/<member path>``
(``docs/format.md``), and the manifest lists the archives.

Descriptors (D17) can be added to every run: ``tags`` (``key=value``, origin
``user``) and ``tags_from_path``, a pattern matched against each run's source
file (origin ``path``). ``docs/convert-verify.md`` describes both.
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from .format import (
    check_dataset_id,
    read_manifest,
    update_descriptors,
    write_manifest,
)
from .format.schema import ALGORITHM_DESCRIPTORS, descriptor_key_problem
from .io import READERS, detect_format
from .io._csv import parse_token
from .io.archive import Staged, staged

_TYPES = ("str", "int", "float", "bool")


class ConvertError(ValueError):
    """The input cannot be converted: unknown format, bad option or bad tag."""


# --------------------------------------------------------------------------- tags


def _typed(text: str, kind: str, where: str) -> Any:
    try:
        return parse_token(text, kind)
    except ValueError as exc:
        raise ConvertError(f"{where}: {exc}") from None


def _key_and_type(spec: str, where: str) -> tuple[str, str]:
    key, _, kind = spec.partition(":")
    kind = kind or "str"
    if not key:
        raise ConvertError(f"{where}: the descriptor key is empty")
    if kind not in _TYPES:
        raise ConvertError(f"{where}: unknown type {kind!r}; use one of {_TYPES}")
    problem = descriptor_key_problem(key)
    if problem is not None:
        raise ConvertError(f"{where}: {problem}")
    return key, kind


def parse_tag(tag: str) -> tuple[str, Any]:
    """``key=value`` (a string) or ``key:type=value`` with type int, float or bool."""
    spec, sep, text = tag.partition("=")
    if not sep:
        raise ConvertError(f"tag {tag!r}: expected key=value")
    key, kind = _key_and_type(spec, f"tag {tag!r}")
    return key, _typed(text, kind, f"tag {tag!r}")


@dataclass(frozen=True)
class PathPattern:
    """Descriptors taken from source paths, such as ``{algorithm_name}/{budget:int}_*``.

    ``{key}`` captures text within one path part, ``{key:int}`` (or ``float``,
    ``bool``) also gives it a type, and ``*`` matches any text within one part.
    The pattern matches a whole source path or its leading parts: ``{alg}``
    matches ``cma/data_f1/IOHprofiler_f1_DIM5.dat`` with ``alg = "cma"``.
    """

    pattern: str
    regex: re.Pattern[str] = field(init=False, compare=False)
    fields: tuple[tuple[str, str], ...] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        regex, fields = _compile_pattern(self.pattern)
        object.__setattr__(self, "regex", regex)
        object.__setattr__(self, "fields", tuple(fields))

    def values(self, path: str) -> dict[str, Any] | None:
        """The typed descriptors of ``path``, or None when it does not match."""
        m = self.regex.fullmatch(path)
        if m is None:
            return None
        where = f"--tags-from-path {self.pattern!r} on {path!r}"
        return {
            key: _typed(m[f"g{i}"], kind, where)
            for i, (key, kind) in enumerate(self.fields)
        }


def _compile_pattern(pattern: str) -> tuple[re.Pattern[str], list[tuple[str, str]]]:
    where = f"--tags-from-path {pattern!r}"
    parts: list[str] = []
    fields: list[tuple[str, str]] = []
    for token in re.split(r"(\{[^{}]*\}|\*)", pattern.strip("/")):
        if token == "*":  # nosec B105  # a glob token, not a credential
            parts.append("[^/]*")
        elif token.startswith("{") and token.endswith("}"):
            key, kind = _key_and_type(token[1:-1], where)
            if key in (k for k, _ in fields):
                raise ConvertError(f"{where}: {key!r} appears twice")
            parts.append(f"(?P<g{len(fields)}>[^/]+?)")
            fields.append((key, kind))
        elif "{" in token or "}" in token:
            raise ConvertError(f"{where}: unbalanced brace")
        else:
            parts.append(re.escape(token))
    if not fields:
        raise ConvertError(f"{where}: the pattern names no {{key}}")
    return re.compile("".join(parts) + "(?:/.*)?", re.DOTALL), fields


def _tag_records(
    runs: pl.DataFrame,
    tags: Sequence[tuple[str, Any]],
    pattern: PathPattern | None,
) -> list[tuple[int, str, Any, str]]:
    """``(run_id, key, value, origin)`` for every run and tag."""
    records: list[tuple[int, str, Any, str]] = []
    for rid, source in runs.select("run_id", "source_file").iter_rows():
        found: dict[str, tuple[Any, str]] = {}
        if pattern is not None:
            values = pattern.values(source)
            if values is None:
                raise ConvertError(
                    f"--tags-from-path {pattern.pattern!r} does not match the "
                    f"source file {source!r}"
                )
            found.update({k: (v, "path") for k, v in values.items()})
        found.update({k: (v, "user") for k, v in tags})
        for key, (value, origin) in found.items():
            if key in ALGORITHM_DESCRIPTORS:
                origin = "user"  # the format allows only algorithm or user here
            records.append((rid, key, value, origin))
    return records


# --------------------------------------------------------------------------- convert


def default_dataset_id(source: str | os.PathLike[str]) -> str:
    """A dataset id from the input's name: its stem with other characters as ``_``."""
    stem = Path(source).resolve().name.split(".")[0] or "dataset"
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", stem)[:128]
    return check_dataset_id(cleaned if cleaned[0].isalnum() else "d" + cleaned[:127])


def _choose_format(path: Path, format_name: str | None, mapping: Any) -> str:
    if format_name is None:
        format_name = "custom-csv" if mapping is not None else detect_format(path)
    if format_name not in READERS:
        known = ", ".join(READERS)
        raise ConvertError(f"unknown format {format_name!r}; known: {known}")
    if format_name == "custom-csv" and mapping is None:
        raise ConvertError("custom-csv needs a column mapping (--mapping)")
    if mapping is not None and format_name != "custom-csv":
        raise ConvertError(f"a mapping is only used by custom-csv, not {format_name}")
    return format_name


def _finish(
    output: Path,
    stage: Staged,
    options: dict[str, Any],
    tags: tuple[Sequence[tuple[str, Any]], PathPattern | None],
) -> dict[str, Any]:
    """Add the tags, the archives and the convert options to the written dataset."""
    runs = pl.read_parquet(output / "runs.parquet", columns=["run_id", "source_file"])
    records = _tag_records(runs, *tags)
    if records:
        update_descriptors(output, records, origin=None)
    manifest = read_manifest(output, warn=False)
    manifest.pop("updated", None)  # tags are part of the conversion
    if stage.archives:
        manifest["archives"] = stage.archives
        paths = sorted((a["path"] for a in stage.archives), key=len, reverse=True)
        for entry in manifest["sources"]:  # the innermost archive that holds it
            entry_archive = next(
                (a for a in paths if entry["path"].startswith(a + "/")), None
            )
            if entry_archive is not None:
                entry["archive"] = entry_archive
    if stage.warnings:
        manifest["warnings"] = manifest.get("warnings", []) + stage.warnings
    manifest.setdefault("extra", {})["convert"] = options
    write_manifest(output, manifest)
    return manifest


def convert(  # pylint: disable=too-many-arguments
    source: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    format: str | None = None,  # pylint: disable=redefined-builtin
    mapping: Mapping[str, Any] | str | os.PathLike[str] | None = None,
    dataset_id: str | None = None,
    tags: Sequence[str | tuple[str, Any]] = (),
    tags_from_path: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert ``source`` (a file, folder or archive) to a dataset in ``output``.

    ``format`` is a name from ``READERS`` (default: detected); ``mapping`` is the
    column mapping of custom CSV (JSON object, text or file). ``dataset_id``
    defaults to the input's name. ``tags`` are ``key=value`` strings (see
    ``parse_tag``) or ``(key, value)`` pairs for every run, and
    ``tags_from_path`` a ``PathPattern``. Returns the manifest written.
    """
    # #lizard forgives  (the public entry point; each step is a helper)
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass overwrite=True to replace it")
    dataset_id = check_dataset_id(dataset_id or default_dataset_id(source))
    parsed = [parse_tag(t) if isinstance(t, str) else (t[0], t[1]) for t in tags]
    pattern = PathPattern(tags_from_path) if tags_from_path else None
    with staged(source) as stage:
        name = _choose_format(stage.path, format, mapping)
        extra = {} if mapping is None else {"mapping": mapping}
        READERS[name].read(stage.path, dataset_id, **extra).write(
            output, overwrite=overwrite
        )
        options: dict[str, Any] = {"format": name, "input": Path(source).name}
        if parsed:
            options["tags"] = [f"{k}={v}" for k, v in parsed]
        if pattern is not None:
            options["tags_from_path"] = pattern.pattern
        try:
            return _finish(output, stage, options, (parsed, pattern))
        except Exception:
            shutil.rmtree(output, ignore_errors=True)
            raise
