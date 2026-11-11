"""Convert benchmark logs to dataset format v1: ``iohinspector convert``.

``convert`` brings the two reader tables of ``iohinspector.io`` together:

- ``READERS``: readers that detect their input and return a ``ReadResult``
  (IOHexperimenter and COCO);
- ``CONVERTERS``: streaming CSV converters that write the dataset themselves.
  Nevergrad files are detected by their columns; custom CSV needs a column
  mapping (``mapping``), so it is used only when a mapping is given.

The input is a file, a folder, or an archive (zip, tar and compressed tar), and
archives inside folders and archives are opened too (``iohinspector.io.archive``).
Archives are extracted into a temporary folder; the originals are never changed.
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
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
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
from .format.schema import descriptor_key_problem
from .io import CONVERTERS, READERS
from .io.archive import is_archive, open_tree
from .io.nevergrad import is_nevergrad_csv

#: Descriptor keys that only the origins ``algorithm`` and ``user`` may hold.
ALGORITHM_KEYS = ("algorithm_name", "algorithm_info")
_TYPES = ("str", "int", "float", "bool")
_BOOLS = {"true": True, "false": False, "1": True, "0": False}


class ConvertError(ValueError):
    """The input cannot be converted: unknown format, bad option or bad tag."""


# --------------------------------------------------------------------------- tags


def _typed(text: str, kind: str, where: str) -> Any:
    try:
        if kind == "int":
            return int(text)
        if kind == "float":
            return float(text)
    except ValueError:
        raise ConvertError(f"{where}: {text!r} is not {kind}") from None
    if kind == "bool":
        if text.lower() not in _BOOLS:
            raise ConvertError(f"{where}: {text!r} is not a bool (true or false)")
        return _BOOLS[text.lower()]
    return text


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
            if key in ALGORITHM_KEYS:
                origin = "user"  # the format allows only algorithm or user here
            records.append((rid, key, value, origin))
    return records


# --------------------------------------------------------------------------- archives


@dataclass
class _Staged:
    """The input as the readers see it: ``path``, plus the archives it came from."""

    path: Path
    archives: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)


def _has_archive(source: Path) -> bool:
    if source.is_dir():
        return any(p.is_file() and is_archive(p.name) for p in source.rglob("*"))
    return is_archive(source.name)


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


@contextmanager
def staged_input(source: str | os.PathLike[str]) -> Iterator[_Staged]:
    """The input with its archives opened, as a folder every reader can read.

    A plain file, or a folder without archives, is used where it is. Otherwise
    ``iohinspector.io.archive.open_tree`` extracts the archives (nested ones too),
    and its files are laid out under a temporary folder by their source paths
    (hard links where possible), so ``algA.zip/data_f1/x.dat`` is read from
    ``<tmp>/algA.zip/data_f1/x.dat`` and keeps that source path.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"{source} not found")
    if not _has_archive(source):
        yield _Staged(source)
        return
    with (
        open_tree(source) as tree,
        tempfile.TemporaryDirectory(prefix="iohinspector-convert-") as tmp,
    ):
        root = Path(tmp)
        for display, local in tree.files.items():
            _link_or_copy(local, root.joinpath(*display.split("/")))
        yield _Staged(root, list(tree.archives), list(tree.warnings))


# --------------------------------------------------------------------------- convert


def default_dataset_id(source: str | os.PathLike[str]) -> str:
    """A dataset id from the input's name: its stem with other characters as ``_``."""
    stem = Path(source).resolve().name.split(".")[0] or "dataset"
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", stem)[:128]
    return check_dataset_id(cleaned if cleaned[0].isalnum() else "d" + cleaned[:127])


def detect(path: str | os.PathLike[str]) -> str:
    """The format of ``path``: the first ``READERS`` entry that matches, then
    Nevergrad CSV."""
    path = Path(path)
    for name, reader in READERS.items():
        if reader.detect(path):
            return name
    if path.is_file() and is_nevergrad_csv(path):
        return "nevergrad-csv"
    known = ", ".join([*READERS, *CONVERTERS])
    raise ConvertError(
        f"{path}: no reader recognises this input; give the format ({known}), "
        "and a mapping for custom CSV"
    )


def _csv_file(staged: _Staged, name: str) -> tuple[Path, str]:
    """The single CSV file a converter reads, and its source path."""
    path = staged.path
    if path.is_file():
        return path, path.name
    files = sorted(
        p for p in path.rglob("*") if p.is_file() and ".csv" in p.name.lower()
    )
    if len(files) != 1:
        raise ConvertError(
            f"{name} converts one CSV file; the input holds {len(files)} CSV files"
        )
    return files[0], files[0].relative_to(path).as_posix()


def _run_format(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    staged: _Staged,
    output: Path,
    name: str,
    dataset_id: str,
    mapping: Any,
    overwrite: bool,
) -> dict[str, Any]:
    if name in READERS:
        result = READERS[name].read(staged.path, dataset_id)
        return result.write(output, overwrite=overwrite)
    file, source_name = _csv_file(staged, name)
    kwargs: dict[str, Any] = {
        "dataset_id": dataset_id,
        "source_name": source_name,
        "overwrite": overwrite,
    }
    if name == "custom-csv":
        return CONVERTERS[name](file, output, mapping, **kwargs)
    return CONVERTERS[name](file, output, **kwargs)


def _choose_format(staged: _Staged, format_name: str | None, mapping: Any) -> str:
    if format_name is None:
        format_name = "custom-csv" if mapping is not None else detect(staged.path)
    if format_name not in READERS and format_name not in CONVERTERS:
        known = ", ".join([*READERS, *CONVERTERS])
        raise ConvertError(f"unknown format {format_name!r}; known: {known}")
    if format_name == "custom-csv" and mapping is None:
        raise ConvertError("custom-csv needs a column mapping (--mapping)")
    if mapping is not None and format_name != "custom-csv":
        raise ConvertError(f"a mapping is only used by custom-csv, not {format_name}")
    return format_name


def _finish(
    output: Path,
    staged: _Staged,
    options: dict[str, Any],
    records: list[tuple[int, str, Any, str]],
) -> dict[str, Any]:
    """Add the tags, the archives and the convert options to the written dataset."""
    if records:
        update_descriptors(output, records, origin=None)
    manifest = read_manifest(output, warn=False)
    manifest.pop("updated", None)  # tags are part of the conversion
    if staged.archives:
        known = {a["path"] for a in manifest.get("archives", [])}
        manifest["archives"] = manifest.get("archives", []) + [
            a for a in staged.archives if a["path"] not in known
        ]
        _mark_archive_members(manifest)
    if staged.warnings:
        manifest["warnings"] = manifest.get("warnings", []) + staged.warnings
    manifest.setdefault("extra", {})["convert"] = options
    write_manifest(output, manifest)
    return manifest


def _mark_archive_members(manifest: dict[str, Any]) -> None:
    """Set ``archive`` on each source to the innermost archive that holds it."""
    paths = sorted((a["path"] for a in manifest["archives"]), key=len, reverse=True)
    for entry in manifest.get("sources", []):
        if "archive" in entry:
            continue
        for archive in paths:
            if entry["path"].startswith(archive + "/"):
                entry["archive"] = archive
                break


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

    ``format`` is a name from ``READERS`` or ``CONVERTERS`` (default: detected);
    ``mapping`` is the column mapping of custom CSV (JSON object, text or file).
    ``dataset_id`` defaults to the input's name. ``tags`` are ``key=value``
    strings (see ``parse_tag``) or ``(key, value)`` pairs for every run, and
    ``tags_from_path`` a ``PathPattern``. Returns the manifest written.
    """
    # #lizard forgives  (the public entry point; each step is a helper)
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass overwrite=True to replace it")
    dataset_id = check_dataset_id(dataset_id or default_dataset_id(source))
    parsed = [parse_tag(t) if isinstance(t, str) else tuple(t) for t in tags]
    pattern = PathPattern(tags_from_path) if tags_from_path else None
    with staged_input(source) as staged:
        name = _choose_format(staged, format, mapping)
        _run_format(staged, output, name, dataset_id, mapping, overwrite)
        options: dict[str, Any] = {"format": name, "input": Path(source).name}
        if parsed:
            options["tags"] = [f"{k}={v}" for k, v in parsed]
        if tags_from_path:
            options["tags_from_path"] = tags_from_path
        try:
            runs = pl.read_parquet(
                output / "runs.parquet", columns=["run_id", "source_file"]
            )
            records = _tag_records(runs, parsed, pattern)  # type: ignore[arg-type]
            return _finish(output, staged, options, records)
        except Exception:
            shutil.rmtree(output, ignore_errors=True)
            raise
