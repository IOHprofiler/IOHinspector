"""Reading, writing and checking ``manifest.json`` (``docs/format.md``, "manifest.json")."""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import os
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .ids import check_dataset_id
from .schema import (
    FORMAT_MAJOR,
    FORMAT_MINOR,
    FORMAT_NAME,
    FORMAT_VERSION,
    MANIFEST_FILE,
)

#: The normative JSON Schema of the manifest, identical to the one in docs/format.md.
MANIFEST_SCHEMA: dict[str, Any] = json.loads(
    r"""
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "urn:iohinspector:dataset-manifest:1",
  "title": "iohinspector dataset manifest, format version 1",
  "type": "object",
  "additionalProperties": false,
  "required": ["format", "format_version", "dataset_id", "created", "reader", "writer", "sources", "counts"],
  "properties": {
    "format": {"const": "iohinspector-dataset"},
    "format_version": {"type": "string", "pattern": "^1\\.(0|[1-9][0-9]*)$"},
    "dataset_id": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"},
    "created": {"type": "string", "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\\.[0-9]{1,9})?Z$"},
    "reader": {"$ref": "#/$defs/software"},
    "writer": {"$ref": "#/$defs/software"},
    "sources": {"type": "array", "items": {"$ref": "#/$defs/source"}},
    "archives": {"type": "array", "items": {"$ref": "#/$defs/archive"}},
    "counts": {
      "type": "object",
      "additionalProperties": false,
      "required": ["runs", "trace_rows"],
      "properties": {
        "runs": {"type": "integer", "minimum": 0},
        "trace_rows": {"type": "integer", "minimum": 0},
        "positions_rows": {"type": "integer", "minimum": 0}
      }
    },
    "renamed_columns": {"type": "array", "items": {"$ref": "#/$defs/rename"}},
    "warnings": {"type": "array", "items": {"$ref": "#/$defs/warning"}},
    "variable_types": {"type": "null"},
    "extra": {"type": "object"}
  },
  "$defs": {
    "path": {
      "type": "string",
      "pattern": "^[^/\\\\\\u0000-\\u001f][^\\\\\\u0000-\\u001f]*$",
      "not": {"pattern": "(^|/)\\.{1,2}(/|$)|//|/$"}
    },
    "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    "software": {
      "type": "object",
      "additionalProperties": false,
      "required": ["name", "version"],
      "properties": {
        "name": {"type": "string", "minLength": 1},
        "version": {"type": "string"},
        "options": {"type": "object"}
      }
    },
    "source": {
      "type": "object",
      "additionalProperties": false,
      "required": ["path", "sha256", "bytes"],
      "properties": {
        "path": {"$ref": "#/$defs/path"},
        "sha256": {"oneOf": [{"$ref": "#/$defs/sha256"}, {"type": "null"}]},
        "bytes": {"type": ["integer", "null"], "minimum": 0},
        "kind": {"type": "string", "minLength": 1},
        "archive": {"$ref": "#/$defs/path"},
        "data_rows": {"type": "integer", "minimum": 0},
        "logger": {"$ref": "#/$defs/software"},
        "triggers": {"type": "array", "items": {"$ref": "#/$defs/trigger"}},
        "precision": {"$ref": "#/$defs/precision"}
      }
    },
    "archive": {
      "type": "object",
      "additionalProperties": false,
      "required": ["path", "sha256", "bytes"],
      "properties": {
        "path": {"$ref": "#/$defs/path"},
        "sha256": {"$ref": "#/$defs/sha256"},
        "bytes": {"type": "integer", "minimum": 0},
        "archive": {"$ref": "#/$defs/path"}
      }
    },
    "trigger": {
      "type": "object",
      "additionalProperties": false,
      "required": ["kind"],
      "properties": {
        "kind": {"enum": ["all", "improvement", "fixed_budget", "fixed_target", "other"]},
        "params": {"type": "object"}
      }
    },
    "precision": {
      "type": "object",
      "additionalProperties": false,
      "required": ["notation"],
      "properties": {
        "notation": {"enum": ["fixed", "scientific", "general", "roundtrip", "unknown"]},
        "digits": {"type": ["integer", "null"], "minimum": 0}
      }
    },
    "rename": {
      "type": "object",
      "additionalProperties": false,
      "required": ["from", "to"],
      "properties": {
        "from": {"type": "string", "minLength": 1},
        "to": {"type": "string", "minLength": 1},
        "source": {"$ref": "#/$defs/path"}
      }
    },
    "warning": {
      "type": "object",
      "additionalProperties": false,
      "required": ["code", "message"],
      "properties": {
        "code": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"},
        "message": {"type": "string"},
        "source": {"$ref": "#/$defs/path"}
      }
    }
  }
}
"""
)


class UnsupportedFormatVersion(ValueError):
    """The dataset has a format name or major version this iohinspector cannot read."""


class ManifestError(ValueError):
    """The manifest is not valid JSON or does not match the schema."""


class FormatWarning(UserWarning):
    """A dataset uses a newer minor format version than this iohinspector knows."""


class DatasetWarning(UserWarning):
    """A warning recorded in a dataset's manifest, for example about source precision."""


def _iohinspector_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("iohinspector")
    except PackageNotFoundError:
        return "unknown"


def utc_now() -> str:
    """The current time in the manifest's ``created`` syntax."""
    now = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0)
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_format_version(version: object) -> tuple[int, int]:
    """Split ``"MAJOR.MINOR"`` into integers; raise ``UnsupportedFormatVersion`` if malformed."""
    if isinstance(version, str):
        major, sep, minor = version.partition(".")
        if sep and major.isdigit() and minor.isdigit():
            return int(major), int(minor)
    raise UnsupportedFormatVersion(
        f"malformed format_version {version!r}; expected 'MAJOR.MINOR'"
    )


def check_format_version(
    manifest: Mapping[str, Any], *, warn: bool = True
) -> tuple[int, int]:
    """Refuse a dataset this iohinspector cannot read.

    Raises ``UnsupportedFormatVersion`` for another format name or an unknown major
    version. Warns (``FormatWarning``) for a newer minor version, which is readable.
    """
    if manifest.get("format") != FORMAT_NAME:
        raise UnsupportedFormatVersion(
            f"not an iohinspector dataset: format is {manifest.get('format')!r}, expected {FORMAT_NAME!r}"
        )
    major, minor = parse_format_version(manifest.get("format_version"))
    if major != FORMAT_MAJOR:
        raise UnsupportedFormatVersion(
            f"dataset format version {major}.{minor} is not supported; this iohinspector "
            f"reads major version {FORMAT_MAJOR} only"
        )
    if minor > FORMAT_MINOR and warn:
        warnings.warn(
            f"dataset format version {major}.{minor} is newer than {FORMAT_VERSION}; "
            "fields and columns this version does not know are ignored",
            FormatWarning,
            stacklevel=3,
        )
    return major, minor


def _relaxed(schema: Any) -> Any:
    """A copy of ``schema`` without ``additionalProperties: false`` anywhere."""
    if isinstance(schema, dict):
        return {
            k: _relaxed(v)
            for k, v in schema.items()
            if not (k == "additionalProperties" and v is False)
        }
    if isinstance(schema, list):
        return [_relaxed(v) for v in schema]
    return schema


_VALIDATORS: dict[bool, Any] = {}


def _validator(relaxed: bool) -> Any:
    if relaxed not in _VALIDATORS:
        from jsonschema import Draft202012Validator

        schema = _relaxed(MANIFEST_SCHEMA) if relaxed else MANIFEST_SCHEMA
        _VALIDATORS[relaxed] = Draft202012Validator(schema)
    return _VALIDATORS[relaxed]


def validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """All problems with ``manifest``, as messages; an empty list means it is valid.

    Checks the JSON Schema and the rules the schema cannot express: unique source
    and archive paths, and that every ``archive`` reference is listed. Unknown fields
    are accepted when the manifest has a newer minor version.
    """
    try:
        _, minor = check_format_version(manifest, warn=False)
    except UnsupportedFormatVersion as exc:
        return [str(exc)]
    problems = []
    for error in sorted(
        _validator(minor > FORMAT_MINOR).iter_errors(manifest),
        key=lambda e: list(e.path),
    ):
        where = "/".join(str(p) for p in error.absolute_path) or "(top level)"
        problems.append(f"manifest {where}: {error.message}")
    if problems:
        return problems

    archives = [a["path"] for a in manifest.get("archives", [])]
    sources = [s["path"] for s in manifest["sources"]]
    for kind, paths in (("source", sources), ("archive", archives)):
        seen: set[str] = set()
        for path in paths:
            if path in seen:
                problems.append(f"manifest: duplicate {kind} path {path!r}")
            seen.add(path)
    known_archives = set(archives)
    for entry in [*manifest["sources"], *manifest.get("archives", [])]:
        ref = entry.get("archive")
        if ref is not None and ref not in known_archives:
            problems.append(
                f"manifest: {entry['path']!r} names archive {ref!r}, which is not in 'archives'"
            )
    return problems


def _manifest_path(path: str | os.PathLike[str]) -> Path:
    path = Path(path)
    return path / MANIFEST_FILE if path.is_dir() else path


def read_manifest(
    path: str | os.PathLike[str], *, validate: bool = True, warn: bool = True
) -> dict[str, Any]:
    """Read the manifest of the dataset directory ``path`` (or a manifest file).

    Raises ``UnsupportedFormatVersion`` for an unknown format or major version, and
    ``ManifestError`` when the manifest is not valid JSON or, with ``validate``, does
    not match the schema.
    """
    file = _manifest_path(path)
    try:
        manifest = json.loads(
            file.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
    except (ValueError, UnicodeDecodeError) as exc:
        raise ManifestError(f"{file}: not a valid JSON manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ManifestError(f"{file}: the manifest must be a JSON object")
    check_format_version(manifest, warn=warn)
    if validate:
        problems = validate_manifest(manifest)
        if problems:
            raise ManifestError(
                f"{file}: invalid manifest:\n  " + "\n  ".join(problems)
            )
    return manifest


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def dump_manifest(manifest: Mapping[str, Any]) -> str:
    """The manifest as text: sorted keys, indent 2, a final newline, no NaN."""
    return (
        json.dumps(
            manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n"
    )


def write_manifest(path: str | os.PathLike[str], manifest: Mapping[str, Any]) -> Path:
    """Validate ``manifest`` and write it to the dataset directory ``path`` (or a file path).

    Raises ``ManifestError`` if it is invalid. Returns the file written.
    """
    problems = validate_manifest(manifest)
    if problems:
        raise ManifestError("invalid manifest:\n  " + "\n  ".join(problems))
    file = _manifest_path(path)
    tmp = file.with_name(f".{file.name}.tmp")
    tmp.write_text(dump_manifest(manifest), encoding="utf-8")
    os.replace(tmp, file)
    return file


def new_manifest(
    dataset_id: str,
    reader_name: str,
    reader_version: str | None = None,
    *,
    reader_options: Mapping[str, Any] | None = None,
    sources: Iterable[Mapping[str, Any]] = (),
    archives: Iterable[Mapping[str, Any]] = (),
    warnings: Iterable[Mapping[str, Any]] = (),
    renamed_columns: Iterable[Mapping[str, Any]] = (),
    extra: Mapping[str, Any] | None = None,
    created: str | None = None,
) -> dict[str, Any]:
    """A manifest for a reader to fill in. ``write_dataset`` sets the counts and the writer."""
    check_dataset_id(dataset_id)
    reader: dict[str, Any] = {
        "name": reader_name,
        "version": reader_version or _iohinspector_version(),
    }
    if reader_options is not None:
        reader["options"] = dict(reader_options)
    manifest: dict[str, Any] = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "dataset_id": dataset_id,
        "created": created or utc_now(),
        "reader": reader,
        "writer": writer_entry(),
        "sources": [dict(s) for s in sources],
        "counts": {"runs": 0, "trace_rows": 0},
    }
    for key, values in (
        ("archives", archives),
        ("warnings", warnings),
        ("renamed_columns", renamed_columns),
    ):
        values = [dict(v) for v in values]
        if values:
            manifest[key] = values
    if extra is not None:
        manifest["extra"] = dict(extra)
    return manifest


def writer_entry() -> dict[str, str]:
    """The manifest's ``writer`` entry for this iohinspector."""
    return {"name": "iohinspector", "version": _iohinspector_version()}


def file_entry(
    local_path: str | os.PathLike[str], path: str, **fields: Any
) -> dict[str, Any]:
    """A ``sources`` (or ``archives``) entry for a file on disk: its sha256 and size.

    ``path`` is the source path recorded in the dataset; ``fields`` adds optional
    entries such as ``kind``, ``data_rows``, ``triggers`` or ``precision``.
    """
    digest = hashlib.sha256()
    size = 0
    with open(local_path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return {"path": path, "sha256": digest.hexdigest(), "bytes": size, **fields}


def copy_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """A deep copy, so that callers' dictionaries are never changed."""
    return copy.deepcopy(dict(manifest))
