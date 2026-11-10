"""Walk the files of a verify input: a file, a folder or an archive, with nested
archives.

Written for ``verify`` alone: it shares no code with the readers or with
``iohinspector.convert``. Source paths follow ``docs/format.md``: relative to the
folder (or to the file's folder), and ``<archive path>/<member path>`` inside an
archive.
"""

from __future__ import annotations

import io
import posixpath
import tarfile
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO

ARCHIVE_ENDINGS = (
    ".zip",
    ".tgz",
    ".tar.gz",
    ".tar",
    ".tbz2",
    ".tar.bz2",
    ".txz",
    ".tar.xz",
)
MAX_DEPTH = 4


@dataclass
class SourceFile:
    """One file of the input. ``load()`` returns its bytes; call it only while
    the walk is at this file (tar members are read in one pass)."""

    path: str
    load: Callable[[], bytes]
    archive: bool


def looks_like_archive(name: str) -> bool:
    return name.lower().endswith(ARCHIVE_ENDINGS)


def walk(source: Path) -> Iterator[SourceFile]:
    """Every file of ``source``, archives and their members included, in path order."""
    if source.is_dir():
        for path in sorted(p for p in source.rglob("*") if p.is_file()):
            yield from _file(path, path.relative_to(source).as_posix())
    else:
        yield from _file(source, source.name)


def _file(path: Path, display: str) -> Iterator[SourceFile]:
    entry = SourceFile(display, path.read_bytes, looks_like_archive(display))
    yield entry
    if entry.archive:
        with open(path, "rb") as handle:
            yield from _members(handle, display, 1)


def _member_name(name: str) -> str | None:
    name = name.replace("\\", "/")
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if not parts or ".." in parts or name.startswith("/"):
        return None
    return posixpath.join(*parts)


def _members(handle: IO[bytes], display: str, depth: int) -> Iterator[SourceFile]:
    """The members of the archive in ``handle`` (a seekable binary file)."""
    if depth > MAX_DEPTH:
        return
    if zipfile.is_zipfile(handle):
        handle.seek(0)
        yield from _zip_members(handle, display, depth)
        return
    handle.seek(0)
    try:
        # closed by the with block below
        archive = tarfile.open(fileobj=handle, mode="r:*")  # noqa: SIM115
    except tarfile.TarError:
        return
    with archive:
        for member in archive:
            name = _member_name(member.name)
            if name is None or not member.isfile():
                continue
            yield from _member(archive, member, f"{display}/{name}", depth)


def _member(
    archive: tarfile.TarFile, member: tarfile.TarInfo, display: str, depth: int
) -> Iterator[SourceFile]:
    cache: list[bytes] = []

    def load() -> bytes:
        if not cache:
            handle = archive.extractfile(member)
            cache.append(handle.read() if handle is not None else b"")
        return cache[0]

    entry = SourceFile(display, load, looks_like_archive(display))
    yield entry
    if entry.archive:
        yield from _members(io.BytesIO(load()), display, depth + 1)


def _zip_members(handle: IO[bytes], display: str, depth: int) -> Iterator[SourceFile]:
    with zipfile.ZipFile(handle) as archive:
        for info in archive.infolist():
            name = _member_name(info.filename)
            if info.is_dir() or name is None:
                continue
            path = f"{display}/{name}"

            def load(info: zipfile.ZipInfo = info) -> bytes:
                return archive.read(info)

            entry = SourceFile(path, load, looks_like_archive(path))
            yield entry
            if entry.archive:
                yield from _members(io.BytesIO(load()), path, depth + 1)
