"""Input trees: a folder, a single file or an archive, with nested archives opened.

A reader asks ``open_tree`` for the files of its input. Archives (zip, tar and
compressed tar) are extracted into a temporary folder, also when they sit inside
another archive or in a folder, and every file gets its source path in the
syntax of ``docs/format.md``: ``<archive path>/<member path>``, for example
``algA.tgz/algA/data_f1/bbobexp_f1_DIM2.dat``. The manifest's ``archives``
entries and each file's innermost archive come with the tree.

Members that are not regular files (links, devices) or whose names leave the
archive (absolute paths, ``..``) are not extracted; a ``skipped_archive_member``
warning names them. Archives nested deeper than ``MAX_DEPTH`` are not opened.
"""

from __future__ import annotations

import os
import posixpath
import shutil
import tarfile
import tempfile
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..format import check_source_path, file_entry
from .base import source_path

#: File name endings read as archives (compared in lower case).
ARCHIVE_SUFFIXES = (
    ".zip",
    ".tgz",
    ".tar.gz",
    ".tar",
    ".tbz2",
    ".tar.bz2",
    ".txz",
    ".tar.xz",
)
#: How deep archives inside archives are opened.
MAX_DEPTH = 4


def is_archive(name: str) -> bool:
    """True when ``name`` ends like an archive this module opens."""
    return name.lower().endswith(ARCHIVE_SUFFIXES)


@dataclass
class SourceTree:  # pylint: disable=too-many-instance-attributes
    """The files of an input, by source path.

    ``files`` holds every wanted file found by walking the input, extracted
    archive members included. ``base`` is the folder that plain source paths
    are relative to, so that ``resolve`` also finds files the walk did not list
    (the data files of a single ``.info`` file given as input).
    """

    base: Path | None
    files: dict[str, Path] = field(default_factory=dict)
    #: The innermost archive of each extracted file.
    archive_of: dict[str, str] = field(default_factory=dict)
    #: The outermost archive of each extracted file.
    outer_archive_of: dict[str, str] = field(default_factory=dict)
    archives: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)
    walked: bool = False

    def resolve(self, path: str) -> Path | None:
        """The local file of source path ``path``, or None when there is none."""
        local = self.files.get(path)
        if local is not None:
            return local
        if self.base is None or path in self.archive_of:
            return None
        candidate = self.base.joinpath(*path.split("/"))
        return candidate if candidate.is_file() else None

    def warn(self, code: str, message: str, source: str | None = None) -> None:
        entry = {"code": code, "message": message}
        if source is not None:
            entry["source"] = source
        self.warnings.append(entry)


@contextmanager
def open_tree(
    source: str | os.PathLike[str],
    root: str | os.PathLike[str] | None = None,
    wanted: Callable[[str], bool] = lambda _: True,
) -> Iterator[SourceTree]:
    """The files of ``source`` (a folder, a file or an archive) as a ``SourceTree``.

    Source paths are relative to ``root``, by default the folder itself or the
    file's folder. Only files for which ``wanted(source path)`` is true are
    listed (archives are always opened). Extracted files live in a temporary
    folder that is removed when the context ends.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"{source} not found")
    base = (
        Path(root)
        if root is not None
        else (source if source.is_dir() else source.parent)
    )
    tree = SourceTree(base)
    with tempfile.TemporaryDirectory(prefix="iohinspector-") as tmp:
        opener = _Opener(tree, Path(tmp), wanted)
        if source.is_dir():
            tree.walked = True
            for path in sorted(p for p in source.rglob("*") if p.is_file()):
                opener.add_file(path, source_path(path, base))
        else:
            opener.add_file(source, source_path(source, base))
        yield tree


@dataclass
class _Opener:
    tree: SourceTree
    tmp: Path
    wanted: Callable[[str], bool]
    count: int = 0

    def add_file(
        self,
        local: Path,
        display: str,
        parents: tuple[str, ...] = (),
    ) -> None:
        if is_archive(display):
            self.open_archive(local, display, parents)
            return
        if not self.wanted(display):
            return
        self.tree.files[display] = local
        if parents:
            self.tree.archive_of[display] = parents[-1]
            self.tree.outer_archive_of[display] = parents[0]

    def open_archive(self, local: Path, display: str, parents: tuple[str, ...]) -> None:
        entry = file_entry(local, display)
        if parents:
            entry["archive"] = parents[-1]
        self.tree.archives.append(entry)
        if len(parents) >= MAX_DEPTH:
            self.tree.warn(
                "skipped_archive", f"nested deeper than {MAX_DEPTH} archives", display
            )
            return
        self.count += 1
        dest = self.tmp / f"a{self.count}"
        dest.mkdir()

        def keep(name: str) -> bool:
            member = _member_path(display, name)
            return member is not None and (is_archive(member) or self.wanted(member))

        try:
            members = _extract(local, dest, keep)
        except (tarfile.TarError, zipfile.BadZipFile, OSError, EOFError) as exc:
            self.tree.warn(
                "skipped_archive", f"cannot open the archive: {exc}", display
            )
            return
        inner = (*parents, display)
        for name, path in members:
            member = _member_path(display, name)
            if path is None or member is None:
                self.tree.warn(
                    "skipped_archive_member",
                    f"member {name!r} is not a regular file inside the archive",
                    display,
                )
                continue
            self.add_file(path, member, inner)


def _member_path(archive: str, name: str) -> str | None:
    """``<archive>/<name>`` when ``name`` is a safe relative member name."""
    name = name.replace("\\", "/")
    while name.startswith("./"):
        name = name[2:]
    if not name or name.startswith("/") or ".." in name.split("/"):
        return None
    name = posixpath.normpath(name)
    try:
        return check_source_path(f"{archive}/{name}")
    except ValueError:
        return None


def _safe_target(dest: Path, name: str) -> Path | None:
    name = name.replace("\\", "/")
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if not parts or ".." in parts or name.startswith("/"):
        return None
    return dest.joinpath(*parts)


def _extract(
    local: Path, dest: Path, keep: Callable[[str], bool]
) -> list[tuple[str, Path | None]]:
    """Extract the regular files of an archive for which ``keep(name)`` holds.

    Returns ``(name, path)`` per extracted member, and ``(name, None)`` per member
    that is not a regular file or whose name leaves the archive.
    """
    if zipfile.is_zipfile(local):
        return _extract_zip(local, dest, keep)
    return _extract_tar(local, dest, keep)


def _extract_zip(
    local: Path, dest: Path, keep: Callable[[str], bool]
) -> list[tuple[str, Path | None]]:
    out: list[tuple[str, Path | None]] = []
    with zipfile.ZipFile(local) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            target = _safe_target(dest, info.filename)
            mode = (info.external_attr >> 16) & 0o170000
            if target is None or mode not in (0, 0o100000):
                out.append((info.filename, None))
                continue
            if not keep(info.filename):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            out.append((info.filename, target))
    return out


def _extract_tar(
    local: Path, dest: Path, keep: Callable[[str], bool]
) -> list[tuple[str, Path | None]]:
    out: list[tuple[str, Path | None]] = []
    with tarfile.open(local, mode="r:*") as archive:
        for member in archive:
            if member.isdir():
                continue
            target = _safe_target(dest, member.name)
            if target is None or not member.isfile():
                out.append((member.name, None))
                continue
            if not keep(member.name):
                continue
            src = archive.extractfile(member)
            if src is None:
                out.append((member.name, None))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            out.append((member.name, target))
    return out
