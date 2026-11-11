"""Archives in the input: ``staged`` opens them so that every reader can read it.

Archives (zip, tar and compressed tar) are extracted into a temporary folder,
also when they sit inside another archive or in a folder, and every file gets
its source path in the syntax of ``docs/format.md``: ``<archive path>/<member
path>``, for example ``algA.tgz/algA/data_f1/bbobexp_f1_DIM2.dat``. The input
is laid out under the temporary folder by those paths, so a reader that reads
the folder gives each file its source path. The originals are never changed.

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
from functools import partial
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
class Staged:
    """The input as the readers see it: ``path``, plus the manifest ``archives``
    entries and the warnings of opening them."""

    path: Path
    archives: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)

    def warn(self, code: str, message: str, source: str) -> None:
        self.warnings.append({"code": code, "message": message, "source": source})


@contextmanager
def staged(source: str | os.PathLike[str]) -> Iterator[Staged]:
    """The input with its archives opened, as a folder every reader can read.

    A plain file, or a folder without archives, is used where it is. Otherwise
    the input's files are laid out under a temporary folder by their source
    paths (hard links where possible), with the archives' members in place of
    the archives. The folder is removed when the context ends.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"{source} not found")
    base = source if source.is_dir() else source.parent
    files = (
        sorted(p for p in source.rglob("*") if p.is_file())
        if source.is_dir()
        else [source]
    )
    if not any(is_archive(p.name) for p in files):
        yield Staged(source)
        return
    with tempfile.TemporaryDirectory(prefix="iohinspector-") as tmp:
        root = Path(tmp) / "input"
        stage = Staged(root)
        for path in files:
            _add(path, source_path(path, base), (), stage, Path(tmp))
        yield stage


def _add(
    local: Path, display: str, parents: tuple[str, ...], stage: Staged, tmp: Path
) -> None:
    if is_archive(display):
        _open(local, display, parents, stage, tmp)
        return
    target = stage.path.joinpath(*display.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(local, target)
    except OSError:
        shutil.copy2(local, target)


def _open(
    local: Path, display: str, parents: tuple[str, ...], stage: Staged, tmp: Path
) -> None:
    entry = file_entry(local, display)
    if parents:
        entry["archive"] = parents[-1]
    stage.archives.append(entry)
    if len(parents) >= MAX_DEPTH:
        stage.warn(
            "skipped_archive", f"nested deeper than {MAX_DEPTH} archives", display
        )
        return
    dest = tmp / f"a{len(stage.archives)}"
    dest.mkdir()
    try:
        members = _extract(local, dest)
    except (tarfile.TarError, zipfile.BadZipFile, OSError, EOFError) as exc:
        stage.warn("skipped_archive", f"cannot open the archive: {exc}", display)
        return
    for name, path in members:
        member = _member_path(display, name)
        if path is None or member is None:
            stage.warn(
                "skipped_archive_member",
                f"member {name!r} is not a regular file inside the archive",
                display,
            )
            continue
        _add(path, member, (*parents, display), stage, tmp)


def _member_path(archive: str, name: str) -> str | None:
    """``<archive>/<name>`` when ``name`` is a safe relative member name."""
    name = name.replace("\\", "/")
    while name.startswith("./"):
        name = name[2:]
    if not name or name.startswith("/") or ".." in name.split("/"):
        return None
    try:
        return check_source_path(f"{archive}/{posixpath.normpath(name)}")
    except ValueError:
        return None


def _extract(local: Path, dest: Path) -> list[tuple[str, Path | None]]:
    """Extract every member that is a regular file into ``dest`` (numbered, as
    names may clash). Returns ``(name, path)`` per member that is not a folder,
    with a None path for links and other members that are not regular files."""
    out: list[tuple[str, Path | None]] = []
    if zipfile.is_zipfile(local):
        with zipfile.ZipFile(local) as archive:
            for info in archive.infolist():
                regular = (info.external_attr >> 16) & 0o170000 in (0, 0o100000)
                if not info.is_dir():
                    opener = partial(archive.open, info) if regular else None
                    out.append(_save(info.filename, opener, dest / str(len(out))))
        return out
    with tarfile.open(local, mode="r:*") as archive:
        for member in archive:
            if not member.isdir():
                opener = (
                    partial(archive.extractfile, member) if member.isfile() else None
                )
                out.append(_save(member.name, opener, dest / str(len(out))))
    return out


def _save(
    name: str, opener: Callable[[], Any] | None, target: Path
) -> tuple[str, Path | None]:
    """``(name, target)`` with the member's content copied to ``target``, or
    ``(name, None)`` when there is no content to open."""
    if opener is None:
        return name, None
    with opener() as content, open(target, "wb") as fh:
        shutil.copyfileobj(content, fh)
    return name, target
