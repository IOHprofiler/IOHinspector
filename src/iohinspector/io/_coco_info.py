"""COCO ``.info`` files: one block of three lines per function, dimension and session.

::

    suite = 'bbob', funcId = 1, DIM = 2, Precision = 1.000e-08, algId = 'BFGS', ...
    % BFGS: free text about the algorithm
    data_f1/bbobexp_f1_DIM2.dat, 1:13|2.7e-13, 2:25|2.3e-13, 3:17|9.2e-12

1. The header: ``key = value`` pairs. Strings are quoted with ``'``; numbers are
   not. Old loggers write only ``funcId``, ``DIM``, ``Precision`` and ``algId``.
2. A comment line starting with ``%``.
3. The data file, relative to the ``.info`` file (and to ``folder`` when the
   header has one), then one entry per run: ``instance:evaluations|value``, or a
   bare ``instance`` for a run the logger did not finalise. ``value`` is the best
   Δf at the end of the run, written with ``%.1e`` (old loggers subtract the
   final target, so solved runs show small negative values).

Real archives deviate: blank lines, CRLF line endings, a data line with stray
text on the next line (bbob 2009, NELDER, f13). The parser skips lines that fit
none of the three kinds and reports them in ``InfoFile.skipped``; it fails with
file:line on a header or entry it cannot read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .base import INTEGER_PATTERN, NUMBER_PATTERN, ReaderError

#: ``key = value`` pairs of a header line; the same pattern as cocopp's ``parseinfo``.
_PAIR = re.compile(r"\s*([^,=]+?)\s*=\s*(\"[^\"]*\"|'[^']*'|[^,]*?)\s*(?=,|$)")
_INTEGER = re.compile(f"^(?:{INTEGER_PATTERN})$")
_NUMBER = re.compile(f"^(?:{NUMBER_PATTERN})$")
_HEADER_START = re.compile(r"^\s*[A-Za-z_][\w ]*=")
_ENTRY = re.compile(
    rf"^(?P<instance>{INTEGER_PATTERN})"
    rf"(?::(?P<evals>{INTEGER_PATTERN})\|(?P<value>{NUMBER_PATTERN}))?$"
)
#: Header keys with a fixed meaning; their aliases as cocopp reads them.
ALIASES = {"function": "funcId", "dim": "DIM", "algorithm": "algId"}


@dataclass
class InfoEntry:
    """One run entry of a data line."""

    instance: int
    #: Evaluations and final value; None when the run was not finalised.
    evals: int | None
    value: float | None
    line: int


@dataclass
class InfoBlock:  # pylint: disable=too-many-instance-attributes
    """One header, comment and data line."""

    line: int
    #: Header pairs in file order, with typed values (str, int or float).
    header: list[tuple[str, Any]]
    comment: str
    function_id: int
    dimension: int
    #: The data file as written (``/`` separators), with ``folder`` prepended.
    data_file: str
    entries: list[InfoEntry] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        return _lookup(self.header, key, default)


def _lookup(header: list[tuple[str, Any]], key: str, default: Any = None) -> Any:
    """The value of ``key`` (or one of its aliases) in a header."""
    return next((v for n, v in header if ALIASES.get(n, n) == key), default)


@dataclass
class InfoFile:
    path: str
    blocks: list[InfoBlock]
    #: Lines that are no header, comment or data line: ``(line, text)``.
    skipped: list[tuple[int, str]] = field(default_factory=list)
    #: Blocks without a data line (cut-off files): their header line numbers.
    incomplete: list[int] = field(default_factory=list)


def header_value(text: str) -> Any:
    """A header value: quoted text is a string, else an int, a float or a string."""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    if _INTEGER.match(text):
        return int(text)
    if _NUMBER.match(text):
        return float(text)
    return text


def parse_header(text: str, path: str, line: int) -> list[tuple[str, Any]]:
    """The ``key = value`` pairs of a header line."""
    pairs = [(k.strip(), header_value(v)) for k, v in _PAIR.findall(text)]
    if not pairs:
        raise ReaderError("no key = value pairs in the header line", path, line)
    return pairs


def _int_key(header: list[tuple[str, Any]], key: str, path: str, line: int) -> int:
    value = _lookup(header, key)
    if value is None:
        raise ReaderError(f"the header has no {key}", path, line)
    if not isinstance(value, int):
        raise ReaderError(f"{key} = {value!r} is not an integer", path, line)
    return value


def parse_entries(parts: list[str], path: str, line: int) -> list[InfoEntry]:
    """Run entries of a data line, after its file name."""
    out = []
    for part in parts:
        part = part.strip()
        if not part or "=" in part:
            continue
        m = _ENTRY.match(part)
        if m is None:
            raise ReaderError(
                f"run entry {part!r} is not instance:evals|value", path, line
            )
        evals, value = m["evals"], m["value"]
        out.append(
            InfoEntry(
                instance=int(m["instance"]),
                evals=None if evals is None else int(evals),
                value=None if value is None else float(value.lower()),
                line=line,
            )
        )
    return out


def _kind(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "blank"
    if stripped.startswith("%"):
        return "comment"
    first = stripped.split(",", 1)[0].strip()
    if first.endswith("dat"):
        return "data"
    if _HEADER_START.match(stripped):
        return "header"
    return "other"


def parse_info(text: str, path: str) -> InfoFile:
    """Parse the text of a COCO ``.info`` file; ``path`` names it in errors."""
    # #lizard forgives  (a small state machine over the line kinds)
    out = InfoFile(path, [])
    header: list[tuple[str, Any]] | None = None
    header_line = 0
    comment = ""
    for number, raw in enumerate(text.split("\n"), 1):
        kind = _kind(raw)
        if kind == "blank":
            continue
        if kind == "header":
            if header is not None:
                out.incomplete.append(header_line)
            header, header_line, comment = parse_header(raw, path, number), number, ""
        elif kind == "comment" and header is not None and not comment:
            comment = raw.strip()
        elif kind == "data" and header is not None:
            out.blocks.append(_block(header, header_line, comment, raw, path, number))
            header = None
        else:
            out.skipped.append((number, raw.strip()))
    if header is not None:
        out.incomplete.append(header_line)
    return out


def _block(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    header: list[tuple[str, Any]],
    header_line: int,
    comment: str,
    data: str,
    path: str,
    line: int,
) -> InfoBlock:
    parts = data.strip().split(",")
    name = parts[0].strip().replace("\\", "/")
    folder = next((v for k, v in header if k == "folder"), None)
    if folder:
        name = f"{str(folder).rstrip('/')}/{name}"
    files = [p.strip() for p in parts[1:] if p.strip().endswith("dat")]
    if files:
        raise ReaderError(
            f"the data line names {len(files) + 1} data files; one is supported",
            path,
            line,
        )
    return InfoBlock(
        line=header_line,
        header=header,
        comment=comment,
        function_id=_int_key(header, "funcId", path, header_line),
        dimension=_int_key(header, "DIM", path, header_line),
        data_file=name,
        entries=parse_entries(parts[1:], path, line),
    )
