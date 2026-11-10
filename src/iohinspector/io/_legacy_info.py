"""Parser of legacy IOHprofiler ``.info`` files (the format before the JSON metadata).

An ``.info`` file is a sequence of blocks of three lines, as read by IOHanalyzer's
``read_index_file__IOH`` (``R/readFiles.R``):

1. a header of ``key = value`` pairs separated by commas, values optionally in
   single or double quotes, for example
   ``suite = 'PBO', funcId = 1, DIM = 100, algId = 'EA', algInfo = 'EA'``;
2. a comment line, usually ``%``;
3. the data file, relative to the ``.info`` file, then one entry per run:
   ``<data file>, <instance>:<evaluations>|<best f>[;<attribute>...], ...``.
   The values after ``;`` belong to the header's ``dynamicAttribute`` names
   (separated by ``|``).

Blank lines between blocks are skipped. Several blocks may name the same data
file (IOHexperimenter rewrites the header when the dimension changes); their runs
follow each other in that file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .base import INTEGER_PATTERN, NUMBER_PATTERN, ReaderError

_PAIR = re.compile(
    r"\s*(?P<key>[^=,]+?)\s*=\s*"
    r"(?:\"(?P<dq>[^\"]*)\"|'(?P<sq>[^']*)'|(?P<bare>[^,]*?))"
    r"\s*(?:,|$)"
)
_RUN = re.compile(
    rf"(?P<instance>{INTEGER_PATTERN}):(?P<evals>{INTEGER_PATTERN})\|(?P<rest>.*)"
)
_NUMBER = re.compile(f"(?:{NUMBER_PATTERN})")


@dataclass(frozen=True)
class HeaderValue:
    """One ``key = value`` pair of a block header; ``quoted`` if the value was."""

    key: str
    value: str
    quoted: bool


@dataclass
class InfoRun:
    """One run entry of a block: ``instance:evals|best_y;attributes``."""

    instance: int
    evals: int
    best_y: float
    attributes: list[str]
    line: int


@dataclass
class InfoBlock:  # pylint: disable=too-many-instance-attributes
    """One three-line block of an ``.info`` file."""

    info_path: str
    line: int
    header: list[HeaderValue]
    comment: str
    datafile: str
    runs: list[InfoRun] = field(default_factory=list)
    #: Run entries with zero evaluations, which IOHanalyzer drops (a misuse of
    #: ``reset_problem`` in old loggers wrote them); they name no data.
    empty_runs: list[str] = field(default_factory=list)
    #: True when the run list had an empty entry (a cut-off file).
    incomplete: bool = False

    def get(self, *keys: str) -> HeaderValue | None:
        """The first header value whose key is one of ``keys``."""
        for key in keys:
            for item in self.header:
                if item.key == key:
                    return item
        return None

    def dynamic_attributes(self) -> list[str]:
        """Names of the per-run attributes, from ``dynamicAttribute``."""
        item = self.get("dynamicAttribute")
        if item is None or not item.value.strip():
            return []
        return [name.strip() for name in item.value.split("|")]


def parse_header(text: str, path: str, line: int) -> list[HeaderValue]:
    """The ``key = value`` pairs of a header line; fails with file:line."""
    out: list[HeaderValue] = []
    pos = 0
    text = text.rstrip()
    while pos < len(text):
        m = _PAIR.match(text, pos)
        if m is None or m.end() == pos:
            raise ReaderError(
                f"cannot read the header near {text[pos:][:40]!r}: expected "
                "key = value pairs separated by commas",
                path,
                line,
            )
        pos = m.end()
        quoted = m.group("bare") is None
        value = m.group("dq") if m.group("dq") is not None else m.group("sq")
        if not quoted:
            value = m.group("bare")
        out.append(HeaderValue(m.group("key"), value, quoted))
    keys = [item.key for item in out]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise ReaderError(f"the header names {dupes} more than once", path, line)
    if not out:
        raise ReaderError("empty header line", path, line)
    return out


def _parse_run(item: str, names: list[str], path: str, line: int) -> InfoRun:
    m = _RUN.fullmatch(item)
    if m is None:
        raise ReaderError(
            f"run entry {item!r} is not <instance>:<evaluations>|<best f>", path, line
        )
    parts = m.group("rest").split(";")
    best = parts[0].strip()
    if _NUMBER.fullmatch(best) is None:
        raise ReaderError(f"run entry {item!r}: {best!r} is not a number", path, line)
    attributes = [p.strip() for p in parts[1:]]
    if len(attributes) != len(names):
        raise ReaderError(
            f"run entry {item!r} has {len(attributes)} attribute value(s), the header "
            f"names {len(names)} (dynamicAttribute)",
            path,
            line,
        )
    return InfoRun(
        int(m.group("instance")),
        int(m.group("evals")),
        float(best.lower()),
        attributes,
        line,
    )


def _parse_record(block: InfoBlock, text: str, line: int) -> None:
    items = text.split(",")
    block.datafile = items[0].strip().replace("\\", "/")
    if not block.datafile:
        raise ReaderError("the data file name is empty", block.info_path, line)
    names = block.dynamic_attributes()
    for item in items[1:]:
        item = item.strip()
        if not item:
            block.incomplete = True
            continue
        run = _parse_run(item, names, block.info_path, line)
        if run.evals <= 0:
            block.empty_runs.append(item)
        else:
            block.runs.append(run)


def parse_info_text(text: str, path: str) -> list[InfoBlock]:
    """The blocks of an ``.info`` file's text; ``path`` names it in errors."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[InfoBlock] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        if i + 2 >= len(lines) or not lines[i + 2].strip():
            raise ReaderError(
                "a block needs three lines: header, comment and data file with runs",
                path,
                i + 1,
            )
        header = parse_header(lines[i], path, i + 1)
        block = InfoBlock(path, i + 1, header, lines[i + 1].strip(), "")
        _parse_record(block, lines[i + 2], i + 3)
        blocks.append(block)
        i += 3
    if not blocks:
        raise ReaderError("no blocks found", path)
    return blocks


def parse_info(local: Path, path: str) -> list[InfoBlock]:
    """The blocks of the ``.info`` file ``local``; ``path`` names it in errors."""
    try:
        text = local.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReaderError(
            f"not UTF-8 text ({exc.reason} at byte {exc.start})", path
        ) from exc
    return parse_info_text(text, path)
