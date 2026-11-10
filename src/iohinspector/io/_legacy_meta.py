"""What a legacy ``.info`` block says about its runs: runs columns and descriptors.

Header keys and where they go:

| Key | Stored as |
|---|---|
| ``suite`` | ``runs.suite`` (``""`` when absent) |
| ``funcId`` | ``runs.function_id``; a non-integer value gets an assigned id |
| ``funcName`` | ``runs.function_name``; else a non-integer ``funcId`` is the name |
| ``DIM`` | ``runs.dimension`` |
| ``maximization`` or ``Maximization`` | ``runs.maximization`` |
| ``algId``, ``algInfo`` | descriptors ``algorithm_name``, ``algorithm_info`` |
| ``dynamicAttribute`` | names of the per-run descriptors (origin ``run``) |
| every other key | a descriptor with origin ``experiment`` |

Without a maximisation key the direction follows IOHanalyzer (``R/DataSet.R``):
minimisation when the suite name contains ``bbob``, maximisation otherwise.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ._compat import INTEGER_PATTERN, NUMBER_PATTERN, ReaderError
from ._legacy_info import HeaderValue, InfoBlock, InfoRun

_INTEGER = re.compile(INTEGER_PATTERN)
_NUMBER = re.compile(f"(?:{NUMBER_PATTERN})")
#: Values R's ``as.logical`` accepts, which IOHanalyzer uses for ``maximization``.
_LOGICAL = {
    "T": True,
    "TRUE": True,
    "true": True,
    "True": True,
    "F": False,
    "FALSE": False,
    "false": False,
    "False": False,
}
_CONSUMED = {
    "suite",
    "funcId",
    "funcName",
    "DIM",
    "maximization",
    "Maximization",
    "algId",
    "algInfo",
    "dynamicAttribute",
}


def _is_integer(text: str) -> bool:
    return _INTEGER.fullmatch(text.strip()) is not None


def header_value(item: HeaderValue) -> Any:
    """The typed value of a header pair: quoted values are strings."""
    if item.key == "constrained" and item.value.strip() in _LOGICAL:
        return _LOGICAL[item.value.strip()]
    if item.quoted:
        return item.value
    text = item.value.strip()
    if _is_integer(text):
        return int(text)
    if _NUMBER.fullmatch(text):
        return float(text.lower())
    return text


def run_value(token: str) -> Any:
    """A per-run attribute: the logger writes doubles, so numbers are floats."""
    return float(token.lower()) if _NUMBER.fullmatch(token) else token


@dataclass
class FunctionIds:
    """Ids for ``funcId`` values that are not integers, unique within the dataset.

    They are numbered from one above the largest integer ``funcId``, in the byte
    order of the names.
    """

    assigned: dict[str, int] = field(default_factory=dict)

    @classmethod
    def of(cls, blocks: Iterable[InfoBlock]) -> FunctionIds:
        numbers: set[int] = set()
        names: set[str] = set()
        for block in blocks:
            item = block.get("funcId")
            if item is None:
                continue
            text = item.value.strip()
            if _is_integer(text):
                numbers.add(int(text))
            else:
                names.add(text)
        start = max(numbers, default=0) + 1
        ordered = sorted(names, key=lambda s: s.encode("utf-8"))
        return cls({name: start + i for i, name in enumerate(ordered)})

    def describe(self) -> str:
        pairs = ", ".join(f"{k!r} = {v}" for k, v in self.assigned.items())
        return f"function ids assigned to funcId values that are not integers: {pairs}"


@dataclass
class BlockMeta:  # pylint: disable=too-many-instance-attributes
    """The runs columns and descriptors one ``.info`` block gives its runs."""

    block: InfoBlock
    suite: str
    function_id: int
    function_name: str | None
    dimension: int
    maximization: bool
    algorithm_name: str
    algorithm_info: str
    experiment: list[tuple[str, Any, str]] = field(default_factory=list)
    run_attributes: list[str] = field(default_factory=list)
    warnings: list[tuple[str, str]] = field(default_factory=list)

    def run_row(  # pylint: disable=too-many-arguments
        self,
        rid: int,
        info: InfoRun | None,
        source: str,
        index: int,
        sha256: str | None,
    ) -> dict[str, Any]:
        """One row of ``runs.parquet``."""
        return {
            "run_id": rid,
            "suite": self.suite,
            "function_id": self.function_id,
            "function_name": self.function_name,
            "instance": None if info is None else info.instance,
            "dimension": self.dimension,
            "maximization": self.maximization,
            "evals_used": None if info is None else info.evals,
            "best_y": None if info is None else info.best_y,
            "n_objectives": 1,
            "source_file": source,
            "source_run_index": index,
            "source_rows_sha256": sha256,
        }


def _required(block: InfoBlock, key: str) -> HeaderValue:
    item = block.get(key)
    if item is None:
        raise ReaderError(f"the header has no {key}", block.info_path, block.line)
    return item


def _dimension(block: InfoBlock) -> int:
    text = _required(block, "DIM").value.strip()
    if not _is_integer(text) or not 1 <= int(text) < 2**31:
        raise ReaderError(
            f"DIM = {text!r} is not a positive integer", block.info_path, block.line
        )
    return int(text)


def _maximization(block: InfoBlock, suite: str) -> tuple[bool, list[tuple[str, str]]]:
    item = block.get("maximization", "Maximization")
    if item is None:
        guess = re.search("bbob", suite, re.IGNORECASE) is None
        direction = "maximisation" if guess else "minimisation"
        message = (
            f"the header does not say the direction; {direction} is assumed as in "
            "IOHanalyzer (minimisation when the suite name contains bbob)"
        )
        return guess, [("maximization_inferred", message)]
    text = item.value.strip()
    if text not in _LOGICAL:
        raise ReaderError(
            f"{item.key} = {text!r} is not T, F, TRUE or FALSE",
            block.info_path,
            block.line,
        )
    return _LOGICAL[text], []


def _function(block: InfoBlock, ids: FunctionIds) -> tuple[int, str | None, list[Any]]:
    """Function id, name, and the descriptors that keep the source text."""
    text = _required(block, "funcId").value.strip()
    name_item = block.get("funcName")
    name = None if name_item is None else name_item.value
    if _is_integer(text):
        if not -(2**31) <= int(text) < 2**31:
            raise ReaderError(
                f"funcId = {text} does not fit in int32", block.info_path, block.line
            )
        return int(text), name, []
    if name is None:
        return ids.assigned[text], text, []
    return ids.assigned[text], name, [("funcId", text, "experiment")]


def _record_warnings(block: InfoBlock) -> list[tuple[str, str]]:
    line = block.line + 2
    out = []
    if block.incomplete:
        message = (
            f"line {line}: the run list has empty entries (a cut-off file); "
            "they are skipped"
        )
        out.append(("incomplete_info_record", message))
    if block.empty_runs:
        message = (
            f"line {line}: entries with 0 evaluations are not matched to data "
            f"runs, as in IOHanalyzer: {', '.join(block.empty_runs)}"
        )
        out.append(("info_run_without_evaluations", message))
    return out


def block_meta(block: InfoBlock, ids: FunctionIds) -> BlockMeta:
    """Runs columns and descriptors of ``block``."""
    suite_item = block.get("suite")
    suite = "" if suite_item is None else suite_item.value
    function_id, function_name, extra = _function(block, ids)
    maximization, warnings = _maximization(block, suite)
    alg = block.get("algId")
    if alg is None:
        warnings.append(("missing_algorithm", "the header has no algId; stored as ''"))
    warnings += _record_warnings(block)
    info = block.get("algInfo")
    experiment = extra + [
        (item.key, header_value(item), "experiment")
        for item in block.header
        if item.key not in _CONSUMED
    ]
    comment = block.comment.lstrip("%").strip()
    if comment:
        experiment.append(("comment", comment, "experiment"))
    return BlockMeta(
        block=block,
        suite=suite,
        function_id=function_id,
        function_name=function_name,
        dimension=_dimension(block),
        maximization=maximization,
        algorithm_name="" if alg is None else alg.value,
        algorithm_info="" if info is None else info.value,
        experiment=experiment,
        run_attributes=block.dynamic_attributes(),
        warnings=warnings,
    )
