"""Helpers to read the golden reference outputs (WP I03).

    from tests.golden.load import golden, golden_cocopp, versions, fixture_path

    df = golden("ioh040_min", "RT_summary_par1")      # IOHanalyzer, polars DataFrame
    coco = golden_cocopp("art_51targets")             # cocopp reference for the COCO fixture

The outputs are references, not truths: where docs/definitions says IOHanalyzer is wrong, a test
asserts the agreed value and names the difference (see tests/golden/README.md).
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import polars as pl

GOLDEN_DIR = Path(__file__).resolve().parent
IOHANALYZER_DIR = GOLDEN_DIR / "iohanalyzer"
COCOPP_DIR = GOLDEN_DIR / "cocopp" / "coco"
#: COCO fixtures with cocopp references; ``coco_old`` is not an IOHanalyzer fixture.
COCO_FIXTURES = ("coco", "coco_old")
FIXTURES_DIR = GOLDEN_DIR / "fixtures"

FIXTURES = ("ioh040_min", "ioh040_max", "ioh040_cons", "ioh040_pos", "extdata", "coco")


def fixture_path(name: str) -> Path:
    """Folder with the raw logs of a fixture (input to readers)."""
    path = FIXTURES_DIR / name
    if not path.is_dir():
        raise FileNotFoundError(f"unknown golden fixture {name!r}; have {FIXTURES}")
    return path


@cache
def versions() -> dict:
    """versions.json of the IOHanalyzer references: package versions, seed, call per file, skips."""
    return json.loads((IOHANALYZER_DIR / "versions.json").read_text())


@cache
def cocopp_versions() -> dict:
    """versions.json of the cocopp references."""
    return json.loads((COCOPP_DIR / "versions.json").read_text())


def metrics(fixture: str) -> list[str]:
    """Names of the metrics that have a reference for ``fixture``."""
    return sorted(p.stem for p in (IOHANALYZER_DIR / fixture).glob("*.parquet"))


def call(fixture: str, metric: str) -> str:
    """The IOHanalyzer call that produced ``fixture/metric``."""
    return versions()["outputs"][f"{fixture}/{metric}.parquet"]


def golden(fixture: str, metric: str) -> pl.DataFrame:
    """IOHanalyzer reference output ``metric`` for ``fixture``."""
    path = IOHANALYZER_DIR / fixture / f"{metric}.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"no golden output {fixture}/{metric}; versions()['skipped'] lists IOHanalyzer errors"
        )
    return pl.read_parquet(path)


def golden_cocopp(name: str, fixture: str = "coco") -> pl.DataFrame:
    """cocopp reference output ``name`` for a COCO fixture (``coco`` or ``coco_old``;
    the names of ``coco_old`` end in ``_2D`` or ``_3D``)."""
    return pl.read_parquet(GOLDEN_DIR / "cocopp" / fixture / f"{name}.parquet")
