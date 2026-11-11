"""The COCO reader against cocopp 2.9.1 (``tests/golden/cocopp``, WP I03).

The runtimes and aRT here are computed from the reader's tables with plain
polars, so the test checks what the reader stores: instances, run order, Δf in
``raw_y``, evaluation counts and the merged ``.dat``/``.tdat`` rows.

cocopp looks targets up on its grid of levels 10^(i/5) and takes the closest
level *not larger* than the requested target (``docs/definitions/alignment.md``,
"How cocopp implements it"). It computes a level as ``10 ** (i / 5)``, which can
be one ulp above the requested ``10 ** (2 - k / 5)``; it then takes the next,
stricter level. ``cocopp_level`` reproduces that, and the comparison with cocopp
uses it. iohinspector's definition (AL-2, AL-8) hits the target itself, with no
grid and no tolerance; ``test_agreed_rule_differs_only_by_the_grid_lookup`` names
the cells where the two differ.
"""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from iohinspector.io.coco import read_coco
from tests.golden.load import fixture_path, golden_cocopp

ROOT = Path(__file__).resolve().parents[1]
#: (reader input, golden fixture, file suffix, dimension)
CASES = [
    (ROOT / "test_coco_data", "coco", "", 2),
    (fixture_path("coco_old"), "coco_old", "_2D", 2),
    (fixture_path("coco_old"), "coco_old", "_3D", 3),
]
IDS = ["coco", "coco_old-2D", "coco_old-3D"]


def cocopp_level(target: float) -> float:
    """The grid level cocopp's ``detEvals`` uses for ``target``."""
    i = round(5 * math.log10(target))
    level = 10 ** (i / 5)
    return level if level <= target else 10 ** ((i - 1) / 5)


def runs_and_trace(folder: Path, dimension: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    result = read_coco(folder, "cocopp")
    runs = result.runs.filter(pl.col("dimension") == dimension).sort("source_run_index")
    trace = result.trace.join(runs.select("run_id", "source_run_index"), on="run_id")
    return runs, trace


def hitting_times(trace: pl.DataFrame, runs: int, target: float) -> list[float | None]:
    """First evaluation per run with ``raw_y <= target`` (None when never)."""
    hits = (
        trace.filter(pl.col("raw_y") <= target)
        .sort("row")
        .group_by("source_run_index")
        .agg(pl.col("evaluations").first())
    )
    found = dict(zip(hits["source_run_index"], hits["evaluations"]))
    return [None if found.get(i) is None else float(found[i]) for i in range(runs)]


def same(a: float | None, b: float) -> bool:
    return (a is None and math.isnan(b)) or a == b


@pytest.mark.parametrize(("folder", "fixture", "suffix", "dimension"), CASES, ids=IDS)
def test_instances_in_cocopp_order(folder, fixture, suffix, dimension):
    runs, _ = runs_and_trace(folder, dimension)
    ref = golden_cocopp(f"runtimes_51targets{suffix}", fixture)
    order = ref.filter(pl.col("target") == 100.0).sort("run")["instance"].to_list()
    assert runs["instance"].to_list() == order


@pytest.mark.parametrize(("folder", "fixture", "suffix", "dimension"), CASES, ids=IDS)
def test_runtimes_match_cocopp(folder, fixture, suffix, dimension):
    runs, trace = runs_and_trace(folder, dimension)
    ref = golden_cocopp(f"runtimes_51targets{suffix}", fixture)
    for target, group in ref.group_by("target"):
        mine = hitting_times(trace, runs.height, cocopp_level(target[0]))
        theirs = group.sort("run")["evals"].to_list()
        assert all(same(a, b) for a, b in zip(mine, theirs)), (target, mine, theirs)


@pytest.mark.parametrize("table", ["art_51targets", "art_table_targets"])
@pytest.mark.parametrize(("folder", "fixture", "suffix", "dimension"), CASES, ids=IDS)
def test_art_matches_cocopp(folder, fixture, suffix, dimension, table):
    """aRT = (evaluations of successful runs to the target + evaluations of the
    unsuccessful runs) / successes, with ``evals_used`` from the ``.info`` file."""
    runs, trace = runs_and_trace(folder, dimension)
    budgets = runs["evals_used"].to_list()
    ref = golden_cocopp(f"{table}{suffix}", fixture)
    for row in ref.iter_rows(named=True):
        times = hitting_times(trace, runs.height, cocopp_level(row["target"]))
        successes = sum(t is not None for t in times)
        total = sum(t if t is not None else b for t, b in zip(times, budgets))
        assert successes == row["n_success"]
        assert row["n_runs"] == runs.height
        if successes == 0:
            assert math.isinf(row["ert"])
        else:
            assert total / successes == pytest.approx(row["ert"], rel=1e-12)


@pytest.mark.parametrize(("folder", "fixture", "suffix", "dimension"), CASES, ids=IDS)
def test_agreed_rule_differs_only_by_the_grid_lookup(
    folder, fixture, suffix, dimension
):
    """Hitting the target itself (AL-2, AL-8) differs from cocopp only at targets
    whose grid level cocopp computes one ulp too large, and only by being earlier."""
    runs, trace = runs_and_trace(folder, dimension)
    ref = golden_cocopp(f"runtimes_51targets{suffix}", fixture)
    shifted = 0
    for target, group in ref.group_by("target"):
        mine = hitting_times(trace, runs.height, target[0])
        theirs = group.sort("run")["evals"].to_list()
        if cocopp_level(target[0]) == target[0]:
            assert all(same(a, b) for a, b in zip(mine, theirs)), target
            continue
        shifted += 1
        for a, b in zip(mine, theirs):
            assert same(a, b) or (a is not None and (math.isnan(b) or a < b))
    assert shifted > 0  # the grid lookup matters on every fixture
