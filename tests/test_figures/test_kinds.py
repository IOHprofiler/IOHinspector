"""The figure kinds of the Workbench (I20): each draws from its frame, which is
its metric's own result, with stable element ids, and writes Python code that
runs."""

import ast
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from iohinspector.figures import (
    Axis,
    Data,
    FigureSpec,
    Layout,
    Panel,
    Series,
    atlas,
    elements,
    panel_frames,
    render,
    to_python,
)
from iohinspector.figures.render import KINDS, _plot
from iohinspector.figures.spec import KINDS as SPEC_KINDS
from iohinspector.metrics import (
    aggregate_convergence,
    get_shapley_values,
    get_value_sample,
    get_win_fractions,
)
from iohinspector.targets import get_ecdf_targets

from .sample import (
    KIND_DATA,
    ONE,
    PARAMETERS,
    gallery_runs,
    parameter_data,
    sample_data,
)

PART = {
    "line": "line",
    "points": "points",
    "bars": "bars",
    "box": "box",
    "violin": "violin",
    "cells": "cells",
}


def spec_of(kind: str, data: Data | None = None, **fields) -> FigureSpec:
    return FigureSpec(
        panels=[Panel(kind, data or KIND_DATA[kind])], formats=["png"], dpi=50, **fields
    )


def plot_of(kind: str, data: Data | None = None):
    spec = spec_of(kind, data)
    runs = gallery_runs(spec)
    (frame,) = panel_frames(spec, runs)
    return _plot(spec.panels[0], runs, frame), frame, runs


def rounded(code: str) -> str:
    return re.sub(r"\d+\.\d{9,}", lambda m: f"{float(m[0]):.8g}", code)


def walk(node):
    yield node
    for child in node["children"]:
        yield from walk(child)


def test_every_kind_is_known_to_the_spec_and_the_renderer():
    assert set(SPEC_KINDS) == set(KINDS)
    assert set(KIND_DATA) | {"ecdf"} == set(KINDS)


@pytest.mark.parametrize("kind", KIND_DATA)
def test_a_kind_draws_the_same_from_its_frame_as_from_the_runs(kind):
    spec = spec_of(kind)
    runs = gallery_runs(spec)
    frames = panel_frames(spec, runs)
    needed = KINDS[kind].columns
    assert all(c in frames[0].columns for c in needed)
    assert render(spec, runs) == render(spec, frames=frames)
    assert render(spec, runs)[:4] == b"\x89PNG"


@pytest.mark.parametrize("kind", KIND_DATA)
def test_the_elements_of_a_kind_have_stable_ids_and_boxes(kind):
    spec = spec_of(kind)
    runs = gallery_runs(spec)
    tree = elements(spec, runs)
    assert tree == elements(spec, runs)
    nodes = {n["id"]: n for n in walk(tree)}
    plot, _, _ = plot_of(kind)
    part = PART[plot.mark]
    series = [
        n for i, n in nodes.items() if i.startswith("p0-s-") and n["kind"] == part
    ]
    assert series and all(len(n["box"]) == 4 for n in series)
    assert all(n["svg"] for n in series)
    assert {"p0-plot", "p0-xaxis", "p0-yaxis", "p0-frame"} <= set(nodes)


@pytest.mark.parametrize("kind", KIND_DATA)
def test_the_python_of_a_kind_runs(kind, tmp_path, monkeypatch):
    spec = spec_of(kind)
    runs = gallery_runs(spec)
    code = to_python(spec, runs)
    ast.parse(code)
    again = to_python(spec, frames=panel_frames(spec, runs))
    assert rounded(again) == rounded(code)  # the metrics' sums differ in the last bit
    monkeypatch.chdir(tmp_path)
    with matplotlib.rc_context():
        exec(compile(code, "figure.py", "exec"), {"data": runs})  # noqa: S102
        plt.close("all")
    assert (tmp_path / "figure.png").stat().st_size > 1000
    assert "# frame = " in code  # the call that computes the numbers


def test_convergence_draws_the_medians_and_quartiles_of_the_metric():
    plot, _, runs = plot_of("convergence")
    summary = aggregate_convergence(runs.filter(**{k: v[0] for k, v in ONE.items()}))
    de = summary.filter(algorithm_name="DE").drop_nulls("median")
    curve = next(c for c in plot.curves if c.label == "DE")
    assert np.allclose(curve.y, de["median"]) and np.allclose(curve.lo, de["q25"])
    assert [(a.kind, a.y) for a in plot.annotations] == [("line", 1e-3)]


def test_a_kind_for_one_pair_says_so_when_given_several():
    spec = spec_of("convergence", Data())
    with pytest.raises(ValueError, match="one function and dimension"):
        render(spec, sample_data())


def test_fixed_budget_has_a_point_per_pair_and_ert_leaves_out_failures():
    plot, frame, _ = plot_of("fixed_budget")
    assert plot.mark == "points" and plot.x["ticklabels"] == [
        "f1 d5",
        "f1 d20",
        "f2 d5",
        "f2 d20",
    ]
    assert all(len(c.x) == 4 for c in plot.curves)
    plot, frame, _ = plot_of("ert", Data(where={"function_id": [1]}, at=1e-12))
    assert frame["ERT"].is_infinite().any()
    assert all(np.isfinite(c.y).all() for c in plot.curves)


def test_the_heatmaps_hold_the_metrics_numbers():
    plot, _, runs = plot_of("win_fractions")
    (cells,) = plot.curves
    names = plot.x["ticklabels"]
    wins = get_win_fractions(runs, by=["algorithm_name"])
    for a, b, w in wins.select(
        "algorithm_name", "algorithm_name_vs", "win_fraction"
    ).iter_rows():
        assert cells.z[names.index(a), names.index(b)] == pytest.approx(w)
    one = plot_of("pairwise_tests", Data(where=ONE))[0].curves[0]
    many = plot_of("pairwise_tests")[0].curves[0]
    assert np.isnan(np.diag(one.z)).all() and np.nanmax(many.z) <= 1


def test_target_by_budget_has_three_modes():
    best, _, _ = plot_of("target_budget")
    assert 1 <= len(best.curves) and best.mark == "cells"
    z = np.array([c.z for c in best.curves])
    assert (np.isfinite(z).sum(axis=0) <= 1).all()  # a cell has one winner
    one, _, _ = plot_of("target_budget", Data(mode="one", focus="DE"))
    assert [c.label for c in one.curves] == ["DE"]
    diff, _, _ = plot_of("target_budget", Data(mode="vs_best", focus="RS"))
    assert np.nanmax(diff.curves[0].z) <= 1 and np.nanmin(diff.curves[0].z) < 0
    assert diff.y["lim"][0] > diff.y["lim"][1]  # harder targets upwards
    with pytest.raises(ValueError, match="mode"):
        plot_of("target_budget", Data(mode="all"))
    with pytest.raises(ValueError, match="focus"):
        plot_of("target_budget", Data(mode="one", focus="x"))


def test_maximising_turns_the_target_axis_round():
    runs = sample_data().with_columns(raw_y=100 - pl.col("raw_y"))
    spec = spec_of("target_budget", Data(maximization=True))
    (frame,) = panel_frames(spec, runs)
    plot = _plot(spec.panels[0], runs, frame)
    assert plot.y["lim"][0] < plot.y["lim"][1]


def test_distributions_are_drawn_from_the_sample_and_its_summaries():
    plot, _, runs = plot_of("box")
    sample = get_value_sample(
        runs.filter(function_id=1, dimension=20), [runs["evaluations"].max()]
    )
    ce = next(c for c in plot.curves if c.label == "CMA-ES")
    assert np.allclose(
        np.sort(ce.y), np.sort(sample.filter(algorithm_name="CMA-ES")["value"])
    )
    violin, _, _ = plot_of("violin")
    assert all(c.mid is not None and (c.hi >= c.lo).all() for c in violin.curves)
    hist, _, _ = plot_of("histogram")
    assert hist.step and hist.curves[0].y[:-1].sum() == 3  # the runs that succeeded


def test_the_box_plot_of_runtimes_leaves_out_failed_runs():
    data = Data(where=ONE, measure="runtime", at=1e-2)
    _, frame, _ = plot_of("box", data)
    assert frame["runtime"].is_finite().all() and frame.height < 25


def test_eaf_has_a_line_per_group_and_percentile_and_a_difference_map():
    plot, _, _ = plot_of("eaf")
    assert [c.label for c in plot.curves][:3] == [
        f"CMA-ES ({p}%)" for p in (10, 50, 90)
    ]
    diff, _, _ = plot_of("eaf_difference")
    (cells,) = diff.curves
    assert cells.label == "DE − GA" and np.isfinite(cells.z).any()
    assert cells.z.shape == (len(cells.y) - 1, len(cells.x) - 1)


def test_the_cumulative_difference_plot_and_the_shapley_bars():
    plot, frame, _ = plot_of("cumulative_difference")
    assert frame["label"][0] == "DE − GA" and plot.curves[0].lo is not None
    bars, _, runs = plot_of("shapley")
    shapley = get_shapley_values(
        runs,
        get_ecdf_targets(runs, "log-linear"),
        ["algorithm_name"],
        divide_by_dimension=True,
    )
    assert [c.label for c in bars.curves] == shapley.sort("shapley", descending=True)[
        "algorithm_name"
    ].to_list()
    assert sum(c.y[0] for c in bars.curves) == pytest.approx(1)


def test_the_parameter_figures_summarise_the_configurations():
    data = Data(**PARAMETERS)
    parallel, frame, _ = plot_of("parallel_coordinates")
    assert len(parallel.curves) == frame.height == 9
    assert parallel.x["ticklabels"] == ["sigma", "mu", "ECDF AUC"]
    assert all(0 <= c.y.min() and c.y.max() <= 1 for c in parallel.curves)
    marginals, _, _ = plot_of("marginals")
    assert [c.label for c in marginals.curves] == ["sigma", "mu"]
    best = frame.group_by("sigma").agg(pl.col("auc").mean()).sort("sigma")["auc"]
    assert marginals.curves[0].y == pytest.approx(best.to_numpy())
    spread, _, _ = plot_of("sensitivity")
    assert [c.y[0] for c in spread.curves] == sorted(
        (c.y[0] for c in spread.curves), reverse=True
    )
    pmap, _, _ = plot_of("parameter_map")
    assert pmap.curves[0].z.shape == (3, 3) and pmap.x["ticklabels"] == [
        "0.1",
        "0.3",
        "1.0",
    ]
    with pytest.raises(ValueError, match="two parameters"):
        plot_of("parameter_map", Data(by=["sigma"]))
    assert data.by == ["sigma", "mu"]


def test_cells_and_categories_have_no_grid_unless_asked():
    runs = parameter_data()
    plain = render(spec_of("sensitivity"), runs)
    grid = FigureSpec(
        panels=[
            Panel(
                "sensitivity",
                KIND_DATA["sensitivity"],
                y=Axis(grid="major"),
                x=Axis(grid="major"),
            )
        ],
        formats=["png"],
        dpi=50,
    )
    assert render(grid, runs) != plain
    with_grid = elements(grid, runs)
    assert {n["id"]: n for n in walk(with_grid)}["p0-grid"]["svg"]


def test_the_atlas_is_a_panel_per_function_and_dimension():
    runs = sample_data()
    spec = atlas(FigureSpec(), runs)
    assert [p.title for p in spec.panels] == ["f1 d5", "f1 d20", "f2 d5", "f2 d20"]
    assert spec.layout == Layout(columns=2, share_x=True, share_y=True)
    assert spec.panels[1].data.where == {"function_id": [1], "dimension": [20]}
    assert spec.legend.shared and render(spec, runs)[:4] == b"%PDF"
    only = atlas(FigureSpec(panels=[Panel(data=Data(where={"dimension": [5]}))]), runs)
    assert len(only.panels) == 2


def test_series_overrides_reach_the_cells():
    spec = spec_of("target_budget", Data(mode="vs_best", focus="CMA-ES"))
    spec.series["CMA-ES"] = Series(color="#cc0000", color_below="#0000cc")
    runs = sample_data()
    assert render(spec, runs) != render(
        spec_of("target_budget", spec.panels[0].data), runs
    )


def test_the_new_data_fields_round_trip_and_are_checked():
    spec = spec_of("histogram")
    assert FigureSpec.from_json(spec.to_json()) == spec
    with pytest.raises(ValueError, match="measure"):
        Data(measure="speed")


def test_the_old_plots_name_kinds_that_exist():
    # iohinspector.plots stays as it is; its docstrings point to these kinds
    import re

    from iohinspector import plots
    from iohinspector.figures.spec import KINDS

    named = set(re.findall(r'``"(\w+)"``', plots.__doc__))
    assert named and named <= set(KINDS)
