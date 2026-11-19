import io

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib.image import imread

from iohinspector.figures import (
    VENUES,
    Axis,
    Data,
    FigureSpec,
    Legend,
    Series,
    figure_size,
    render,
    series_styles,
)
from iohinspector.figures.render import _figure, _prepare, style
from iohinspector.figures.venues import TEX_PT_PER_INCH

from .sample import ALGORITHMS, sample_data


@pytest.fixture(scope="module")
def data():
    return sample_data()


def draw(spec, data):
    """The figure ``render`` would save, with its axes."""
    with style(spec):
        fig = _figure(spec, *_prepare(spec, data))
        fig.draw_without_rendering()
    return fig, fig.axes[0]


def test_venue_widths_follow_the_class_files():
    assert VENUES["lncs"].width_in("double") == pytest.approx(12.2 / 2.54)
    assert VENUES["lncs"].width_in("single") == VENUES["lncs"].width_in("double")
    assert VENUES["ieee"].column_pt * 2 + 12 == VENUES["ieee"].text_pt  # 1pc gap
    assert VENUES["acm"].column_pt * 2 + 24 == pytest.approx(VENUES["acm"].text_pt)
    assert VENUES["elsevier"].column_pt * 2 + 18 == VENUES["elsevier"].text_pt


def test_figure_size():
    assert figure_size(FigureSpec(venue="ieee")) == pytest.approx(
        (252 / TEX_PT_PER_INCH, 0.75 * 252 / TEX_PT_PER_INCH)
    )
    spec = FigureSpec(width_cm=2.54, height_cm=5.08)
    assert figure_size(spec) == pytest.approx((1, 2))


@pytest.mark.parametrize(
    "fmt, magic", [("pdf", b"%PDF"), ("svg", b"<?xml"), ("png", b"\x89PNG")]
)
def test_render_is_deterministic(data, fmt, magic):
    spec = FigureSpec(venue="lncs", formats=[fmt])
    first = render(spec, data)
    assert first.startswith(magic)
    assert render(spec, data) == first


def test_png_has_the_physical_size(data):
    spec = FigureSpec(venue="ieee", formats=["png"], dpi=200)
    height, width = imread(io.BytesIO(render(spec, data))).shape[:2]
    w, h = figure_size(spec)
    assert (width, height) == (round(w * 200), round(h * 200))


def test_user_rc_params_do_not_leak_in(data):
    spec = FigureSpec(formats=["svg"])
    expected = render(spec, data)
    with mpl.rc_context({"lines.linewidth": 5, "font.size": 20, "axes.grid": True}):
        assert render(spec, data, fmt="svg") == expected
    assert render(spec, data, fmt="png") != expected


def test_direct_labels_are_ordered_by_final_value_and_spread(data):
    spec = FigureSpec(venue="acm")
    fig, ax = draw(spec, data)
    by_height = sorted(ax.texts, key=lambda t: -t.get_position()[1])
    names = [t.get_text() for t in by_height]
    assert names == ["CMA-ES", "DE", "GA", "PSO", "RS"]  # by final value
    y = [t.get_position()[1] for t in by_height]
    gap = 1.2 * 9 / (ax.bbox.height * 72 / fig.dpi)
    assert np.all(-np.diff(y) >= gap - 1e-9)
    assert all(t.get_position()[0] > 1 for t in ax.texts)  # right of the axes


def test_lines_follow_the_ecdf(data):
    from iohinspector.figures.ecdf import ecdf_plot

    plot = ecdf_plot(FigureSpec(), data)
    assert [c.label for c in plot.curves] == sorted(ALGORITHMS)
    for curve in plot.curves:
        assert np.all(np.diff(curve.x) >= 0) and np.all(np.diff(curve.y) >= 0)
        assert curve.x[0] == pytest.approx(1 / 20)  # first evaluation, d = 20
    pso = plot.curves[sorted(ALGORITHMS).index("PSO")]
    assert pso.x[-1] == 100  # its budget per dimension
    assert plot.x["lim"] == [pytest.approx(1 / 20), 300]


def test_label_template_and_series_overrides(data):
    spec = FigureSpec(
        data=Data(
            by=["algorithm_name", "dimension"], label="{algorithm_name} d={dimension}"
        ),
        series={
            "DE d=5": Series(name="DE (small)", color="#ff0000", marker="s"),
            "RS d=20": Series(visible=False),
        },
    )
    _, ax = draw(spec, data)
    names = {t.get_text() for t in ax.texts}
    assert "DE (small)" in names and "CMA-ES d=20" in names
    assert "RS d=20" not in names and len(names) == 9
    styles = series_styles(spec, ["DE d=5", "DE d=20"])
    assert styles["DE d=5"]["color"] == "#ff0000"
    assert styles["DE d=20"]["color"] == "#D55E00"  # second palette colour


def test_where_selects_rows(data):
    spec = FigureSpec(
        data=Data(where={"dimension": [5], "algorithm_name": ["DE", "RS"]})
    )
    _, ax = draw(spec, data)
    assert {t.get_text() for t in ax.texts} == {"DE", "RS"}
    with pytest.raises(ValueError, match="selects no rows"):
        render(FigureSpec(data=Data(where={"dimension": [3]})), data)


@pytest.mark.parametrize("mode", ["box", "outside", "none"])
def test_legend_modes(data, mode):
    fig, ax = draw(FigureSpec(legend=Legend(mode=mode)), data)
    legends = [*fig.legends, *([ax.get_legend()] if ax.get_legend() else [])]
    assert not ax.texts
    assert len(legends) == (mode != "none")
    if legends:
        names = [t.get_text() for t in legends[0].get_texts()]
        assert names == ["CMA-ES", "DE", "GA", "PSO", "RS"]  # by final value


@pytest.mark.parametrize(
    "ticks, label, expected",
    [
        ("auto", "$\\log_{10}$(# f-evals / dimension)", "$1$"),
        ("power", "# f-evals / dimension", "$10^{1}$"),
        ("plain", "# f-evals / dimension", "$10$"),
    ],
)
def test_log_tick_formats(data, ticks, label, expected):
    _, ax = draw(FigureSpec(x=Axis(ticks=ticks)), data)
    assert ax.get_xlabel() == label
    assert expected in [t.get_text() for t in ax.get_xticklabels()]


def test_axis_overrides(data):
    spec = FigureSpec(
        data=Data(per_dimension=False),
        x=Axis(label="", scale="linear", lim=[0, None], grid="none"),
        y=Axis(label="ECDF", lim=[0, 0.8]),
        title="f1 and f2",
        markers=0,
    )
    _, ax = draw(spec, data)
    assert ax.get_xscale() == "linear" and ax.get_xlabel() == ""
    assert ax.get_xlim()[0] == 0 and ax.get_ylim() == (0, 0.8)
    assert ax.get_ylabel() == "ECDF" and ax.get_title() == "f1 and f2"
    assert len(ax.lines) == 2 * 5  # a line and its end marker per curve


def test_render_rejects_an_unknown_format(data):
    with pytest.raises(ValueError, match="format 'jpg'"):
        render(FigureSpec(), data, fmt="jpg")
