"""The Studio element model (I35): frames, panels, annotations, the extra series
and legend fields, backgrounds and palettes, and the element tree."""

import re

import numpy as np
import polars as pl
import pytest

from iohinspector.figures import (
    Annotation,
    Axis,
    Data,
    FigureSpec,
    Layout,
    Legend,
    Panel,
    Series,
    elements,
    panel_frames,
    render,
    series_styles,
)
from iohinspector.figures.ecdf import ECDF
from iohinspector.figures.render import _figure, _prepare, style
from iohinspector.figures.venues import HIGHLIGHT, PALETTES

from .sample import ALGORITHMS, sample_data

F1, F2 = (Panel(data=Data(where={"function_id": [f]}), title=f"f{f}") for f in (1, 2))


@pytest.fixture(scope="module")
def data():
    return sample_data()


def draw(spec, data=None, frames=None):
    with style(spec):
        return _figure(spec, *_prepare(spec, data, frames))


def walk(node):
    yield node
    for child in node["children"]:
        yield from walk(child)


def test_the_ecdf_frame_is_get_ecdf_with_the_budget(data):
    frame = panel_frames(FigureSpec(), data)[0]
    assert {"algorithm_name", "evaluations", "ecdf"} <= set(frame.columns)
    last = frame.group_by("algorithm_name").agg(pl.col("evaluations").max())
    assert last["evaluations"].to_list() == [300.0] * len(ALGORITHMS)


@pytest.mark.parametrize("fmt", ["svg", "png"])
def test_render_from_frames_matches_render_from_runs(data, fmt):
    spec = FigureSpec(formats=[fmt])
    assert render(spec, frames=panel_frames(spec, data)) == render(spec, data)


def test_render_from_a_frame_with_labels_and_a_band():
    """A frame from elsewhere (the website): labels, steps, a band; drawn as
    given, from its smallest positive x to its largest."""
    frame = pl.DataFrame(
        {
            "label": ["A"] * 3 + ["B"] * 3,
            "evaluations": [0.0, 2, 50] * 2,
            "ecdf": [0.0, 0.5, 0.9, 0.0, 0.2, 0.4],
            "lo": [0.0, 0.4, 0.8, 0.0, 0.1, 0.3],
            "hi": [0.0, 0.6, 1.0, 0.0, 0.3, 0.5],
        }
    )
    spec = FigureSpec(series={"B": Series(band=False)})
    fig = draw(spec, frames=frame)
    ax = fig.axes[0]
    assert ax.get_xlim() == (2, 50)
    assert len(ax.collections) == 1  # A's band only
    a = next(line for line in ax.lines if line.get_gid() == "p0-s-A-line")
    np.testing.assert_array_equal(a.get_xdata(), [2, 50])
    np.testing.assert_array_equal(a.get_ydata(), [0.5, 0.9])
    with pytest.raises(ValueError, match=r"no columns \['ecdf'\]"):
        render(spec, frames=frame.drop("ecdf"))
    with pytest.raises(ValueError, match="no label column"):
        render(spec, frames=frame.drop("label"))
    with pytest.raises(ValueError, match="runs .* or the frames"):
        render(spec)


def test_panels_share_looks_and_take_their_own_data(data):
    spec = FigureSpec(
        panels=[F1, F2, Panel()],
        layout=Layout(columns=2, gap=0.5, labels="letters", share_y=True),
        title="BBOB",
    )
    fig = draw(spec, data)
    axes = fig.axes
    assert len(axes) == 3 and fig.get_suptitle() == "BBOB"
    assert [ax.get_title(loc="left") for ax in axes] == ["(a) f1", "(b) f2", "(c)"]
    colors = [{ln.get_gid(): ln.get_color() for ln in ax.lines} for ax in axes]
    assert colors[0]["p0-s-RS-line"] == colors[1]["p1-s-RS-line"]
    assert not any(t.label1.get_visible() for t in axes[1].yaxis.get_major_ticks())
    assert axes[1].get_ylabel() == "" and axes[2].get_ylabel()
    frames = panel_frames(spec, data)
    assert render(spec, frames=frames) == render(spec, data)
    assert render(spec, frames=[frames[0], None, frames[2]], data=data)
    with pytest.raises(ValueError, match="one per panel, 3; got 2"):
        render(spec, frames=frames[:2])


def test_annotations(data):
    panel = Panel(
        annotations=[
            Annotation("text", "note", x=1, y=0.5),
            Annotation("arrow", "best", x=10, y=0.8, x2=30, y2=0.4),
            Annotation("line", "target", y=0.5, color="#ff0000"),
            Annotation("region", x=1, x2=10),
        ]
    )
    ax = draw(FigureSpec(panels=[panel]), data).axes[0]
    gids = {a.get_gid() for a in [*ax.texts, *ax.lines, *ax.patches]}
    assert {"p0-a0", "p0-a1", "p0-a2-0", "p0-a2-1", "p0-a3"} <= gids
    line = next(a for a in ax.lines if a.get_gid() == "p0-a2-0")
    assert line.get_color() == "#ff0000" and list(line.get_ydata()) == [0.5, 0.5]


def test_series_fields(data):
    spec = FigureSpec(
        series={
            "DE": Series(marker_size=6, every=5, order=5),
            "GA": Series(order=-1),
        }
    )
    ax = draw(spec, data).axes[0]
    drawn = [ln.get_gid() for ln in ax.lines if ln.get_gid().endswith("-line")]
    assert drawn[0] == "p0-s-GA-line" and drawn[-1] == "p0-s-DE-line"
    markers = next(ln for ln in ax.lines if ln.get_gid() == "p0-s-DE-markers")
    line = next(ln for ln in ax.lines if ln.get_gid() == "p0-s-DE-line")
    assert markers.get_markersize() == 6
    np.testing.assert_array_equal(markers.get_xdata(), line.get_xdata()[::5])


@pytest.mark.parametrize(
    "order, expected",
    [
        ("value", ["CMA-ES", "DE", "GA", "PSO", "RS"]),
        ("yours", ["PSO", "CMA-ES", "DE", "GA", "RS"]),
        ("name", ["CMA-ES", "DE", "GA", "PSO", "RS"]),
        ("drawing", ["PSO", "RS", "GA", "DE", "CMA-ES"]),
    ],
)
def test_legend_order(data, order, expected):
    spec = FigureSpec(
        series={"PSO": Series(highlight="yours")},
        legend=Legend(mode="box", order=order),
    )
    texts = draw(spec, data).axes[0].get_legend().get_texts()
    assert [t.get_text() for t in texts] == expected


def test_highlight_palette_and_n_others(data):
    spec = FigureSpec(
        palette="highlight",
        series={
            "PSO": Series(highlight="yours"),
            "GA": Series(highlight="other"),
            "RS": Series(highlight="other"),
        },
        legend=Legend(mode="box", order="yours", others=True),
    )
    ax = draw(spec, data).axes[0]
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert texts == ["PSO", "CMA-ES", "DE", "2 others"]
    lines = {ln.get_gid(): ln for ln in ax.lines}
    assert lines["p0-s-PSO-line"].get_color() == HIGHLIGHT["white"]["yours"]
    assert lines["p0-s-PSO-line"].get_linewidth() == 1.5
    assert lines["p0-s-GA-line"].get_color() == HIGHLIGHT["white"]["other"]
    assert lines["p0-s-CMA-ES-line"].get_color() == PALETTES["highlight"][0]
    assert lines["p0-s-DE-line"].get_color() == PALETTES["highlight"][1]
    drawn = [g for g in lines if g.endswith("-line")]
    assert drawn[:2] == ["p0-s-GA-line", "p0-s-RS-line"]  # others behind
    assert drawn[-1] == "p0-s-PSO-line"  # yours in front


def test_family_palette():
    spec = FigureSpec(palette="family")
    labels = ["A 5", "A 20", "B 5", "B 20"]
    looks = series_styles(spec, labels, ["A", "A", "B", "B"])
    assert [looks[k]["color"] for k in labels] == [
        PALETTES["family"][i] for i in (0, 0, 1, 1)
    ]
    assert [looks[k]["linestyle"] for k in labels] == ["-", "--", "-", "--"]


def test_series_ids_are_unique_slugs():
    looks = series_styles(FigureSpec(), ["a b", "a_b", "$x$", "é"])
    assert [s["id"] for s in looks.values()] == ["a_b", "a_b_", "x", "series"]


def test_shared_and_outside_legends(data):
    layout = Layout(columns=2)
    for legend in (Legend(mode="box", shared=True, columns=5), Legend(mode="outside")):
        spec = FigureSpec(panels=[F1, F2], layout=layout, span="double", legend=legend)
        fig = draw(spec, data)
        assert len(fig.legends) == 1 and not any(ax.get_legend() for ax in fig.axes)
        assert fig.legends[0].get_gid() == "legend"


def test_backgrounds_frames_and_ticks(data):
    spec = FigureSpec(
        background="dark",
        frame="spines",
        palette="okabe-ito",
        panels=[Panel(x=Axis(ticks="sci", tick_count=3), y=Axis(tick_count=3))],
        formats=["svg"],
    )
    fig = draw(spec, data)
    ax = fig.axes[0]
    assert fig.get_facecolor()[:3] == pytest.approx(
        (0x12 / 255, 0x13 / 255, 0x16 / 255)
    )
    assert [s for s, v in ax.spines.items() if v.get_visible()] == ["left", "bottom"]
    assert "1e2" in {t.get_text() for t in ax.get_xticklabels()}
    assert len(ax.get_yticks()) <= 4
    looks = series_styles(spec, list("abcdefg"))
    assert looks["g"]["color"] == "#FFFFFF"  # okabe-ito's black, on a dark slide
    transparent = render(FigureSpec(background="none", formats=["svg"]), data)
    assert b"fill: #ffffff" not in transparent.split(b'id="p0"')[0]


def test_every_element_has_an_id_in_the_svg(data):
    spec = FigureSpec(
        panels=[
            Panel(title="f1", annotations=[Annotation("line", "t", y=0.5)]),
            F2,
        ],
        layout=Layout(columns=2),
        title="ECDF",
        formats=["svg"],
    )
    tree = elements(spec, data)
    svg = render(spec, data).decode()
    ids = set(re.findall(r' id="([^"]+)"', svg))
    nodes = list(walk(tree))
    assert all(set(n["svg"]) <= ids for n in nodes)
    kinds = {n["id"]: n["kind"] for n in nodes}
    for i in (0, 1):
        for part, kind in [
            ("", "panel"),
            ("-plot", "plot"),
            ("-frame", "frame"),
            ("-grid", "grid"),
            ("-title", "title"),
            ("-xaxis", "axis"),
            ("-xlabel", "label"),
            ("-xticks", "ticks"),
            ("-yticks", "ticks"),
            ("-s-CMA-ES", "series"),
            ("-s-CMA-ES-line", "line"),
            ("-s-CMA-ES-end", "end"),
            ("-s-CMA-ES-markers", "markers"),
            ("-s-CMA-ES-label", "label"),
        ]:
            assert kinds[f"p{i}{part}"] == kind
    assert kinds["title"] == "title" and kinds["p0-a0"] == "annotation"
    assert render(spec, data) == render(spec, data)


def test_boxes_are_in_points_from_the_top_left(data):
    spec = FigureSpec(width=8, height=6)
    tree = elements(spec, data)
    w, h = 8 / 2.54 * 72, 6 / 2.54 * 72
    assert tree["box"] == pytest.approx([0, 0, w, h], abs=0.01)
    nodes = {n["id"]: n for n in walk(tree)}
    plot = nodes["p0-plot"]["box"]
    x0, y0, x1, y1 = plot
    assert 0 < x0 < x1 < w and 0 < y0 < y1 < h
    assert nodes["p0-xlabel"]["box"][1] > y1  # below the plot area
    assert nodes["p0-ylabel"]["box"][2] < x0  # left of it
    assert nodes["p0-s-CMA-ES-label"]["box"][0] >= x1  # direct labels at the right
    series = [n["label"] for n in nodes["p0"]["children"] if n["kind"] == "series"]
    assert series == sorted(ALGORITHMS, reverse=True)  # front first
    assert nodes["p0-s-DE"]["name"] == "DE" and nodes["p0"]["panel"] == 0
    for n in nodes.values():
        b = n["box"]
        assert b[0] <= b[2] and b[1] <= b[3]


def test_frames_of_kinds_are_validated():
    assert ECDF.columns == ("evaluations", "ecdf")
