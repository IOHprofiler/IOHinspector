import pytest

from iohinspector.figures import (
    STYLES,
    Annotation,
    Axis,
    Data,
    FigureSpec,
    Layout,
    Legend,
    Panel,
    Series,
    Text,
    apply_style,
)

# A version 1 spec (I19, I21) as the website stored it: no version key.
V1 = {
    "kind": "ecdf",
    "data": {"by": ["algorithm_name", "dimension"], "where": {"function_id": [1]}},
    "venue": "ieee",
    "span": "double",
    "width_cm": 12,
    "height_cm": 5,
    "title": "ECDF",
    "text": {"family": "DejaVu Serif", "size": 8},
    "x": {"label": "budget", "ticks": "power", "lim": [1, None], "grid": "both"},
    "y": {"scale": "linear"},
    "palette": ["#000000", "tab:blue"],
    "series": {"DE, 5": {"name": "DE", "color": "red", "visible": False}},
    "legend": {"mode": "box", "loc": "lower right", "columns": 2},
    "markers": 0,
    "formats": ["svg"],
    "dpi": 600,
}


def full_spec() -> FigureSpec:
    return FigureSpec(
        panels=[
            Panel(
                data=Data(
                    by=["algorithm_name", "dimension"],
                    label="{algorithm_name} ($d$ = {dimension})",
                    where={"function_id": [1, 2]},
                    targets="bbob",
                ),
                title="f1",
                x=Axis(label="budget", ticks="sci", tick_count=4, grid="both"),
                y=Axis(scale="linear", lim=[0, None]),
                annotations=[
                    Annotation("line", "target", y=0.5),
                    Annotation("arrow", "here", x=10, y=0.2, x2=30, y2=0.1),
                    Annotation("region", x=1, x2=10, color="#cccccc"),
                ],
            ),
            Panel(data=Data(where={"function_id": [3]}), title="f3"),
        ],
        layout=Layout(columns=2, gap=0.3, labels="letters", share_y=True),
        venue="ieee",
        span="double",
        unit="in",
        width=7,
        height=2.5,
        background="dark",
        title="ECDF",
        text=Text(family="DejaVu Serif", size=8, annotation_size=6),
        frame="spines",
        palette="highlight",
        series={
            "DE (d = 5)": Series(
                name="DE",
                color="red",
                color_below="blue",
                marker="s",
                marker_size=5,
                every=3,
                order=2,
                band_opacity=0.3,
                highlight="yours",
            ),
            "RS (d = 5)": Series(highlight="other", visible=False),
        },
        legend=Legend(mode="box", columns=2, order="yours", others=True, shared=True),
        markers=0,
        formats=["svg", "png"],
        dpi=600,
    )


def test_default_spec_is_valid():
    spec = FigureSpec()
    assert spec.version == 2 and spec.panels == [Panel()]
    assert spec.panels[0].kind == "ecdf" and spec.legend.mode == "direct"
    assert FigureSpec.from_dict({"version": 2}) == spec


@pytest.mark.parametrize("spec", [FigureSpec(), full_spec()])
def test_json_round_trip(spec):
    text = spec.to_json()
    assert FigureSpec.from_json(text) == spec
    assert FigureSpec.from_json(text).to_json() == text


def test_version_1_specs_are_upgraded():
    spec = FigureSpec.from_dict(V1)
    assert spec.version == 2 and len(spec.panels) == 1
    panel = spec.panels[0]
    assert panel.data.by == ["algorithm_name", "dimension"] and panel.title == "ECDF"
    assert panel.x == Axis(label="budget", ticks="power", lim=[1, None], grid="both")
    assert (spec.unit, spec.width, spec.height) == ("cm", 12, 5)
    assert spec.title is None and spec.layout == Layout()
    assert spec.series["DE, 5"].visible is False and spec.legend.columns == 2
    assert FigureSpec.from_dict({}) == FigureSpec()  # the v1 default spec
    assert FigureSpec.from_json(spec.to_json()) == spec  # stays v2


def test_unknown_versions_raise():
    with pytest.raises(ValueError, match="version 3: 1 or 2"):
        FigureSpec.from_dict({"version": 3})
    with pytest.raises(ValueError, match="version 1: 2"):
        FigureSpec(version=1)


@pytest.mark.parametrize(
    "value, match",
    [
        ({"version": 2, "colour": "red"}, r"spec: unknown keys \['colour'\]"),
        ({"legend": {"position": "left"}}, r"spec.legend: unknown keys"),
        ({"series": {"DE": {"colour": "red"}}}, r"spec.series.DE: unknown keys"),
        (
            {"version": 2, "panels": [{}, {"kind": "ecdf", "titel": "f"}]},
            r"spec.panels\[1\]: unknown keys \['titel'\]",
        ),
    ],
)
def test_from_dict_rejects_unknown_keys(value, match):
    with pytest.raises(ValueError, match=match):
        FigureSpec.from_dict(value)


@pytest.mark.parametrize(
    "value, match",
    [
        ({"text": 8}, "spec.text: expected an object"),
        ({"series": ["DE"]}, "spec.series: expected an object"),
        ({"version": 2, "panels": {}}, "spec.panels: expected a list"),
        ([], "spec: expected an object"),
    ],
)
def test_from_dict_rejects_wrong_shapes(value, match):
    with pytest.raises(TypeError, match=match):
        FigureSpec.from_dict(value)


@pytest.mark.parametrize(
    "make, match",
    [
        (lambda: Panel(kind="radar"), "kind 'radar'"),
        (lambda: FigureSpec(panels=[]), "panels: at least one"),
        (lambda: FigureSpec(venue="nature"), "venue 'nature'"),
        (lambda: FigureSpec(span="triple"), "span"),
        (lambda: FigureSpec(unit="mm"), "unit 'mm'"),
        (lambda: FigureSpec(width=0), "width 0: must be positive"),
        (lambda: FigureSpec(background="black"), "background"),
        (lambda: FigureSpec(frame="round"), "frame"),
        (lambda: FigureSpec(palette="rainbow"), "palette 'rainbow'"),
        (lambda: FigureSpec(palette=["red", "nocolour"]), "a name or a list"),
        (lambda: FigureSpec(markers=-1), "markers"),
        (lambda: FigureSpec(formats=[]), "formats: at least one"),
        (lambda: FigureSpec(formats=["jpg"]), "format 'jpg'"),
        (lambda: Layout(columns=0), "layout columns"),
        (lambda: Layout(gap=-1), "layout gap"),
        (lambda: Layout(labels="roman"), "layout labels"),
        (lambda: Data(by=[]), "data.by"),
        (lambda: Data(targets="cocopp"), "data.targets"),
        (lambda: Text(size=-1), "text.size"),
        (lambda: Text(annotation_size=0), "text.annotation_size"),
        (lambda: Axis(scale="symlog"), "axis scale"),
        (lambda: Axis(ticks="1e3"), "axis ticks"),
        (lambda: Axis(tick_count=0), "axis tick_count"),
        (lambda: Axis(lim=[1]), r"axis lim \[1\]"),
        (lambda: Annotation("circle"), "annotation kind"),
        (lambda: Annotation("text", "a", x=1), "annotation text: needs x, y"),
        (lambda: Annotation("line", x=1, y=2), "needs x or y"),
        (lambda: Annotation("region", x=1, y2=2), "needs x, x2 or y, y2"),
        (lambda: Annotation(color="nocolour", x=1, y=1), "annotation color"),
        (lambda: Series(color="nocolour"), "series color"),
        (lambda: Series(color_below="nocolour"), "series color_below"),
        (lambda: Series(linestyle="wavy"), "series linestyle"),
        (lambda: Series(marker="star"), "series marker"),
        (lambda: Series(every=0), "series every"),
        (lambda: Series(band_opacity=2), "series band_opacity"),
        (lambda: Series(highlight="mine"), "series highlight"),
        (lambda: Legend(mode="side"), "legend mode"),
        (lambda: Legend(loc="top"), "legend loc"),
        (lambda: Legend(columns=0), "legend columns"),
        (lambda: Legend(order="random"), "legend order"),
    ],
)
def test_invalid_values_raise(make, match):
    with pytest.raises(ValueError, match=match):
        make()


def test_styles_are_spec_fragments():
    spec = FigureSpec(width=8.4, height=5.6, panels=[Panel(), Panel()])
    paper = apply_style(spec, "paper")
    assert paper.text.size == 8 and paper.text.tick_size == 7
    assert (paper.width, paper.height) == (8.4, 5.6)  # paper has no size
    slides = apply_style(paper, "slides-dark")
    assert (slides.width, slides.background, slides.palette) == (
        24,
        "dark",
        "highlight",
    )
    assert slides.text.size == 16
    own = apply_style(
        spec, {"panel": {"x": {"grid": "none"}}, "legend": {"mode": "box"}}
    )
    assert [p.x.grid for p in own.panels] == ["none", "none"]
    assert own.legend == Legend(mode="box") and own.width == 8.4
    for name in STYLES:
        assert apply_style(FigureSpec(), name).version == 2
