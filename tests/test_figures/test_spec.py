import pytest

from iohinspector.figures import Axis, Data, FigureSpec, Legend, Series, Text


def full_spec() -> FigureSpec:
    return FigureSpec(
        data=Data(
            by=["algorithm_name", "dimension"],
            label="{algorithm_name} ($d$ = {dimension})",
            where={"function_id": [1, 2]},
            targets="bbob",
        ),
        venue="ieee",
        span="double",
        height_cm=5,
        title="ECDF",
        text=Text(family="DejaVu Serif", size=8, tick_size=7),
        x=Axis(label="budget", ticks="power", lim=[1, None], grid="both"),
        y=Axis(scale="linear"),
        palette=["#000000", "tab:blue"],
        series={"DE (d = 5)": Series(name="DE", color="red", marker="s", visible=False)},
        legend=Legend(mode="box", loc="lower right", columns=2),
        markers=0,
        formats=["svg", "png"],
        dpi=600,
    )


def test_default_spec_is_valid():
    spec = FigureSpec()
    assert spec.kind == "ecdf" and spec.legend.mode == "direct"
    assert FigureSpec.from_dict({}) == spec


@pytest.mark.parametrize("spec", [FigureSpec(), full_spec()])
def test_json_round_trip(spec):
    text = spec.to_json()
    assert FigureSpec.from_json(text) == spec
    assert FigureSpec.from_json(text).to_json() == text


@pytest.mark.parametrize(
    "value, match",
    [
        ({"colour": "red"}, r"spec: unknown keys \['colour'\]"),
        ({"legend": {"position": "left"}}, r"spec.legend: unknown keys"),
        ({"series": {"DE": {"colour": "red"}}}, r"spec.series.DE: unknown keys"),
        ({"text": 8}, "spec.text: expected an object"),
        ({"series": ["DE"]}, "spec.series: expected an object"),
    ],
)
def test_from_dict_rejects_unknown_keys_and_shapes(value, match):
    with pytest.raises(ValueError, match=match):
        FigureSpec.from_dict(value)


@pytest.mark.parametrize(
    "make, match",
    [
        (lambda: FigureSpec(kind="radar"), "kind 'radar'"),
        (lambda: FigureSpec(venue="nature"), "venue 'nature'"),
        (lambda: FigureSpec(span="triple"), "span"),
        (lambda: FigureSpec(width_cm=0), "width_cm 0: must be positive"),
        (lambda: FigureSpec(palette="rainbow"), "palette 'rainbow'"),
        (lambda: FigureSpec(palette=["red", "nocolour"]), "a name or a list"),
        (lambda: FigureSpec(markers=-1), "markers"),
        (lambda: FigureSpec(formats=[]), "formats: at least one"),
        (lambda: FigureSpec(formats=["jpg"]), "format 'jpg'"),
        (lambda: Data(by=[]), "data.by"),
        (lambda: Data(targets="cocopp"), "data.targets"),
        (lambda: Text(size=-1), "text.size"),
        (lambda: Axis(scale="symlog"), "axis scale"),
        (lambda: Axis(ticks="1e3"), "axis ticks"),
        (lambda: Axis(lim=[1]), r"axis lim \[1\]"),
        (lambda: Series(color="nocolour"), "series color"),
        (lambda: Series(linestyle="wavy"), "series linestyle"),
        (lambda: Series(marker="star"), "series marker"),
        (lambda: Legend(mode="side"), "legend mode"),
        (lambda: Legend(loc="top"), "legend loc"),
        (lambda: Legend(columns=0), "legend columns"),
    ],
)
def test_invalid_values_raise(make, match):
    with pytest.raises(ValueError, match=match):
        make()
