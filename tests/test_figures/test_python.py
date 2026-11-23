"""``to_python``: the generated matplotlib code runs and draws the figure."""

import ast

import matplotlib
import matplotlib.pyplot as plt
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
    panel_frames,
    to_python,
)

from .sample import sample_data

F1 = Panel(
    data=Data(where={"function_id": [1]}),
    title="f1",
    x=Axis(ticks="sci", tick_count=4, grid="both"),
    annotations=[
        Annotation("line", "target", y=0.5),
        Annotation("arrow", "PSO", x=30, y=0.6, x2=3, y2=0.85),
        Annotation("region", x=0.1, x2=1),
        Annotation("text", "note", x=1, y=0.9),
    ],
)
SPECS = {
    "default": FigureSpec(),
    "panels": FigureSpec(
        panels=[F1, Panel(data=Data(where={"function_id": [2]}), title="f2")],
        layout=Layout(columns=2, gap=0.4, labels="letters", share_y=True),
        span="double",
        title="BBOB",
        palette="highlight",
        frame="spines",
        series={
            "PSO": Series(highlight="yours", every=4),
            "RS": Series(highlight="other"),
            "GA": Series(highlight="other"),
        },
        legend=Legend(mode="box", order="yours", others=True, shared=True, columns=4),
        formats=["png"],
    ),
    "outside": FigureSpec(
        legend=Legend(mode="outside"), background="dark", formats=["svg"]
    ),
}


@pytest.fixture(scope="module")
def data():
    return sample_data()


@pytest.mark.parametrize("name", SPECS)
def test_the_code_runs_and_saves_the_figure(data, tmp_path, monkeypatch, name):
    spec = SPECS[name]
    code = to_python(spec, data)
    ast.parse(code)
    monkeypatch.chdir(tmp_path)
    with matplotlib.rc_context():
        exec(compile(code, "figure.py", "exec"), {"data": data})  # noqa: S102
        plt.close("all")
    fmt = spec.formats[0]
    assert (tmp_path / f"figure.{fmt}").stat().st_size > 1000
    assert to_python(spec, frames=panel_frames(spec, data)) == code


def test_the_code_carries_the_rc_params_and_the_style_table(data):
    code = to_python(SPECS["panels"], data)
    assert "mpl.rcParams.update(" in code and "'font.size': 9" in code
    assert "'PSO': {'name': 'PSO', 'color': '#c05800'" in code
    assert "'markevery': 4" in code  # every n points
    assert "get_ecdf(" in code and "is_in([1])" in code
    assert "ax.axhline(0.5" in code and "ax.axvspan(0.1, 1" in code
    assert "handle('GA', '2 others')" in code
    assert "fig.legend(" in code and "loc='outside lower center'" in code
    assert "ax.spines['top'].set_visible(False)" in code
    assert 'lambda v, _: f"1e{round(math.log10(v), 6):g}"' in code
