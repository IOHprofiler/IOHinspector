"""The TeX exports: ``render(fmt="pgf")`` and ``pgfplots``. Compiling needs a
TeX installation (the optional ``tex.yml`` workflow has one); without it those
tests are skipped."""

import re
import shutil
import subprocess

import numpy as np
import pytest

from iohinspector.figures import Axis, Data, FigureSpec, Legend, Series, render
from iohinspector.figures.pgfplots import pgfplots
from iohinspector.figures.render import tex_escape

from .sample import ALGORITHMS, sample_data

needs_tex = pytest.mark.skipif(not shutil.which("pdflatex"), reason="no TeX")
DOCUMENT = "\\documentclass{article}%s\\begin{document}\\input{%s}\\end{document}"


@pytest.fixture(scope="module")
def data():
    return sample_data()


def compile_tex(
    folder, name, package="\\usepackage{pgfplots}\\pgfplotsset{compat=1.18}"
):
    (folder / "doc.tex").write_text(DOCUMENT % (package, name))
    run = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "doc.tex"],
        cwd=folder,
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout[-2000:]
    assert (folder / "doc.pdf").stat().st_size > 1000


def test_tex_escape_leaves_math_alone():
    assert tex_escape("# f-evals_1 $x_1\\%$ 50%") == "\\# f-evals\\_1 $x_1\\%$ 50\\%"


@needs_tex
@pytest.mark.parametrize("venue", ["lncs", "ieee", "acm"])
def test_pgf_uses_the_documents_font_and_compiles(data, tmp_path, venue):
    spec = FigureSpec(venue=venue, formats=["pgf"], title="f_1 & f_2")
    pgf = render(spec, data)
    assert pgf == render(spec, data)
    assert b"%% Creator: Matplotlib, PGF backend" in pgf
    assert b"\\fontfamily{\\familydefault}" in pgf  # the document's, not ours
    (tmp_path / "figure.pgf").write_bytes(pgf)
    compile_tex(tmp_path, "figure.pgf", "\\usepackage{pgf}")


@pytest.fixture(scope="module")
def tex_files(data):
    return pgfplots(FigureSpec(venue="ieee"), data, "ecdf")


def test_pgfplots_writes_a_table_and_a_picture(tex_files):
    assert set(tex_files) == {"ecdf.tex", "ecdf.csv"}
    rows = tex_files["ecdf.csv"].decode().splitlines()
    assert rows[0] == "series,x,y"
    table = np.array([r.split(",") for r in rows[1:]], dtype=float)
    assert sorted(set(table[:, 0])) == list(range(len(ALGORITHMS)))
    best = table[table[:, 0] == 0]  # highest final value first
    assert best[-1, 2] == max(table[:, 2]) and np.all(np.diff(best[:, 1]) >= 0)


def test_pgfplots_matches_the_figure(tex_files):
    tex = tex_files["ecdf.tex"].decode()
    assert tex.count("\\addplot[") == 2 * len(ALGORITHMS)  # lines, markers
    assert "xmode=log" in tex and "ymode=log" not in tex
    assert "\\log_{10}$(\\# f-evals / dimension)" in tex
    assert "dash pattern=on 3.7bp off 1.6bp" in tex  # matplotlib's dashed, width 1
    assert "color={rgb,255:red,0;green,114;blue,178}" in tex  # first palette colour
    assert "\\fontsize{8}{9.6}" in tex  # IEEE figure captions
    labels = re.findall(r"\{(CMA-ES|DE|GA|PSO|RS)\};", tex)
    assert labels == ["CMA-ES", "DE", "GA", "PSO", "RS"]  # by final value
    assert "fontfamily" not in tex  # the document's font


def test_pgfplots_options(data):
    spec = FigureSpec(
        legend=Legend(mode="box", loc="upper left"),
        x=Axis(ticks="plain", grid="both"),
        series={"DE": Series(name="D_E", linestyle=":", linewidth=2)},
        title="f1",
    )
    tex = pgfplots(spec, data)["figure.tex"].decode()
    assert "legend pos=north west" in tex and "\\addlegendentry{D\\_E}" in tex
    assert "xminorgrids" in tex and "title={f1}" in tex
    assert "line width=2bp, line cap=butt, dash pattern=on 2bp" in tex
    assert "\\draw" not in tex  # no direct labels


@needs_tex
@pytest.mark.parametrize(
    "spec",
    [
        FigureSpec(venue="acm"),
        FigureSpec(legend=Legend(mode="outside"), x=Axis(ticks="power")),
        FigureSpec(
            data=Data(by=["algorithm_name", "dimension"]), x=Axis(ticks="plain")
        ),
    ],
)
def test_pgfplots_compile(data, tmp_path, spec):
    for name, content in pgfplots(spec, data).items():
        (tmp_path / name).write_bytes(content)
    compile_tex(tmp_path, "figure.tex")
