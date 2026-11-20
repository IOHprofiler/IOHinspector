"""A small closed-form benchmark for figure tests and the gallery, and the
gallery itself: ``python -m tests.test_figures.sample`` writes every venue
preset, single and double column, and two figures of the element model to
``docs/figures/`` (the snapshot baselines).
"""

import itertools
from pathlib import Path

import numpy as np
import polars as pl

from iohinspector.figures import (
    VENUES,
    Annotation,
    Data,
    FigureSpec,
    Layout,
    Legend,
    Panel,
    Series,
    apply_style,
    render,
)

GALLERY = Path(__file__).parents[2] / "docs" / "figures"
# name: (convergence rate, budget per dimension)
ALGORITHMS = {
    "CMA-ES": (1.6, 300),
    "DE": (0.9, 300),
    "GA": (0.85, 300),
    "PSO": (0.7, 100),
    "RS": (0.2, 300),
}


def sample_data() -> pl.DataFrame:
    """Runs whose best value falls as 10^(2 - rate log10(evaluations)) with a
    run- and function-dependent rate: no randomness, the same everywhere."""
    rows = []
    for run_id, (name, f, d, k) in enumerate(
        itertools.product(ALGORITHMS, (1, 2), (5, 20), range(5))
    ):
        rate, budget = ALGORITHMS[name]
        evals = np.unique(np.geomspace(1, budget * d, 60).round().astype(int))
        r = rate * (1 + 0.15 * k - 0.3) / (f * np.log10(d))
        for e in evals:
            rows.append((run_id, name, f, d, int(e), 10 ** (2 - r * np.log10(e))))
    return pl.DataFrame(
        rows,
        schema=[
            "run_id",
            "algorithm_name",
            "function_id",
            "dimension",
            "evaluations",
            "raw_y",
        ],
        orient="row",
    )


def gallery_specs() -> dict[str, FigureSpec]:
    specs = {
        f"ecdf-{venue}-{span}": FigureSpec(
            venue=venue, span=span, formats=["png"], dpi=150
        )
        for venue in VENUES
        for span in ("single", "double")
    }
    # the element model (I35): panels, highlighting, annotations, a dark slide
    panels = FigureSpec(
        panels=[
            Panel(
                data=Data(where={"dimension": [5]}),
                title="$d$ = 5",
                annotations=[
                    Annotation("line", "half", y=0.5),
                    Annotation("arrow", "PSO restarts", x=100, y=0.45, x2=3, y2=0.8),
                ],
            ),
            Panel(data=Data(where={"dimension": [20]}), title="$d$ = 20"),
        ],
        layout=Layout(columns=2, labels="letters", share_y=True),
        span="double",
        palette="highlight",
        series={
            "PSO": Series(highlight="yours"),
            "GA": Series(highlight="other"),
            "RS": Series(highlight="other"),
        },
        legend=Legend(others=True),
        formats=["png"],
        dpi=150,
    )
    specs["ecdf-panels-highlight"] = panels
    specs["ecdf-slides-dark"] = apply_style(panels, "slides-dark")
    return specs


def main() -> None:
    GALLERY.mkdir(parents=True, exist_ok=True)
    data = sample_data()
    for name, spec in gallery_specs().items():
        (GALLERY / f"{name}.png").write_bytes(render(spec, data))


if __name__ == "__main__":
    main()
