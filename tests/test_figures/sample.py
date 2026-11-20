"""A small closed-form benchmark for figure tests and the gallery, and the
gallery itself: ``python -m tests.test_figures.sample`` writes every venue
preset, single and double column, two figures of the element model and one of
each figure kind to ``docs/figures/`` (the snapshot baselines).
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
    atlas,
    render,
)

ONE = {"function_id": [1], "dimension": [20]}
TWO = {**ONE, "algorithm_name": ["DE", "GA"]}
PARAMETERS = {"by": ["sigma", "mu"]}
KIND_DATA = {  # each figure kind of the Workbench and what it is drawn from
    "convergence": Data(where=ONE, at=1e-3),
    "fixed_budget": Data(),
    "ert": Data(where={"function_id": [1]}),
    "fixed_budget_ecdf": Data(),
    "target_budget": Data(),
    "pairwise_tests": Data(),
    "win_fractions": Data(),
    "box": Data(where=ONE),
    "violin": Data(where=ONE),
    "histogram": Data(where=ONE, measure="runtime", at=1e-2),
    "eaf": Data(where={**ONE, "algorithm_name": ["CMA-ES", "RS"]}),
    "eaf_difference": Data(where=TWO),
    "shapley": Data(),
    "cumulative_difference": Data(where=TWO),
    "parallel_coordinates": Data(**PARAMETERS),
    "marginals": Data(**PARAMETERS),
    "parameter_map": Data(**PARAMETERS),
    "sensitivity": Data(**PARAMETERS),
}
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


def parameter_data() -> pl.DataFrame:
    """Runs of one algorithm in 3 x 3 configurations of two parameters, ``sigma``
    and ``mu``, on one function and dimension: the best value falls fastest at
    ``sigma`` 0.3 and a larger ``mu``, with no randomness."""
    rows = []
    for run_id, (sigma, mu, k) in enumerate(
        itertools.product((0.1, 0.3, 1.0), (4, 8, 16), range(3))
    ):
        rate = 1.5 - (np.log10(sigma) + 0.5) ** 2 + 0.03 * mu + 0.1 * k
        for e in np.unique(np.geomspace(1, 2000, 40).round().astype(int)):
            rows.append(
                (
                    run_id,
                    "ES",
                    sigma,
                    mu,
                    1,
                    10,
                    int(e),
                    10 ** (2 - rate * np.log10(e) / 2),
                )
            )
    return pl.DataFrame(
        rows,
        schema=[
            "run_id",
            "algorithm_name",
            "sigma",
            "mu",
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
    for kind, data in KIND_DATA.items():
        specs[f"kind-{kind}"] = FigureSpec(
            panels=[Panel(kind, data)], formats=["png"], dpi=100
        )
    specs["kind-target_budget-vs_best"] = FigureSpec(
        panels=[Panel("target_budget", Data(mode="vs_best", focus="CMA-ES"))],
        formats=["png"],
        dpi=100,
    )
    specs["kind-atlas"] = atlas(
        FigureSpec(span="double", formats=["png"], dpi=100), sample_data()
    )
    specs["ecdf-panels-highlight"] = panels
    specs["ecdf-slides-dark"] = apply_style(panels, "slides-dark")
    return specs


def gallery_runs(spec: FigureSpec) -> pl.DataFrame:
    """The runs a gallery figure is drawn from."""
    parameters = "sigma" in spec.panels[0].data.by
    return parameter_data() if parameters else sample_data()


def main() -> None:
    GALLERY.mkdir(parents=True, exist_ok=True)
    for name, spec in gallery_specs().items():
        (GALLERY / f"{name}.png").write_bytes(render(spec, gallery_runs(spec)))


if __name__ == "__main__":
    main()
