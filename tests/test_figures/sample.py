"""A small closed-form benchmark for figure tests and the gallery, and the
gallery itself: ``python -m tests.test_figures.sample`` writes every venue
preset, single and double column, to ``docs/figures/`` (the snapshot baselines).
"""

import itertools
from pathlib import Path

import numpy as np
import polars as pl

from iohinspector.figures import VENUES, FigureSpec, render

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
    return {
        f"ecdf-{venue}-{span}": FigureSpec(
            venue=venue, span=span, formats=["png"], dpi=150
        )
        for venue in VENUES
        for span in ("single", "double")
    }


def main() -> None:
    GALLERY.mkdir(parents=True, exist_ok=True)
    data = sample_data()
    for name, spec in gallery_specs().items():
        (GALLERY / f"{name}.png").write_bytes(render(spec, data))


if __name__ == "__main__":
    main()
