"""Deterministically generate the ioh 0.4.0 fixture logs used by the golden references (WP I03).

Run with ``ioh==0.4.0`` installed (``make golden-fixtures``). Seeds are fixed, so a second run
produces the same logs. Output goes to ``tests/golden/fixtures/<set>/<algorithm>/``.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import ioh
import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"
INSTANCES = range(1, 6)


class Tracked:
    """Holds a parameter that the logger watches."""

    def __init__(self) -> None:
        self.sigma = 0.0
        self.x0 = self.x1 = self.x2 = 0.0


def real_algorithm(name, problem, budget, rng, tracked):
    lb, ub = np.array(problem.bounds.lb), np.array(problem.bounds.ub)
    dim = problem.meta_data.n_variables
    x = rng.uniform(lb, ub)
    fx = problem(x)
    sigma = {"rs": None, "es_small": 0.1, "es_large": 1.0}[name]
    while problem.state.evaluations < budget:
        if sigma is None:
            y = rng.uniform(lb, ub)
        else:
            tracked.sigma = sigma
            y = np.clip(x + sigma * rng.standard_normal(dim), lb, ub)
            if dim == 3:
                tracked.x0, tracked.x1, tracked.x2 = (float(v) for v in y)
        fy = problem(y)
        if fy <= fx:
            x, fx = y, fy
            if sigma is not None:
                sigma *= 1.5
        elif sigma is not None:
            sigma *= 0.9


def bit_algorithm(name, problem, budget, rng, tracked):
    dim = problem.meta_data.n_variables
    x = rng.integers(0, 2, dim)
    fx = problem(x.tolist())
    while problem.state.evaluations < budget:
        if name == "rs":
            y = rng.integers(0, 2, dim)
        elif name == "rls":
            y = x.copy()
            y[rng.integers(dim)] ^= 1
        else:  # (1+1) EA, p = 1/n
            flip = rng.random(dim) < 1.0 / dim
            y = np.where(flip, 1 - x, x)
        fy = problem(y.tolist())
        tracked.sigma = 1.0 / dim
        if fy >= fx:
            x, fx = y, fy


def run_set(setname, algorithms, problems, runner, budget_of, trigger=None, extra=(), seed=0):
    root = OUT / setname
    if root.exists():
        shutil.rmtree(root)
    for ai, alg in enumerate(algorithms):
        tracked = Tracked()
        kwargs = {}
        if trigger is not None:
            kwargs["triggers"] = [trigger]
        logger = ioh.logger.Analyzer(
            root=str(root), folder_name=alg, algorithm_name=alg,
            algorithm_info=f"golden fixture {setname}", additional_properties=list(extra), **kwargs,
        )
        if alg != "rs":
            logger.watch(tracked, ["sigma"] + (["x0", "x1", "x2"] if setname == "ioh040_pos" else []))
        for pi, make in enumerate(problems):
            for inst in INSTANCES:
                rng = np.random.default_rng([seed, ai, pi, inst])
                problem = make(inst)
                problem.attach_logger(logger)
                runner(alg, problem, budget_of(problem), rng, tracked)
                problem.detach_logger()
        logger.close()


def main() -> None:
    if ioh.__file__ and not hasattr(ioh, "get_problem"):
        sys.exit("ioh not importable")
    bb = ioh.ProblemClass.BBOB
    # minimisation: three BBOB functions, two dimensions, three algorithms
    problems = [
        (lambda i, f=f, d=d: ioh.get_problem(f, i, d, bb)) for f in (1, 2, 3) for d in (5, 10)
    ]
    run_set("ioh040_min", ["rs", "es_small", "es_large"], problems, real_algorithm,
            lambda p: 100 * p.meta_data.n_variables, seed=1)
    # maximisation: PBO OneMax and LeadingOnes
    pbo = ioh.ProblemClass.PBO
    problems = [
        (lambda i, f=f, d=d: ioh.get_problem(f, i, d, pbo)) for f in (1, 2) for d in (16, 32)
    ]
    run_set("ioh040_max", ["rs", "rls", "ea"], problems, bit_algorithm,
            lambda p: 20 * p.meta_data.n_variables, seed=2)
    # positions logged as parameters x0..x2 (IOHanalyzer reads them as parameters)
    run_set("ioh040_pos", ["es_small"], [lambda i: ioh.get_problem(1, i, 3, bb)], real_algorithm,
            lambda p: 60, seed=4)
    # constrained: sphere with a soft linear constraint, every evaluation logged
    cons = lambda i: _constrained(i)
    run_set("ioh040_cons", ["rs", "es_small"], [cons], real_algorithm, lambda p: 200,
            trigger=ioh.logger.trigger.ALWAYS,
            extra=[ioh.logger.property.VIOLATION, ioh.logger.property.PENALTY], seed=3)


def _constrained(inst):
    p = ioh.get_problem(1, inst, 5, ioh.ProblemClass.BBOB)
    p.constraints.add(
        ioh.RealConstraint(lambda x: max(0.0, float(np.sum(x)) - 1.0), weight=10.0,
                           enforced=ioh.ConstraintEnforcement.SOFT, name="sum<=1")
    )
    return p


if __name__ == "__main__":
    main()
