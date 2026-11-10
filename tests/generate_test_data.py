"""Generate IOHexperimenter test logs with ``ioh==0.4.0``.

``python tests/generate_test_data.py`` writes the random logs in ``tests/test_data``
(not seeded). ``python tests/generate_test_data.py --reader-fixtures`` writes the
seeded logs in ``tests/io/fixtures/ioh040``, which the reader tests (WP I05) use. Each
reader fixture reproduces one defect of the iohinspector 0.0.8 loader or one quirk
of the logger. Run with ``ioh==0.3.22`` installed, it writes the same logs in
``tests/io/fixtures/ioh0322``.
"""

import os
import shutil
import sys

import ioh
import numpy as np

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.realpath(os.path.join(BASE_DIR, "test_data"))


def _ioh_version() -> str:
    from importlib.metadata import version

    return version("ioh")


#: ``ioh040`` for ioh 0.4.0, ``ioh0322`` for ioh 0.3.22.
READER_DIR = os.path.realpath(
    os.path.join(BASE_DIR, "io", "fixtures", "ioh" + _ioh_version().replace(".", ""))
)


def run(problem, algorithm_name):
    logger = ioh.logger.Analyzer(
        root=DATA_DIR, folder_name=algorithm_name, algorithm_name=algorithm_name
    )
    problem.attach_logger(logger)

    for _ in range(5):
        for _ in range(100):
            x = np.random.uniform(problem.bounds.lb, problem.bounds.ub)
            problem(x)
        problem.reset()


class Tracked:
    """Values the logger watches or reads as run attributes."""

    def __init__(self):
        self.seed = 0.0
        self.sigma = 0.0
        self.x0 = 0.0
        self.unset = float("nan")
        self.big = float("inf")


def _random_runs(problem, rng, runs, evals):
    for _ in range(runs):
        for _ in range(evals):
            problem(rng.uniform(problem.bounds.lb, problem.bounds.ub))
        problem.reset()


def _analyzer(name, **kwargs):
    return ioh.logger.Analyzer(
        root=READER_DIR, folder_name=name, algorithm_name=name, **kwargs
    )


def every_third():
    """Trigger ``Each(3)``: evaluation 1 is never logged (defect: split on evaluations == 1)."""
    trigger = ioh.logger.trigger.Each(3)  # the logger keeps only a reference
    logger = _analyzer("every_third", triggers=[trigger])
    problem = ioh.get_problem(1, 1, 2)
    problem.attach_logger(logger)
    _random_runs(problem, np.random.default_rng(1), runs=3, evals=10)
    logger.close()


def cross_run():
    """Run 2 starts where run 1 ended (defect: improvement filter across runs).

    Run 1 is logged on improvement and ends with a non-improving evaluation (defect:
    the final row is dropped). Run 2 first evaluates the best point of run 1, so
    its first best-so-far value repeats the last one of run 1.
    """
    logger = _analyzer("cross_run")
    problem = ioh.get_problem(1, 1, 2)
    problem.attach_logger(logger)
    rng = np.random.default_rng(2)
    xs = [rng.uniform(-5, 5, 2) for _ in range(8)]
    for x in xs:
        problem(x)
    best = min(xs, key=lambda x: float(np.sum((x - problem.optimum.x) ** 2)))
    last = best + 3.0
    problem(last)
    problem.reset()
    problem(best)
    for _ in range(5):
        problem(rng.uniform(-5, 5, 2))
    problem.reset()
    logger.close()


def f_zero():
    """Runs that reach f = 0, written as ``0.0000000000`` (defect: assert on f = 0)."""
    logger = _analyzer("f_zero")
    problem = ioh.get_problem(1, 1, 2)
    problem.attach_logger(logger)
    rng = np.random.default_rng(3)
    for _ in range(2):
        problem(rng.uniform(-5, 5, 2))
        problem(problem.optimum.x + 1e-7)  # below 5e-11 after rounding
        problem(problem.optimum.x)
        problem(rng.uniform(-5, 5, 2))
        problem.reset()
    logger.close()


def attributes():
    """Maximisation with experiment and run attributes, watched values and positions.

    The watched ``x0`` clashes with a reserved trace name, the run attributes
    include ``nan`` and ``inf`` (written as bare JSON tokens), the experiment
    attribute ``instance`` clashes with a runs column, and the algorithm info has
    a backslash and a quote, which the logger does not escape.
    """
    tracked = Tracked()
    logger = ioh.logger.Analyzer(
        root=READER_DIR,
        folder_name="attributes",
        algorithm_name="ea",
        algorithm_info='C:\\runs\\ea "v2"',
        store_positions=True,
    )
    logger.add_experiment_attribute("budget", "8")
    logger.add_experiment_attribute("instance", "seven")
    logger.add_run_attribute(tracked, "seed")
    logger.add_run_attribute(tracked, "unset")
    logger.add_run_attribute(tracked, "big")
    logger.watch(tracked, ["sigma", "x0"])
    rng = np.random.default_rng(4)
    for dim in (4, 6):
        problem = ioh.get_problem(1, 1, dim, ioh.ProblemClass.PBO)
        problem.attach_logger(logger)
        for r in range(2):
            tracked.seed = float(2**40 + r)
            for _ in range(8):
                x = rng.integers(0, 2, dim)
                tracked.sigma = float(rng.random())
                tracked.x0 = float(x[0])
                problem(x.tolist())
            problem.reset()
        problem.detach_logger()
    logger.close()


def reader_fixtures():
    if os.path.isdir(READER_DIR):
        shutil.rmtree(READER_DIR)
    every_third()
    cross_run()
    f_zero()
    attributes()


if __name__ == "__main__":
    if "--reader-fixtures" in sys.argv[1:]:
        reader_fixtures()
        sys.exit(0)

    p1 = ioh.get_problem(1, 1, 2)
    p2 = ioh.get_problem(2, 1, 2)

    run(p1, "algorithm_A")
    run(p1, "algorithm_A")
    run(p1, "algorithm_B")

    run(p2, "algorithm_A")
    run(p2, "algorithm_B")
