"""Deep Statistical Comparison (DSC), computed locally (``docs/definitions/dsc.md``,
DS-1 to DS-14).

Four steps, as the DSCTool service has them: ``get_dsc_sample`` prepares one value
per run and problem (step 1), ``get_dsc_ranking`` ranks the algorithms on each
problem (step 2, with practical DSC), ``get_dsc_omnibus`` tests the ranks over the
problems (step 3) and ``get_dsc_posthoc`` compares every algorithm with every other
as control (step 4). An algorithm is a group of ``by`` (D17); a problem is a
(function, dimension) pair. Smaller values are better.
"""

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import polars as pl
from scipy import stats

from ..targets import PAIR, get_target_dt
from .distributions import get_runtime_sample, get_value_sample
from .stats import hochberg, holm
from .utils import runs_of

__all__ = [
    "DscOmnibus",
    "DscPosthoc",
    "DscRanking",
    "ad_pvalue",
    "get_dsc_omnibus",
    "get_dsc_posthoc",
    "get_dsc_ranking",
    "get_dsc_sample",
]

#: The omnibus methods by number of algorithms (DS-12): k = 2 and k >= 3.
METHODS_2 = ["wilcoxon-signed-rank-less"]
METHODS_K = ["friedman", "friedman-aligned-rank", "iman-davenport"]
NOT_ENOUGH = "Not enough problems for multiple-problem analysis!"
REJECTED = "The null hypothesis is rejected."
NOT_REJECTED = "The null hypothesis is not rejected."
#: The standard deviation of the limiting Anderson-Darling distribution (DS-9).
SIGMA_INF = math.sqrt(2 * (math.pi**2 - 9) / 3)
P_MIN, P_MAX = 1e-5, 1 - 1e-5


# Step 1: client-side data preparation (DS-2, DS-3)


def get_dsc_sample(  # pylint: disable=too-many-arguments
    data: pl.DataFrame,
    *,
    perspective: str = "fixed_target",
    targets: pl.DataFrame | None = None,
    by: Sequence[str] = ("algorithm_name",),
    na: str = "par",
    k: float = 10,
    maximization: bool = False,
    fval_var: str = "raw_y",
    eval_var: str = "evaluations",
) -> pl.DataFrame:
    """One value per run and (function, dimension) pair, smaller is better (DS-2).

    ``perspective="fixed_target"``: the run's runtime at the pair's target. A
    failed run counts as ``k`` times the run set's largest budget with
    ``na="par"`` (PAR-k, RT-3), as the run set's ERT with ``na="ert"`` (the sum of
    its budgets when no run succeeds), or is left out with ``na="remove"`` (DS-3).
    ``"fixed_budget"``: the run's best value at the pair's budget, negated under
    maximisation; a run without a value there is an error.

    Args:
        data: the runs, with ``function_id``, ``dimension``, ``run_id``, the
            ``by`` columns, ``eval_var`` and ``fval_var``.
        perspective: ``"fixed_target"`` or ``"fixed_budget"``.
        targets: ``function_id``, ``dimension`` and ``target`` (a target or a
            budget); ``get_target_dt`` when None (TG-4).
        by: the columns that make an algorithm.
        na: ``"par"``, ``"ert"`` or ``"remove"``: failed runs, fixed target only.
        k: the PAR-k factor.
        maximization: whether larger function values are better.
        fval_var: the column of function values.
        eval_var: the column of evaluation numbers.

    Returns:
        pl.DataFrame: ``function_id``, ``dimension``, the ``by`` columns,
            ``run_id``, ``target`` and ``value``.
    """
    if perspective not in ("fixed_target", "fixed_budget"):
        raise ValueError(f"perspective {perspective!r}: fixed_target or fixed_budget")
    if na not in ("par", "ert", "remove"):
        raise ValueError(f"na {na!r}: par, ert or remove")
    options = {"fval_var": fval_var, "eval_var": eval_var, "maximization": maximization}
    if targets is None:
        targets = get_target_dt(data, perspective, by=by, **options)
    parts = data.partition_by(PAIR, as_dict=True)
    frames = []
    for f, d, target in targets.select(*PAIR, "target").iter_rows():
        if (f, d) not in parts:
            continue
        if perspective == "fixed_target":
            sample = _runtimes(parts[f, d], target, by, na=na, k=k, **options)
        else:
            sample = _values(parts[f, d], target, by, **options)
            _check_values(sample, f"function {f}, dimension {d}, budget {target}")
        frames.append(sample.with_columns(function_id=pl.lit(f), dimension=pl.lit(d)))
    columns = [*PAIR, *by, "run_id", "target", "value"]
    return pl.concat([s.select(columns) for s in frames]).sort(*PAIR, *by, "run_id")


def _runtimes(frame, target, by, *, na, k, **options) -> pl.DataFrame:
    sample = get_runtime_sample(frame, [target], by, **options)
    budgets = runs_of(frame, by, options["eval_var"]).select("run_id", "evals_used")
    sample = sample.join(budgets, on="run_id")
    t, budget = pl.col("runtime"), pl.col("evals_used").cast(pl.Float64)
    ok = t.is_finite()
    if na == "remove":
        return sample.filter(ok).with_columns(value=t)
    if na == "par":
        failed = k * budget.max().over(by)
    else:
        n_success = ok.sum().over(by)
        charged = pl.when(ok).then(t).otherwise(budget).sum().over(by)
        failed = pl.when(n_success > 0).then(charged / n_success)
        failed = failed.otherwise(budget.sum().over(by))
    return sample.with_columns(value=pl.when(ok).then(t).otherwise(failed))


def _values(frame, budget, by, **options) -> pl.DataFrame:
    sample = get_value_sample(frame, [int(budget)], by, **options)
    sign = -1.0 if options["maximization"] else 1.0
    return sample.with_columns(target=pl.lit(budget), value=pl.col("value") * sign)


def _check_values(sample: pl.DataFrame, where: str):
    missing = sample.filter(pl.col("value").is_null())
    if len(missing):
        runs = missing["run_id"].to_list()
        raise ValueError(
            f"{where}: runs {runs} have no value before their first evaluation"
        )


# Step 2: ranking per problem, and practical DSC (DS-4, DS-5, DS-9 to DS-11)


@dataclass(frozen=True)
class DscRanking:
    """The ``rank`` response: ``ranked_matrix`` has the problem columns, the ``by``
    columns and ``rank``; ``dropped_problems`` lists the problems that not every
    algorithm has (DS-2). ``valid_methods`` is empty below two problems (DS-14)."""

    ranked_matrix: pl.DataFrame
    number_algorithms: int
    valid_methods: list[str]
    dropped_problems: pl.DataFrame
    by: tuple[str, ...]
    problem: tuple[str, ...]
    parametric_tests: bool = False


def get_dsc_ranking(  # pylint: disable=too-many-arguments
    sample: pl.DataFrame,
    *,
    by: Sequence[str] = ("algorithm_name",),
    problem: Sequence[str] = tuple(PAIR),
    value: str = "value",
    test: str = "AD",
    alpha: float = 0.05,
    epsilon: float = 0.0,
    monte_carlo_iterations: int = 0,
    seed: int | None = None,
) -> DscRanking:
    """DSC ranks of the groups of ``by`` on each problem (``dsc.md`` step 2).

    Every pair of algorithms is tested with a two-sample KS or AD test (DS-9); two
    are equal when p >= ``alpha`` / m, m = k(k - 1)/2. When equality is transitive,
    the groups of equal algorithms are ordered by the mean of their members'
    sample means; otherwise each algorithm by its own mean. Ranks are the average
    positions. With ``epsilon`` > 0, practical DSC: run j of one algorithm and run
    j of the other become their midpoint when they differ by less than
    ``epsilon``, in the given run order, or over ``monte_carlo_iterations``
    seeded shuffles, the pair's p-value then the median (DS-11).

    Args:
        sample: one value per run, for example from ``get_dsc_sample``.
        by: the columns that make an algorithm.
        problem: the columns that make a problem.
        value: the column of values, smaller is better.
        test: ``"AD"`` or ``"KS"`` (any case).
        alpha: the level of the pairwise tests.
        epsilon: the practical-significance threshold, in the values' units.
        monte_carlo_iterations: shuffles for Monte Carlo practical DSC.
        seed: the seed of the shuffles.

    Returns:
        DscRanking: the ranks and the omnibus methods they admit.
    """
    by, problem = list(by), list(problem)
    test = test.upper()
    if test not in ("AD", "KS"):
        raise ValueError(f"test {test!r}: AD or KS")
    if epsilon < 0 or monte_carlo_iterations < 0:
        raise ValueError("epsilon and monte_carlo_iterations must not be negative")
    names = sample.select(by).unique().sort(by).rows()
    k = len(names)
    if k < 2:
        raise ValueError(f"DSC needs at least two groups of {by}, got {k}")
    if k > 4:
        warnings.warn(f"DSC of {k} algorithms: comparisons with a control are advised")
    kept, dropped = _common_problems(sample, by, problem, k)
    rng = np.random.default_rng(seed)

    def same(x, y):
        p = _pair_p(x, y, test, epsilon, monte_carlo_iterations, rng)
        return p >= alpha / (k * (k - 1) / 2)

    rows = []
    for key, frame in kept.partition_by(problem, as_dict=True).items():
        samples = _samples(frame, by, names, value, key)
        rows += [(*key, *name, r) for name, r in zip(names, _rank(samples, same))]
    schema = [*problem, *by, "rank"]
    matrix = pl.DataFrame(rows, schema=schema, orient="row", infer_schema_length=None)
    matrix = matrix.cast(sample.select(*problem, *by).schema).sort(problem)
    n = len(rows) // k
    methods = [] if n < 2 else METHODS_2 if k == 2 else METHODS_K
    return DscRanking(matrix, k, list(methods), dropped, tuple(by), tuple(problem))


def _common_problems(sample, by, problem, k) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The rows of the problems every algorithm has, sorted by problem in a stable
    way, and the other problems (DS-2)."""
    counts = sample.group_by(problem).agg(pl.struct(by).n_unique().alias("n"))
    common = counts.filter(pl.col("n") == k).select(problem)
    dropped = counts.filter(pl.col("n") < k).select(problem).sort(problem)
    kept = sample.join(common, on=problem, maintain_order="left")
    return kept.sort(problem, maintain_order=True), dropped


def _samples(frame, by, names, value, key) -> list[np.ndarray]:
    groups = frame.partition_by(by, as_dict=True)
    samples = [groups[name][value].cast(pl.Float64).to_numpy() for name in names]
    for name, x in zip(names, samples):
        if len(x) < 2 or np.isnan(x).any():
            raise ValueError(f"problem {key}, algorithm {name}: fewer than two values")
    return samples


def _rank(samples: list[np.ndarray], same) -> np.ndarray:
    """Average-position ranks of the samples by mean, equal ones by their group's
    mean when the relation ``same`` is transitive (DQ-4, DQ-5, DS-10)."""
    k = len(samples)
    eq = np.eye(k, dtype=bool)
    for i, j in combinations(range(k), 2):
        eq[i, j] = eq[j, i] = same(samples[i], samples[j])
    means = np.array([x.mean() for x in samples])
    if np.array_equal(eq.astype(int) @ eq.astype(int) > 0, eq):
        means = np.where(eq, means, 0.0).sum(axis=1) / eq.sum(axis=1)
    return stats.rankdata(means)


def _pair_p(x, y, test, epsilon, iterations, rng) -> float:
    """The p-value of one pair, under practical DSC when ``epsilon`` > 0 (DS-11)."""
    if epsilon == 0:
        return _p(test, x, y)
    if iterations == 0:
        return _p(test, *_midpoints(x, y, epsilon))
    ps = [
        _p(test, *_midpoints(rng.permutation(x), rng.permutation(y), epsilon))
        for _ in range(iterations)
    ]
    return _median(ps)


def _midpoints(x, y, epsilon) -> tuple[np.ndarray, np.ndarray]:
    """Runs j of x and y replaced by their midpoint when closer than ``epsilon``,
    for j up to the shorter sample's length."""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    n = min(len(x), len(y))
    close = np.abs(x[:n] - y[:n]) < epsilon
    mid = (x[:n] + y[:n]) / 2
    x[:n], y[:n] = np.where(close, mid, x[:n]), np.where(close, mid, y[:n])
    return x, y


def _median(ps: Sequence[float]) -> float:
    """The upper median: below a level exactly when more than half are (DS-11)."""
    return float(np.sort(ps)[len(ps) // 2])


def _p(test: str, x: np.ndarray, y: np.ndarray) -> float:
    """The two-sided p-value of "x and y come from one distribution" (DS-9); 1 for
    samples equal as multisets (DS-10)."""
    if len(x) == len(y) and np.array_equal(np.sort(x), np.sort(y)):
        return 1.0
    if test == "KS":
        method = "exact" if len(x) * len(y) < 10_000 else "asymp"
        return float(stats.ks_2samp(x, y, method=method).pvalue)  # type: ignore
    return ad_pvalue(_ad_statistic(x, y))


def _ad_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """The standardised two-sample AD statistic T with midranks [Scholz1987]."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # scipy's capped p-value, not used
        try:
            result = stats.anderson_ksamp([x, y], variant="midrank")
        except TypeError:  # pragma: no cover  (scipy < 1.17)
            result = stats.anderson_ksamp([x, y], midrank=True)
    return float(result.statistic)  # type: ignore


def _ad_inf(z: float) -> float:
    """The limiting Anderson-Darling distribution function [Marsaglia2004],
    within 2e-6."""
    if z <= 0:
        return 0.0
    if z < 2:
        poly = (
            0.247105
            - (0.0649821 - (0.0347962 - (0.011672 - 0.00168691 * z) * z) * z) * z
        )
        return math.exp(-1.2337141 / z) / math.sqrt(z) * (2.00012 + poly * z)
    poly = 2.30695 - (0.43424 - (0.082433 - (0.008056 - 0.0003146 * z) * z) * z) * z
    return math.exp(-math.exp(1.0776 - poly * z))


def ad_pvalue(t: float) -> float:
    """The p-value of the standardised two-sample AD statistic ``t``:
    1 - F(1 + sigma t) with F the limiting distribution, in [1e-5, 1 - 1e-5]
    (DS-9)."""
    return min(max(1 - _ad_inf(1 + SIGMA_INF * t), P_MIN), P_MAX)


# Step 3: omnibus test (DS-6, DS-12 to DS-14)


@dataclass(frozen=True)
class DscOmnibus:
    """The ``omnibus`` response: ``algorithm_means`` has the ``by`` columns and the
    ``mean`` DSC rank over the ``n`` problems, in name order (DQ-16)."""

    method: str
    alpha: float
    message: str
    p_value: float
    t: float
    algorithm_means: pl.DataFrame
    n: int
    by: tuple[str, ...]


def get_dsc_omnibus(
    ranking: DscRanking, *, method: str | None = None, alpha: float = 0.05
) -> DscOmnibus:
    """The omnibus test on the DSC ranks (``dsc.md`` step 3): ``method`` one of
    ``ranking.valid_methods``, the first by default. ``message`` says whether
    p <= ``alpha``."""
    if not ranking.valid_methods:
        raise ValueError(NOT_ENOUGH)
    method = ranking.valid_methods[0] if method is None else method
    if method not in ranking.valid_methods:
        raise ValueError(f"method {method!r}: one of {ranking.valid_methods}")
    keys = [*ranking.problem, *ranking.by]
    matrix = ranking.ranked_matrix.sort(keys)
    r = matrix["rank"].to_numpy().reshape(-1, ranking.number_algorithms)
    p, t = OMNIBUS[method](r)
    means = matrix.group_by(ranking.by).agg(pl.col("rank").mean().alias("mean"))
    message = REJECTED if p <= alpha else NOT_REJECTED
    return DscOmnibus(
        method, alpha, message, p, t, means.sort(ranking.by), len(r), ranking.by
    )


def _friedman(r: np.ndarray) -> tuple[float, float]:
    n, k = r.shape
    chi2 = 12 * n / (k * (k + 1)) * (np.sum(r.mean(axis=0) ** 2) - k * (k + 1) ** 2 / 4)
    return float(stats.chi2.sf(chi2, k - 1)), float(chi2)


def _iman_davenport(r: np.ndarray) -> tuple[float, float]:
    n, k = r.shape
    chi2 = _friedman(r)[1]
    denominator = n * (k - 1) - chi2
    f = (n - 1) * chi2 / denominator if denominator > 0 else math.inf
    return float(stats.f.sf(f, k - 1, (k - 1) * (n - 1))), f


def _aligned_ranks(r: np.ndarray) -> tuple[float, float]:
    """Friedman aligned ranks [Garcia2010]: rank all values minus their problem's
    mean together."""
    n, k = r.shape
    a = stats.rankdata(r - r.mean(axis=1, keepdims=True)).reshape(n, k)
    big = k * n
    numerator = (k - 1) * (np.sum(a.sum(axis=0) ** 2) - k * n**2 / 4 * (big + 1) ** 2)
    denominator = big * (big + 1) * (2 * big + 1) / 6 - np.sum(a.sum(axis=1) ** 2) / k
    t = numerator / denominator
    return float(stats.chi2.sf(t, k - 1)), float(t)


def _wilcoxon(r: np.ndarray) -> tuple[float, float]:
    """One-sided Wilcoxon signed-rank test of the lower-mean algorithm against
    the other (DS-13); p = 1 when every difference is zero."""
    lower = int(np.argmin(r.mean(axis=0)))
    d = r[:, lower] - r[:, 1 - lower]
    if not d.any():
        return 1.0, 0.0
    return float(stats.wilcoxon(d, alternative="less").pvalue), 0.0  # type: ignore


OMNIBUS = {
    "friedman": _friedman,
    "friedman-aligned-rank": _aligned_ranks,
    "iman-davenport": _iman_davenport,
    "wilcoxon-signed-rank-less": _wilcoxon,
}


# Step 4: post-hoc test (DS-7)


@dataclass(frozen=True)
class DscPosthoc:
    """The ``posthoc`` responses for every control, with the omnibus result they
    follow. ``adjusted_p_values`` has the control's ``by`` columns, the other
    algorithm's with the suffix ``_vs``, and ``ZValue``, ``UnadjustedPValue``,
    ``Holm`` and ``Hochberg``: a small value means the control is better."""

    omnibus: DscOmnibus
    method: str
    adjusted_p_values: pl.DataFrame


def get_dsc_posthoc(omnibus: DscOmnibus) -> DscPosthoc:
    """Every algorithm as control against the others (``dsc.md`` step 4):
    z = (m_control - m_other) / SE from the mean DSC ranks, p = Phi(z), and the
    Holm and Hochberg adjustments of the k - 1 p-values of one control. The
    statistic follows the omnibus method: ``friedman-aligned-rank`` after aligned
    ranks, ``friedman`` otherwise (DS-7)."""
    by, means = list(omnibus.by), omnibus.algorithm_means
    m, n = means["mean"].to_numpy(), omnibus.n
    k = len(m)
    aligned = omnibus.method == "friedman-aligned-rank"
    method = "friedman-aligned-rank" if aligned else "friedman"
    se = math.sqrt(k * (n + 1) / 6 if aligned else k * (k + 1) / (6 * n))
    names = means.select(by).rows()
    rows = []
    for c in range(k):
        others = [a for a in range(k) if a != c]
        z = (m[c] - m[others]) / se
        p = stats.norm.cdf(z)
        rows += [
            (*names[c], *names[a], *v)
            for a, *v in zip(others, z, p, holm(p), hochberg(p))
        ]
    vs = [f"{c}_vs" for c in by]
    values = ["ZValue", "UnadjustedPValue", "Holm", "Hochberg"]
    frame = pl.DataFrame(rows, schema=[*by, *vs, *values], orient="row")
    types: dict = {**means.select(by).schema, **dict(zip(vs, means.select(by).dtypes))}
    return DscPosthoc(
        omnibus, method, frame.cast(types | dict.fromkeys(values, pl.Float64))
    )
