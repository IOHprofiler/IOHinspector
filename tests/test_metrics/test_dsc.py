"""DSC (``docs/definitions/dsc.md``): the published DSCTool example and hand-checkable
cases for every decision DS-1 to DS-14."""

import json
import math
import os
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy import stats

from iohinspector.metrics import dsc
from iohinspector.metrics.dsc import (
    DscRanking,
    ad_pvalue,
    get_dsc_omnibus,
    get_dsc_posthoc,
    get_dsc_ranking,
    get_dsc_sample,
)

ALG = ["algorithm"]
PROBLEM = ["problem"]
EXAMPLE = "https://web.archive.org/web/20220615id_/https://cs.ijs.si/dl/dsctool/"
#: [DSCExample] as cited in dsc.md (step 3 and step 4).
MEANS = {
    "BSif": 2.3181818181818183,
    "BSqi": 2.227272727272727,
    "CMA-MSR": 1.4545454545454546,
}
POSTHOC = {  # control CMA-MSR: z, unadjusted, Holm, Hochberg
    "BSif": (
        -2.864357773488755,
        0.0020892787840832367,
        0.004178557568166473,
        0.004178557568166473,
    ),
    "BSqi": (
        -2.56284642891099,
        0.005190897894850896,
        0.005190897894850896,
        0.005190897894850896,
    ),
}


def long(samples: dict) -> pl.DataFrame:
    """{(algorithm, problem): values} as a sample frame."""
    rows = [(a, p, float(v)) for (a, p), vs in samples.items() for v in vs]
    return pl.DataFrame(rows, schema=[*ALG, *PROBLEM, "value"], orient="row")


def rank(samples: dict, **options) -> DscRanking:
    return get_dsc_ranking(long(samples), by=ALG, problem=PROBLEM, **options)


def ranks(ranking: DscRanking) -> dict:
    return {(a, p): r for p, a, r in ranking.ranked_matrix.rows()}


def ranking_of(matrix: dict) -> DscRanking:
    """A ranking with the given ranks {algorithm: [rank per problem]}."""
    rows = [(i, a, r) for a, rs in matrix.items() for i, r in enumerate(rs)]
    frame = pl.DataFrame(rows, schema=[*PROBLEM, *ALG, "rank"], orient="row")
    k = len(matrix)
    methods = dsc.METHODS_2 if k == 2 else dsc.METHODS_K
    return DscRanking(
        frame, k, methods, frame.clear().select(PROBLEM), tuple(ALG), tuple(PROBLEM)
    )


def example_matrix() -> dict:
    """22 problems with the rank sums of [DSCExample] (51, 49, 32)."""
    rows = [(3, 2, 1)] * 7 + [(2, 3, 1)] * 5 + [(3, 1, 2)] * 5 + [(1, 3, 2)] * 5
    return dict(zip(MEANS, np.array(rows, dtype=float).T.tolist()))


# The published example (downloaded; skipped offline, Q-DS-4)


#: the downloaded example, kept between runs (not committed: it is not ours)
CACHE = Path(__file__).parents[1] / ".cache" / "dsctool"


@pytest.fixture(scope="module")
def example():
    files = {}
    try:
        for name in ("dsc.json", "dsc.result", "omnibus.result", "posthoc.result"):
            cached = CACHE / name
            if not cached.exists():
                url = EXAMPLE + name
                assert url.startswith("https://")
                with urllib.request.urlopen(url, timeout=60) as response:  # nosec B310
                    body = response.read()
                CACHE.mkdir(parents=True, exist_ok=True)
                partial = cached.with_suffix(f".{os.getpid()}")  # workers in parallel
                partial.write_bytes(body)
                partial.replace(cached)
            files[name] = json.loads(cached.read_bytes())
    except (OSError, ValueError) as error:
        pytest.skip(f"DSCTool example not available: {error}")
    return files


def test_example_reproduced(example):
    request = example["dsc.json"]
    samples = {
        (a["algorithm"], p["name"]): [float(v) for v in p["data"]]
        for a in request["data"]
        for p in a["problems"]
    }
    method = request["method"]
    ranking = rank(samples, test=method["name"], alpha=method["alpha"])
    result = example["dsc.result"]["result"]
    expected = {
        (x["algorithm"], m["problem"]): x["rank"]
        for m in result["ranked_matrix"]
        for x in m["result"]
    }
    assert ranks(ranking) == expected
    assert ranking.valid_methods == result["valid_methods"]
    assert ranking.parametric_tests == result["parametric_tests"]
    assert ranking.number_algorithms == result["number_algorithms"]

    omnibus = get_dsc_omnibus(ranking)
    reference = example["omnibus.result"]["result"]
    assert omnibus.message == reference["message"]
    assert omnibus.p_value == pytest.approx(reference["p_value"], rel=1e-9)
    assert omnibus.t == pytest.approx(reference["t"], rel=1e-9)
    means = {m["algorithm"]: m["mean"] for m in reference["algorithm_means"]}
    assert dict(omnibus.algorithm_means.rows()) == pytest.approx(means, rel=1e-12)

    posthoc = get_dsc_posthoc(omnibus).adjusted_p_values
    mine = posthoc.filter(pl.col("algorithm") == "CMA-MSR")
    for entry in example["posthoc.result"]["result"]["adjusted_p_values"]:
        values = {x["algorithm"]: x["value"] for x in entry["algorithms"]}
        got = dict(zip(mine["algorithm_vs"], mine[entry["name"]]))
        assert got == pytest.approx(values, rel=1e-9)


# Step 1: data preparation (DS-2, DS-3)


def runs_data() -> pl.DataFrame:
    """f1 d2: A reaches 1.0 at 5 (run 1), fails with budget 10 (run 2); B reaches it
    at 2 (run 3) and 4 (run 4). f2 d2: C never reaches 1.0 (budgets 6 and 8)."""
    runs = [  # function, algorithm, run, [(evaluations, raw_y)]
        (1, "A", 1, [(1, 5.0), (5, 1.0)]),
        (1, "A", 2, [(1, 4.0), (10, 3.0)]),
        (1, "B", 3, [(2, 0.5), (3, 0.2)]),
        (1, "B", 4, [(4, 0.9), (7, 0.1)]),
        (2, "C", 5, [(2, 5.0), (6, 4.0)]),
        (2, "C", 6, [(3, 6.0), (8, 2.0)]),
    ]
    rows = [(f, 2, a, r, e, y) for f, a, r, trace in runs for e, y in trace]
    schema = [
        "function_id",
        "dimension",
        "algorithm_name",
        "run_id",
        "evaluations",
        "raw_y",
    ]
    return pl.DataFrame(rows, schema=schema, orient="row")


TARGETS = pl.DataFrame(
    {"function_id": [1, 2], "dimension": [2, 2], "target": [1.0, 1.0]}
)


def values(sample: pl.DataFrame) -> dict:
    return dict(zip(sample["run_id"], sample["value"]))


def test_sample_par_k():
    sample = get_dsc_sample(runs_data(), targets=TARGETS)
    assert sample.columns == [
        "function_id",
        "dimension",
        "algorithm_name",
        "run_id",
        "target",
        "value",
    ]
    # PAR-10 by default: k times the run set's largest budget (RT-3)
    assert values(sample) == {1: 5, 2: 100, 3: 2, 4: 4, 5: 80, 6: 80}
    assert values(get_dsc_sample(runs_data(), targets=TARGETS, k=1))[2] == 10


def test_sample_ert_and_remove():
    ert = values(get_dsc_sample(runs_data(), targets=TARGETS, na="ert"))
    # ERT of A: (5 + 10) / 1; no success for C: the sum of the budgets (DS-3)
    assert ert == {1: 5, 2: 15, 3: 2, 4: 4, 5: 14, 6: 14}
    removed = get_dsc_sample(runs_data(), targets=TARGETS, na="remove")
    assert values(removed) == {1: 5, 3: 2, 4: 4}


def test_sample_budget_counts_from_run_metadata():
    data = runs_data().with_columns(evals_used=pl.lit(50))
    sample = get_dsc_sample(data, targets=TARGETS)
    assert values(sample)[2] == 500


def test_sample_fixed_budget_orientation():
    budgets = TARGETS.with_columns(target=pl.lit(5.0))
    low = values(
        get_dsc_sample(runs_data(), perspective="fixed_budget", targets=budgets)
    )
    assert low[1] == 1.0 and low[3] == 0.2
    high = get_dsc_sample(
        runs_data(), perspective="fixed_budget", targets=budgets, maximization=True
    )
    assert values(high)[2] == -4.0  # the best (largest) value so far, negated


def test_sample_fixed_budget_before_first_evaluation_fails():
    budgets = TARGETS.with_columns(target=pl.lit(1.0))
    with pytest.raises(ValueError, match="function 1, dimension 2"):
        get_dsc_sample(runs_data(), perspective="fixed_budget", targets=budgets)


def test_sample_default_targets_and_bad_options():
    sample = get_dsc_sample(runs_data())  # get_target_dt (TG-4)
    assert sample.select("function_id", "dimension").unique().height == 2
    with pytest.raises(ValueError, match="perspective"):
        get_dsc_sample(runs_data(), perspective="by_RT")
    with pytest.raises(ValueError, match="na"):
        get_dsc_sample(runs_data(), na="PAR-10")


# Step 2: ranking (DS-4, DS-9, DS-10)


def test_separated_samples_rank_by_mean():
    base = np.arange(10.0)
    samples = {("a", 1): base + 100, ("b", 1): base, ("c", 1): base + 50}
    for test in ("AD", "KS", "ks"):
        assert ranks(rank(samples, test=test)) == {
            ("a", 1): 3,
            ("b", 1): 1,
            ("c", 1): 2,
        }


def test_identical_samples_are_equal():
    samples = {("a", 1): [0, 0, 0], ("b", 1): [0, 0, 0], ("c", 1): [1, 2, 3]}
    assert ranks(rank(samples)) == {("a", 1): 1.5, ("b", 1): 1.5, ("c", 1): 3}
    assert dsc._p("AD", np.array([3.0, 1, 2]), np.array([1.0, 2, 3])) == 1.0


def relation(*pairs):
    equal = {frozenset(p) for p in pairs}
    return lambda x, y: frozenset((x[0], y[0])) in equal


def test_transitive_groups_ordered_by_mean_of_means():
    samples = [np.array([m]) for m in (0.0, 10.0, 4.0)]  # a ~ b, group mean 5
    assert dsc._rank(samples, relation((0.0, 10.0))).tolist() == [2.5, 2.5, 1]


def test_not_transitive_ranks_by_means_alone():
    samples = [np.array([m]) for m in (0.0, 1.0, 2.0, 1.0)]
    same = relation((0.0, 1.0), (1.0, 2.0))  # a ~ b ~ c, but a !~ c
    assert dsc._rank(samples, same).tolist() == [1, 2.5, 4, 2.5]


def test_bonferroni_on_the_level():
    x, y = np.arange(8.0), np.arange(8.0) + 3.5
    p = dsc._p("KS", x, y)
    samples = {("a", 1): x, ("b", 1): y, ("c", 1): x + 1000}
    # equal when p >= alpha / m, m = 3: here p < alpha
    assert ranks(rank(samples, test="KS", alpha=3 * p * 0.99))[("a", 1)] == 1.5
    assert ranks(rank(samples, test="KS", alpha=3 * p * 1.01))[("a", 1)] == 1


def test_ks_pvalue_is_scipys():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=30), rng.normal(0.5, size=40)
    exact = stats.ks_2samp(x, y, method="exact").pvalue
    assert dsc._p("KS", x, y) == pytest.approx(exact, rel=1e-12)
    x, y = rng.normal(size=100), rng.normal(0.2, size=100)  # n m = 10000
    asymp = stats.ks_2samp(x, y, method="asymp").pvalue
    assert dsc._p("KS", x, y) == pytest.approx(asymp, rel=1e-12)


def test_ad_pvalue_limit_distribution():
    sigma = math.sqrt(2 * (math.pi**2 - 9) / 3)
    # known upper 10 % and 5 % points of the limiting distribution: 1.933, 2.492
    assert ad_pvalue((1.933 - 1) / sigma) == pytest.approx(0.10, abs=5e-5)
    assert ad_pvalue((2.492 - 1) / sigma) == pytest.approx(0.05, abs=5e-5)
    assert dsc._ad_inf(2 - 1e-12) == pytest.approx(dsc._ad_inf(2), abs=1e-5)
    assert ad_pvalue(-10) == 1 - 1e-5
    assert ad_pvalue(1e3) == 1e-5


def test_ad_statistic_is_scipys_midrank():
    x, y = np.array([1.0, 2, 3, 4, 5]), np.array([2.0, 3, 4, 6, 7, 8])
    # scipy 1.18.1: anderson_ksamp([x, y], variant="midrank").statistic
    assert dsc._ad_statistic(x, y) == pytest.approx(0.47794477729102014, rel=1e-12)


def test_edge_cases():
    with pytest.raises(ValueError, match="at least two groups"):
        rank({("a", 1): [1, 2]})
    with pytest.raises(ValueError, match="fewer than two values"):
        rank({("a", 1): [1], ("b", 1): [1, 2]})
    with pytest.raises(ValueError, match="AD or KS"):
        rank({("a", 1): [1, 2], ("b", 1): [1, 2]}, test="t")
    with pytest.raises(ValueError, match="negative"):
        rank({("a", 1): [1, 2], ("b", 1): [1, 2]}, epsilon=-1)
    five = {(a, 1): [1, 2] for a in "abcde"}
    with pytest.warns(UserWarning, match="5 algorithms"):
        rank(five)


def test_problems_matched_by_name_and_dropped_reported():
    samples = {
        ("a", 1): [1, 2],
        ("b", 1): [3, 4],
        ("a", 2): [1, 2],
        ("b", 3): [1, 2],
        ("b", 4): [5, 6],
        ("a", 4): [7, 8],
    }
    ranking = rank(samples)
    assert ranking.dropped_problems["problem"].to_list() == [2, 3]
    assert set(ranking.ranked_matrix["problem"]) == {1, 4}
    assert ranking.valid_methods == dsc.METHODS_2


def test_too_few_problems():
    ranking = rank({("a", 1): [1, 2], ("b", 1): [3, 4]})
    assert ranking.valid_methods == []
    assert ranking.ranked_matrix.height == 2  # the ranking is still returned
    with pytest.raises(ValueError, match="Not enough problems"):
        get_dsc_omnibus(ranking)


def test_groups_of_several_columns():
    sample = long({("a", 1): [1, 2], ("b", 1): [30, 40]})
    sample = sample.with_columns(sigma=pl.lit(1))
    ranking = get_dsc_ranking(sample, by=["algorithm", "sigma"], problem=PROBLEM)
    assert ranking.ranked_matrix.columns == ["problem", "algorithm", "sigma", "rank"]


# Practical DSC (DS-5, DS-11)


def test_midpoints_sequential():
    x, y = dsc._midpoints([1.0, 5.0, 9.0], [1.4, 9.0], epsilon=0.5)
    assert x.tolist() == [1.2, 5.0, 9.0] and y.tolist() == [1.2, 9.0]
    x, y = dsc._midpoints([1.0], [1.5], epsilon=0.5)  # strictly below epsilon
    assert x.tolist() == [1.0] and y.tolist() == [1.5]


def test_practical_dsc_equates_close_samples():
    base = np.arange(20.0)
    samples = {("a", 1): base, ("b", 1): base + 100}
    assert ranks(rank(samples))[("a", 1)] == 1
    assert ranks(rank(samples, epsilon=200))[("a", 1)] == 1.5
    monte = rank(samples, epsilon=200, monte_carlo_iterations=5, seed=1)
    assert ranks(monte)[("a", 1)] == 1.5


def test_epsilon_zero_is_dsc_and_seed_reproduces():
    rng = np.random.default_rng(3)
    samples = {
        (a, p): rng.normal(i * 0.3, size=15)
        for i, a in enumerate("abc")
        for p in (1, 2)
    }
    plain = rank(samples)
    assert ranks(rank(samples, monte_carlo_iterations=7, seed=2)) == ranks(plain)
    one = rank(samples, epsilon=0.3, monte_carlo_iterations=9, seed=5)
    two = rank(samples, epsilon=0.3, monte_carlo_iterations=9, seed=5)
    assert ranks(one) == ranks(two)


def test_median_rule():
    # different exactly when more than half of the iterations are below the level
    assert dsc._median([0.001, 0.001, 0.5, 0.5]) == 0.5  # half: not different
    assert dsc._median([0.001, 0.001, 0.001, 0.5]) == 0.001
    assert dsc._median([0.3, 0.001, 0.2]) == 0.2


# Step 3: omnibus (DS-6, DS-12 to DS-14)


def test_friedman_example_numbers():
    omnibus = get_dsc_omnibus(ranking_of(example_matrix()))
    assert omnibus.method == "friedman"
    assert omnibus.t == pytest.approx(9.909090909090903, rel=1e-9)
    assert omnibus.p_value == pytest.approx(0.007051284680703929, rel=1e-9)
    assert dict(omnibus.algorithm_means.rows()) == pytest.approx(MEANS, rel=1e-12)
    assert omnibus.message == "The null hypothesis is rejected."
    assert omnibus.n == 22
    weak = get_dsc_omnibus(ranking_of(example_matrix()), alpha=0.007)
    assert weak.message == "The null hypothesis is not rejected."


def test_iman_davenport():
    omnibus = get_dsc_omnibus(ranking_of(example_matrix()), method="iman-davenport")
    chi2, n, k = 9.909090909090903, 22, 3
    f = (n - 1) * chi2 / (n * (k - 1) - chi2)
    assert omnibus.t == pytest.approx(f, rel=1e-12)
    assert omnibus.p_value == pytest.approx(
        stats.f.sf(f, k - 1, (k - 1) * (n - 1)), rel=1e-12
    )
    perfect = {"a": [1.0, 1.0], "b": [2.0, 2.0], "c": [3.0, 3.0]}
    omnibus = get_dsc_omnibus(ranking_of(perfect), method="iman-davenport")
    assert omnibus.t == math.inf and omnibus.p_value == 0


def test_friedman_aligned_ranks():
    matrix = example_matrix()
    r = np.array(list(matrix.values())).T
    n, k = r.shape
    aligned = [r[i, j] - r[i].mean() for i in range(n) for j in range(k)]
    a = stats.rankdata(aligned).reshape(n, k)
    col = sum(a[:, j].sum() ** 2 for j in range(k))
    row = sum(a[i, :].sum() ** 2 for i in range(n))
    t = (
        (k - 1)
        * (col - k * n * n / 4 * (k * n + 1) ** 2)
        / (k * n * (k * n + 1) * (2 * k * n + 1) / 6 - row / k)
    )
    omnibus = get_dsc_omnibus(ranking_of(matrix), method="friedman-aligned-rank")
    assert omnibus.t == pytest.approx(t, rel=1e-12)
    assert omnibus.p_value == pytest.approx(stats.chi2.sf(t, k - 1), rel=1e-12)


def test_wilcoxon_two_algorithms():
    matrix = {"a": [2, 2, 1, 2, 2, 1.5, 2, 2], "b": [1, 1, 2, 1, 1, 1.5, 1, 1]}
    omnibus = get_dsc_omnibus(ranking_of(matrix))
    assert omnibus.method == "wilcoxon-signed-rank-less" and omnibus.t == 0
    d = np.array(matrix["b"]) - np.array(matrix["a"])  # b has the lower mean
    expected = stats.wilcoxon(d, alternative="less").pvalue
    assert omnibus.p_value == pytest.approx(expected, rel=1e-12)
    tied = get_dsc_omnibus(ranking_of({"a": [1.5, 1.5], "b": [1.5, 1.5]}))
    assert tied.p_value == 1


def test_omnibus_method_must_be_valid():
    with pytest.raises(ValueError, match="method 'ANOVA'"):
        get_dsc_omnibus(ranking_of(example_matrix()), method="ANOVA")
    with pytest.raises(ValueError, match="method 'friedman'"):
        get_dsc_omnibus(ranking_of({"a": [1, 2.0], "b": [2, 1.0]}), method="friedman")


# Step 4: post-hoc (DS-7)


def posthoc_of(omnibus, control="CMA-MSR") -> dict:
    frame = get_dsc_posthoc(omnibus).adjusted_p_values.filter(
        pl.col("algorithm") == control
    )
    columns = ["ZValue", "UnadjustedPValue", "Holm", "Hochberg"]
    return {row[0]: row[1:] for row in frame.select("algorithm_vs", *columns).rows()}


def test_posthoc_example_numbers():
    omnibus = get_dsc_omnibus(ranking_of(example_matrix()))
    mine = posthoc_of(omnibus)
    for name, expected in POSTHOC.items():
        assert mine[name] == pytest.approx(expected, rel=1e-9)
    posthoc = get_dsc_posthoc(omnibus)
    assert posthoc.method == "friedman" and posthoc.omnibus is omnibus
    assert posthoc.adjusted_p_values.height == 6  # every algorithm as control


def test_posthoc_follows_the_omnibus_method():
    ranking = ranking_of(example_matrix())
    omnibus = get_dsc_omnibus(ranking, method="friedman-aligned-rank")
    assert get_dsc_posthoc(omnibus).method == "friedman-aligned-rank"
    se = math.sqrt(3 * 23 / 6)
    z = (MEANS["CMA-MSR"] - MEANS["BSif"]) / se
    assert posthoc_of(omnibus)["BSif"][:2] == pytest.approx(
        (z, stats.norm.cdf(z)), rel=1e-12
    )
    davenport = get_dsc_omnibus(ranking, method="iman-davenport")
    assert get_dsc_posthoc(davenport).method == "friedman"
