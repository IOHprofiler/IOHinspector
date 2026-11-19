import numpy as np
import polars as pl
import pytest

from iohinspector.metrics.ecdf import (
    get_data_ecdf,
    get_ecdf,
    get_ecdf_auc,
    get_fixed_budget_ecdf,
    get_fixed_budget_ecdf_auc,
    get_rank_table,
)
from iohinspector.metrics.fixed_budget import aggregate_convergence
from iohinspector.targets import get_ecdf_budgets, get_ecdf_targets
from tests.golden.load import golden, golden_renamed, logged
from tests.test_metrics.test_fixed_budget import STOPPED

pytestmark = pytest.mark.filterwarnings("ignore:get_data_ecdf:DeprecationWarning")

# A simple synthetic dataset
DF = pl.DataFrame(
    {
        "evaluations": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "raw_y": [10, 8, 6, 4, 2, 18, 16, 14, 12, 10],
        "algorithm_name": ["algo1"] * 5 + ["algo2"] * 5,
        "run_id": [1] * 5 + [2] * 5,
    }
)


def test_basic_ecdf():
    result = get_data_ecdf(DF, scale_eval_log=False, scale_f_log=False)
    algo1_eaf = (
        result.filter(pl.col("algorithm_name") == "algo1")["eaf"].to_numpy().copy()
    )
    algo1_eaf.sort()
    np.testing.assert_allclose(algo1_eaf, [0.5, 0.625, 0.75, 0.875, 1])

    algo2_eaf = (
        result.filter(pl.col("algorithm_name") == "algo2")["eaf"].to_numpy().copy()
    )
    algo2_eaf.sort()
    np.testing.assert_allclose(algo2_eaf, [0, 0.125, 0.25, 0.375, 0.5])


def test_get_data_ecdf_is_deprecated():
    with pytest.warns(DeprecationWarning, match="get_ecdf"):
        get_data_ecdf(DF)


# Since 0.2.0 get_data_ecdf is get_ecdf(continuous=True): the bounds are the
# observed range of the pair (2 and 18), not of the values at the budgets.
def test_ecdf_with_custom_eval_values():
    eval_values = [2, 4]
    result = get_data_ecdf(
        DF, eval_values=eval_values, scale_eval_log=False, scale_f_log=False
    )
    algo1_eaf = (
        result.filter(pl.col("algorithm_name") == "algo1")["eaf"].to_numpy().copy()
    )
    algo1_eaf.sort()
    np.testing.assert_allclose(algo1_eaf, [10 / 16, 14 / 16])

    algo2_eaf = (
        result.filter(pl.col("algorithm_name") == "algo2")["eaf"].to_numpy().copy()
    )
    algo2_eaf.sort()
    np.testing.assert_allclose(algo2_eaf, [2 / 16, 6 / 16])


def test_ecdf_with_maximization():
    result = get_data_ecdf(DF, maximization=True)
    # best-so-far 10 (algo1) and 18 (algo2) between the bounds 2 and 18, log10
    algo1_eaf = result.filter(pl.col("algorithm_name") == "algo1")["eaf"].to_numpy()
    np.testing.assert_allclose(algo1_eaf, [np.log10(5) / np.log10(9)] * 5)

    algo2_eaf = result.filter(pl.col("algorithm_name") == "algo2")["eaf"].to_numpy()
    np.testing.assert_allclose(algo2_eaf, [1, 1, 1, 1, 1])


def test_ecdf_with_custom_bounds():
    result = get_data_ecdf(
        DF, f_min=0, f_max=100, scale_eval_log=False, scale_f_log=False
    )
    algo1_eaf = (
        result.filter(pl.col("algorithm_name") == "algo1")["eaf"].to_numpy().copy()
    )
    algo1_eaf.sort()
    np.testing.assert_allclose(
        algo1_eaf, [90 / 100, 92 / 100, 94 / 100, 96 / 100, 98 / 100]
    )

    algo2_eaf = (
        result.filter(pl.col("algorithm_name") == "algo2")["eaf"].to_numpy().copy()
    )
    algo2_eaf.sort()
    np.testing.assert_allclose(
        algo2_eaf, [82 / 100, 84 / 100, 86 / 100, 88 / 100, 90 / 100]
    )


def test_ecdf_with_eval_min_eval_max():
    result = get_data_ecdf(
        DF, eval_min=2, eval_max=4, scale_eval_log=False, scale_f_log=False
    )
    algo1_eaf = (
        result.filter(pl.col("algorithm_name") == "algo1")["eaf"].to_numpy().copy()
    )
    algo1_eaf.sort()
    np.testing.assert_allclose(algo1_eaf, [10 / 16, 12 / 16, 14 / 16])

    algo2_eaf = (
        result.filter(pl.col("algorithm_name") == "algo2")["eaf"].to_numpy().copy()
    )
    algo2_eaf.sort()
    np.testing.assert_allclose(algo2_eaf, [2 / 16, 4 / 16, 6 / 16])


def test_run_without_a_value_yet_keeps_its_group():
    # run 2 is first logged at evaluation 5, so it has no value at budget 1; that row
    # still belongs to algorithm A (it used to form a separate group with a null name)
    df = pl.DataFrame(
        {
            "run_id": [1, 1, 2, 2],
            "algorithm_name": ["A"] * 4,
            "evaluations": [1, 10, 5, 10],
            "raw_y": [4.0, 2.0, 3.0, 1.0],
        }
    )
    result = get_data_ecdf(
        df, f_min=1, f_max=4, scale_f_log=False, eval_values=[1, 5, 10]
    )
    assert result["algorithm_name"].to_list() == ["A"] * 3
    assert result["evaluations"].to_list() == [1, 5, 10]


# The target ECDF (docs/definitions/ecdf.md, EC-1 to EC-3, EC-6)


def _runs(values, evaluations=(1, 10, 100), function_id=1, algorithm="A", first=1):
    """One row per (run, evaluation): run i logs ``values[i]``."""
    rows = [
        (first + i, function_id, 2, algorithm, e, y)
        for i, ys in enumerate(values)
        for e, y in zip(evaluations, ys)
    ]
    columns = ["run_id", "function_id", "dimension", "algorithm_name"]
    return pl.DataFrame(rows, schema=[*columns, "evaluations", "raw_y"], orient="row")


# The worked example of ecdf.md: three runs, 10 log-linear targets from 1e-4 to 1e4
THREE = _runs([[1e4, 1e0, 1e0], [1e4, 1e-2, 1e-2], [1e4, 1e-4, 1e-4]])


def _strict(targets, maximization=False):
    """IOHanalyzer's strict f < t as f <= the next float below t (above, maximising)."""
    end = np.inf if maximization else -np.inf
    return targets.with_columns(
        pl.col("target").map_batches(lambda t: pl.Series(np.nextafter(t, end)))
    )


def _at(result, budget):
    return result.filter(pl.col("evaluations") == budget)["ecdf"].item()


def test_worked_example_070_0733_075():
    targets = get_ecdf_targets(THREE)
    assert targets["target"][[0, -1]].to_list() == [1e-4, 1e4]
    # IOHanalyzer's strict comparison: no run reaches the target equal to its value
    assert _at(get_ecdf(THREE, _strict(targets), x_values=[100]), 100) == 0.7
    # EC-1: inclusive, the agreed target ECDF
    assert _at(get_ecdf(THREE, x_values=[100]), 100) == pytest.approx(22 / 30, 1e-12)
    # EC-5, EC-6: the continuum ECDF with bounds 1e-4 and 1e4, as the legacy one
    assert _at(get_ecdf(THREE, continuous=True, x_values=[100]), 100) == 0.75
    legacy = get_data_ecdf(THREE, f_min=1e-4, f_max=1e4, eval_values=[100])
    assert legacy["eaf"].item() == 0.75


def test_exact_steps_and_samples():
    # every run reaches 1 of 10 targets at evaluation 1, then 4, 6 and 8 more at 10
    result = get_ecdf(THREE)
    assert result["evaluations"].to_list() == [0.0, 1.0, 10.0]
    np.testing.assert_allclose(result["ecdf"], [0, 0.1, 22 / 30], rtol=1e-12)
    sampled = get_ecdf(THREE, x_values=[0.5, 1, 9, 10, 1000])
    np.testing.assert_allclose(sampled["ecdf"], [0, 0.1, 0.1, 22 / 30, 22 / 30])


@pytest.mark.parametrize("maximization", [False, True])
def test_hit_rule_is_inclusive(maximization):
    df = _runs([[5.0, 3.0, 1.0]])
    targets = pl.DataFrame({"function_id": [1], "dimension": [2], "target": [3.0]})
    result = get_ecdf(df, targets, maximization=maximization)
    first = 1.0 if maximization else 10.0  # 5 >= 3 at once; 3 <= 3 at 10
    assert result.rows() == [("A", 0.0, 0.0), ("A", first, 1.0)]


def test_a_group_that_reaches_nothing_is_kept():
    df = pl.concat([THREE, _runs([[1e5] * 3], algorithm="B", first=9)])
    result = get_ecdf(df, get_ecdf_targets(THREE), x_values=[100])
    assert result.sort("algorithm_name")["ecdf"].to_list()[1] == 0.0


def test_pairs_have_equal_weight():
    # pair 1: one run reaches its only target; pair 2: one of three runs does
    df = pl.concat(
        [_runs([[1.0]], (1,)), _runs([[1.0], [9.0], [9.0]], (1,), 2, first=2)]
    )
    targets = pl.DataFrame(
        {"function_id": [1, 2], "dimension": [2, 2], "target": [1.0, 1.0]}
    )
    result = get_ecdf(df, targets, x_values=[1])
    assert result["ecdf"].item() == pytest.approx((1 + 1 / 3) / 2, 1e-12)
    by_pair = get_ecdf(df, targets, ["algorithm_name", "function_id"], x_values=[1])
    assert by_pair.sort("function_id")["ecdf"].to_list() == pytest.approx([1, 1 / 3])


def test_a_pair_split_over_files_counts_once():
    split = pl.concat([THREE.filter(pl.col("run_id") < 3), THREE.filter(run_id=3)])
    assert get_ecdf(split).equals(get_ecdf(THREE))


def test_common_pairs_by_default_with_a_warning():
    # A has pairs 1 and 2, B only pair 1
    df = pl.concat([THREE, _runs([[1e4] * 3], function_id=2, first=4)])
    df = pl.concat([df, _runs([[1e5] * 3], algorithm="B", first=7)])
    targets = get_ecdf_targets(THREE)
    targets = pl.concat(
        [targets, targets.with_columns(function_id=2)], how="vertical_relaxed"
    )
    with pytest.warns(UserWarning, match=r"pairs='common'.*B has no \(2, 2\)"):
        common = get_ecdf(df, targets, x_values=[100])
    with pytest.warns(UserWarning, match=r"pairs='all'.*B has no \(2, 2\)"):
        own = get_ecdf(df, targets, x_values=[100], pairs="all")
    a = pl.col("algorithm_name") == "A"
    assert common.filter(a)["ecdf"].item() == pytest.approx(22 / 30, 1e-12)
    # pair 2: the run reaches only the largest target
    assert own.filter(a)["ecdf"].item() == pytest.approx((22 / 30 + 0.1) / 2, 1e-12)
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="share no pair"):
        get_ecdf(df.filter(~a | (pl.col("function_id") == 2)))
    with pytest.raises(ValueError, match="common or all"):
        get_ecdf(THREE, pairs="some")


def test_divide_by_dimension():
    result = get_ecdf(THREE, divide_by_dimension=True)
    assert result["evaluations"].to_list() == [0.0, 0.5, 5.0]


@pytest.mark.parametrize("maximization", [False, True])
def test_continuum_ecdf_quality(maximization):
    # bounds 1 and 100 on a log scale: 10 is halfway
    df = _runs([[100.0, 10.0, 1.0] if not maximization else [1.0, 10.0, 100.0]])
    targets = pl.DataFrame(
        {"function_id": [1, 1], "dimension": [2, 2], "target": [1.0, 100.0]}
    )
    result = get_ecdf(df, targets, continuous=True, maximization=maximization)
    assert result.rows() == [("A", 0.0, 0.0), ("A", 10.0, 0.5), ("A", 100.0, 1.0)]
    linear = get_ecdf(
        df, targets, continuous=True, maximization=maximization, scale_f_log=False
    )
    half = 9 / 99 if maximization else 90 / 99
    assert linear["ecdf"].to_list() == pytest.approx([0, half, 1], 1e-12)
    one = targets.head(1).with_columns(target=pl.lit(10.0))
    result = get_ecdf(df, one, continuous=True, maximization=maximization)
    assert result.rows() == [("A", 0.0, 0.0), ("A", 10.0, 1.0)]


@pytest.mark.parametrize("maximization", [False, True])
def test_continuum_ecdf_of_integer_values(maximization):
    # integer values: the bounds 1 and 100 must not be truncated by clip
    df = _runs([[100, 10, 1] if not maximization else [1, 10, 100]])
    targets = pl.DataFrame(
        {"function_id": [1, 1], "dimension": [2, 2], "target": [1.0, 99.5]}
    )
    result = get_ecdf(df, targets, continuous=True, maximization=maximization)
    floats = get_ecdf(
        df.cast({"raw_y": pl.Float64}),
        targets,
        continuous=True,
        maximization=maximization,
    )
    assert result.equals(floats)


# AUC (EC-4) and AOCC (EC-5)


def test_auc_is_the_exact_normalised_area():
    # ECDF 0.1 on [1, 10), 22/30 on [10, 100]
    auc = get_ecdf_auc(THREE)["auc"].item()
    assert auc == pytest.approx((0.1 * 9 + 22 / 30 * 90) / 99, 1e-12)
    auc = get_ecdf_auc(THREE, b_lo=5, b_hi=1000)["auc"].item()
    assert auc == pytest.approx((0.1 * 5 + 22 / 30 * 990) / 995, 1e-12)
    log = get_ecdf_auc(THREE, scale_eval_log=True)["auc"].item()
    assert log == pytest.approx((0.1 + 22 / 30) / 2, 1e-12)
    over = get_ecdf_auc(THREE, divide_by_dimension=True)["auc"].item()
    assert over == pytest.approx((0.1 * 4 + 22 / 30 * 45) / 49, 1e-12)
    with pytest.raises(ValueError, match="empty"):
        get_ecdf_auc(THREE, b_lo=100, b_hi=10)
    with pytest.raises(ValueError, match="positive"):
        get_ecdf_auc(THREE, b_lo=0, scale_eval_log=True)


def test_aocc_is_the_mean_area_under_the_quality():
    # q: run 1 0.5, run 2 0.75, run 3 1 from evaluation 10 on (bounds 1e-4, 1e4)
    aocc = get_ecdf_auc(THREE, continuous=True)["auc"].item()
    assert aocc == pytest.approx(0.75 * 90 / 99, 1e-12)
    log = get_ecdf_auc(THREE, continuous=True, scale_eval_log=True)
    assert log["auc"].item() == pytest.approx(0.375, 1e-12)


@pytest.mark.parametrize("maximization", [False, True])
def test_aocc_is_the_limit_of_the_target_auc(maximization):
    # EC-5 equivalence: the AUC over m log-spaced targets tends to the AOCC
    rng = np.random.default_rng(1)
    df = _runs((10 ** rng.uniform(-6, 3, (8, 30))).tolist(), range(1, 31))
    options = {"scale_eval_log": True, "maximization": maximization}
    aocc = get_ecdf_auc(df, continuous=True, **options)["auc"].item()
    for m in (10, 1001):  # each run's fraction of targets is within 1/m of q
        targets = get_ecdf_targets(df, n=m)
        auc = get_ecdf_auc(df, targets, **options)["auc"].item()
        assert abs(auc - aocc) < 1 / m


# Multi-function rank tables (EC-8)


def test_rank_table_ties_inf_and_missing():
    summary = pl.DataFrame(
        {
            "function_id": [1] * 6 + [2] * 2,
            "dimension": [2] * 8,
            "ERT": [5.0, 3.0, 3.0, np.inf, None, np.nan, 7.0, 1.0],
        }
    )
    ranks = get_rank_table(summary)["rank"].to_list()
    assert ranks == [3.0, 1.5, 1.5, 4.0, 5.5, 5.5, 2.0, 1.0]
    mean = summary.rename({"ERT": "mean"}).with_columns(pl.col("mean").fill_nan(9.0))
    ranks = get_rank_table(mean, "mean", higher_is_better=True)["rank"].to_list()
    assert ranks == [3.0, 4.5, 4.5, 1.0, 6.0, 2.0, 1.0, 2.0]


@pytest.mark.parametrize(
    ("fixture", "metric"),
    [
        ("ioh040_min", "Aggr_DIM"),
        ("ioh040_min", "Aggr_funcId"),
        ("ioh040_max", "Aggr_DIM"),
        ("ioh040_max", "Aggr_funcId"),
        ("extdata", "Aggr_funcId"),
    ],
)
def test_rank_table_matches_iohanalyzer(fixture, metric):
    # generate.Aggr ranks ERT per (DIM, funcId) with frank(na.last = TRUE); its ERT
    # values carry RT-1 (and Aggr_funcId mixes the dimensions' targets), so the
    # ranks are compared on its own values
    gold = golden(fixture, metric)
    summary = gold.drop("rank").rename({"DIM": "dimension", "funcId": "function_id"})
    ours = get_rank_table(summary, "value")
    assert ours["rank"].to_list() == gold["rank"].to_list()


# Golden tests against IOHanalyzer (WP I03): generate_data.ECDF and generate_data.AUC
# with 5 log-linear targets per pair, on its 50-point linear grid. IOHanalyzer
# compares strictly (f < t); with the next float below each target (above, when
# maximising) the inclusive ECDF is the same function, so the results must agree
# exactly. The agreed, inclusive ECDF (EC-1) is never lower.
GOLDEN = pytest.mark.parametrize("fixture", ["ioh040_min", "ioh040_max"])
PAIR = ["function_id", "dimension"]
BY_PAIR = ["algorithm_name", *PAIR]


def _golden(fixture, metric):
    """A golden output, with the pair of a per-pair call as ``PAIR``."""
    gold = golden_renamed(fixture, metric)
    if "pair_f" not in gold.columns:
        return gold
    pair = [
        pl.col(c).cast(pl.Int32).alias(n) for c, n in zip(["pair_f", "pair_d"], PAIR)
    ]
    return gold.with_columns(pair)


def _golden_ecdf(fixture, metric, by, strict=True):
    maximization = fixture == "ioh040_max"
    targets = golden_renamed(fixture, "RT_sample").select(*PAIR, "target").unique()
    if strict:
        targets = _strict(targets, maximization)
    gold = _golden(fixture, metric).rename({"x": "evaluations"})
    ours = get_ecdf(
        logged(fixture),
        targets,
        by,
        x_values=gold["evaluations"].unique(),
        maximization=maximization,
    )
    joined = gold.join(ours, on=[*by, "evaluations"], how="left")
    assert joined.height == gold.height
    return joined


@GOLDEN
@pytest.mark.parametrize("by", [BY_PAIR, ["algorithm_name"]], ids=["pair", "pooled"])
def test_ecdf_matches_iohanalyzer(fixture, by):
    metric = "ECDF_by_RT" if by == BY_PAIR else "ECDF_default_targets"
    joined = _golden_ecdf(fixture, metric, by)
    np.testing.assert_allclose(joined["ecdf"], joined["mean"], rtol=0, atol=1e-12)
    inclusive = _golden_ecdf(fixture, metric, by, strict=False)
    assert (inclusive["ecdf"] >= inclusive["mean"] - 1e-12).all()
    assert (inclusive["ecdf"] > inclusive["mean"] + 1e-12).any()


@GOLDEN
def test_auc_matches_iohanalyzer_trapezoid(fixture):
    # EC-4: IOHanalyzer integrates its grid with the trapezoid rule and divides by
    # the upper budget, not by the range; only this test reproduces that
    x, y = pl.col("evaluations"), pl.col("ecdf")
    area = ((y + y.shift(1)) / 2 * x.diff()).over(BY_PAIR)
    ours = (
        _golden_ecdf(fixture, "ECDF_by_RT", BY_PAIR)
        .sort(BY_PAIR + ["evaluations"])
        .with_columns(area=area)
        .group_by(BY_PAIR)
        .agg(raw=pl.col("area").sum(), x=x.max())
    )
    for metric, ours_auc in [("AUC_by_RT_raw", "raw"), ("AUC_by_RT", "normalised")]:
        gold = _golden(fixture, metric)
        joined = gold.join(
            ours.with_columns(normalised=pl.col("raw") / pl.col("x")),
            on=[*BY_PAIR, "x"],
        )
        assert joined.height == gold.height
        np.testing.assert_allclose(joined[ours_auc], joined["auc"], rtol=1e-12)


# The fixed-budget ECDF (EC-9) and its AUC (EC-10)

BUDGETS = pl.DataFrame({"function_id": [1, 1], "dimension": [2, 2], "budget": [1, 10]})


@pytest.mark.parametrize("maximization", [False, True])
def test_fixed_budget_ecdf_is_exact(maximization):
    # at budget 1 every run has 1e4; at 10 the runs have 1, 1e-2 and 1e-4
    result = get_fixed_budget_ecdf(THREE, BUDGETS, maximization=maximization)
    if maximization:  # the best-so-far value stays 1e4
        assert result["raw_y"].to_list() == [np.inf, 1e4]
        assert result["ecdf"].to_list() == pytest.approx([0, 1], 1e-12)
        return
    assert result["raw_y"].to_list() == [-np.inf, 1e-4, 1e-2, 1.0, 1e4]
    np.testing.assert_allclose(result["ecdf"], [0, 1 / 6, 2 / 6, 3 / 6, 1])
    sampled = get_fixed_budget_ecdf(THREE, BUDGETS, x_values=[1e-5, 1e-3, 1, 1e5])
    np.testing.assert_allclose(sampled["ecdf"], [0, 1 / 6, 3 / 6, 1])


@pytest.mark.parametrize("maximization", [False, True])
def test_fixed_budget_hit_rule_is_inclusive(maximization):
    # one run: 5, 3, 1 at evaluations 1, 10, 100; budget 10 and value 3 count
    df = _runs([[5.0, 3.0, 1.0]])
    budgets = BUDGETS.tail(1)
    result = get_fixed_budget_ecdf(
        df, budgets, x_values=[3.0], maximization=maximization
    )
    assert result["ecdf"].item() == 1.0
    before = get_fixed_budget_ecdf(
        df, budgets.with_columns(budget=9), x_values=[3], maximization=maximization
    )
    assert before["ecdf"].item() == float(maximization)  # 5 >= 3, but 5 > 3


def test_fixed_budget_ecdf_without_a_value_yet():
    # run 2 is first logged at evaluation 5: at budget 1 it has not reached anything
    df = pl.DataFrame(
        {
            "run_id": [1, 1, 2, 2],
            "function_id": [1] * 4,
            "dimension": [2] * 4,
            "algorithm_name": ["A"] * 4,
            "evaluations": [1, 10, 5, 10],
            "raw_y": [4.0, 2.0, 3.0, 1.0],
        }
    )
    result = get_fixed_budget_ecdf(df, BUDGETS.head(1), x_values=[10.0])
    assert result["ecdf"].item() == 0.5


def test_default_budgets():
    budgets = get_ecdf_budgets(THREE)
    assert budgets["budget"].to_list() == [1, 2, 3, 5, 8, 13, 22, 36, 60, 100]
    assert get_fixed_budget_ecdf(THREE).equals(get_fixed_budget_ecdf(THREE, budgets))


@pytest.mark.parametrize("maximization", [False, True])
@pytest.mark.parametrize("scale_f_log", [False, True])
def test_fixed_budget_auc_is_the_mean_continuum_ecdf(maximization, scale_f_log):
    # EC-10: the area over [f_lo, f_hi] is the mean over the budgets of the
    # continuum ECDF with the bounds f_lo and f_hi
    rng = np.random.default_rng(2)
    df = pl.concat(
        [
            _runs((10 ** rng.uniform(-6, 3, (5, 30))).tolist(), range(1, 31)),
            _runs(
                (10 ** rng.uniform(-6, 3, (4, 30))).tolist(), range(1, 31), 2, first=9
            ),
        ]
    )
    budgets = get_ecdf_budgets(df, 7)
    lo, hi = 1e-4, 1e2
    options = {"maximization": maximization, "scale_f_log": scale_f_log}
    auc = get_fixed_budget_ecdf_auc(df, budgets, f_lo=lo, f_hi=hi, **options)
    bounds = (
        budgets.select(PAIR)
        .unique()
        .join(pl.DataFrame({"target": [lo, hi]}), how="cross")
    )
    continuum = get_ecdf(
        df, bounds, continuous=True, x_values=budgets["budget"].unique(), **options
    )
    assert auc["auc"].item() == pytest.approx(continuum["ecdf"].mean(), rel=1e-12)


def test_fixed_budget_auc_defaults_and_errors():
    # default range: the observed values 1e-4 to 1e4, on log10
    assert get_fixed_budget_ecdf_auc(THREE, BUDGETS)["auc"].item() == 0.375
    linear = get_fixed_budget_ecdf_auc(THREE, BUDGETS, scale_f_log=False)
    expected = ((1e4 - 1) + (1e4 - 1e-2) + (1e4 - 1e-4)) / 6 / (1e4 - 1e-4)
    assert linear["auc"].item() == pytest.approx(expected, rel=1e-12)
    with pytest.raises(ValueError, match="empty"):
        get_fixed_budget_ecdf_auc(THREE, f_lo=1, f_hi=1)
    with pytest.raises(ValueError, match="positive"):
        get_fixed_budget_ecdf_auc(THREE, f_lo=0, f_hi=1)


def _golden_fixed_budget(fixture, strict=True):
    """Our fixed-budget ECDF per pair on IOHanalyzer's value grid, joined to
    ``ECDF_by_FV``. IOHanalyzer's T < b is T <= b - 1 for integer evaluations."""
    maximization = fixture == "ioh040_max"
    budgets = golden_renamed(fixture, "FV_sample").select(
        *PAIR, budget=pl.col("runtime") - int(strict)
    )
    gold = _golden(fixture, "ECDF_by_FV").with_row_index()
    ours = get_fixed_budget_ecdf(
        logged(fixture),
        budgets.unique(),
        BY_PAIR,
        x_values=gold["x"].unique(),
        maximization=maximization,
        stopped="carry_forward",  # IOHanalyzer's rule (EC-7)
    )
    joined = gold.join(
        ours.rename({"raw_y": "x"}), on=[*BY_PAIR, "x"], how="left"
    ).sort("index")
    assert joined.height == gold.height
    return joined


@GOLDEN
def test_fixed_budget_ecdf_matches_iohanalyzer(fixture):
    joined = _golden_fixed_budget(fixture)
    np.testing.assert_allclose(joined["ecdf"], joined["mean"], rtol=0, atol=1e-12)
    inclusive = _golden_fixed_budget(fixture, strict=False)  # EC-9
    assert (inclusive["ecdf"] >= inclusive["mean"] - 1e-12).all()
    assert (inclusive["ecdf"] > inclusive["mean"] + 1e-12).any()


@GOLDEN
def test_fixed_budget_auc_matches_iohanalyzer_trapezoid(fixture):
    # EC-10: IOHanalyzer runs the trapezoid rule over its value grid in the grid's
    # order (largest value first when minimising) and divides by the last value;
    # only this test reproduces that. The agreed AUC lies in [0, 1]
    x, y = pl.col("x"), pl.col("ecdf")
    area = ((y + y.shift(1)) / 2 * x.diff()).over(BY_PAIR)
    ours = (
        _golden_fixed_budget(fixture)
        .with_columns(area=area)
        .group_by(BY_PAIR, maintain_order=True)
        .agg(auc=pl.col("area").sum() / x.last(), x=x.last())
    )
    gold = _golden(fixture, "AUC_by_FV")
    joined = gold.join(ours, on=[*BY_PAIR, "x"], suffix="_ours")
    assert joined.height == gold.height
    np.testing.assert_allclose(joined["auc_ours"], joined["auc"], rtol=1e-12)
    maximization = fixture == "ioh040_max"
    agreed = get_fixed_budget_ecdf_auc(
        logged(fixture), by=BY_PAIR, maximization=maximization
    )["auc"]
    assert ((agreed >= 0) & (agreed <= 1)).all()


# Simulated restarts (EC-7, FB-6), the default of the fixed-budget ECDF

# runs 1 and 2 stop at 3 (values 1 and 2), run 3 is first logged at 6 (value 5)
STOPPED_PAIR = STOPPED.with_columns(function_id=1, dimension=2)
STOPPED_BUDGETS = pl.DataFrame({"function_id": 1, "dimension": 2, "budget": [3, 4, 6]})


def test_fixed_budget_restarts_are_those_of_aggregate_convergence():
    # one procedure: with the same seed the restarted values are the ones the
    # fixed-budget summary has, so the ECDF's mass at each value adds up to the
    # same sum, minimum and maximum
    for seed in range(5):
        ecdf = get_fixed_budget_ecdf(STOPPED_PAIR, STOPPED_BUDGETS, seed=seed)
        summary = aggregate_convergence(
            STOPPED, budgets=[3, 4, 6], stopped="restart", seed=seed
        )
        mass = ecdf["ecdf"].diff()[1:] * 9  # 3 runs x 3 budgets
        values = ecdf["raw_y"][1:]
        total = (summary["mean"] * summary["n_defined"]).sum()
        assert (values * mass).sum() == pytest.approx(total, rel=1e-12)
        assert (values.min(), values.max()) == (
            summary["best"].min(),
            summary["worst"].max(),
        )
        assert ecdf["n_restarted"].unique().to_list() == [2]


def test_restarts_only_improve_on_carrying_forward():
    xs = [1.0, 2.0, 5.0, 8.0, 9.0]
    carried = get_fixed_budget_ecdf(
        STOPPED_PAIR, STOPPED_BUDGETS, x_values=xs, stopped="carry_forward"
    )
    assert "n_restarted" not in carried.columns
    # at 3 and 4 the stopped runs keep 1 and 2, at 6 also run 3 has 5
    np.testing.assert_allclose(carried["ecdf"], [3 / 9, 6 / 9, 7 / 9, 7 / 9, 7 / 9])
    for seed in range(5):
        restarted = get_fixed_budget_ecdf(
            STOPPED_PAIR, STOPPED_BUDGETS, x_values=xs, seed=seed
        )
        assert (restarted["ecdf"] >= carried["ecdf"] - 1e-12).all()
        auc = get_fixed_budget_ecdf_auc(STOPPED_PAIR, STOPPED_BUDGETS, seed=seed)
        before = get_fixed_budget_ecdf_auc(
            STOPPED_PAIR, STOPPED_BUDGETS, stopped="carry_forward"
        )
        assert auc["auc"].item() >= before["auc"].item()
        assert auc["n_restarted"].item() == 2


def test_restarts_are_seeded_mirrored_and_need_no_evals_used():
    options = {"budgets": STOPPED_BUDGETS, "seed": 1}
    first = get_fixed_budget_ecdf(STOPPED_PAIR, **options)
    assert first.equals(get_fixed_budget_ecdf(STOPPED_PAIR, **options))
    # without evals_used B_i is the last logged evaluation, here the same
    unused = STOPPED_PAIR.drop("evals_used")
    assert first.equals(get_fixed_budget_ecdf(unused, **options))
    high = get_fixed_budget_ecdf(
        STOPPED_PAIR.with_columns(-pl.col("raw_y")), maximization=True, **options
    )
    assert high.with_columns(-pl.col("raw_y")).equals(first)
    alone = get_fixed_budget_ecdf_auc(STOPPED_PAIR, by=[], **options)
    assert alone.columns == ["auc", "n_restarted"]
    with pytest.raises(ValueError, match="carry_forward"):
        get_fixed_budget_ecdf(STOPPED_PAIR, stopped="exclude")
