import functools
import warnings

import numpy as np
import polars as pl
import pytest

from iohinspector import DataManager
from iohinspector.metrics import RestartCostWarning, get_restart_cost, restarts
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
from tests.golden.load import (
    cocopp_level,
    fixture_path,
    golden,
    golden_cocopp,
    golden_renamed,
    logged,
)
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
    sampled = get_ecdf(THREE, x_values=[0.5, 1, 9, 10, 100])
    np.testing.assert_allclose(sampled["ecdf"], [0, 0.1, 0.1, 22 / 30, 22 / 30])
    # without restarts (EC-7) it stays there; with them it rises after 100
    carried = get_ecdf(THREE, x_values=[1000], stopped="carry_forward")
    assert carried["ecdf"].item() == pytest.approx(22 / 30, rel=1e-12)
    assert "n_restarted" not in carried.columns
    restarted = get_ecdf(THREE, x_values=[1000])
    assert restarted["ecdf"].item() > 0.99
    # the three runs stop at 100, before 1000 but not before the default horizon
    assert restarted["n_restarted"].item() == 3
    assert result["n_restarted"].unique().to_list() == [0]
    auc = get_ecdf_auc(THREE, b_hi=1000, continuous=True)
    assert auc.columns == ["algorithm_name", "auc", "n_restarted"]
    assert auc["n_restarted"].item() == 3


@pytest.mark.parametrize("maximization", [False, True])
def test_hit_rule_is_inclusive(maximization):
    df = _runs([[5.0, 3.0, 1.0]])
    targets = pl.DataFrame({"function_id": [1], "dimension": [2], "target": [3.0]})
    result = get_ecdf(df, targets, maximization=maximization)
    first = 1.0 if maximization else 10.0  # 5 >= 3 at once; 3 <= 3 at 10
    assert result.rows() == [("A", 0.0, 0.0, 0), ("A", first, 1.0, 0)]


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
    assert result.rows() == [
        ("A", 0.0, 0.0, 0),
        ("A", 10.0, 0.5, 0),
        ("A", 100.0, 1.0, 0),
    ]
    linear = get_ecdf(
        df, targets, continuous=True, maximization=maximization, scale_f_log=False
    )
    half = 9 / 99 if maximization else 90 / 99
    assert linear["ecdf"].to_list() == pytest.approx([0, half, 1], 1e-12)
    one = targets.head(1).with_columns(target=pl.lit(10.0))
    result = get_ecdf(df, one, continuous=True, maximization=maximization)
    assert result.rows() == [("A", 0.0, 0.0, 0), ("A", 10.0, 1.0, 0)]


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
    auc = get_ecdf_auc(THREE, b_lo=5, b_hi=1000, stopped="carry_forward")["auc"]
    auc = auc.item()
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
        stopped="carry_forward",  # IOHanalyzer's rule (EC-7)
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


# Simulated restarts (EC-7), the default of every ECDF: sampled, or exact
EXACT = restarts.EXACT

# runs 1 and 2 stop at 3 (values 1 and 2), run 3 is first logged at 6 (value 5)
STOPPED_PAIR = STOPPED.with_columns(function_id=1, dimension=2)
STOPPED_BUDGETS = pl.DataFrame({"function_id": 1, "dimension": 2, "budget": [3, 4, 6]})


def _restarted(hits, used, b):
    """E_t(b) by its definition (ecdf.md, EC-7), recursively: ``hits`` the runs'
    hitting times of t (inf when not reached), ``used`` their B_i."""
    failed = [u for t, u in zip(hits, used) if not np.isfinite(t)]

    @functools.cache
    def e(x):
        if x < 0:
            return 0.0
        return (sum(t <= x for t in hits) + sum(e(x - u) for u in failed)) / len(hits)

    return e(b)


def _hitting_times(df, target, maximization=False):
    """Per run of ``df`` (one pair and group): its first evaluation with a value at
    least as good as ``target`` (inf if none), and its last evaluation (B_i)."""
    good = pl.col("raw_y") >= target if maximization else pl.col("raw_y") <= target
    runs = df.group_by("run_id").agg(
        hit=pl.col("evaluations").filter(good).min(), used=pl.col("evaluations").max()
    )
    return runs["hit"].fill_null(np.inf).to_list(), runs["used"].to_list()


def _ragged(seed, n=5):
    """``n`` runs of one pair with random values at evaluations 1..L_i, L_i random."""
    rng = np.random.default_rng(seed)
    rows = [
        (i, 1, 2, "A", e, float(10 ** rng.uniform(-3, 2)))
        for i in range(n)
        for e in range(1, int(rng.integers(3, 13)) + 1)
    ]
    columns = ["run_id", "function_id", "dimension", "algorithm_name"]
    return pl.DataFrame(rows, schema=[*columns, "evaluations", "raw_y"], orient="row")


@pytest.mark.parametrize("maximization", [False, True])
def test_target_ecdf_restarts_are_the_expected_restarted_runs(maximization):
    # E_t(b) computed directly from its definition, per target, averaged
    df = _ragged(3)
    targets = get_ecdf_targets(df, n=6)
    budgets = [1, 4, 9, 13, 20, 37, 60]
    exact = {"maximization": maximization, "stopped": EXACT}
    ours = get_ecdf(df, targets, x_values=budgets, **exact)
    times = [_hitting_times(df, t, maximization) for t in targets["target"]]
    for b, value in zip(budgets, ours["ecdf"]):
        expected = np.mean([_restarted(h, u, b) for h, u in times])
        assert value == pytest.approx(expected, rel=1e-12)
    # the exact steps agree with the samples, up to the largest budget asked for
    steps = get_ecdf(df, targets, x_values=None, **exact)
    carried = get_ecdf(
        df,
        targets,
        x_values=budgets,
        maximization=maximization,
        stopped="carry_forward",
    )
    assert (ours["ecdf"] >= carried["ecdf"] - 1e-12).all()
    at = ours.filter(pl.col("evaluations") <= steps["evaluations"].max())
    for _, b, value, _ in at.rows():
        last = steps.filter(pl.col("evaluations") <= b)["ecdf"][-1]
        assert last == pytest.approx(value, rel=1e-12)


@pytest.mark.parametrize("stopped", ["restart", EXACT])
@pytest.mark.parametrize("divide", [False, True])
def test_restarts_in_chunks_and_sampled_are_the_exact_steps(
    monkeypatch, divide, stopped
):
    # the renewal pass in chunks of 7 evaluations gives the same rows (up to rounding
    # where they are summed); sampled, the restarts are summed at the samples, so
    # the ECDF there is that of the steps
    df = _ragged(11).with_columns(dimension=3)
    targets = get_ecdf_targets(df, n=6)
    xs = [0.5, 2 / 3, 1, 7 / 3, 3.3, 4.1, 9, 13 + 1 / 3, 20, 37, 60]
    options = {"targets": targets, "divide_by_dimension": divide, "stopped": stopped}
    steps = get_ecdf(df, x_values=None, **options)
    sampled = get_ecdf(df, x_values=xs, **options)
    budgets = get_ecdf_budgets(df, 5).with_columns(pl.col("budget") * 4)
    fixed = get_fixed_budget_ecdf(df, budgets, stopped=stopped)
    monkeypatch.setattr(restarts, "CHUNK", 7)
    assert get_ecdf(df, x_values=None, **options).equals(steps)
    assert get_ecdf(df, x_values=xs, **options).equals(sampled)
    chunked = get_fixed_budget_ecdf(df, budgets, stopped=stopped)
    np.testing.assert_allclose(chunked["ecdf"], fixed["ecdf"], rtol=1e-12)
    within = sampled.filter(pl.col("evaluations") <= steps["evaluations"].max())
    assert within.height >= 5
    for _, x, value, _ in within.rows():
        last = steps.filter(pl.col("evaluations") <= x)["ecdf"][-1]
        assert value == pytest.approx(last, rel=1e-12)


@pytest.mark.parametrize("stopped", ["restart", EXACT])
def test_long_curves_do_not_leak_into_a_batch_of_short_ones(monkeypatch, stopped):
    # curves of up to CHUNK evaluations are computed in one array; a longer curve
    # between them (computed on its own) must place no hits in that array
    b = _ragged(12).with_columns(
        algorithm_name=pl.lit("B"), run_id=pl.col("run_id") + 9
    )
    df = pl.concat([_ragged(11), b])
    targets = get_ecdf_targets(df, n=8)
    xs = [1, 4, 9, 13, 20, 37, 60, 150]
    expected = get_ecdf(df, targets, x_values=xs, stopped=stopped)
    for chunk in range(4, 160, 3):
        monkeypatch.setattr(restarts, "CHUNK", chunk)
        ours = get_ecdf(df, targets, x_values=xs, stopped=stopped)
        np.testing.assert_allclose(ours["ecdf"], expected["ecdf"], rtol=1e-12)


def _two_groups():
    b = _ragged(12).with_columns(
        algorithm_name=pl.lit("B"), run_id=pl.col("run_id") + 9
    )
    return pl.concat([_ragged(11), b])


def test_sampled_restarts_are_close_to_the_exact_ones(monkeypatch):
    # the sampled restarts estimate the exact expectation; the error falls as the
    # draws grow, in the target ECDF, its AUC and the fixed-budget ECDF
    df = _two_groups()
    targets = get_ecdf_targets(df, n=8)
    xs = [1, 4, 9, 13, 20, 37, 60, 150, 1000]
    budgets = get_ecdf_budgets(df, 5).with_columns(pl.col("budget") * 4)

    def both(stopped):
        return (
            get_ecdf(df, targets, x_values=xs, stopped=stopped)["ecdf"],
            get_ecdf_auc(df, targets, b_hi=1000, stopped=stopped)["auc"],
            get_fixed_budget_ecdf(df, budgets, stopped=stopped)["ecdf"],
        )

    exact = both(EXACT)
    for ours, expected in zip(both("restart"), exact):
        assert (ours - expected).abs().max() < 0.03
    monkeypatch.setattr(restarts, "SAMPLES", 100_000)
    for ours, expected in zip(both("restart"), exact):
        assert (ours - expected).abs().max() < 0.003


def test_sampled_restarts_reach_their_limit_exactly():
    # once every draw lies within the budget, a target that one run reaches counts
    # as reached by all runs, as with the exact restarts
    targets = pl.DataFrame({"function_id": 1, "dimension": 2, "target": [1e-4, 1e0]})
    ecdf = get_ecdf(THREE, targets, x_values=[1e6])["ecdf"].item()
    assert ecdf == pytest.approx(1.0, rel=1e-12)


def test_sampled_restarts_depend_only_on_the_groups_own_runs():
    # the same uniforms serve every run set: adding a group changes no other group
    df = _two_groups()
    targets = get_ecdf_targets(df, n=8)
    xs = [4, 20, 150]
    alone = get_ecdf(df.filter(algorithm_name="A"), targets, x_values=xs)
    with_b = get_ecdf(df, targets, x_values=xs).filter(algorithm_name="A")
    np.testing.assert_allclose(with_b["ecdf"], alone["ecdf"], rtol=1e-12)
    again = get_ecdf(df.filter(algorithm_name="A"), targets, x_values=xs)
    assert again.equals(alone)


def test_sampled_restarts_do_not_depend_on_the_order_of_tied_runs():
    # four runs end at the same value after different budgets; the draws index
    # the failed runs, so their order must not depend on the order of the input
    finals = [1.0, 1.0, 1.0, 1.0, 1e-4, 1e-4]
    lengths = [10, 20, 40, 80, 30, 50]
    rows = [
        (i, 1, 2, "A", e, finals[i] if e == n else 10.0)
        for i, n in enumerate(lengths)
        for e in range(1, n + 1)
    ]
    columns = ["run_id", "function_id", "dimension", "algorithm_name"]
    df = pl.DataFrame(rows, schema=[*columns, "evaluations", "raw_y"], orient="row")
    targets = pl.DataFrame({"function_id": 1, "dimension": 2, "target": [1e-3]})
    xs = [50, 100, 200, 400, 800]
    expected = get_ecdf(df, targets, x_values=xs)
    for seed in range(20):
        order = np.random.default_rng(seed).permutation(len(lengths)).tolist()
        shuffled = pl.concat([df.filter(run_id=i) for i in order])
        ours = get_ecdf(shuffled, targets, x_values=xs)
        np.testing.assert_array_equal(ours["ecdf"], expected["ecdf"])


def test_exact_restarts_estimate_their_cost_and_warn(monkeypatch):
    df = _ragged(11)
    short, long = get_restart_cost(df, horizon=20), get_restart_cost(df, horizon=2000)
    assert set(short) == {"curves", "evaluations", "seconds", "gigabytes"}
    assert 0 < short["evaluations"] < long["evaluations"]
    assert 0 < short["seconds"] < long["seconds"]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RestartCostWarning)
        get_ecdf(df, x_values=[1, 100], stopped=EXACT)  # cheap: no warning
    monkeypatch.setattr(restarts, "WARN_SECONDS", 0.0)
    with pytest.warns(RestartCostWarning, match="restart_exact"):
        get_ecdf(df, x_values=[1, 100], stopped=EXACT)
    with pytest.warns(RestartCostWarning):
        get_fixed_budget_ecdf(df, stopped=EXACT)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RestartCostWarning)
        get_ecdf(df, x_values=[1, 100])  # sampled: never
    with pytest.raises(ValueError, match="simulated chains"):
        aggregate_convergence(df, stopped=EXACT)


@pytest.mark.parametrize("chunk", [None, 7])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"stopped": EXACT},
        {"scale_eval_log": True, "b_lo": 2},
        {"scale_eval_log": True, "b_lo": 2, "stopped": EXACT},
        {"divide_by_dimension": True, "b_lo": 0.5},
        {"continuous": True, "scale_eval_log": True},
    ],
)
def test_restarted_auc_is_the_area_under_the_steps(monkeypatch, options, chunk):
    # the AUC sums the restarts' weighted increments per chunk instead of building
    # the step function: the same as the exact area under get_ecdf's steps
    b = _ragged(12).with_columns(
        algorithm_name=pl.lit("B"), run_id=pl.col("run_id") + 9
    )
    df = pl.concat([_ragged(11), b]).with_columns(dimension=3)
    if chunk:
        monkeypatch.setattr(restarts, "CHUNK", chunk)
    auc = get_ecdf_auc(df, **options).sort("algorithm_name")
    lo, log = options.get("b_lo", 1.0), options.get("scale_eval_log", False)
    keep = ("divide_by_dimension", "continuous", "stopped")
    steps = get_ecdf(df, **{k: v for k, v in options.items() if k in keep})
    for (name,), curve in steps.sort("algorithm_name").group_by(
        "algorithm_name", maintain_order=True
    ):
        x, ecdf = curve["evaluations"].to_numpy(), curve["ecdf"].to_numpy()
        hi = x.max()
        axis = np.log10 if log else (lambda v: v)
        ends = axis(np.clip(np.r_[x[1:], hi], lo, hi))
        area = np.sum(ecdf * (ends - axis(np.clip(x, lo, hi))))
        value = auc.filter(pl.col("algorithm_name") == name)["auc"].item()
        assert value == pytest.approx(area / (axis(hi) - axis(lo)), rel=1e-12)


def test_runs_that_reach_a_target_are_not_restarted():
    # every run reaches 1e4 at evaluation 1: restarts change nothing; no run reaches
    # 1e-5: it stays unreached; for 1e0 two runs stop (at 100) without reaching it
    targets = pl.DataFrame({"function_id": 1, "dimension": 2, "target": [1e4]})
    xs = [1, 100, 1e4]
    assert get_ecdf(THREE, targets, x_values=xs)["ecdf"].to_list() == [1, 1, 1]
    never = targets.with_columns(target=pl.lit(1e-5))
    assert get_ecdf(THREE, never, x_values=xs)["ecdf"].to_list() == [0, 0, 0]
    once = targets.with_columns(target=pl.lit(1e-4))
    ecdf = get_ecdf(THREE, once, x_values=[100, 200, 300], stopped=EXACT)
    ecdf = ecdf["ecdf"].to_list()
    # E(b) = 1[b >= 10]/3 + 2/3 E(b - 100)
    assert ecdf == pytest.approx([1 / 3, 1 / 3 + 2 / 9, 1 / 3 + 2 / 9 + 4 / 27])


def test_continuum_and_fixed_budget_restarts_are_the_same_restarts():
    # EC-5 and EC-10 with restarts: the AUC over m targets tends to the AOCC, and the
    # fixed-budget AUC is the mean continuum ECDF over its budgets
    other = _ragged(6, 4).with_columns(run_id=pl.col("run_id") + 9, function_id=2)
    df = pl.concat([_ragged(5, 6), other.cast({"function_id": pl.Int64})])
    aocc = get_ecdf_auc(df, continuous=True, b_hi=40)["auc"].item()
    carried = get_ecdf_auc(df, continuous=True, b_hi=40, stopped="carry_forward")
    assert aocc > carried["auc"].item() + 0.05
    for m in (10, 301):
        auc = get_ecdf_auc(df, get_ecdf_targets(df, n=m), b_hi=40)["auc"].item()
        assert abs(auc - aocc) < 1 / m
    budgets = get_ecdf_budgets(df, 6).with_columns(pl.col("budget") * 3)
    lo, hi = 1e-3, 1e2
    auc = get_fixed_budget_ecdf_auc(df, budgets, f_lo=lo, f_hi=hi)
    bounds = (
        budgets.select(PAIR)
        .unique()
        .join(pl.DataFrame({"target": [lo, hi]}), how="cross")
    )
    per_pair = [
        get_ecdf(
            df.filter(function_id=f),
            bounds.filter(function_id=f),
            continuous=True,
            x_values=budgets.filter(function_id=f)["budget"],
        )["ecdf"].mean()
        for f in (1, 2)
    ]
    assert auc["auc"].item() == pytest.approx(np.mean(per_pair), rel=1e-12)


@pytest.mark.parametrize("maximization", [False, True])
def test_fixed_budget_restarts_are_the_expected_restarted_runs(maximization):
    df = _ragged(7)
    budgets = pl.DataFrame({"function_id": 1, "dimension": 2, "budget": [2, 7, 15, 31]})
    ours = get_fixed_budget_ecdf(df, budgets, maximization=maximization, stopped=EXACT)
    for _, v, value, _ in ours.rows()[1:]:
        h, u = _hitting_times(df, v, maximization)
        expected = np.mean([_restarted(h, u, b) for b in budgets["budget"]])
        assert value == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_restarts_only_improve_on_carrying_forward():
    xs = [1.0, 2.0, 5.0, 8.0, 9.0]
    carried = get_fixed_budget_ecdf(
        STOPPED_PAIR, STOPPED_BUDGETS, x_values=xs, stopped="carry_forward"
    )
    assert "n_restarted" not in carried.columns
    # at 3 and 4 the stopped runs keep 1 and 2, at 6 also run 3 has 5
    np.testing.assert_allclose(carried["ecdf"], [3 / 9, 6 / 9, 7 / 9, 7 / 9, 7 / 9])
    # v = 1: E(b) = 1[b >= 3]/3 + E(b - 3)/3 + E(b - 6)/3 is 1/3, 1/3 and 4/9
    restarted = get_fixed_budget_ecdf(
        STOPPED_PAIR, STOPPED_BUDGETS, x_values=xs, stopped=EXACT
    )
    np.testing.assert_allclose(
        restarted["ecdf"], [10 / 27, 6 / 9, 7 / 9, 7 / 9, 7 / 9], rtol=1e-12
    )
    assert restarted["n_restarted"].unique().to_list() == [2]
    auc = get_fixed_budget_ecdf_auc(STOPPED_PAIR, STOPPED_BUDGETS)
    before = get_fixed_budget_ecdf_auc(
        STOPPED_PAIR, STOPPED_BUDGETS, stopped="carry_forward"
    )
    assert auc["auc"].item() > before["auc"].item()
    assert auc["n_restarted"].item() == 2


def test_restarts_are_the_mean_of_aggregate_convergences_chains():
    # one procedure: FB-6 draws the restarts, EC-7 takes their expected value; with
    # 30000 chains the fraction of values at most 1 is within 5 standard errors
    copies = 10000
    run_id = pl.col("run_id") * copies + pl.col("copy")
    many = STOPPED.join(pl.DataFrame({"copy": range(copies)}), how="cross")
    many = many.with_columns(run_id=run_id).drop("copy")
    summary = aggregate_convergence(
        many,
        budgets=[3, 4, 6],
        stopped="restart",
        custom_op=lambda s: float((s <= 1).sum()),
    )
    drawn = (summary["<lambda>"] / summary["n_runs"]).mean()
    assert abs(drawn - 10 / 27) < 5 * np.sqrt(10 / 27 * 17 / 27 / 30000)


def test_restarts_are_deterministic_mirrored_and_need_no_evals_used():
    first = get_fixed_budget_ecdf(STOPPED_PAIR, STOPPED_BUDGETS)
    assert first.equals(get_fixed_budget_ecdf(STOPPED_PAIR, STOPPED_BUDGETS))
    # without evals_used B_i is the last logged evaluation, here the same
    unused = STOPPED_PAIR.drop("evals_used")
    assert first.equals(get_fixed_budget_ecdf(unused, STOPPED_BUDGETS))
    high = get_fixed_budget_ecdf(
        STOPPED_PAIR.with_columns(-pl.col("raw_y")), STOPPED_BUDGETS, maximization=True
    )
    assert high.with_columns(-pl.col("raw_y")).equals(first)
    alone = get_fixed_budget_ecdf_auc(STOPPED_PAIR, STOPPED_BUDGETS, by=[])
    assert alone.columns == ["auc", "n_restarted"]
    for function in (get_fixed_budget_ecdf, get_ecdf):
        with pytest.raises(ValueError, match="carry_forward"):
            function(STOPPED_PAIR, stopped="exclude")


# Validation against cocopp 2.9.1 (ecdf.md, "Implementation (I33)")

COCOPP_SAMPLES = 20000  # tests/golden/cocopp/generate.py


def _coco_old(dimension):
    """DIRECT on f6 (five runs, mostly unsuccessful for small targets); ``raw_y`` is
    the precision."""
    manager = DataManager()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        manager.add_folder(fixture_path("coco_old"))
    columns = ["function_id", "dimension", "algorithm_name", "evals_used"]
    return manager.load(include_columns=columns).filter(dimension=dimension)


def _within_cocopp(joined, samples=None):
    """Every cell within 5 standard errors of cocopp's sample of 20000, and of
    ours of ``samples`` if sampled."""
    p = joined["ours"].clip(0, 1)
    se = (p * (1 - p) / COCOPP_SAMPLES).sqrt()
    if samples:  # p from both: a sampled 0 or 1 has no error of its own
        p = ((joined["ours"] + joined["ecdf"]) / 2).clip(0, 1)
        se = (p * (1 - p) * (1 / COCOPP_SAMPLES + 1 / samples)).sqrt()
    diff = (joined["ours"] - joined["ecdf"]).abs()
    assert (diff <= 5 * se + 1e-12).all()
    return diff


@pytest.mark.parametrize("stopped", [EXACT, "restart"])
@pytest.mark.parametrize("dimension", [2, 3])
def test_restarts_match_cocopp(dimension, stopped):
    # cocopp's runtime ECDF with simulated restarts (drawSP, as pprldmany) gives per
    # target t and budget b the fraction of 20000 simulated runtimes T(t) <= b. Ours
    # is its expected value: the target ECDF of t at b, and, as T(t) <= b exactly
    # when V(b) <= t (EC-9), the fixed-budget ECDF with the one budget b at v = t
    # (cocopp's grid level of t)
    gold = golden_cocopp(f"ecdf_restarts_{dimension}D", "coco_old")
    data = _coco_old(dimension)
    pair = data.select(PAIR).unique()
    targets = gold["target"].unique().sort()
    levels = [cocopp_level(t) for t in targets]
    budget = gold["budget"].unique().sort()
    target_ecdf = pl.concat(
        get_ecdf(
            data, pair.with_columns(target=level), x_values=budget, stopped=stopped
        ).select(target=pl.lit(t), budget="evaluations", ours="ecdf")
        for t, level in zip(targets, levels)
    )
    fixed_budget = pl.concat(
        get_fixed_budget_ecdf(
            data, pair.with_columns(budget=int(b)), x_values=levels, stopped=stopped
        ).select(target=targets, budget=pl.lit(b), ours="ecdf")
        for b in budget
    )
    cells = []
    for ours in (target_ecdf, fixed_budget):
        joined = gold.join(ours, on=["target", "budget"], maintain_order="left")
        assert joined.height == gold.height == 510
        _within_cocopp(joined, None if stopped == EXACT else restarts.SAMPLES)
        cells.append(joined["ours"])
    np.testing.assert_allclose(*cells, rtol=1e-12, atol=1e-15)
    # the fixed-budget ECDF over the ten budgets is the mean of a target's cells
    budgets = pair.join(pl.DataFrame({"budget": budget.cast(pl.Int64)}), how="cross")
    mean = gold.group_by("target").agg(pl.col("ecdf").mean()).sort("target")["ecdf"]
    ecdf = get_fixed_budget_ecdf(data, budgets, x_values=levels, stopped=stopped)
    assert (ecdf["ecdf"] - mean).abs().max() < 0.005
    carried = get_fixed_budget_ecdf(
        data, budgets, x_values=levels, stopped="carry_forward"
    )["ecdf"]
    assert (carried - mean).abs().max() > 0.05  # what restarts change
