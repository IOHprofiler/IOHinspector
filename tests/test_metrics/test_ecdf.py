import numpy as np
import polars as pl
import pytest

from iohinspector.metrics.ecdf import get_data_ecdf, get_ecdf, get_ecdf_auc
from iohinspector.targets import get_ecdf_targets

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


def test_ecdf_with_custom_eval_values():
    eval_values = [2, 4]
    result = get_data_ecdf(
        DF, eval_values=eval_values, scale_eval_log=False, scale_f_log=False
    )
    algo1_eaf = (
        result.filter(pl.col("algorithm_name") == "algo1")["eaf"].to_numpy().copy()
    )
    algo1_eaf.sort()
    np.testing.assert_allclose(algo1_eaf, [2 / 3, 1])

    algo2_eaf = (
        result.filter(pl.col("algorithm_name") == "algo2")["eaf"].to_numpy().copy()
    )
    algo2_eaf.sort()
    np.testing.assert_allclose(algo2_eaf, [0, 1 / 3])


def test_ecdf_with_maximization():
    result = get_data_ecdf(DF, maximization=True)
    # eaf_raw_y should be between 0 and 1
    algo1_eaf = result.filter(pl.col("algorithm_name") == "algo1")["eaf"].to_numpy()
    # Assert that all values in algo1_eaf are 0 and the array is not empty
    np.testing.assert_allclose(algo1_eaf, [0, 0, 0, 0, 0])

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
    np.testing.assert_allclose(algo1_eaf, [2 / 3, 5 / 6, 1])

    algo2_eaf = (
        result.filter(pl.col("algorithm_name") == "algo2")["eaf"].to_numpy().copy()
    )
    algo2_eaf.sort()
    np.testing.assert_allclose(algo2_eaf, [0, 1 / 6, 1 / 3])


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
