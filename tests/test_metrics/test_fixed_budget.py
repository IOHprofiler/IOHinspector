import numpy as np
import polars as pl
import pytest

from iohinspector.metrics.fixed_budget import aggregate_convergence, get_overview

DF = pl.DataFrame(
    {
        "evaluations": [1, 2, 3, 1, 2, 3, 1, 3, 1, 3],
        "raw_y": [30, 20, 10, 35, 25, 15, 40, 30, 20, 10],
        "algorithm_name": ["A", "A", "A", "A", "A", "A", "B", "B", "B", "B"],
        "run_id": [0, 0, 0, 1, 1, 1, 2, 2, 3, 3],
    }
)


def test_basic_aggregation():
    result = aggregate_convergence(DF)
    for col in ["mean", "best", "worst", "median", "sd", "q50", "geometric_mean"]:
        assert col in result.columns
    # Should have 6 rows (3 evals x 2 algs)
    assert len(result) == 6
    # Check that means are correct for one group
    mean_a = result.filter(pl.col("algorithm_name") == "A")["mean"].to_numpy()
    np.testing.assert_allclose(mean_a, [32.5, 22.5, 12.5])
    mean_b = result.filter(pl.col("algorithm_name") == "B")["mean"].to_numpy()
    np.testing.assert_allclose(mean_b, [30, 30, 20])


def test_custom_op():
    def custom_sum(s):
        return s.sum()  # Sum the Series and return as float

    result = aggregate_convergence(DF, custom_op=custom_sum)
    assert "custom_sum" in result.columns

    # Check that custom_sum is correct for one group
    sum_a = result.filter(pl.col("algorithm_name") == "A")["custom_sum"].to_numpy()
    np.testing.assert_allclose(sum_a, [65, 45, 25])
    sum_a = result.filter(pl.col("algorithm_name") == "B")["custom_sum"].to_numpy()
    np.testing.assert_allclose(sum_a, [60, 60, 40])


def test_maximization():
    # Should not affect aggregation, but test for code path
    result = aggregate_convergence(DF, maximization=True)
    assert "mean" in result.columns


def test_eval_min_eval_max():
    # Limit to a subset of evaluations
    result = aggregate_convergence(DF, eval_min=2, eval_max=3)
    assert (result["evaluations"] >= 2).all()
    assert (result["evaluations"] <= 3).all()


def test_return_polars():
    result = aggregate_convergence(DF)
    assert isinstance(result, pl.DataFrame)


def test_free_variables():
    # Use a different free variable
    df = DF.with_columns(pl.lit("foo").alias("other_var"))
    result = aggregate_convergence(df, by=["other_var"])
    assert "other_var" in result.columns


def test_empty_data():
    empty_df = DF.filter(pl.col("evaluations") > 100)
    with pytest.raises(ValueError):
        aggregate_convergence(empty_df)


def test_single_row():
    single_df = pl.DataFrame(
        {"evaluations": [1], "raw_y": [42], "algorithm_name": ["A"], "run_id": [0]}
    )
    result = aggregate_convergence(single_df)
    assert len(result) == 1
    assert result["mean"][0] == pytest.approx(42.0, abs=5e-8)


def test_geometric_mean_needs_positive_values():
    # FB-3
    df = DF.with_columns(
        raw_y=pl.when(pl.col("run_id") == 3).then(0.0).otherwise("raw_y")
    )
    result = aggregate_convergence(df, budgets=[3])
    assert result.filter(algorithm_name="A")["geometric_mean"].item() > 0
    assert result.filter(algorithm_name="B")["geometric_mean"].item() is None


def test_best_and_worst_follow_the_direction():
    result = aggregate_convergence(DF, budgets=[3], maximization=True)
    assert result.filter(algorithm_name="A").select("best", "worst").row(0) == (35, 30)


# two runs of 3 evaluations, values 9 then 1 and 8 then 2; run 3 used 6 evaluations
STOPPED = pl.DataFrame(
    {
        "run_id": [1, 1, 2, 2, 3],
        "evaluations": [1, 3, 1, 3, 6],
        "raw_y": [9.0, 1.0, 8.0, 2.0, 5.0],
        "evals_used": [3, 3, 3, 3, 6],
        "algorithm_name": "A",
    }
)


def test_restart_continues_stopped_runs():
    # FB-6: at 4 a stopped run restarts with run 1 or 2 (9 or 8, worse than its own
    # final value) or run 3 (null at its evaluation 1); at 6 a restart with run 1 or
    # 2 reaches 1 or 2. Run 3 (B = 6) is never restarted.
    for seed in range(5):
        result = aggregate_convergence(
            STOPPED, budgets=[3, 4, 6], stopped="restart", seed=seed
        )
        carried = aggregate_convergence(STOPPED, budgets=[3, 4, 6])
        assert result["n_restarted"].to_list() == [0, 2, 2]
        assert (
            result["n_active"].to_list() == carried["n_active"].to_list() == [3, 1, 1]
        )
        assert result["mean"][:2].to_list() == carried["mean"][:2].to_list()
        assert result["mean"][2] <= carried["mean"][2]
    first = aggregate_convergence(STOPPED, budgets=[3, 4, 6], stopped="restart", seed=1)
    again = aggregate_convergence(STOPPED, budgets=[3, 4, 6], stopped="restart", seed=1)
    assert first.equals(again)


def test_restart_needs_evals_used():
    with pytest.raises(ValueError, match="evals_used"):
        aggregate_convergence(DF, stopped="restart")
    with pytest.raises(ValueError, match="carry_forward"):
        aggregate_convergence(DF, stopped="exclude")


def test_overview():
    row = get_overview(STOPPED).row(0, named=True)
    assert row == {
        "algorithm_name": "A",
        "n_runs": 3,
        "budget": 6,
        "best_reached": 1.0,
        "worst_reached": 5.0,
        "mean_reached": 8 / 3,
        "median_reached": 2.0,
        "n_best": 1,
        "worst_recorded": 9.0,
        "min_runtime": 1,
        "max_runtime": 6,
    }
