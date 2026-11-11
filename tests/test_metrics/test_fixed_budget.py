import numpy as np
import polars as pl
import pytest

from iohinspector.metrics.fixed_budget import aggregate_convergence

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
    # Should contain columns for mean, min, max, median, std, geometric_mean
    for col in ["mean", "min", "max", "median", "std", "geometric_mean"]:
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
    assert result["mean"][0] == pytest.approx(42.0)
