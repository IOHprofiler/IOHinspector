import numpy as np
import polars as pl
import pytest

from iohinspector.metrics.eaf import (
    get_discritized_eaf_single_objective,
    get_eaf_data,
    get_eaf_diff_data,
    get_eaf_pareto_data,
)

DATA = pl.DataFrame(
    {
        "evaluations": [1, 10, 100, 1000],
        "raw_y": [1.0, 0.1, 0.01, 0.001],
        "run_id": [1, 1, 1, 1],
    }
)

MULTI_DATA = pl.DataFrame(
    {
        "evaluations": [1, 10, 100, 1000, 1, 10, 100, 1000],
        "raw_y": [1.0, 0.1, 0.01, 0.001, 1.5, 0.15, 0.015, 0.0015],
        "run_id": [1, 1, 1, 1, 2, 2, 2, 2],
    }
)

# Simple predictable data: constant improvement
SIMPLE_DATA = pl.DataFrame(
    {
        "evaluations": [1, 2, 3, 4, 5],
        "raw_y": [5.0, 4.0, 3.0, 2.0, 1.0],  # Decreasing linearly
        "run_id": [1, 1, 1, 1, 1],
    }
)

# Two identical runs for predictable EAF
DUAL_DATA = pl.DataFrame(
    {
        "evaluations": [1, 2, 3, 1, 2, 3],
        "raw_y": [10.0, 5.0, 1.0, 10.0, 5.0, 1.0],  # Same values for both runs
        "run_id": [1, 1, 1, 2, 2, 2],
    }
)

# Simple predictable Pareto front data
# Run 1: Points that clearly dominate each other
# Run 2: Same structure but slightly worse
SIMPLE_MO_DATA = pl.DataFrame(
    {
        "obj1": [3.0, 2.0, 1.0, 3.5, 2.5, 0.5],  # Minimization objective
        "obj2": [1.0, 2.0, 3.0, 1.5, 2.5, 3.0],  # Minimization objective
        "run_id": [1, 1, 1, 2, 2, 2],
    }
)

SIMPLE_RESULTS = pl.DataFrame(
    {
        "obj1": [3.0, 2.0, 1.0, 3.5, 2.5, 0.5],
        "obj2": [1.0, 2.0, 3.0, 1.5, 2.5, 3.0],
        "eaf": [0.5, 0.5, 1.0, 1.0, 1.0, 0.5],
    }
)

# Identical runs for predictable EAF = 50%
IDENTICAL_MO_DATA = pl.DataFrame(
    {
        "obj1": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "obj2": [3.0, 2.0, 1.0, 3.0, 2.0, 1.0],
        "run_id": [1, 1, 1, 2, 2, 2],
    }
)

# Dataset 1: Better performance (lower values for minimization)
BETTER_DATA = pl.DataFrame(
    {"obj1": [1.0, 2.0, 3.0], "obj2": [3.0, 2.0, 1.0], "run_id": [1, 1, 1]}
)

# Dataset 2: Worse performance (higher values)
WORSE_DATA = pl.DataFrame(
    {"obj1": [2.0, 3.0, 4.0], "obj2": [4.0, 3.0, 2.0], "run_id": [1, 1, 1]}
)

# Identical datasets for predictable diff = 0
IDENTICAL_DATA1 = pl.DataFrame(
    {"obj1": [1.0, 2.0], "obj2": [2.0, 1.0], "run_id": [1, 1]}
)

IDENTICAL_DATA2 = pl.DataFrame(
    {"obj1": [2.0, 1.0], "obj2": [1.0, 2.0], "run_id": [1, 1]}
)


def test_basic_single_run_id():
    result = get_discritized_eaf_single_objective(DATA)

    assert isinstance(result, pl.DataFrame)
    assert result.columns[0] == "eaf_target"
    assert result.width == 11  # eaf_target and the default x_targets
    assert result.height == 101  # default y_targets
    # Assert all values are 1 or 0
    columns = [str(e) for e in DATA["evaluations"]]
    assert np.isin(result[columns].to_numpy(), [1, 0]).all()
    assert result["1"][-1] == 0
    assert result["1000"][0] == 1


def test_basic_multi_run_id():
    result = get_discritized_eaf_single_objective(MULTI_DATA)
    assert result.columns[0] == "eaf_target"
    assert result.width == 11  # eaf_target and the default x_targets
    assert result.height == 101  # default y_targets
    # Assert all values are 1, 0.5 or 0
    columns = [str(e) for e in MULTI_DATA["evaluations"].unique()]
    assert np.isin(result[columns].to_numpy(), [1, 0.5, 0]).all()
    assert result["1"][-1] == 0
    assert result["1000"][0] == 1


def test_custom_eval_values():
    eval_values = [1, 3, 5]
    result = get_discritized_eaf_single_objective(DATA, eval_values=eval_values)
    assert all(str(x) in result.columns for x in eval_values)


def test_custom_eval_min_max():
    result = get_discritized_eaf_single_objective(
        DATA, eval_min=2, eval_max=4, eval_targets=2
    )
    assert all(str(x) in result.columns for x in [2, 4])


def test_custom_f_min_max_targets():
    result = get_discritized_eaf_single_objective(
        DATA, f_min=1e-12, f_max=1.0, f_targets=5
    )
    assert result.shape[0] == 5
    assert result["eaf_target"].min() == pytest.approx(0.0)
    assert result["eaf_target"].max() == pytest.approx(1.0)


def test_scale_eval_log_and_f_log():
    result = get_discritized_eaf_single_objective(
        DATA, scale_f_log=False, scale_eval_log=False
    )
    # Check that all values except the last row are 1, and the last row is 0
    values = result.drop("eaf_target").to_numpy()
    assert np.all(values[:-1] == 1)
    assert np.all(values[-1] == 0)

    budgets = [float(c) for c in result.columns[1:]]
    np.testing.assert_allclose(budgets, np.linspace(1, 1000, 10))


def test_basic_with_simple_data():
    """Test with simple predictable data"""
    result = get_eaf_data(SIMPLE_DATA, eval_min=1, eval_max=5, scale_eval_log=False)
    assert isinstance(result, pl.DataFrame)
    assert "evaluations" in result.columns
    assert "raw_y" in result.columns
    assert "run_id" in result.columns

    # Check that we have the expected number of rows (should be same as input)
    assert len(result) == len(SIMPLE_DATA)

    # Check that run_id is preserved
    assert result["run_id"].unique().to_list() == [1]

    # Check evaluation values are within expected range
    assert (result["evaluations"] >= 1).all()
    assert (result["evaluations"] <= 5).all()


def test_dual_runs_predictable_eaf():
    result = get_eaf_data(DUAL_DATA, eval_min=1, eval_max=3, scale_eval_log=False)

    assert isinstance(result, pl.DataFrame)
    assert "evaluations" in result.columns
    assert "raw_y" in result.columns
    assert "run_id" in result.columns

    # Should have both run_ids
    assert len(result["run_id"].unique()) == 2
    assert set(result["run_id"].unique()) == {1, 2}

    # Should have expected number of rows
    assert len(result) == len(DUAL_DATA)


def test_eaf_data_return_type():
    assert isinstance(get_eaf_data(SIMPLE_DATA), pl.DataFrame)


def test_simple_pareto_fronts():
    """Test with simple, predictable Pareto front data"""
    result = get_eaf_pareto_data(SIMPLE_MO_DATA, "obj1", "obj2")
    assert isinstance(result, pl.DataFrame)
    assert "eaf" in result.columns
    assert "obj1" in result.columns
    assert "obj2" in result.columns
    # Should have some data points
    assert len(result) > 0

    for row in result.iter_rows(named=True):
        obj1 = row["obj1"]
        obj2 = row["obj2"]
        eaf_value = row["eaf"]
        expected_eaf = SIMPLE_RESULTS.filter(
            (pl.col("obj1") == obj1) & (pl.col("obj2") == obj2)
        )["eaf"].to_list()[0]

        assert eaf_value == pytest.approx(expected_eaf)


def test_identical_runs_eaf():
    result = get_eaf_pareto_data(IDENTICAL_MO_DATA, "obj1", "obj2")
    assert isinstance(result, pl.DataFrame)
    assert "eaf" in result.columns
    assert "obj1" in result.columns
    assert "obj2" in result.columns

    # Should have some data points
    assert len(result) > 0

    max_per_pair = result.group_by(["obj1", "obj2"]).agg(pl.col("eaf").max())
    assert np.allclose(max_per_pair["eaf"].to_numpy(), 1.0)


def test_eaf_pareto_data_return_type():
    result = get_eaf_pareto_data(SIMPLE_MO_DATA, "obj1", "obj2")
    assert isinstance(result, pl.DataFrame)


def test_clear_performance_difference():
    """Test with clearly better vs worse datasets"""
    result = get_eaf_diff_data(BETTER_DATA, WORSE_DATA, "obj1", "obj2")

    assert isinstance(result, pl.DataFrame)
    assert "eaf_diff" in result.columns
    assert "x_min" in result.columns
    assert "y_min" in result.columns
    assert "x_max" in result.columns
    assert "y_max" in result.columns

    # Should have some rectangles with differences
    assert len(result) > 0

    # Check that rectangle coordinates are valid
    assert (result["x_min"] <= result["x_max"]).all()
    assert (result["y_min"] <= result["y_max"]).all()

    # Check for no NaN values
    assert result.null_count().sum_horizontal()[0] == 0
    assert not np.isnan(result.to_numpy()).any()

    # Since better_data dominates worse_data, should have positive differences
    assert result["eaf_diff"].max() > 0


def test_identical_datasets_zero_diff():
    """Test with identical datasets - should get minimal or no differences"""
    result = get_eaf_diff_data(IDENTICAL_DATA1, IDENTICAL_DATA2, "obj1", "obj2")
    assert isinstance(result, pl.DataFrame)
    assert "eaf_diff" in result.columns

    # Result should be either empty or contain only very small differences
    assert len(result) == 0 or abs(result["eaf_diff"]).max() < 0.1


def test_eaf_diff_data_return_type():
    result = get_eaf_diff_data(BETTER_DATA, WORSE_DATA, "obj1", "obj2")
    assert isinstance(result, pl.DataFrame)
