import numpy as np
import polars as pl

from iohinspector.metrics.ecdf import get_data_ecdf

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
