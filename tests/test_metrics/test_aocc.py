import numpy as np
import polars as pl

from iohinspector.metrics import get_aocc

# Simple dataset with two groups and two run_ids
DF = pl.DataFrame(
    {
        "run_id": [1, 1, 1, 2, 2, 2],
        "function_name": ["f1", "f1", "f1", "f1", "f1", "f1"],
        "algorithm_name": ["alg1", "alg1", "alg1", "alg1", "alg1", "alg1"],
        "evaluations": [0, 5, 10, 0, 5, 10],
        "eaf": [10.0, 7.0, 4.0, 12.0, 9.0, 6.0],
    }
)


def test_basic_aocc():
    # AOCC should be computed for the group
    result = get_aocc(DF, eval_max=10)
    assert isinstance(result, pl.DataFrame)

    assert "AOCC" in result.columns
    aocc_val = result["AOCC"][0]
    assert aocc_val == 6.5

    result = get_aocc(DF, eval_max=10)
    assert isinstance(result, pl.DataFrame)


def test_multiple_groups():
    # Add a second group
    df = DF.with_columns(
        [pl.Series("function_name", ["f1", "f1", "f1", "f2", "f2", "f2"])]
    )

    result = get_aocc(df, eval_max=10)
    assert "AOCC" in result.columns
    aocc_f1_val = result.filter(pl.col("function_name") == "f1")["AOCC"][0]
    aocc_f2_val = result.filter(pl.col("function_name") == "f2")["AOCC"][0]
    assert aocc_f1_val == 5.5
    assert aocc_f2_val == 7.5


def test_custom_fval_col():
    # Use a different column for fval_var
    df = DF.rename({"eaf": "custom_col"})
    result = get_aocc(df, eval_max=10, fval_var="custom_col")
    assert "AOCC" in result.columns
    aocc_val = result["AOCC"][0]
    assert aocc_val == 6.5


def test_custom_free_vars():
    # Use only algorithm_name as free var
    result = get_aocc(DF, eval_max=10, by=["algorithm_name"])
    aocc_val = result["AOCC"][0]
    assert aocc_val == 6.5


def test_aocc_with_missing_evaluations():
    # Remove some evaluation steps to test fill_null
    df = pl.DataFrame(
        {
            "run_id": [1, 1, 1, 2, 2],
            "function_name": ["f1", "f1", "f1", "f1", "f1"],
            "algorithm_name": ["alg1", "alg1", "alg1", "alg1", "alg1"],
            "evaluations": [0, 5, 10, 0, 10],
            "eaf": [10.0, 8.0, 4.0, 12.0, 6.0],
        }
    )
    result = get_aocc(df, eval_max=10)

    assert "AOCC" in result.columns
    aocc_val = result["AOCC"][0]
    assert aocc_val == 6


def test_aocc_zero_budget():
    # Test with max_budget=0 (should handle gracefully)
    df = DF
    result = get_aocc(df, eval_max=0)
    assert "AOCC" in result.columns
    # AOCC should be nan or 0
    aocc_val = result["AOCC"][0]
    assert np.isnan(aocc_val) or aocc_val == 0


def test_aocc_log():
    df = pl.DataFrame(
        {
            "run_id": [1, 1, 1, 2, 2, 2],
            "function_name": ["f1", "f1", "f1", "f1", "f1", "f1"],
            "algorithm_name": ["alg1", "alg1", "alg1", "alg1", "alg1", "alg1"],
            "evaluations": [1, 10, 100, 1, 10, 100],
            "eaf": [10.0, 7.0, 4.0, 12.0, 9.0, 6.0],
        }
    )
    result = get_aocc(df, eval_max=100, scale_eval_log=True)
    aocc_val = result["AOCC"][0]
    assert aocc_val == 6.5
