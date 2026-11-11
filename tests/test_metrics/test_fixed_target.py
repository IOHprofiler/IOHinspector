import polars as pl

from iohinspector.metrics.fixed_target import aggregate_running_time

DF = pl.DataFrame(
    {
        "evaluations": [1, 10, 20, 1, 15, 26],
        "raw_y": [1.0, 0.7, 0.1, 0.9, 0.3, 0.2],
        "algorithm_name": ["A", "A", "A", "B", "B", "B"],
        "run_id": [1, 1, 1, 2, 2, 2],
    }
)


def test_basic_aggregation():
    result = aggregate_running_time(DF)
    assert "mean" in result.columns
    assert "ERT" in result.columns
    assert "PAR-10" in result.columns
    assert result.height > 0

    # Assert the value of success_count for A is 1 and for B is 0
    # You can use filter as shown, or use row indexing with .row or .to_dicts()
    a_success_count = result.filter(
        (pl.col("algorithm_name") == "A") & (pl.col("raw_y") == 0.1)
    )["success_count"].to_list()[0]
    b_success_count = result.filter(
        (pl.col("algorithm_name") == "B") & (pl.col("raw_y") == 0.1)
    )["success_count"].to_list()[0]

    assert a_success_count == 1
    assert b_success_count == 0


def test_return_as_pandas():
    result = aggregate_running_time(DF)
    assert hasattr(result, "to_numpy")  # pandas DataFrame


def test_custom_op():
    def my_sum(s):
        return float(s.sum())

    result = aggregate_running_time(DF, custom_op=my_sum)
    assert "my_sum" in result.columns


def test_maximization():
    # Should not raise error
    df = pl.DataFrame(
        {
            "evaluations": [1, 10, 20, 1, 15, 26],
            "raw_y": [0.1, 0.7, 1.0, 0.2, 0.3, 0.9],
            "algorithm_name": ["A", "A", "A", "B", "B", "B"],
            "run_id": [1, 1, 1, 2, 2, 2],
        }
    )

    result = aggregate_running_time(df, maximization=True)
    assert result.height > 0

    a_success_count = result.filter(
        (pl.col("algorithm_name") == "A") & (pl.col("raw_y") >= 0.9)
    )["success_count"].to_list()[0]
    b_success_count = result.filter(
        (pl.col("algorithm_name") == "B") & (pl.col("raw_y") >= 0.9)
    )["success_count"].to_list()[0]

    assert a_success_count == 1
    assert b_success_count == 0


def test_with_f_min_f_max():
    result = aggregate_running_time(DF, f_min=0.2, f_max=0.5)
    assert result["raw_y"].min() >= 0.2
    assert result["raw_y"].max() <= 0.5


def test_with_different_free_variables():
    result = aggregate_running_time(DF, by=["algorithm_name", "run_id"])
    assert "algorithm_name" in result.columns
    assert "run_id" in result.columns


def test_success_ratio_and_count():
    # Add a non-finite value
    df = DF.with_columns(
        [
            pl.when(pl.col("evaluations") == 3)
            .then(float("nan"))
            .otherwise(pl.col("evaluations"))
            .alias("evaluations")
        ]
    )
    result = aggregate_running_time(df)
    assert "success_ratio" in result.columns
    assert "success_count" in result.columns
    assert (result["success_ratio"] <= 1).all()
