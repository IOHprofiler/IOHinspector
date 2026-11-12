import numpy as np
import polars as pl
import pytest

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
    assert "par_10" in result.columns
    assert result.height > 0

    # Assert the value of success_count for A is 1 and for B is 0
    # You can use filter as shown, or use row indexing with .row or .to_dicts()
    a_success_count = result.filter(
        (pl.col("algorithm_name") == "A") & (pl.col("raw_y") == 0.1)
    )["n_success"].to_list()[0]
    b_success_count = result.filter(
        (pl.col("algorithm_name") == "B") & (pl.col("raw_y") == 0.1)
    )["n_success"].to_list()[0]

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
    )["n_success"].to_list()[0]
    b_success_count = result.filter(
        (pl.col("algorithm_name") == "B") & (pl.col("raw_y") >= 0.9)
    )["n_success"].to_list()[0]

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
    assert "n_success" in result.columns
    assert (result["success_ratio"] <= 1).all()


# docs/definitions/runtime.md: three runs that used 30, 50 and 40 evaluations;
# runs 1 and 2 reach 0.1 at 20 and 45, run 3 never does
RUNS = pl.DataFrame(
    {
        "run_id": [1, 1, 2, 2, 3],
        "evaluations": [1, 20, 1, 45, 1],
        "raw_y": [1.0, 0.1, 1.0, 0.1, 1.0],
        "evals_used": [30, 30, 50, 50, 40],
        "algorithm_name": "A",
    }
)


def _summary(**options):
    return aggregate_running_time(RUNS, targets=[0.1], **options).row(0, named=True)


def test_ert_charges_each_failure_its_own_budget():
    # RT-1: (20 + 45 + 40) / 2; PAR-10 cutoff is the largest budget (RT-3)
    row = _summary()
    assert (row["n_runs"], row["n_success"], row["success_ratio"]) == (3, 2, 2 / 3)
    assert row["ERT"] == 52.5
    assert row["par_10"] == (20 + 45 + 10 * 50) / 3
    assert (row["mean"], row["median"]) == (32.5, 32.5)
    assert row["sd"] == pytest.approx(np.std([20, 45], ddof=1), rel=1e-12)


def test_cap_makes_late_hits_failures():
    # C = 30: run 2's hit at 45 fails and is charged 30, as is run 3 (RT-1)
    row = _summary(eval_max=30, k=1)
    assert row["n_success"] == 1
    assert row["ERT"] == 20 + 30 + 30
    assert row["par_1"] == (20 + 30 + 30) / 3
    assert (row["mean"], row["sd"]) == (20, None)


def test_no_success_is_inf_and_null():
    row = aggregate_running_time(RUNS, targets=[0.01]).row(0, named=True)
    assert row["ERT"] == np.inf  # RT-2
    assert row["mean"] is row["median"] is row["sd"] is None
    assert row["q2"] == np.inf
    row = aggregate_running_time(RUNS, targets=[0.01], include_unfinished=False)
    assert row.row(0, named=True)["q2"] is None


def test_quantiles_count_unfinished_runs():
    # RT-5: the sample is 20, 45, inf
    row = _summary()
    assert [row[f"q{p}"] for p in (25, 50, 90)] == [20, 45, np.inf]
    row = _summary(include_unfinished=False)
    assert [row[f"q{p}"] for p in (25, 50, 90)] == [20, 20, 45]


@pytest.mark.parametrize("n", range(1, 27))
def test_quantiles_are_type_3(n):
    # numpy's closest_observation is Hyndman-Fan type 3 (runtime.md)
    rng = np.random.default_rng(n)
    hits = rng.integers(1, 100, n)
    df = pl.DataFrame(
        {"run_id": range(n), "evaluations": hits, "raw_y": 0.0, "algorithm_name": "A"}
    ).with_columns(evals_used=100)
    df = pl.concat(
        [df.with_columns(evaluations=0, raw_y=1.0), df], how="vertical_relaxed"
    )
    row = aggregate_running_time(df, targets=[0.0]).row(0, named=True)
    probabilities = [2, 5, 10, 25, 50, 75, 90, 95, 98]
    expected = np.quantile(
        hits, np.array(probabilities) / 100, method="closest_observation"
    )
    assert [row[f"q{p}"] for p in probabilities] == expected.tolist()
