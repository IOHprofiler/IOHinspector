"""The deprecated get_aocc: get_ecdf_auc(continuous=True) on a quality in [0, 1]."""

import polars as pl
import pytest

from iohinspector.metrics import get_aocc

# two runs of one algorithm on f1; quality 0.2, 0.5, 0.8 and 0.4, 0.4, 1.0
DF = pl.DataFrame(
    {
        "run_id": [1, 1, 1, 2, 2, 2],
        "function_name": ["f1"] * 6,
        "algorithm_name": ["alg1"] * 6,
        "evaluations": [1, 5, 9, 1, 5, 9],
        "eaf": [0.2, 0.5, 0.8, 0.4, 0.4, 1.0],
    }
)
pytestmark = pytest.mark.filterwarnings("ignore:get_aocc:DeprecationWarning")


def test_deprecated():
    with pytest.warns(DeprecationWarning, match="get_ecdf_auc"):
        get_aocc(DF, eval_max=9)


def test_mean_area_over_the_runs():
    # over [1, 9]: run 1 (0.2 * 4 + 0.5 * 4) / 8, run 2 (0.4 * 8) / 8
    result = get_aocc(DF, eval_max=9)
    assert result.columns == ["function_name", "algorithm_name", "AOCC"]
    assert result["AOCC"].item() == pytest.approx((0.35 + 0.4) / 2, 1e-12)


def test_groups_and_column():
    df = DF.with_columns(function_name=pl.Series(["f1"] * 3 + ["f2"] * 3))
    result = get_aocc(df.rename({"eaf": "q"}), eval_max=9, fval_var="q").sort(
        "function_name"
    )
    assert result["AOCC"].to_list() == pytest.approx([0.35, 0.4], 1e-12)
    by_algorithm = get_aocc(df, eval_max=9, by=["algorithm_name"])
    assert by_algorithm["AOCC"].item() == pytest.approx(0.375, 1e-12)


def test_log_scale_and_late_start():
    # log10 of [1, 100]: run 1 starts at 10 (0 before), run 2 at 1
    df = pl.DataFrame(
        {
            "run_id": [1, 1, 2],
            "function_name": ["f1"] * 3,
            "algorithm_name": ["alg1"] * 3,
            "evaluations": [10, 100, 1],
            "eaf": [0.5, 1.0, 0.25],
        }
    )
    result = get_aocc(df, eval_max=100, scale_eval_log=True)
    assert result["AOCC"].item() == pytest.approx((0.25 + 0.25) / 2, 1e-12)


def test_empty_budget_range():
    with pytest.raises(ValueError, match="empty"):
        get_aocc(DF, eval_max=1)
