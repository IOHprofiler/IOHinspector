import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iohinspector.metrics import cumulative_difference, get_cumulative_difference
from iohinspector.metrics.cdp import _densities
from tests.golden.load import GOLDEN_DIR, golden, logged

TOL = 1e-12  # cdp.md, "Test tolerance"
OURS = ["y_a", "y_a_lower", "y_a_upper", "y_b", "y_b_lower", "y_b_upper"]
OURS += ["diff", "diff_lower", "diff_upper"]
RVCOMPARE = [
    f"{y}_cumulative_{part}"
    for y in ("Y_A", "Y_B")
    for part in ("estimation", "lower", "upper")
] + ["diff_estimation", "diff_lower", "diff_upper"]


def test_tie_groups_share_the_density():
    # RVCompare's own example in ranksOfObserved: c(0.1, 0.2, 0.5, 0.5) and
    # c(0.0, 0.2, 0.4, 0.5)
    mult = _densities(
        np.array([[0.1, 0.2, 0.5, 0.5]]), np.array([[0, 0.2, 0.4, 0.5]]), 1e-20
    )
    np.testing.assert_allclose(mult[0], [0, 1, 0.5, 0.5, 0, 2 / 3, 2 / 3, 2 / 3])
    # values closer than epsilon tie; infinite values tie with each other (CD-2)
    mult = _densities(
        np.array([[1.0, math.inf]]), np.array([[1 + 1e-9, math.inf]]), 1e-6
    )
    np.testing.assert_allclose(mult[0], [0.5, 0.5, 0.5, 0.5])


def test_curves_and_their_symmetry():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 30), rng.normal(0.5, 1, 30)
    ab = cumulative_difference(a, b, n_bootstrap=50, seed=1)
    assert ab.columns == ["p", *OURS] and ab.height == 2000
    assert ab["p"][0] == 0 and ab["p"][-1] == 1
    # RVCompare normalises the density by its sum, not by the trapezoid integral, so
    # the cumulative functions end just below 1
    np.testing.assert_allclose(ab.row(-1)[1:7], 1, atol=1e-3)
    assert (ab["diff"] >= -1).all() and (ab["diff"] <= 1).all()
    for name in ["y_a", "y_b", "diff"]:
        assert (ab[f"{name}_lower"] <= ab[name] + TOL).all()
        assert (ab[name] <= ab[f"{name}_upper"] + TOL).all()
    ba = cumulative_difference(b, a, n_bootstrap=50, seed=1)
    np.testing.assert_allclose(ab["y_a"], ba["y_b"], atol=TOL)
    np.testing.assert_allclose(ab["diff"], -ba["diff"], atol=TOL)
    same = cumulative_difference(a, a, n_bootstrap=1)
    np.testing.assert_allclose(same["diff"], 0, atol=TOL)
    again = cumulative_difference(a, b, n_bootstrap=50, seed=1)
    assert again.equals(ab)  # seeded


def test_sample_errors():
    with pytest.raises(ValueError, match="equal size"):
        cumulative_difference([1, 2], [1])
    with pytest.raises(ValueError, match="NaN"):
        cumulative_difference([1, math.nan], [1, 2])
    with pytest.raises(ValueError, match="epsilon"):
        cumulative_difference([1], [2], epsilon=0)


def _runs(rows):
    names = ["run_id", "algorithm_name", "evaluations", "raw_y"]
    return pl.DataFrame(rows, schema=names, orient="row").with_columns(
        function_id=pl.lit(1), dimension=pl.lit(2)
    )


DATA = _runs(
    [
        (1, "A", 1, 5.0), (1, "A", 3, 1.0),
        (2, "A", 1, 4.0), (2, "A", 6, 2.0),
        (3, "A", 2, 9.0),
        (4, "B", 1, 3.0), (4, "B", 2, 0.5),
        (5, "B", 1, 8.0),
        (6, "B", 1, 1.0),
    ]
)  # fmt: skip


def test_fixed_target_counts_failed_runs_as_worst():
    result = get_cumulative_difference(DATA, target=2.0, seed=0, n_bootstrap=10)
    runs = result.runs.sort("run_id")
    assert runs["value"].to_list() == [3, 6, math.inf, 2, math.inf, 1]
    assert runs["group"].to_list() == ["A"] * 3 + ["B"] * 3
    assert runs.columns == ["run_id", "algorithm_name", "group", "value", "used"]
    assert not result.subsampled
    expected = cumulative_difference([3, 6, math.inf], [2, math.inf, 1], n_bootstrap=1)
    np.testing.assert_allclose(result.curves["y_a"], expected["y_a"], rtol=TOL)


def test_fixed_budget_orients_values():
    result = get_cumulative_difference(DATA, budget=2, seed=0, n_bootstrap=10)
    assert result.runs.sort("run_id")["value"].to_list() == [5, 4, 9, 0.5, 8, 1]
    maxi = get_cumulative_difference(DATA, budget=1, maximization=True, n_bootstrap=10)
    # negated; run 3 starts at evaluation 2, so it has no value yet: the worst
    assert maxi.runs.sort("run_id")["value"].to_list() == [-5, -4, math.inf, -3, -8, -1]


def test_unequal_groups_are_subsampled_and_reported():
    data = pl.concat([DATA, _runs([(7, "B", 1, 2.0)])])
    with pytest.warns(UserWarning, match="group B has 4 runs and the other 3"):
        result = get_cumulative_difference(data, target=2.0, seed=3, n_bootstrap=10)
    assert result.subsampled
    used = result.runs.group_by("group").agg(pl.col("used").sum()).sort("group")
    assert used["used"].to_list() == [3, 3]
    assert result.runs.filter(group="A")["used"].all()  # the smaller group is whole
    with pytest.warns(UserWarning):
        again = get_cumulative_difference(data, target=2.0, seed=3, n_bootstrap=10)
    assert again.runs.equals(result.runs) and again.curves.equals(result.curves)
    with pytest.raises(ValueError, match="3 and 4 runs"):
        get_cumulative_difference(data, target=2.0, subsample=False)


def test_get_cumulative_difference_errors():
    with pytest.raises(ValueError, match="either target or budget"):
        get_cumulative_difference(DATA)
    with pytest.raises(ValueError, match="exactly two groups"):
        get_cumulative_difference(
            DATA.with_columns(algorithm_name=pl.col("run_id").cast(pl.String)),
            target=2.0,
        )
    two_pairs = pl.concat([DATA, DATA.with_columns(function_id=pl.lit(2))])
    with pytest.raises(ValueError, match="one \\(function, dimension\\) pair"):
        get_cumulative_difference(two_pairs, target=2.0)


# Golden tests. IOHanalyzer's generate_data.CDP calls RVCompare 0.1.8 with 200
# bootstrap samples; its fixtures have 5 runs per algorithm, so its bands are too
# discrete to compare across random streams, and only the estimates (the observed
# sample) are compared. The bands are compared on RVCompare's own reference: two
# samples of 100 values with ties, 1000 bootstrap samples (tests/golden/r/rvcompare.R).


def _cases():
    for fixture in ["ioh040_min", "ioh040_max", "ioh040_cons"]:
        for metric in ["CDP_by_FV", "CDP_by_RT", "CDP_by_FV_with_failures"]:
            if (GOLDEN_DIR / "iohanalyzer" / fixture / f"{metric}.parquet").exists():
                yield fixture, metric


@pytest.mark.parametrize(("fixture", "metric"), list(_cases()))
def test_estimates_match_iohanalyzer(fixture, metric):
    maximization = fixture == "ioh040_max"
    data = logged(fixture)
    two = sorted(data["algorithm_name"].unique())[:2]  # sort(ids)[1:2]
    gold = golden(fixture, metric)
    for (f, d), ref in gold.partition_by(["pair_f", "pair_d"], as_dict=True).items():
        pair = data.filter(
            pl.col("function_id") == f,
            pl.col("dimension") == d,
            pl.col("algorithm_name").is_in(two),
        )
        if metric == "CDP_by_RT":  # the 3rd of 5 log budgets
            budgets = golden(fixture, "FV_sample").filter(pair_f=f, pair_d=d)
            where = {"budget": int(budgets["runtime"].unique().sort()[2])}
        elif metric == "CDP_by_FV":  # the worst final value: every run reaches it
            best = pl.col("raw_y").max() if maximization else pl.col("raw_y").min()
            final = pair.group_by("run_id").agg(best)["raw_y"]
            where = {"target": final.min() if maximization else final.max()}
        else:  # the 3rd of 5 log targets
            targets = golden(fixture, "RT_sample").filter(pair_f=f, pair_d=d)
            where = {"target": targets["target"].unique().sort()[2]}
        ours = get_cumulative_difference(
            pair, maximization=maximization, n_bootstrap=1, **where
        ).curves
        np.testing.assert_allclose(ours["p"], ref["p"], rtol=TOL)
        for name, gold_name in zip(OURS[::3], RVCOMPARE[::3]):
            np.testing.assert_allclose(ours[name], ref[gold_name], rtol=0, atol=TOL)


def test_bands_match_rvcompare():
    folder = Path(GOLDEN_DIR) / "rvcompare"
    samples = pl.read_csv(folder / "samples.csv")
    ref = pl.read_csv(folder / "bootstrap.csv")  # every 10th point, 15 digits
    ours = cumulative_difference(samples["a"], samples["b"], seed=20240423)[::10]
    np.testing.assert_allclose(ours["p"], ref["p"], rtol=1e-14)
    for name, gold_name in zip(OURS, RVCOMPARE):
        error = np.abs(ours[name].to_numpy() - ref[gold_name].to_numpy())
        if name in ("y_a", "y_b", "diff"):
            assert error.max() < 1e-13  # the observed sample: exact up to 15 digits
        else:
            # Monte Carlo: Y bands within 0.02 (cdp.md); the difference band spreads
            # twice as much (up to 0.037 between our own seeds), so 0.04
            assert error.max() < (0.04 if name.startswith("diff") else 0.02), name
            assert error.mean() < 0.01, name
