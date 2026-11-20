import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from iohinspector.align import align_data
from iohinspector.metrics import eaf as eaf_module
from iohinspector.metrics import get_ecdf, get_ecdf_auc
from iohinspector.metrics.eaf import (
    RECTANGLE,
    get_discritized_eaf_single_objective,
    get_eaf,
    get_eaf_data,
    get_eaf_diff_data,
    get_eaf_difference,
    get_eaf_margin,
    get_eaf_pareto_data,
)
from iohinspector.targets import seq_rt
from tests.golden.load import golden_renamed, logged

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
    assert result["eaf_target"].min() == pytest.approx(0.0, abs=5e-8)
    assert result["eaf_target"].max() == pytest.approx(1.0, abs=5e-8)


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

        assert eaf_value == pytest.approx(expected_eaf, abs=5e-8)


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


# The single-objective EAF of the definitions (docs/definitions/eaf.md, WP I17)

# two runs of A: (1, 10), a worse row, (3, 2); and (2, 5). One run of B: (2, 4)
TWO = pl.DataFrame(
    {
        "run_id": [1, 1, 2, 1, 3],
        "algorithm_name": ["A", "A", "A", "A", "B"],
        "evaluations": [1, 2, 2, 3, 2],
        "raw_y": [10.0, 12.0, 5.0, 2.0, 4.0],
    }
)


def _rows(df, *columns):
    return sorted(df.select(columns).rows())


@pytest.mark.parametrize("maximization", [False, True])
def test_eaf_surfaces_of_best_so_far_traces(maximization):
    # rows of the runs interleaved: moocore reads a set from consecutive rows
    sign = -1.0 if maximization else 1.0
    data = TWO.with_columns(pl.col("raw_y") * sign)
    result = get_eaf(data, percentiles=[50, 100], maximization=maximization)
    a = result.filter(pl.col("algorithm_name") == "A")
    assert a["evaluations"].dtype == data["evaluations"].dtype
    expected = [(1, 10.0, 50.0), (2, 5.0, 50.0), (3, 2.0, 50.0)]
    expected += [(2, 10.0, 100.0), (3, 5.0, 100.0)]
    expected = [(e, sign * y, p) for e, y, p in expected]
    assert _rows(a, "evaluations", "raw_y", "percentile") == sorted(expected)


def test_eaf_levels_as_iohanalyzer():
    # 0 and 10 % are the level of one run of two, as eaf::eafs maps them
    result = get_eaf(TWO.filter(pl.col("algorithm_name") == "A"))
    levels = result.group_by("percentile").agg(pl.col("raw_y").sort())
    by_level = dict(levels.rows())
    assert by_level[0.0] == by_level[50.0] == [2.0, 5.0, 10.0]
    assert by_level[60.0] == by_level[100.0] == [5.0, 10.0]
    assert sorted(by_level) == [float(p) for p in range(0, 101, 10)]


def test_discretised_eaf():
    result = get_eaf(
        TWO, mode="discretised", eval_values=[1, 2, 3], f_values=[2.0, 5.0, 10.0]
    )
    a = result.filter(pl.col("algorithm_name") == "A")
    # run 2 has no value before evaluation 2: it has attained nothing (AL-5)
    alpha = {(e, y): v for _, e, y, v in a.rows()}
    assert alpha[1, 10.0] == 0.5 and alpha[1, 5.0] == 0.0
    assert alpha[2, 10.0] == 1.0 and alpha[2, 5.0] == 0.5 and alpha[2, 2.0] == 0
    assert alpha[3, 2.0] == 0.5 and alpha[3, 5.0] == 1.0
    assert result.columns == ["algorithm_name", "evaluations", "raw_y", "eaf"]
    assert result.height == 2 * 9


def test_discretised_eaf_default_grid():
    data = MULTI_DATA.with_columns(algorithm_name=pl.lit("A"))
    result = get_eaf(data, mode="discretised", eval_values=5, f_values=4)
    assert result["evaluations"].unique().sort().to_list() == [1, 6, 32, 178, 1000]
    values = result["raw_y"].unique().sort().to_numpy()
    np.testing.assert_allclose(values, np.geomspace(0.001, 1.5, 4), rtol=1e-12)


def test_several_pairs_use_the_normalised_quality():
    other = TWO.with_columns(pl.col("run_id") + 10, pl.col("raw_y") * 100)
    data = pl.concat(
        [TWO.with_columns(function_id=1), other.with_columns(function_id=2)]
    ).with_columns(dimension=pl.lit(5))
    result = get_eaf(data)
    assert "quality" in result.columns
    assert result["quality"].is_between(0, 1).all()
    # the same runs in both pairs, scaled: each pair has the same qualities
    q = get_eaf(data, mode="discretised", f_values=[0.0, 0.5, 1.0])
    assert (
        q.filter(pl.col("quality") == 0, pl.col("evaluations") == 3)["eaf"].min() == 1
    )
    per_pair = get_eaf(data, ["algorithm_name", "function_id"])
    assert "raw_y" in per_pair.columns and "quality" not in per_pair.columns


def test_eaf_options_are_checked():
    with pytest.raises(TypeError, match="maximisation"):
        get_eaf(TWO, maximisation=True)
    with pytest.raises(ValueError, match="mode"):
        get_eaf(TWO, mode="exact")


def test_eaf_difference_of_two_groups():
    result = get_eaf_difference(TWO)
    a = result.filter(pl.col("algorithm_name") == "A").drop("algorithm_name")
    b = result.filter(pl.col("algorithm_name") == "B").drop("algorithm_name")
    assert_frame_equal(b, a.with_columns(-pl.col("eaf_diff")))
    assert result.columns == ["algorithm_name", *RECTANGLE]
    # A attains (1, 10) alone; B reaches 4 at 2 where half of A reaches 5
    assert a.filter(pl.col("x_min") == 1)["eaf_diff"].to_list() == [0.5]
    assert -0.5 in a["eaf_diff"].to_list()


def _delta_at(rectangles, x, y):
    inside = (
        (pl.col("x_min") <= x)
        & (pl.col("x_max") > x)
        & (pl.col("y_min") <= y)
        & (pl.col("y_max") > y)
    )
    hits = rectangles.filter(inside)["eaf_diff"]
    assert hits.len() <= 1
    return hits.sum()


@pytest.mark.parametrize("maximization", [False, True])
def test_precise_and_discretised_differences_agree(maximization):
    # 3 against 4 runs (moocore compares equally many: the runs are repeated);
    # evaluations even and the grid odd, so no grid point is on a rectangle side
    rng = np.random.default_rng(1)
    n = 7 * 12
    data = pl.DataFrame(
        {
            "run_id": np.repeat(np.arange(7), 12),
            "algorithm_name": np.repeat(["A"] * 3 + ["B"] * 4, 12),
            "evaluations": np.tile(np.arange(2, 26, 2), 7),
            "raw_y": rng.uniform(0, 100, n).round(3),
        }
    )
    grid = {"eval_values": list(range(1, 27, 2)), "f_values": np.arange(0.5, 100, 2.5)}
    options = dict(maximization=maximization, **grid)
    precise = get_eaf_difference(data, maximization=maximization)
    discretised = get_eaf_difference(data, mode="discretised", **options)
    for name, x, y, delta in discretised.rows():
        mine = precise.filter(pl.col("algorithm_name") == name)
        assert _delta_at(mine, x, y) == pytest.approx(delta, abs=1e-12)
    assert set(discretised["eaf_diff"].round(10)) - {0.0}  # not trivially equal


def test_margin_against_the_best_of_the_others():
    c = TWO.filter(pl.col("run_id") == 2).with_columns(
        pl.col("run_id") + 2, algorithm_name=pl.lit("C")
    )
    data = pl.concat([TWO, c])
    grid = {"eval_values": [1, 2, 3], "f_values": [2.0, 5.0, 10.0]}
    alpha = get_eaf(data, mode="discretised", **grid)
    margin = get_eaf_margin(data, **grid)
    cells = ["evaluations", "raw_y"]
    wide = alpha.pivot(on="algorithm_name", index=cells, values="eaf")
    expected = pl.concat(
        wide.select(
            *cells,
            algorithm_name=pl.lit(g),
            eaf_margin=pl.col(g) - pl.max_horizontal([c for c in "ABC" if c != g]),
        )
        for g in "ABC"
    )
    assert_frame_equal(
        margin, expected, check_row_order=False, check_column_order=False
    )
    # for two groups the margin is the difference
    two = get_eaf_margin(TWO, **grid).rename({"eaf_margin": "eaf_diff"})
    assert_frame_equal(two, get_eaf_difference(TWO, mode="discretised", **grid))
    with pytest.raises(ValueError, match="at least two groups"):
        get_eaf_margin(TWO, by=[])


@pytest.mark.parametrize("mode", ["precise", "discretised"])
def test_difference_of_exactly_two_groups(mode):
    c = TWO.filter(pl.col("run_id") == 2).with_columns(
        pl.col("run_id") + 2, algorithm_name=pl.lit("C")
    )
    with pytest.raises(ValueError, match=r"exactly two groups of by, found \[3\]"):
        get_eaf_difference(pl.concat([TWO, c]), mode=mode)
    with pytest.raises(ValueError, match="exactly two groups"):
        get_eaf_difference(TWO, by=[], mode=mode)


def test_equal_eafs_have_no_difference():
    copy = TWO.filter(pl.col("algorithm_name") == "A")
    b = copy.with_columns(pl.col("run_id") + 10, algorithm_name=pl.lit("B"))
    data = pl.concat([copy, b])
    for mode in ("precise", "discretised"):
        assert (get_eaf_difference(data, mode=mode)["eaf_diff"] == 0).all()
    # moocore fails when there is no rectangle at all
    same = pl.DataFrame({"run_id": [1], "x": [1.0], "y": [1.0]})
    assert get_eaf_diff_data(same, same, "x", "y").height == 0


def _uneven(seed=2, stop=False):
    """Two algorithms on two pairs, f integer in [0, 10] at evaluations 1..10;
    A has 2 runs on function 1 and 1 on function 2, B 1 and 3. With ``stop``, run
    i stops after 3 + (2 i mod 8) evaluations, most of them early."""
    rng = np.random.default_rng(seed)
    runs = [("A", 1)] * 2 + [("A", 2)] + [("B", 1)] + [("B", 2)] * 3
    data = pl.DataFrame(
        {
            "run_id": np.repeat(np.arange(len(runs)), 10),
            "algorithm_name": np.repeat([a for a, _ in runs], 10),
            "function_id": np.repeat([f for _, f in runs], 10),
            "dimension": 5,
            "evaluations": np.tile(np.arange(1, 11), len(runs)),
            "raw_y": rng.integers(0, 11, 10 * len(runs)).astype(float),
        }
    )
    if stop:
        data = data.filter(pl.col("evaluations") <= 3 + 2 * pl.col("run_id") % 8)
    return data


@pytest.mark.parametrize(
    "stop, stopped",
    [(False, "carry_forward"), (True, "carry_forward"), (True, "restart")],
)
def test_area_under_the_eaf_is_the_aocc(stop, stopped):
    # q = (10 - f) / 10 is a multiple of 0.1 and alpha(b, .) is constant on
    # (q_(j-1), q_j], and on [b, b + 1) in b: the sums below are exact integrals.
    # With runs that stop early, the same stopped on both sides (EA-6, EC-7)
    data = _uneven(stop=stop)
    bounds = pl.DataFrame({"function_id": [1, 2], "dimension": 5}).join(
        pl.DataFrame({"target": [0.0, 10.0]}), how="cross"
    )
    common = {"targets": bounds, "scale_f_log": False, "stopped": stopped}
    aocc = get_ecdf_auc(data, b_hi=10, continuous=True, **common)
    grid = {"eval_values": range(1, 10), "f_values": [k / 10 for k in range(1, 11)]}

    def area(weighting):
        eaf = get_eaf(data, mode="discretised", weighting=weighting, **grid, **common)
        volume = eaf.group_by("algorithm_name").agg(auc=pl.col("eaf").sum() * 0.1)
        return volume.with_columns(pl.col("auc") / 9).sort("algorithm_name")

    with pytest.warns(UserWarning, match=r"weighting='pairs'\), as in the AOCC"):
        pairs = area("pairs")
    aocc = aocc.sort("algorithm_name")
    assert pairs["algorithm_name"].to_list() == aocc["algorithm_name"].to_list()
    np.testing.assert_allclose(pairs["auc"], aocc["auc"], rtol=1e-12)
    with pytest.warns(UserWarning, match="A: \\[1, 2\\]; B: \\[1, 3\\]"):
        runs = area("runs")
    assert not np.allclose(runs["auc"], pairs["auc"])
    if stopped == "restart":  # the restarts change it, and only improve on it
        common["stopped"] = "carry_forward"
        with pytest.warns(UserWarning):
            carried = area("pairs")
        assert (pairs["auc"] > carried["auc"] + 1e-3).all()


def test_eaf_restarts_are_discretised_and_shared():
    data = _uneven(stop=True).filter(pl.col("function_id") == 1)
    with pytest.raises(ValueError, match="mode='discretised'.*carry_forward"):
        get_eaf(data, stopped="restart")
    with pytest.raises(ValueError, match="carry_forward"):
        get_eaf(data, mode="discretised", stopped="exclude")
    grid = {"eval_values": range(1, 10), "f_values": range(11)}
    options = {"mode": "discretised", "stopped": "restart", **grid}
    eaf = get_eaf(data, **options)
    carried = get_eaf(data, mode="discretised", **grid)
    assert (eaf["eaf"] >= carried["eaf"] - 1e-12).all()
    assert (eaf["eaf"] > carried["eaf"] + 1e-12).any()
    # alpha at y is the restarted single-target ECDF of y (EC-7), per group
    for (algorithm,), group in eaf.partition_by("algorithm_name", as_dict=True).items():
        runs = data.filter(pl.col("algorithm_name") == algorithm)
        for y in (0.0, 4.0, 7.0):
            target = pl.DataFrame({"function_id": [1], "dimension": [5], "target": [y]})
            ecdf = get_ecdf(runs, target, x_values=range(1, 10))
            at = group.filter(pl.col("raw_y") == y).sort("evaluations")["eaf"]
            np.testing.assert_allclose(at, ecdf["ecdf"], rtol=1e-12, atol=1e-15)
    # the difference and the margin of two groups are differences of these EAFs
    difference = get_eaf_difference(data, **options)
    margin = get_eaf_margin(data, **{k: v for k, v in options.items() if k != "mode"})
    a = eaf.filter(pl.col("algorithm_name") == "A")["eaf"]
    b = eaf.filter(pl.col("algorithm_name") == "B")["eaf"]
    a_diff = difference.filter(pl.col("algorithm_name") == "A")["eaf_diff"]
    np.testing.assert_allclose(a_diff, a - b, atol=1e-12)
    a_margin = margin.filter(pl.col("algorithm_name") == "A")["eaf_margin"]
    np.testing.assert_allclose(a_margin, a - b, atol=1e-12)


def test_precise_eaf_weighs_pairs_by_repeating_runs():
    # with weighting="pairs", the default, the precise EAF is that of each pair's
    # runs repeated up to the least common multiple of the group's run counts
    data = _uneven()
    copies = {("A", 1): 1, ("A", 2): 2, ("B", 1): 3, ("B", 2): 1}
    repeated = pl.concat(
        part.with_columns(pl.col("run_id") * 10 + k)
        for (a, f), part in data.group_by("algorithm_name", "function_id")
        for k in range(copies[a, f])
    )
    with pytest.warns(UserWarning, match="different numbers of runs"):
        weighted = get_eaf(data)
    keys = ["algorithm_name", "percentile", "evaluations", "quality"]
    assert_frame_equal(weighted.sort(keys), get_eaf(repeated).sort(keys))
    with pytest.warns(UserWarning, match="weighting='runs'"):
        assert get_eaf(data, weighting="runs").height != weighted.height
    with pytest.raises(ValueError, match="weighting"):
        get_eaf(data, weighting="functions")


def test_precise_pair_weighting_is_capped(monkeypatch):
    # _uneven repeats A's runs to 2 per pair and B's to 3: 4 + 6 = 10 sets
    data = _uneven()
    monkeypatch.setattr(eaf_module, "MAX_SETS", 10)
    with pytest.warns(UserWarning):
        assert get_eaf(data).height > 0
    monkeypatch.setattr(eaf_module, "MAX_SETS", 9)
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="10 attainment"):
        get_eaf(data)
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="discretised"):
        get_eaf_difference(data)
    with pytest.warns(UserWarning):  # discretised repeats nothing
        assert get_eaf(data, mode="discretised", eval_values=2, f_values=2).height
    with pytest.warns(UserWarning):  # nor does weighting="runs"
        assert get_eaf(data, weighting="runs").height


def test_precise_difference_repetition_is_capped(monkeypatch):
    # A has 2 runs and B 1: B's run is repeated, 2 + 2 = 4 sets
    monkeypatch.setattr(eaf_module, "MAX_SETS", 4)
    assert get_eaf_difference(TWO).height > 0
    monkeypatch.setattr(eaf_module, "MAX_SETS", 3)
    with pytest.raises(ValueError, match="4 attainment sets.*discretised"):
        get_eaf_difference(TWO)
    assert get_eaf_difference(TWO, mode="discretised").height > 0


def test_discretised_eaf_of_0_0_8_differs_at_quality_0():
    # get_discritized_eaf_single_objective is kept, not an alias of
    # get_eaf(mode="discretised"): on its quality levels between 1e-8 and 1e2 the
    # two agree except at q = 0, which 0.0.8 counts as attained by every run, even
    # one worse than 1e2 (transform_fval clips), and it has its own layout
    data = logged("ioh040_cons").filter(
        pl.col("function_id") == 1, pl.col("algorithm_name") == "rs"
    )
    old = get_discritized_eaf_single_objective(data)
    q = old["eaf_target"].to_numpy()
    budgets = [int(c) for c in old.columns[1:]]
    new = get_eaf(
        data, [], mode="discretised", eval_values=budgets, f_values=10 ** (2 - 10 * q)
    )
    new = new.pivot(on="evaluations", index="raw_y", values="eaf")
    new = new.sort("raw_y", descending=True).drop("raw_y").to_numpy()
    old = old.drop("eaf_target").to_numpy()
    np.testing.assert_array_equal(new[1:], old[1:])
    assert (old[0] == 1).all() and not (new[0] == 1).all()


def test_pareto_eaf_reads_each_run_as_one_set():
    shuffled = SIMPLE_MO_DATA.sample(fraction=1.0, shuffle=True, seed=3)
    assert_frame_equal(
        get_eaf_pareto_data(shuffled, "obj1", "obj2"),
        get_eaf_pareto_data(SIMPLE_MO_DATA, "obj1", "obj2"),
    )


# Golden tests against IOHanalyzer (WP I03): generate_data.EAF, EAF_Difference and
# EAF_diff_Approximate, per (function, dimension) pair. The levels come from the
# R package eaf 2.5 and must match moocore's exactly (EA-1).
PAIR = ["function_id", "dimension"]
BY_PAIR = ["algorithm_name", *PAIR]
MAXIMIZE = {"ioh040_min": False, "ioh040_max": True, "ioh040_cons": False}


def _gold(fixture, metric):
    gold = golden_renamed(fixture, metric).rename(
        {"algId": "algorithm_name"}, strict=False
    )
    pair = [
        pl.col(c).cast(pl.Int32).alias(n) for c, n in zip(["pair_f", "pair_d"], PAIR)
    ]
    return gold.with_columns(pair).drop("pair_f", "pair_d")


@pytest.mark.parametrize("fixture", [*MAXIMIZE, "ioh040_pos"])
@pytest.mark.parametrize("metric", ["EAF", "EAF_logx"])
def test_eaf_matches_iohanalyzer(fixture, metric):
    # IOHanalyzer samples each run's best-so-far value at 20 budgets of the pair;
    # the precise EAF of those samples is its EAF. One value of ioh040_min differs
    # in the 17th digit: IOHanalyzer takes a run's best value from the JSON summary
    data = logged(fixture)
    maximization = MAXIMIZE.get(fixture, False)
    sampled = []
    for _, pair in data.group_by(PAIR):
        budgets = seq_rt(pair["evaluations"], 20, log=metric == "EAF_logx")
        aligned = align_data(pair, budgets, maximization=maximization)
        sampled.append(aligned.drop_nulls("raw_y"))
    ours = get_eaf(pl.concat(sampled), BY_PAIR, maximization=maximization)
    gold = _gold(fixture, metric).rename({"runtime": "evaluations", "f(x)": "raw_y"})
    keys = [*BY_PAIR, "percentage", "evaluations", "raw_y"]
    ours = ours.rename({"percentile": "percentage"}).sort(keys)
    gold = gold.sort(keys)
    assert_frame_equal(
        ours.select(keys[:-1]), gold.select(keys[:-1]), check_dtypes=False
    )
    np.testing.assert_allclose(ours["raw_y"], gold["raw_y"], rtol=1e-15, atol=0)


@pytest.mark.parametrize("fixture", MAXIMIZE)
def test_eaf_difference_matches_iohanalyzer(fixture):
    # the first two algorithms; IOHanalyzer counts the difference in runs (5 each)
    data = logged(fixture)
    two = sorted(data["algorithm_name"].unique())[:2]
    data = data.filter(pl.col("algorithm_name").is_in(two))
    gold = _gold(fixture, "EAF_Difference")
    rectangles = ["xmin", "ymin", "xmax", "ymax", "diff"]
    gold = gold.select(*PAIR, *rectangles).sort(pl.all())

    def ours(maximization):
        result = get_eaf_difference(data, BY_PAIR, maximization=maximization)
        result = result.filter(pl.col("algorithm_name") == two[0])
        result = result.with_columns(pl.col("eaf_diff") * 5)
        return result.select(*PAIR, *RECTANGLE).sort(pl.all())

    # IOHanalyzer's EAF_Difference ignores maximisation: it minimises
    result = ours(False)
    assert_frame_equal(result.select(PAIR), gold.select(PAIR), check_dtypes=False)
    np.testing.assert_allclose(
        result.select(RECTANGLE).to_numpy(),
        gold.select(rectangles).to_numpy(),
        rtol=1e-15,
    )
    if MAXIMIZE[fixture]:
        assert ours(True).height != result.height


@pytest.mark.parametrize("fixture", MAXIMIZE)
def test_margin_matches_iohanalyzer(fixture):
    # EAF_diff_Approximate, each algorithm against the best of the others, on
    # IOHanalyzer's grid: 50 log-spaced budgets and values per pair, its values
    # 10^(log10 spaced) with ends 1 ulp off the data's (TG-1) and 1e-12 for a lower
    # end of 0 (TG-3). When maximising IOHanalyzer labels the values in reverse
    # order (its success rates come sorted by target)
    data = logged(fixture)
    maximization = MAXIMIZE[fixture]
    gold = _gold(fixture, "EAF_diff_Approximate")
    for (f, d), pair in data.group_by(PAIR):
        lower = 1e-12 if pair["raw_y"].min() <= 0 else pair["raw_y"].min()
        log = np.log10([lower, pair["raw_y"].max()])
        values = 10.0 ** np.linspace(*log, 50)
        budgets = seq_rt(pair["evaluations"], 50)
        ours = get_eaf_margin(  # the difference with two algorithms
            pair,
            eval_values=budgets,
            f_values=values,
            maximization=maximization,
        )
        expected = gold.filter(pl.col("function_id") == f, pl.col("dimension") == d)
        labels = expected["fvalue"].cast(pl.Float64).unique().sort().to_numpy()
        np.testing.assert_allclose(labels, values, rtol=1e-9)
        values = values[::-1] if maximization else values
        expected = expected.select(
            "algorithm_name",
            evaluations=pl.col("runtime").cast(ours["evaluations"].dtype),
            raw_y=pl.col("fvalue").cast(pl.Float64).replace_strict(labels, values),
            gold="diff",
        )
        joined = expected.join(ours, on=["algorithm_name", "evaluations", "raw_y"])
        assert joined.height == expected.height == ours.height
        np.testing.assert_allclose(
            joined["eaf_margin"], joined["gold"], rtol=0, atol=1e-12
        )
