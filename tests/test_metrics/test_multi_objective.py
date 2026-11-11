import numpy as np
import polars as pl
import pytest

from iohinspector.indicators.anytime import Epsilon, HyperVolume, IGDPlus
from iohinspector.metrics.multi_objective import (
    get_indicator_over_time_data,
    get_pareto_front_2d,
)

PARETO_DF = pl.DataFrame(
    {
        # For minimization, Pareto front = non-dominated points with lowest
        # values in both objectives
        # A: Only (0.1, 0.9) is non-dominated for A
        # B: (0.2, 0.8) and (0.4, 0.6) are non-dominated for B
        # C: (0.3, 0.7), (0.6, 0.4), (0.7, 0.3) are all non-dominated for C
        "raw_y": [0.1, 0.5, 0.9, 0.2, 0.5, 0.9, 0.3, 0.6, 0.9],
        "F2": [0.2, 0.5, 0.8, 0.8, 0.2, 0.9, 0.7, 0.4, 0.1],
        "algorithm_name": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "evaluations": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "run_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
    }
)

# Minimal DataFrame with two objectives and a single algorithm
# All points belong to algorithm "A" with 10 evaluations
# The points are constructed to simulate a progression towards the Pareto front
INDICATOR_DF = pl.DataFrame(
    {
        "raw_y": [0.9, 0.7, 0.5, 0.3, 0.1],
        "F2": [0.8, 0.6, 0.4, 0.2, 0.1],
        "algorithm_name": ["A"] * 5,
        "evaluations": [1, 10, 100, 1000, 10000],
        "run_id": [1] * 5,
    }
)
# A dict mapping evaluation to (raw_y, F2) point
EVAL_POINTS = dict(
    zip(INDICATOR_DF["evaluations"], zip(INDICATOR_DF["raw_y"], INDICATOR_DF["F2"]))
)


def test_basic_call():
    result = get_pareto_front_2d(PARETO_DF)
    assert "final_nondominated" in result.columns

    expected = {
        ("A", 0.1, 0.2): True,
        ("B", 0.2, 0.8): True,
        ("B", 0.5, 0.2): True,
        ("C", 0.3, 0.7): True,
        ("C", 0.6, 0.4): True,
        ("C", 0.9, 0.1): True,
    }
    for row in result.iter_rows(named=True):
        key = (row["algorithm_name"], row["raw_y"], row["F2"])
        assert row["final_nondominated"] == expected[key]


def test_custom_obj_vars():
    # Test with custom objective variable names
    df_custom = PARETO_DF.rename({"raw_y": "obj1", "F2": "obj2"})
    result = get_pareto_front_2d(df_custom, obj1_var="obj1", obj2_var="obj2")
    assert "final_nondominated" in result.columns
    # Check that the correct points are marked as non-dominated for each
    # algorithm, point by point
    expected = {
        ("A", 0.1, 0.2): True,
        ("B", 0.2, 0.8): True,
        ("B", 0.5, 0.2): True,
        ("C", 0.3, 0.7): True,
        ("C", 0.6, 0.4): True,
        ("C", 0.9, 0.1): True,
    }
    for row in result.iter_rows(named=True):
        key = (row["algorithm_name"], row["obj1"], row["obj2"])
        assert row["final_nondominated"] == expected[key]


def test_plot_indicator_over_time_hypervolume():
    # Use a simple indicator and check output DataFrame
    indicator = HyperVolume(reference_point=[1.0, 1.0])
    result = get_indicator_over_time_data(
        INDICATOR_DF,
        indicator=indicator,
        eval_steps=5,
        eval_min=1,
        eval_max=10_000,
        scale_eval_log=True,
        obj_vars=["raw_y", "F2"],
    )
    # Make a dict of {evaluation: hypervolume}
    hv_dict = dict(zip(result["evaluations"], result["HyperVolume"]))

    for eval in [1, 10, 100, 1000, 10000]:
        point = EVAL_POINTS[eval]
        hv = (1.0 - point[0]) * (1.0 - point[1])  # Since we minimize both objectives
        assert hv_dict[eval] == pytest.approx(hv, abs=5e-6)


def test_plot_indicator_over_time_epsilon_additive():
    # Use a simple indicator and check output DataFrame
    indicator = Epsilon(reference_point=[1.0, 1.0])
    result = get_indicator_over_time_data(
        INDICATOR_DF,
        indicator=indicator,
        eval_steps=5,
        eval_min=1,
        eval_max=10_000,
        scale_eval_log=True,
        obj_vars=["raw_y", "F2"],
    )
    # Make a dict of {evaluation: hypervolume}
    ae = dict(zip(result["evaluations"], result["Epsilon_Additive"]))
    for eval in [1, 10, 100, 1000, 10000]:
        point = EVAL_POINTS[eval]
        eps = max(point[0] - 1.0, point[1] - 1.0)  # Since we minimize both objectives
        assert ae[eval] == pytest.approx(eps, abs=5e-6)


def test_plot_indicator_over_time_epsilon_multiplicative():
    # Use a simple indicator and check output DataFrame
    indicator = Epsilon(reference_point=[1.0, 1.0], version="multiplicative")
    result = get_indicator_over_time_data(
        INDICATOR_DF,
        indicator=indicator,
        eval_steps=5,
        eval_min=1,
        eval_max=10_000,
        scale_eval_log=True,
        obj_vars=["raw_y", "F2"],
    )
    # Make a dict of {evaluation: hypervolume}
    ae = dict(zip(result["evaluations"], result["Epsilon_Mult"]))
    for eval in [1, 10, 100, 1000, 10000]:
        point = EVAL_POINTS[eval]
        eps = max(point[0] / 1.0, point[1] / 1.0)  # Since we minimize both objectives
        assert ae[eval] == pytest.approx(eps, abs=5e-6)


def test_plot_indicator_over_time_igd_plus():
    # Use a simple indicator and check output DataFrame
    indicator = IGDPlus(reference_set=[[0.0, 0.0]])
    result = get_indicator_over_time_data(
        INDICATOR_DF,
        indicator=indicator,
        eval_steps=5,
        eval_min=1,
        eval_max=10_000,
        scale_eval_log=True,
        obj_vars=["raw_y", "F2"],
    )
    # Make a dict of {evaluation: hypervolume}
    ae = dict(zip(result["evaluations"], result["IGD+"]))
    for eval in [1, 10, 100, 1000, 10000]:
        point = EVAL_POINTS[eval]
        # Since we minimize both objectives
        idg_plus = np.sqrt((point[0] - 0.0) ** 2 + (point[1] - 0.0) ** 2)
        assert ae[eval] == pytest.approx(idg_plus, abs=5e-6)
