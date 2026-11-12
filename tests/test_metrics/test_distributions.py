import math

import numpy as np
import polars as pl
import pytest

from iohinspector.metrics.distributions import (
    get_histogram,
    get_runtime_sample,
    get_value_sample,
    get_violin,
    nclass_fd,
    pretty,
)
from tests.golden.load import golden, logged

FIXTURES = ["ioh040_min", "ioh040_max", "ioh040_cons"]
PAIR = ["pair_f", "pair_d"]


def pair_data(fixture, f, d):
    data = logged(fixture)
    return data.filter((pl.col("function_id") == f) & (pl.col("dimension") == d))


def pairs(fixture):
    return golden(fixture, "RT_sample").select(PAIR).unique().sort(PAIR).rows()


def maximization(fixture):
    return bool(golden(fixture, "meta")["maximization"][0])


def cases(metric):
    return [(fx, f, d) for fx in FIXTURES for f, d in pairs(fx)]


def sorted_values(df, keys, value):
    return df.sort([*keys, value]).select(*keys, value)


@pytest.mark.parametrize("fixture,f,d", cases("RT_sample"))
def test_runtime_sample_matches_golden(fixture, f, d):
    ref = golden(fixture, "RT_sample").filter(
        (pl.col("pair_f") == f) & (pl.col("pair_d") == d)
    )
    targets = ref["target"].unique().to_list()
    mine = get_runtime_sample(
        pair_data(fixture, int(f), int(d)),
        targets,
        maximization=maximization(fixture),
    )
    # failed runs: NA in IOHanalyzer, inf here (runtime, RT-1)
    mine = mine.filter(pl.col("runtime").is_finite())
    ref = ref.drop_nulls("RT").rename({"ID": "algorithm_name", "RT": "runtime"})
    keys = ["algorithm_name", "target"]
    a = sorted_values(mine, keys, "runtime").with_columns(pl.col("runtime") * 1.0)
    b = sorted_values(ref, keys, "runtime").with_columns(pl.col("runtime") * 1.0)
    assert a.equals(b)


def test_samples_have_only_their_columns():
    # other columns of the data are not carried along (null for a failed run)
    data = pl.DataFrame(
        {
            "run_id": [1, 1, 2],
            "algorithm_name": ["A", "A", "B"],
            "function_id": [1, 1, 1],
            "x0": [0.5, 0.25, 0.75],
            "evaluations": [1, 3, 2],
            "raw_y": [5.0, 2.0, 4.0],
        }
    )
    runtimes = get_runtime_sample(data, [3.0])
    assert runtimes.rows() == [(1, "A", 3.0, 3.0), (2, "B", 3.0, math.inf)]
    assert runtimes.columns == ["run_id", "algorithm_name", "target", "runtime"]
    values = get_value_sample(data, [1, 2], by=[])
    assert values.rows() == [(1, 1, 5.0), (1, 2, 5.0), (2, 1, None), (2, 2, 4.0)]
    assert values.columns == ["run_id", "budget", "value"]


@pytest.mark.parametrize("fixture,f,d", cases("FV_sample"))
def test_value_sample_matches_golden(fixture, f, d):
    ref = golden(fixture, "FV_sample").filter(
        (pl.col("pair_f") == f) & (pl.col("pair_d") == d)
    )
    budgets = [int(b) for b in ref["runtime"].unique()]
    mine = get_value_sample(
        pair_data(fixture, int(f), int(d)),
        budgets,
        maximization=maximization(fixture),
    ).drop_nulls("value")
    ref = ref.drop_nulls("f(x)").rename(
        {"ID": "algorithm_name", "runtime": "budget", "f(x)": "value"}
    )
    keys = ["algorithm_name", "budget"]
    a = sorted_values(mine.cast({"budget": pl.Float64}), keys, "value")
    b = sorted_values(ref, keys, "value")
    assert a.select(keys).equals(b.select(keys))
    np.testing.assert_allclose(a["value"], b["value"], rtol=1e-12)


@pytest.mark.parametrize("which", ["RT", "FV"])
@pytest.mark.parametrize("fixture,f,d", cases("hist_by_RT"))
def test_histogram_matches_golden(fixture, f, d, which):
    """IOHanalyzer's per-algorithm histograms (bins="per_algorithm")."""
    ref = golden(fixture, f"hist_by_{which}").filter(
        (pl.col("pair_f") == f) & (pl.col("pair_d") == d)
    )
    sample_ref = golden(fixture, "RT_sample" if which == "RT" else "FV_sample")
    sample_ref = sample_ref.filter((pl.col("pair_f") == f) & (pl.col("pair_d") == d))
    data = pair_data(fixture, int(f), int(d))
    if which == "RT":  # the third target, from the largest
        target = sorted(sample_ref["target"].unique(), reverse=True)[2]
        sample = get_runtime_sample(data, [target], maximization=maximization(fixture))
        value = "runtime"
    else:
        budget = sorted(sample_ref["runtime"].unique())[2]
        sample = get_value_sample(
            data, [int(budget)], maximization=maximization(fixture)
        )
        value = "value"
    mine = get_histogram(sample, value, bins="per_algorithm")
    mine = mine.sort("algorithm_name", "mid")
    ref = ref.sort("ID", "x")
    assert mine["algorithm_name"].to_list() == ref["ID"].to_list()
    np.testing.assert_allclose(mine["mid"], ref["x"], rtol=1e-12)
    assert mine["count"].to_list() == ref["y"].to_list()
    np.testing.assert_allclose(mine["width"], ref["width"], rtol=1e-9)


@pytest.mark.parametrize(
    "lo,up,n,expected",
    [  # values of R 4.3.3: pretty(c(lo, up), n, min.n = 1)
        (0, 1, 5, [0, 0.2, 0.4, 0.6, 0.8, 1]),
        (150, 350, 2, [100, 200, 300, 400]),
        (0.13, 7.7, 3, [0, 2, 4, 6, 8]),
        (-3.2, 4.1, 6, [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5]),
        (5, 5, 1, [0, 5]),
        (0, 0, 1, [-1, 0]),
        (1e-9, 3e-9, 4, [1e-9, 1.5e-9, 2e-9, 2.5e-9, 3e-9]),
    ],
)
def test_pretty(lo, up, n, expected):
    np.testing.assert_allclose(pretty(lo, up, n), expected, atol=1e-12)


def test_nclass_fd():
    x = np.arange(1.0, 11)  # IQR 4.5, h = 9: ceiling(9 / (9 * 10^(-1/3)))
    assert nclass_fd(x) == math.ceil(10 ** (1 / 3))
    assert nclass_fd(np.array([3.0, 3.0, 3.0])) == 1


def sample(values, groups, column="v"):
    return pl.DataFrame({"algorithm_name": groups, column: values})


def test_histogram_equal_bins_and_per_algorithm():
    s = sample([1.0, 2, 2, 3, 4, 9, 9, 10, 10, 10], list("aaaaabbbbb"))

    def edges(h):
        return {
            k[0]: f["left"].to_list()
            for k, f in h.partition_by("algorithm_name", as_dict=True).items()
        }

    equal, own = (
        edges(get_histogram(s, "v")),
        edges(get_histogram(s, "v", bins="per_algorithm")),
    )
    assert equal["a"] == equal["b"]  # DI-4: one set of breaks for both groups
    assert own["a"] != own["b"]  # per algorithm: each its own breaks


def test_histogram_right_closed():
    # R: hist(c(0, 1, 2, 3, 4), breaks = 2)$counts is 2 2 (bins [0, 2], (2, 4])
    h = get_histogram(sample([0.0, 1, 2, 3, 4], ["a"] * 5), "v")
    assert h["left"].to_list() == [0.0, 2.0] and h["count"].to_list() == [3, 2]


def test_histogram_counts_failed_and_skips_small_groups():
    s = sample([1.0, 2, 3, np.inf, 5.0], ["a", "a", "a", "a", "b"], "runtime")
    h = get_histogram(s, "runtime")
    assert h["algorithm_name"].unique().to_list() == ["a"]  # b has one value
    assert h["n_failed"].unique().to_list() == [1]
    assert h["count"].sum() == 3
    assert h["n"].unique().to_list() == [3]


def test_histogram_view_columns():
    s = sample([1.0, 2, 3, 4, 100, 110, 120, 130], ["a"] * 8).with_columns(
        target=pl.Series([1] * 4 + [2] * 4)
    )
    h = get_histogram(s, "v", view=["target"])
    lo = h.group_by("target").agg(pl.col("left").min())
    assert lo.sort("target")["left"].to_list() == [0.0, 100.0]


def test_violin():
    x = np.array([1.0, 2.0, 2.5, 4.0, 7.0])
    s = sample(x, ["a"] * 5)
    v = get_violin(s, "v").row(0, named=True)
    sd, (q1, q3) = x.std(ddof=1), np.quantile(x, [0.25, 0.75])
    bw = 1.059 * min(sd, (q3 - q1) / 1.349) * 5**-0.2
    assert v["bandwidth"] == pytest.approx(bw, rel=1e-12)
    assert (v["q1"], v["median"], v["q3"]) == (q1, 2.5, q3)
    grid = np.array(v["x"])
    assert grid[0] == 1.0 and grid[-1] == 7.0  # hard span
    assert np.diff(grid).max() <= bw / 3 * (1 + 1e-12)
    i = 3
    ref = sum(math.exp(-0.5 * ((grid[i] - xi) / bw) ** 2) for xi in x)
    assert v["density"][i] == pytest.approx(ref / (bw * math.sqrt(2 * math.pi)))


def test_violin_degenerate_and_failed():
    s = sample([3.0, 3.0, 3.0, np.inf], ["a"] * 4)
    v = get_violin(s, "v").row(0, named=True)
    assert (v["n"], v["n_failed"], v["x"], v["density"]) == (3, 1, [3.0], [0.0])
