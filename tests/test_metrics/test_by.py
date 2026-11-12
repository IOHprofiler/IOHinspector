"""Grouping with ``by`` (D17): any runs column or descriptor, not only the algorithm."""

import warnings

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from iohinspector import DataManager
from iohinspector.metrics import (
    aggregate_convergence,
    aggregate_running_time,
    get_data_ecdf,
    get_ecdf,
    get_ecdf_auc,
    get_overview,
    get_trajectory,
    group_label,
)
from tests.golden.load import fixture_path

pytestmark = pytest.mark.filterwarnings("ignore:get_data_ecdf:DeprecationWarning")

METRICS = [
    aggregate_convergence,
    aggregate_running_time,
    get_data_ecdf,
    get_ecdf,
    get_ecdf_auc,
    get_overview,
    get_trajectory,
]


@pytest.fixture(scope="module")
def manager():
    """``ioh040_min`` with a user tag ``variant`` and a numeric parameter ``sigma0``
    on the two ES variants, and neither on random search."""
    m = DataManager()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, tags in [
            ("es_small", ["variant=small", "sigma0:float=0.1"]),
            ("es_large", ["variant=large", "sigma0:float=1"]),
            ("rs", []),
        ]:
            m.add_folder(fixture_path("ioh040_min") / name, tags=tags)
    return m


@pytest.mark.parametrize("metric", METRICS, ids=lambda f: f.__name__)
@pytest.mark.parametrize(
    ("by", "groups"),
    [
        (["algorithm_name"], {("es_large",), ("es_small",), ("rs",)}),
        (["sigma0"], {(0.1,), (1.0,), (None,)}),
        (["function_id"], {(1,), (2,), (3,)}),
        (["variant"], {("small",), ("large",), (None,)}),
        (
            ["variant", "dimension"],
            {(v, d) for v in ("small", "large", None) for d in (5, 10)},
        ),
    ],
)
def test_metrics_group_by_any_descriptor(manager, metric, by, groups):
    pair = ["function_id", "dimension"] if "_ecdf" in metric.__name__ else []
    df = manager.load(include_columns=list(dict.fromkeys([*by, *pair])))
    assert set(metric(df, by=by).select(by).unique().rows()) == groups


@pytest.mark.parametrize("metric", METRICS, ids=lambda f: f.__name__)
def test_no_by_is_one_group(manager, metric):
    df = manager.load(include_columns=["function_id", "dimension"])
    one = metric(df.with_columns(group=pl.lit("all")), by=["group"]).drop("group")
    assert_frame_equal(metric(df, by=[]), one, check_row_order=False)


@pytest.mark.parametrize("metric", METRICS, ids=lambda f: f.__name__)
def test_a_missing_descriptor_is_a_group_like_any_other(manager, metric):
    # random search has no variant: its runs are the group null, with their values
    df = manager.load(include_columns=["variant", "function_id", "dimension"])
    filled = df.with_columns(pl.col("variant").fill_null("none"))
    expected = metric(filled, by=["variant"])
    expected = expected.with_columns(pl.col("variant").replace("none", None))
    result = metric(df, by=["variant"])
    assert_frame_equal(result, expected, check_row_order=False)


def test_group_label():
    df = pl.DataFrame({"algorithm_name": ["es", "rs"], "sigma0": [0.1, None]})
    by = ["algorithm_name", "sigma0"]
    assert df.select(group_label(by))["label"].to_list() == ["es, 0.1", "rs, null"]
    template = "{algorithm_name} (sigma {sigma0})"
    labels = df.select(group_label(by, template))["label"].to_list()
    assert labels == ["es (sigma 0.1)", "rs (sigma null)"]
    with pytest.raises(ValueError, match="only"):
        group_label(by, "{sigma0:.2f}")
