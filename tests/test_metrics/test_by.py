"""Grouping with ``by`` (D17): any runs column or descriptor, not only the algorithm."""

import warnings

import polars as pl
import pytest

from iohinspector import DataManager
from iohinspector.metrics import (
    aggregate_convergence,
    aggregate_running_time,
    get_data_ecdf,
    get_ecdf,
    get_overview,
    get_trajectory,
    group_label,
)
from tests.golden.load import fixture_path

METRICS = [
    aggregate_convergence,
    aggregate_running_time,
    get_data_ecdf,
    get_ecdf,
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
    pair = ["function_id", "dimension"] if metric is get_ecdf else []  # its pairs
    df = manager.load(include_columns=list(dict.fromkeys([*by, *pair])))
    assert set(metric(df, by=by).select(by).unique().rows()) == groups


def test_group_label():
    df = pl.DataFrame({"algorithm_name": ["es", "rs"], "sigma0": [0.1, None]})
    by = ["algorithm_name", "sigma0"]
    assert df.select(group_label(by))["label"].to_list() == ["es, 0.1", "rs, null"]
    template = "{algorithm_name} (sigma {sigma0})"
    labels = df.select(group_label(by, template))["label"].to_list()
    assert labels == ["es (sigma 0.1)", "rs (sigma null)"]
    with pytest.raises(ValueError, match="only"):
        group_label(by, "{sigma0:.2f}")
