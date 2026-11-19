import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from iohinspector.figures.labels import spread


def test_separated_positions_stay():
    np.testing.assert_allclose(spread([0.1, 0.5, 0.9], 0.1), [0.1, 0.5, 0.9])


def test_overlapping_positions_move_apart_symmetrically():
    np.testing.assert_allclose(spread([0.5, 0.5], 0.1), [0.45, 0.55])
    np.testing.assert_allclose(spread([0.52, 0.5, 0.48], 0.1), [0.6, 0.5, 0.4])


def test_positions_stay_in_bounds():
    np.testing.assert_allclose(spread([1.0, 0.98], 0.1), [1.0, 0.9])
    np.testing.assert_allclose(spread([0.0, 0.0, 0.0], 0.1), [0.0, 0.1, 0.2])


def test_too_many_labels_are_centred_on_the_range():
    np.testing.assert_allclose(spread([0.5] * 3, 0.6), [-0.1, 0.5, 1.1])


def test_no_labels():
    assert spread([], 0.1).shape == (0,)


@given(
    st.lists(st.floats(0, 1), min_size=1, max_size=12),
    st.floats(0.001, 0.2),
)
def test_spread_keeps_order_gap_and_bounds(y, gap):
    p = spread(y, gap)
    order = np.argsort(y, kind="stable")
    assert np.all(np.diff(p[order]) >= gap - 1e-9)
    if gap * (len(y) - 1) <= 1:
        assert p.min() >= -1e-9 and p.max() <= 1 + 1e-9
