"""BBOB-style direct labels: each curve's name at the right of the axes, at the
height of its final value, spread so that no two overlap and joined to the
curve's end by a short leader."""

import numpy as np


def spread(y, gap: float, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Positions closest to ``y`` (least squares) that keep the order of ``y``, are
    at least ``gap`` apart and lie in [``lo``, ``hi``] (wider, centred on it, when
    they do not fit).

    With z_i = p_i - i gap for the sorted positions p, the gap constraints say z
    is non-decreasing, so the answer is the isotonic regression of y_i - i gap
    (pool adjacent violators), clipped to the bounds and shifted back.
    """
    y = np.asarray(y, dtype=float)
    order = np.argsort(y, kind="stable")
    steps = gap * np.arange(len(y))
    room = hi - lo - steps[-1] if len(y) else 0.0
    if room < 0:
        lo, room = lo + room / 2, 0.0
    z = np.clip(_isotonic(y[order] - steps), lo, lo + room)
    out = np.empty_like(y)
    out[order] = z + steps
    return out


def _isotonic(v: np.ndarray) -> np.ndarray:
    """The non-decreasing sequence closest to ``v`` (pool adjacent violators)."""
    blocks: list[list[float]] = []  # [mean, count]
    for value in v:
        blocks.append([value, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, n2 = blocks.pop()
            m1, n1 = blocks.pop()
            blocks.append([(m1 * n1 + m2 * n2) / (n1 + n2), n1 + n2])
    return np.repeat([b[0] for b in blocks], [int(b[1]) for b in blocks])


def add_direct_labels(ax, ends, size: float) -> list:
    """One label per ``ends`` item (name, x, y, colour), placed at its final value
    for now; ``place_direct_labels`` spreads them once the layout is known."""
    return [
        ax.annotate(
            name,
            xy=(x, y),
            xycoords="data",
            xytext=(1.04, 0.5),
            textcoords="axes fraction",
            va="center",
            ha="left",
            fontsize=size,
            annotation_clip=False,
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "linewidth": 0.6,
                "shrinkA": 1,
                "shrinkB": 0,
            },
        )
        for name, x, y, color in ends
    ]


def place_direct_labels(ax, texts, size: float) -> None:
    """Spread ``texts`` over the height of ``ax``, ordered by the final values
    their leaders point at, one line height (1.2 ``size``) apart at least."""
    if not texts:
        return
    to_axes = ax.transData + ax.transAxes.inverted()
    y = [to_axes.transform(t.xy)[1] for t in texts]
    height = ax.bbox.height * 72 / ax.figure.dpi  # points
    for text, pos in zip(texts, spread(y, 1.2 * size / height)):
        text.set_position((1.04, pos))
