"""What a figure kind hands to the renderer: curves and axis defaults."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Curve:
    """One group's line, in data coordinates; it ends at its last point."""

    label: str
    x: np.ndarray
    y: np.ndarray


@dataclass
class Plot:
    """The curves of a figure kind and its defaults for ``spec.Axis`` fields
    (``label``, ``scale``, ``lim``, ``ticks``). ``step``: draw the curves as
    steps, each value holding until the next x."""

    curves: list[Curve]
    x: dict = field(default_factory=dict)
    y: dict = field(default_factory=dict)
    step: bool = False
