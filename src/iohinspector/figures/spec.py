"""``FigureSpec``: a JSON-serialisable description of a publication figure (D11).

A spec says what is plotted (``Data``), for which venue and column, and how it
looks: text, axes, palette, per-series overrides, legend and output. Every field
has a default, so ``FigureSpec()`` is a valid BBOB-style ECDF. ``to_json`` and
``FigureSpec.from_json`` round-trip; unknown keys and invalid values raise
``ValueError``, a list or number where an object belongs ``TypeError``.
"""

import dataclasses
import json
import typing
from dataclasses import dataclass, field
from typing import Any

from matplotlib import colors, legend, lines, markers

from .venues import PALETTES, VENUES

KINDS = ("ecdf",)
SPANS = ("single", "double")
LEGENDS = ("direct", "box", "outside", "none")
SCALES = ("linear", "log")
TICKS = ("auto", "exponent", "power", "plain")
GRIDS = ("none", "major", "both")
FORMATS = ("pdf", "svg", "png")
TARGETS = ("log-linear", "linear", "bbob")


def _one_of(name: str, value, allowed) -> None:
    if value not in allowed:
        raise ValueError(f"{name} {value!r}: one of {', '.join(map(str, allowed))}")


def _positive(name: str, value) -> None:
    if value is not None and not value > 0:
        raise ValueError(f"{name} {value!r}: must be positive")


@dataclass
class Data:
    """What is plotted. ``by`` makes the groups (one curve each) and ``label`` is
    their label template (D17, ``metrics.group_label``), such as
    ``"{algorithm_name} ($\\sigma$ = {sigma})"``. ``where`` keeps the rows whose
    column value is listed, for example ``{"dimension": [10]}``."""

    by: list[str] = field(default_factory=lambda: ["algorithm_name"])
    label: str | None = None
    where: dict[str, list] = field(default_factory=dict)
    targets: str = "log-linear"
    per_dimension: bool = True
    maximization: bool = False

    def __post_init__(self):
        if not self.by:
            raise ValueError("data.by: at least one column")
        _one_of("data.targets", self.targets, TARGETS)


@dataclass
class Text:
    """Font family and sizes in points. ``None`` takes the venue's: its font and
    its figure caption size (the title at its body size)."""

    family: str | None = None
    size: float | None = None
    title_size: float | None = None
    tick_size: float | None = None
    legend_size: float | None = None

    def __post_init__(self):
        for name in ("size", "title_size", "tick_size", "legend_size"):
            _positive(f"text.{name}", getattr(self, name))


@dataclass
class Axis:
    """One axis. ``label`` and ``scale`` default to the figure kind's (``""`` is no
    label); ``lim`` is ``[lo, hi]`` with ``None`` for automatic. ``ticks`` formats
    the tick labels of a log axis: ``exponent`` (3), ``power`` (10^3) or ``plain``
    (1000)."""

    label: str | None = None
    scale: str | None = None
    lim: list[float | None] | None = None
    ticks: str = "auto"
    grid: str = "major"

    def __post_init__(self):
        if self.scale is not None:
            _one_of("axis scale", self.scale, SCALES)
        _one_of("axis ticks", self.ticks, TICKS)
        _one_of("axis grid", self.grid, GRIDS)
        if self.lim is not None and len(self.lim) != 2:
            raise ValueError(f"axis lim {self.lim!r}: [lo, hi]")


@dataclass
class Series:
    """Overrides for one group, keyed by its label in ``FigureSpec.series``.
    ``name`` is the text shown for it."""

    name: str | None = None
    color: str | None = None
    linestyle: str | None = None
    linewidth: float | None = None
    marker: str | None = None
    visible: bool = True

    def __post_init__(self):
        if self.color is not None and not colors.is_color_like(self.color):
            raise ValueError(f"series color {self.color!r}: not a colour")
        if self.linestyle is not None:
            _one_of("series linestyle", self.linestyle, lines.lineStyles)
        if self.marker is not None:
            _one_of("series marker", self.marker, markers.MarkerStyle.markers)
        _positive("series linewidth", self.linewidth)


@dataclass
class Legend:
    """``direct``: BBOB-style labels at the right of the lines, ordered by final
    value. ``box``: a legend in the axes at ``loc``; ``outside``: to its right."""

    mode: str = "direct"
    loc: str = "best"
    columns: int = 1

    def __post_init__(self):
        _one_of("legend mode", self.mode, LEGENDS)
        _one_of("legend loc", self.loc, legend.Legend.codes)
        _positive("legend columns", self.columns)


@dataclass
class FigureSpec:
    """A publication figure: what, for which venue, and how it looks.

    ``span`` is one column or the full text width of the venue; ``width_cm``
    overrides it. The height is ``height_cm``, else width times ``aspect``.
    ``palette`` is a palette name (``figures.PALETTES``) or a list of colours.
    ``markers`` is the number of sparse markers per line. ``render`` writes
    ``formats[0]`` unless told otherwise; ``dpi`` applies to PNG.
    """

    kind: str = "ecdf"
    data: Data = field(default_factory=Data)
    venue: str = "generic"
    span: str = "single"
    width_cm: float | None = None
    height_cm: float | None = None
    aspect: float = 0.75
    title: str | None = None
    text: Text = field(default_factory=Text)
    x: Axis = field(default_factory=Axis)
    y: Axis = field(default_factory=Axis)
    palette: str | list[str] = "okabe-ito"
    series: dict[str, Series] = field(default_factory=dict)
    legend: Legend = field(default_factory=Legend)
    markers: int = 4
    formats: list[str] = field(default_factory=lambda: ["pdf"])
    dpi: int = 300

    def __post_init__(self):
        _one_of("kind", self.kind, KINDS)
        _one_of("venue", self.venue, VENUES)
        _one_of("span", self.span, SPANS)
        for name in ("width_cm", "height_cm", "aspect", "dpi"):
            _positive(name, getattr(self, name))
        if isinstance(self.palette, str):
            _one_of("palette", self.palette, PALETTES)
        elif not self.palette or not all(map(colors.is_color_like, self.palette)):
            raise ValueError(f"palette {self.palette!r}: a name or a list of colours")
        if self.markers < 0:
            raise ValueError(f"markers {self.markers!r}: 0 or more")
        if not self.formats:
            raise ValueError("formats: at least one")
        for fmt in self.formats:
            _one_of("format", fmt, FORMATS)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FigureSpec":
        return _build(cls, value, "spec")

    @classmethod
    def from_json(cls, text: str) -> "FigureSpec":
        return cls.from_dict(json.loads(text))


def _build(cls, value, path: str):
    """``cls`` from the dict ``value``, nested dataclasses included; errors name
    the ``path`` of the offending key."""
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected an object, got {value!r}")
    hints = typing.get_type_hints(cls)
    unknown = set(value) - set(hints)
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    return cls(**{k: _field(hints[k], v, f"{path}.{k}") for k, v in value.items()})


def _field(hint, value, path: str):
    if dataclasses.is_dataclass(hint):
        return _build(hint, value, path)
    if typing.get_origin(hint) is dict and typing.get_args(hint)[1] is Series:
        if not isinstance(value, dict):
            raise TypeError(f"{path}: expected an object, got {value!r}")
        return {k: _build(Series, v, f"{path}.{k}") for k, v in value.items()}
    return value
