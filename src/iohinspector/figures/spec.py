"""``FigureSpec``: a JSON-serialisable description of a publication figure (D11).

A spec has one or more ``panels``, each saying what is plotted (``kind`` and
``Data``) on which axes, with its annotations; the figure says for which venue
and column it is, its size and background, and how every panel looks: text,
frame, palette, per-series overrides, legend and output. Every field has a
default, so ``FigureSpec()`` is a valid BBOB-style ECDF. ``to_json`` and
``FigureSpec.from_json`` round-trip; unknown keys and invalid values raise
``ValueError``, a list or number where an object belongs ``TypeError``.

Specs carry a ``version``. ``from_dict`` upgrades older specs (version 1 had no
``version`` key: one panel's fields at the top, sizes in ``width_cm`` and
``height_cm``). ``STYLES`` are named spec fragments; ``apply_style`` merges one
into a spec.
"""

import dataclasses
import json
import typing
from dataclasses import dataclass, field
from typing import Any

from matplotlib import colors, legend, lines, markers

from .venues import PALETTES, VENUES

VERSION = 2
KINDS = ("ecdf",)
SPANS = ("single", "double")
UNITS = {"cm": 1 / 2.54, "in": 1.0, "pt": 1 / 72}  # inches per unit (pt: 1/72 in)
BACKGROUNDS = ("white", "none", "dark")
FRAMES = ("box", "spines", "none")
PANEL_LABELS = ("letters", "names", "none")
LEGENDS = ("direct", "box", "outside", "none")
ORDERS = ("value", "yours", "name", "drawing")
HIGHLIGHTS = ("yours", "other")
SCALES = ("linear", "log")
TICKS = ("auto", "exponent", "power", "sci", "plain")
GRIDS = ("none", "major", "both")
FORMATS = ("pdf", "svg", "png", "pgf")
TARGETS = ("log-linear", "linear", "bbob")
ANNOTATIONS = {  # kind: the coordinates it takes, one set or another
    "text": [{"x", "y"}],
    "arrow": [{"x", "y", "x2", "y2"}],
    "line": [{"x"}, {"y"}],
    "region": [{"x", "x2"}, {"y", "y2"}],
}


def _one_of(name: str, value, allowed) -> None:
    if value not in allowed:
        raise ValueError(f"{name} {value!r}: one of {', '.join(map(str, allowed))}")


def _positive(name: str, value) -> None:
    if value is not None and not value > 0:
        raise ValueError(f"{name} {value!r}: must be positive")


def _colour(name: str, value) -> None:
    if value is not None and not colors.is_color_like(value):
        raise ValueError(f"{name} {value!r}: not a colour")


@dataclass
class Data:
    """What is plotted. ``by`` makes the groups (one curve each) and ``label`` is
    their label template (D17, ``metrics.group_label``), such as
    ``"{algorithm_name} ($\\sigma$ = {sigma})"``. ``where`` keeps the runs whose
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
    its figure caption size (the title at its body size); ``annotation_size``
    defaults to ``size``. A family other than matplotlib's own fonts must be
    installed where the figure is rendered."""

    family: str | None = None
    size: float | None = None
    title_size: float | None = None
    tick_size: float | None = None
    legend_size: float | None = None
    annotation_size: float | None = None

    def __post_init__(self):
        for name in ("size", "title", "tick", "legend", "annotation"):
            name = name if name == "size" else f"{name}_size"
            _positive(f"text.{name}", getattr(self, name))


@dataclass
class Axis:
    """One axis. ``label`` and ``scale`` default to the figure kind's (``""`` is no
    label); ``lim`` is ``[lo, hi]`` with ``None`` for automatic. ``ticks`` formats
    the tick labels of a log axis: ``exponent`` (3), ``power`` (10^3), ``sci``
    (1e3) or ``plain`` (1000). ``tick_count`` asks for about that many major
    ticks. ``grid``: major grid lines, ``both`` with the minor ones, or none."""

    label: str | None = None
    scale: str | None = None
    lim: list[float | None] | None = None
    ticks: str = "auto"
    tick_count: int | None = None
    grid: str = "major"

    def __post_init__(self):
        if self.scale is not None:
            _one_of("axis scale", self.scale, SCALES)
        _one_of("axis ticks", self.ticks, TICKS)
        _one_of("axis grid", self.grid, GRIDS)
        _positive("axis tick_count", self.tick_count)
        if self.lim is not None and len(self.lim) != 2:
            raise ValueError(f"axis lim {self.lim!r}: [lo, hi]")


@dataclass
class Annotation:
    """Text, an arrow, a reference line or a shaded region, in the axes' data
    units. ``text``: ``text`` at (``x``, ``y``). ``arrow``: from (``x2``, ``y2``),
    where ``text`` sits, to (``x``, ``y``). ``line``: at ``x`` (vertical) or at
    ``y`` (horizontal). ``region``: from ``x`` to ``x2`` (vertical band) or from
    ``y`` to ``y2``. ``color`` defaults to the text colour."""

    kind: str = "text"
    text: str = ""
    x: float | None = None
    y: float | None = None
    x2: float | None = None
    y2: float | None = None
    color: str | None = None

    def __post_init__(self):
        _one_of("annotation kind", self.kind, ANNOTATIONS)
        _colour("annotation color", self.color)
        given = {k for k in ("x", "y", "x2", "y2") if getattr(self, k) is not None}
        if given not in ANNOTATIONS[self.kind]:
            need = " or ".join(", ".join(sorted(n)) for n in ANNOTATIONS[self.kind])
            raise ValueError(f"annotation {self.kind}: needs {need}")


@dataclass
class Panel:
    """One plot of the figure: its ``kind``, ``data``, axes, ``title`` (its name)
    and ``annotations``."""

    kind: str = "ecdf"
    data: Data = field(default_factory=Data)
    title: str | None = None
    x: Axis = field(default_factory=Axis)
    y: Axis = field(default_factory=Axis)
    annotations: list[Annotation] = field(default_factory=list)

    def __post_init__(self):
        _one_of("kind", self.kind, KINDS)


@dataclass
class Layout:
    """How the panels sit: ``columns`` per row, ``gap`` between them in the
    figure's unit (``None``: matplotlib's), panel ``labels`` ((a), (b), ...; their
    titles; or none) and shared x and y axes."""

    columns: int = 1
    gap: float | None = None
    labels: str = "names"
    share_x: bool = False
    share_y: bool = False

    def __post_init__(self):
        _positive("layout columns", self.columns)
        if self.gap is not None and self.gap < 0:
            raise ValueError(f"layout gap {self.gap!r}: 0 or more")
        _one_of("layout labels", self.labels, PANEL_LABELS)


@dataclass
class Series:
    """Overrides for one group, keyed by its label in ``FigureSpec.series``.
    ``name`` is the text shown for it. ``every``: a marker on every n-th point
    instead of the figure's sparse markers. ``order``: higher is drawn later, on
    top. ``band``: draw the frame's ``lo``-``hi`` band, at ``band_opacity``.
    ``color_below`` is the second colour of a diverging cell series (below 0).
    ``highlight``: the series is ``yours`` or one of the ``other`` ones, for the
    ``highlight`` palette and the legend."""

    name: str | None = None
    color: str | None = None
    color_below: str | None = None
    linestyle: str | None = None
    linewidth: float | None = None
    marker: str | None = None
    marker_size: float | None = None
    every: int | None = None
    visible: bool = True
    order: int = 0
    band: bool = True
    band_opacity: float = 0.2
    highlight: str | None = None

    def __post_init__(self):
        _colour("series color", self.color)
        _colour("series color_below", self.color_below)
        if self.linestyle is not None:
            _one_of("series linestyle", self.linestyle, lines.lineStyles)
        if self.marker is not None:
            _one_of("series marker", self.marker, markers.MarkerStyle.markers)
        for name in ("linewidth", "marker_size", "every"):
            _positive(f"series {name}", getattr(self, name))
        if not 0 <= self.band_opacity <= 1:
            raise ValueError(f"series band_opacity {self.band_opacity!r}: 0 to 1")
        if self.highlight is not None:
            _one_of("series highlight", self.highlight, HIGHLIGHTS)


@dataclass
class Legend:
    """``direct``: BBOB-style labels at the right of the lines, placed by final
    value. ``box``: a legend in the axes at ``loc``; ``outside``: one legend at
    the right of the figure. ``order``: by final value, yours first, by name or
    in drawing order (front first). ``others``: one "n others" entry for the
    series highlighted ``other``. ``shared``: one box legend for all panels,
    below them."""

    mode: str = "direct"
    loc: str = "best"
    columns: int = 1
    order: str = "value"
    others: bool = False
    shared: bool = False

    def __post_init__(self):
        _one_of("legend mode", self.mode, LEGENDS)
        _one_of("legend loc", self.loc, legend.Legend.codes)
        _positive("legend columns", self.columns)
        _one_of("legend order", self.order, ORDERS)


@dataclass
class FigureSpec:
    """A publication figure: what, for which venue, and how it looks.

    ``span`` is one column or the full text width of the venue; ``width`` (in
    ``unit``) overrides it. The height is ``height``, else width times
    ``aspect``. ``background``: white, none (transparent) or a dark slide.
    ``frame``: a box around each plot, the left and bottom spines, or none.
    ``palette`` is a palette name (``figures.PALETTES``; ``highlight`` also
    styles the series highlighted yours and other, ``family`` colours by the
    first ``by`` column and varies the line style within it) or a list of
    colours. ``markers`` is the number of sparse markers per line. ``render``
    writes ``formats[0]`` unless told otherwise; ``dpi`` applies to PNG.
    """

    version: int = VERSION
    panels: list[Panel] = field(default_factory=lambda: [Panel()])
    layout: Layout = field(default_factory=Layout)
    venue: str = "generic"
    span: str = "single"
    unit: str = "cm"
    width: float | None = None
    height: float | None = None
    aspect: float = 0.75
    background: str = "white"
    title: str | None = None
    text: Text = field(default_factory=Text)
    frame: str = "box"
    palette: str | list[str] = "okabe-ito"
    series: dict[str, Series] = field(default_factory=dict)
    legend: Legend = field(default_factory=Legend)
    markers: int = 4
    formats: list[str] = field(default_factory=lambda: ["pdf"])
    dpi: int = 300

    def __post_init__(self):
        if self.version != VERSION:
            raise ValueError(
                f"version {self.version!r}: {VERSION} (from_dict upgrades)"
            )
        if not self.panels:
            raise ValueError("panels: at least one")
        _one_of("venue", self.venue, VENUES)
        _one_of("span", self.span, SPANS)
        _one_of("unit", self.unit, UNITS)
        _one_of("background", self.background, BACKGROUNDS)
        _one_of("frame", self.frame, FRAMES)
        for name in ("width", "height", "aspect", "dpi"):
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
        """The spec ``value`` describes, upgraded from an older version."""
        return _build(cls, upgrade(value), "spec")

    @classmethod
    def from_json(cls, text: str) -> "FigureSpec":
        return cls.from_dict(json.loads(text))


def upgrade(value: dict[str, Any]) -> dict[str, Any]:
    """The dict of a spec of any version as the current version: version 1 (no
    ``version`` key) moves ``kind``, ``data``, ``title``, ``x`` and ``y`` into the
    one panel and ``width_cm`` and ``height_cm`` to ``width`` and ``height``."""
    if not isinstance(value, dict):
        raise TypeError(f"spec: expected an object, got {value!r}")
    version = value.get("version", 1)
    if version == VERSION:
        return value
    if version != 1:
        raise ValueError(f"version {version!r}: 1 or {VERSION}")
    value = dict(value)
    panel = {k: value.pop(k) for k in ("kind", "data", "title", "x", "y") if k in value}
    for old in ("width_cm", "height_cm"):
        if old in value:
            value[old.removesuffix("_cm")] = value.pop(old)
    return {**value, "version": VERSION, "unit": "cm", "panels": [panel]}


# Named looks, as spec fragments (``apply_style``). The key ``panel`` applies to
# every panel. A style without a size keeps the figure's. The fonts ship with
# matplotlib, so a style renders the same everywhere: STIX (a Times design)
# stands in for the design's Source Serif 4, DejaVu Sans for Geist.
STYLES: dict[str, dict[str, Any]] = {
    "paper": {
        "text": {
            "family": "STIXGeneral",
            "title_size": 9,
            "size": 8,
            "tick_size": 7,
            "legend_size": 7,
        },
        "background": "white",
        "frame": "box",
        "legend": {"mode": "direct"},
        "panel": {"x": {"grid": "major"}, "y": {"grid": "major"}},
    },
    "gecco": {  # GECCO: ACM sigconf, one column, the venue's fonts
        "venue": "acm",
        "span": "single",
        "width": None,
        "height": None,
        "text": {"family": None, "size": None, "title_size": None},
    },
    "slides-dark": {
        "unit": "cm",
        "width": 24.0,
        "height": 13.5,
        "background": "dark",
        "palette": "highlight",
        "text": {
            "family": "DejaVu Sans",
            "title_size": 20,
            "size": 16,
            "tick_size": 14,
            "legend_size": 14,
        },
    },
}


def apply_style(spec: FigureSpec, style: str | dict[str, Any]) -> FigureSpec:
    """``spec`` with the style ``style`` (a ``STYLES`` name or a fragment: a
    partial spec dict, whose ``panel`` key applies to every panel) merged in."""
    fragment = dict(STYLES[style] if isinstance(style, str) else style)
    panel = fragment.pop("panel", None)
    merged = _merge(spec.to_dict(), fragment)
    if panel is not None:
        merged["panels"] = [_merge(p, panel) for p in merged["panels"]]
    return FigureSpec.from_dict(merged)


def _merge(base: dict, fragment: dict) -> dict:
    out = dict(base)
    for key, value in fragment.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            value = _merge(out[key], value)
        out[key] = value
    return out


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
    """``value`` as the type ``hint``: dataclasses, and lists and dicts of them,
    are built; anything else is left to the dataclass to check."""
    if dataclasses.is_dataclass(hint):
        return _build(hint, value, path)
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    item = args[-1] if origin in (list, dict) else None
    if not dataclasses.is_dataclass(item):
        return value
    if not isinstance(value, origin):
        kind = "a list" if origin is list else "an object"
        raise TypeError(f"{path}: expected {kind}, got {value!r}")
    if origin is list:
        return [_build(item, v, f"{path}[{i}]") for i, v in enumerate(value)]
    return {k: _build(item, v, f"{path}.{k}") for k, v in value.items()}
