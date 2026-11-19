"""Publication figures (D11): a ``FigureSpec`` and ``render(spec, data) -> bytes``.

The website and Python users get the same figure from the same spec. Venue
presets set the width and fonts (``VENUES``); ``PALETTES`` are colour-blind safe
and every series also has its own line style and marker. For LaTeX, ``render``
writes ``fmt="pgf"`` and ``pgfplots`` a pgfplots picture with its CSV, both set
in the document's own font.
"""

from .render import figure_size, render, series_styles
from .spec import Axis, Data, FigureSpec, Legend, Series, Text
from .tex import pgfplots
from .venues import PALETTES, VENUES, Venue

__all__ = [
    "PALETTES",
    "VENUES",
    "Axis",
    "Data",
    "FigureSpec",
    "Legend",
    "Series",
    "Text",
    "Venue",
    "figure_size",
    "pgfplots",
    "render",
    "series_styles",
]
