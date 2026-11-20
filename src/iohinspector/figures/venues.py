"""Venue presets and palettes for publication figures.

Every width and size is taken from the venue's LaTeX class with its default
options; the comment on each preset names the class and the lines that set
them (checked against the CTAN versions of November 2026). Widths are in TeX
points (72.27 per inch).

The figure is drawn with a font that ships with matplotlib, so that a render is
the same on every machine: ``cmr10`` for Computer Modern and ``STIXGeneral``
for Times (STIX is a Times design). ACM's Linux Libertine does not ship with
matplotlib; STIX stands in for it. ``Text.family`` can name any installed font
instead, and the PGF export (I21) uses the document's own fonts.
"""

from dataclasses import dataclass

TEX_PT_PER_INCH = 72.27


@dataclass(frozen=True, kw_only=True)
class Venue:
    """A venue's text and column widths (TeX points), its text font and the
    matplotlib font that stands in for it, its body (``\\normalsize``) and figure
    caption sizes in points."""

    name: str
    document_class: str
    column_pt: float
    text_pt: float
    font: str
    mpl_font: str
    mathtext: str
    size: float
    caption_size: float

    def width_in(self, span: str) -> float:
        """The width of a figure spanning one column (``single``) or the full
        text width (``double``), in inches."""
        width = self.column_pt if span == "single" else self.text_pt
        return width / TEX_PT_PER_INCH


def _cm(value: float) -> float:
    return value / 2.54 * TEX_PT_PER_INCH


VENUES = {
    # llncs.cls: \textwidth 12.2cm (one column, so both spans are equal); it
    # loads article at its default 10pt; captions are \small, 9pt (@ixpt).
    "lncs": Venue(
        name="Springer LNCS",
        document_class="llncs",
        column_pt=_cm(12.2),
        text_pt=_cm(12.2),
        font="Computer Modern",
        mpl_font="cmr10",
        mathtext="cm",
        size=10,
        caption_size=9,
    ),
    # IEEEtran.cls, conference option: \textwidth 43pc = 516pt, \columnsep 1pc,
    # so \columnwidth 252pt; default 10pt; Times (\rmdefault ptm); figure
    # captions \footnotesize, 8pt at 10pt.
    "ieee": Venue(
        name="IEEE conference",
        document_class="IEEEtran[conference]",
        column_pt=252,
        text_pt=516,
        font="Times",
        mpl_font="STIXGeneral",
        mathtext="stix",
        size=10,
        caption_size=8,
    ),
    # acmart.cls, sigconf: geometry paperwidth 8.5in with inner and outer 54pt,
    # so \textwidth 506.295pt; columnsep 2pc, so \columnwidth 241.1475pt;
    # default 9pt; Linux Libertine; captions at the body size.
    "acm": Venue(
        name="ACM sigconf",
        document_class="acmart[sigconf]",
        column_pt=241.1475,
        text_pt=506.295,
        font="Linux Libertine",
        mpl_font="STIXGeneral",
        mathtext="stix",
        size=9,
        caption_size=9,
    ),
    # elsarticle.cls, 5p (the final two-column layout): textwidth 522pt,
    # columnsep 18pt, so \columnwidth 252pt; default 10pt Computer Modern;
    # captions \footnotesize, 8pt.
    "elsevier": Venue(
        name="Elsevier (elsarticle 5p)",
        document_class="elsarticle[5p]",
        column_pt=252,
        text_pt=522,
        font="Computer Modern",
        mpl_font="cmr10",
        mathtext="cm",
        size=10,
        caption_size=8,
    ),
    # article.cls (size10.clo), twocolumn on letter paper: \textwidth is
    # \paperwidth - 2in = 469.755pt rounded down to 469pt, \columnsep 10pt, so
    # \columnwidth 229.5pt; captions at the body size, 10pt. Not tied to a
    # venue's font, so matplotlib's default sans-serif.
    "generic": Venue(
        name="Generic (article)",
        document_class="article[twocolumn]",
        column_pt=229.5,
        text_pt=469,
        font="DejaVu Sans",
        mpl_font="DejaVu Sans",
        mathtext="dejavusans",
        size=10,
        caption_size=10,
    ),
}

# Categorical palettes. Every series also gets its own line style and marker,
# so a figure stays readable in greyscale and for colour-blind readers.
PALETTES = {
    # Okabe and Ito (2008), ordered so neighbours stay apart under colour-vision
    # deficiency; yellow is left out (too light on white).
    "okabe-ito": [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#E69F00",
        "#CC79A7",
        "#56B4E9",
        "#000000",
    ],
    # Paul Tol's "bright" scheme, in his order.
    "tol-bright": [
        "#4477AA",
        "#EE6677",
        "#228833",
        "#CCBB44",
        "#66CCEE",
        "#AA3377",
        "#BBBBBB",
    ],
    # For greyscale print: the line styles and markers tell the series apart.
    "grey": ["#000000", "#555555", "#888888", "#AAAAAA"],
    # The website's highlighting on a white page: the data colours of its
    # Graphite light scheme (CVD-checked in the W19 design, section 10); the
    # series highlighted yours and other take HIGHLIGHT's colours.
    "highlight": ["#1844b9", "#c63170", "#7a4306", "#7868a9", "#88206b", "#1a6fe1"],
}
# Colour by family (the first ``by`` column), line style within it.
PALETTES["family"] = PALETTES["okabe-ito"]
# On a dark slide: Graphite dark's data colours, and light greys and white for
# the palettes with black; the others stay.
DARK_PALETTES = {
    "highlight": ["#1f98ed", "#5aedd4", "#d664a3", "#d0dd16", "#adb0f6", "#7cbd89"],
    "okabe-ito": [*PALETTES["okabe-ito"][:-1], "#FFFFFF"],
    "grey": ["#FFFFFF", "#C8C8C8", "#969696", "#707070"],
}
DARK_PALETTES["family"] = DARK_PALETTES["okabe-ito"]
# The colours of yours and the others in the highlight palette, on white and on
# a dark slide (Graphite light and dark).
HIGHLIGHT = {
    "white": {"yours": "#c05800", "other": "#a3a9b1"},
    "dark": {"yours": "#f6943f", "other": "#5c626b"},
}
LINESTYLES = ["-", "--", ":", "-."]
MARKERS = ["o", "s", "^", "D", "v", "p", "X", "*"]
