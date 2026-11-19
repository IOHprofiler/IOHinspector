# WP I21: Figures: PGF and pgfplots export
Branch: wp/I21-pgf-export
Status: in progress

## Done
(nothing yet)

## Next step
1. render(fmt="pgf") via matplotlib's PGF backend (needs TeX at render time); 2. figures/pgfplots.py; 3. optional TeX Live workflow.

## Decisions
- Pinned image: texlive/texlive:TL2024-historic (python3 inside; has pgfplots).
- pgfplots export reuses the matplotlib layout (axes size, limits, label positions) from `_figure`.

## Open questions / Blocked
- I19's open questions (generic caption size, STIX for Libertine) not decided here.
