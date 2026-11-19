# WP I19: Figures: spec, venue presets, BBOB-style ECDF
Branch: wp/I19-figure-spec
Status: in progress

## Done
- steps 1-2: `FigureSpec` (JSON dataclasses, validation) and venue presets and palettes
- step 3: `render(spec, data) -> bytes`, the BBOB-style ECDF, direct labels

## Next step
Step 4: snapshot tests against `docs/figures/*.png`, written by the gallery
(`python -m tests.test_figures.sample`).

## Decisions
- Package `iohinspector.figures`: `spec.py` (FigureSpec), `venues.py` (presets,
  palettes), `base.py` (Curve, Plot: what a kind hands the renderer), `ecdf.py`
  (the ECDF kind), `labels.py` (direct labels), `render.py`.
- The spec carries what the figure studio (W12, W19 brief round 2b) edits: size
  (`venue`, `span`, `width_cm`, `height_cm`, `aspect`, `dpi`), text (family and
  sizes for title, labels, ticks, legend), axes (label, scale, lim, tick format,
  grid), palette, per-series overrides keyed by the group label (name, colour,
  line style and width, marker, visibility), legend (direct, box, outside, none)
  and output formats. Annotations, panels and styles are left to I20 and W12.
- Venue widths and sizes come from the class files on CTAN (llncs, IEEEtran
  conference, acmart sigconf, elsarticle 5p, article twocolumn for "generic");
  the source lines are in comments in `venues.py`.
- Rendering uses fonts that ship with matplotlib (cmr10, STIXGeneral, DejaVu
  Sans) so a render is the same on every machine; Linux Libertine (ACM) is not
  shipped, so STIX stands in. `Text.family` can name any installed font.
- Palettes: Okabe-Ito (reordered so neighbours stay apart under CVD; checked
  with the dataviz palette validator), Tol bright, grey. Every series also gets
  its own line style and marker.

- `render` runs in `style.context("default")` plus the spec's rc, so the user's
  matplotlib settings do not leak in; PDF, SVG and PNG are byte-identical across
  calls (no dates, fixed SVG hash salt).
- Direct labels: positions are the least-squares placement that keeps their
  order and a 1.2-line gap (isotonic regression, `labels.spread`), set after a
  first layout pass so the axes height is known. Label text is black; the leader
  and the line carry the colour.
- ECDF: `get_ecdf` steps from the first logged evaluation to each group's
  largest budget, the final value marked; x is log10(evaluations / dimension)
  with exponent tick labels by default; sparse markers are spread on the x
  scale with a per-curve phase so curves do not share positions.

## Open questions / Blocked
- none
