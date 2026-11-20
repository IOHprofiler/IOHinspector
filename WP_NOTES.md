# WP I35: Figures: the Studio element model
Branch: wp/I35-figure-elements
Status: ready for review

## Done
- step 1: spec v2 + v1 upgrade, kinds as frame/plot, ids on every element, `elements()` (commit 39dc710)
- step 2: tests of the model (test_model.py), shared-axis labels, styles on shipped fonts (b3731b2)
- step 3: `to_python` (figures/python.py), test_python.py runs the code (50d06ac)
- step 4: gallery `ecdf-panels-highlight`, `ecdf-slides-dark`; CHANGELOG. Full suite polars 2.1.0: 1305 passed, 9 skipped; figures on polars 1.44.2: 118 passed, 6 skipped (TeX).

- step 5: `add_panels` (a panel from another spec); this file removed

## Next step
None: review and merge.

## Plan
1. Spec v2 and the v1 upgrade; render and tex on panels.
2. Frames: each kind is `frame(panel, runs)` and `plot(panel, frame)`; `render(spec, data, frames=...)`, `figures.frames(spec, data)`.
3. Rendering of the new fields: panels grid, annotations, series fields, ticks, frame, background, legend order / others / shared, palettes.
4. Element ids in the SVG and `elements(spec, data)`.
5. `to_python(spec, ...)`.
6. Gallery, CHANGELOG, polars 1 run; remove this file.

## Decisions
- Panel = content (kind, data, title, axes, annotations); the figure = look (size, text, frame, palette, series, legend). Series overrides are figure-wide, so a series looks the same in every panel.
- ECDF frame = `get_ecdf` output (`by` columns or `label`, `evaluations`, `ecdf`, optional `lo`/`hi`); `ecdf_frame` adds one row per group at the largest budget. Raw rendering = plot(frame(runs)), so both paths give the same bytes.
- Ids: `figure`, `title`, `legend`, `p<i>`, `p<i>-plot|frame|grid|title|xaxis|xlabel|xticks|...`, series `p<i>-s-<slug>` with `-line|-band|-end|-markers|-label`, annotations `p<i>-a<j>`. Several artists of one element get `<id>-<k>`. No hit area in the SVG: Studio widens the line path itself.
- Styles use fonts that ship with matplotlib (STIX for Source Serif 4, DejaVu Sans for Geist), so renders stay identical everywhere and matplotlib does not log missing fonts.
- `to_python` emits the frame code per kind (`Kind.code`), rcParams, `STYLE`, a `draw` helper and the layout-derived numbers (limits, direct-label heights). Annotations go through the same `_annotate` with a call recorder, so they match `render`. Sparse markers become `markevery=1/markers` (matplotlib's display-space spacing), not the exact BBOB positions.
- pgfplots stays one panel, without annotations and bands (raises for more panels).

## Open questions / Blocked
- none
