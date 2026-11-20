# WP I35: Figures: the Studio element model
Branch: wp/I35-figure-elements
Status: in progress

## Done
- (nothing committed yet)

## Next step
Step 1: spec v2 (`version`, `panels`, units, background, frame, series and legend fields, annotations, styles) with the v1 upgrade in `FigureSpec.from_dict`; render and tex read `spec.panels`.

## Plan
1. Spec v2 and the v1 upgrade; render and tex on panels.
2. Frames: each kind is `frame(panel, runs)` and `plot(panel, frame)`; `render(spec, data, frames=...)`, `figures.frames(spec, data)`.
3. Rendering of the new fields: panels grid, annotations, series fields, ticks, frame, background, legend order / others / shared, palettes.
4. Element ids in the SVG and `elements(spec, data)`.
5. `to_python(spec, ...)`.
6. Gallery, CHANGELOG, polars 1 run; remove this file.

## Decisions
- ...

## Open questions / Blocked
- none
