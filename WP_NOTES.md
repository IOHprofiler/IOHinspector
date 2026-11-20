# WP I20: Figures: every Workbench kind on the element model
Branch: wp/I20-figure-kinds
Status: in progress

## Done
- step 1: the kinds (convergence, fixed budget, ERT, fixed-budget ECDF, target x budget, pairwise tests, win fractions, box, violin, histogram, EAF, EAF difference, Shapley, cumulative difference, four parameter figures), marks, atlas, to_python, gallery and tests.

## Next step
Run the full suite on polars 2, the figures tests on polars 1; CHANGELOG; remove this file.

## Decisions
- Plot gained `mark` (line, points, bars, box, violin, cells), `annotations`, `legend`, `fmt`; Curve gained `mid`, `z`. One drawing path: `marks.py` for the new marks, `render._draw` for lines.
- Kind code for to_python: a comment with the frame function call, then the curves written as numbers (`kind.literal`).
- `plots/` left as it is: the functions return (ax, frame), take `ax=` and seaborn options, so thin wrappers would break their API.
