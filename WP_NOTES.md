# WP I01: Baseline and CI
Branch: wp/I01-baseline-ci
Status: in progress

## Done
- step 1: pyproject: requires-python >=3.10, `dev` extra (pytest, hypothesis, ruff, pyright, coverage, ioh, pymoo)
- step 2: CI matrix py3.10-3.13 x polars 1.x / 2.x; runs pytest; ruff and pyright informational
- step 3: polars 2 fix in metrics/ecdf.py (group_by().mean() on str columns raised; now null for non-numeric, as in polars 1, same column order). Renamed module-level `test_notebook_runner` in tests/test_tutorials.py to `run_notebook` (pytest collected it as a test and errored on missing fixture `self`).

## Test results (local, py3.12, polars 1.44.2 and 2.1.0)
- Before: polars 1: 154 pass, 1 error (the collection issue above). polars 2: 4 failed (test_plot_ecdf x2, notebook SO_Examples, README), all from the ecdf mean() on str.
- After: 154 pass on both. No xfail needed; no numerical failures.

## Next step
CI green check, then finish.

## Decisions
- ruff (276 findings) and pyright (212 errors) fail on the existing code; kept informational, not fixed (out of scope).

## Open questions / Blocked
- None
