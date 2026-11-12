# WP I12: ECDF, AUC, AOCC and aggregation
Branch: wp/I12-ecdf (stacked on wp/I10-alignment, IOHinspector#32)
Status: ready for review

## Done
- step 1/4: `metrics.get_ecdf`: target ECDF (EC-1, EC-3, exact steps or sampled
  at `x_values`), continuum ECDF (`continuous=True`, EC-5/EC-6), aggregation
  over pairs (EC-2, `pairs="common"|"all"`, warns). Also `get_ecdf_auc` and
  `get_rank_table` are written (tests in steps 2 and 3).
- step 2/4: tests of `get_ecdf_auc` (EC-4), AOCC (`continuous=True`, EC-5) and
  the equivalence (target AUC within 1/m of the AOCC); legacy note on `get_aocc`.
- step 3/4: `get_rank_table` (EC-8) tests: ties, inf, null/NaN last, direction,
  and IOHanalyzer's `Aggr_*` ranks on its own values (all five files match).
- step 4/4: golden tests: `ECDF_by_RT`, `ECDF_default_targets` (pooled, equal
  weight per pair), `AUC_by_RT`, `AUC_by_RT_raw` on ioh040_min and ioh040_max
  match exactly with strict-equivalent targets (nextafter); the inclusive ECDF
  is never lower and is higher somewhere. Worked example 0.70 / 0.733 / 0.75.
  Docs: implementation section in `docs/definitions/ecdf.md`, CHANGELOG,
  FUNCTION_REFERENCE.

## Next step
None: ready for the controller (push, PR, dvermetten review).

## Decisions
- Venv: `uv venv .venv --python 3.14 && uv pip install -e ".[dev]"` (polars 2.1).
- Commit locally only; the controller pushes and opens the pull request.
- One module, `metrics/ecdf.py`; no change to `align.py` or `targets.py` (I11
  works on the same base). The target ECDF calls `align_data` once per pair
  (per-pair targets), not per run.
- The ECDF is a sum of weighted increments: a hit adds 1/(|P| n m) at its first
  hitting time; the continuum ECDF adds the change of q/(|P| n) at each
  improvement. AUC and AOCC integrate the same step function, so AOCC is
  `get_ecdf_auc(..., continuous=True)`.
- Continuum bounds come from the targets frame (min and max target per pair):
  default the observed range per pair (TG-1/TG-2); BBOB bounds by passing
  `get_ecdf_targets(data, "bbob")`. No detection of BBOB data from the suite name
  (that is still open for H1, see eaf.md).
- `get_data_ecdf` and `get_aocc` stay unchanged (legacy, docstring says so); no
  DeprecationWarning yet, because `plot_ecdf` still uses `get_data_ecdf` until
  the figures move (I19/I20).
- Joins on the trace keep the left row order (`maintain_order="left"`), because
  best-so-far (here and in `align_data`) is a cumulative min in row order.
- `get_rank_table` ranks a given summary table; it does not compute ERT or the
  fixed-budget mean (I11's code, not merged). The website joins I11's ERT at
  `get_target_dt`'s targets with it.

## Open questions / Blocked
- Q-EC-3 (added to ecdf.md): IOHanalyzer's fixed-budget ECDF (`ECDF_by_FV`,
  `AUC_by_FV`) is not defined in `docs/definitions/`, so it is not implemented.
- Q-EC-1 stays open: the data behind 0.67 / 0.90 is unknown; the worked example
  of ecdf.md is the golden case.
- BBOB recognition for the default continuum/AOCC bounds (EA-3) is still open
  for H1; for now the caller passes `get_ecdf_targets(data, "bbob")`.
