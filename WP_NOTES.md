# WP I12: ECDF, AUC, AOCC and aggregation
Branch: wp/I12-ecdf (stacked on wp/I10-alignment, IOHinspector#32)
Status: in progress

## Done
- step 1/4: `metrics.get_ecdf`: target ECDF (EC-1, EC-3, exact steps or sampled
  at `x_values`), continuum ECDF (`continuous=True`, EC-5/EC-6), aggregation
  over pairs (EC-2, `pairs="common"|"all"`, warns). Also `get_ecdf_auc` and
  `get_rank_table` are written (tests in steps 2 and 3).
- step 2/4: tests of `get_ecdf_auc` (EC-4), AOCC (`continuous=True`, EC-5) and
  the equivalence (target AUC within 1/m of the AOCC); legacy note on `get_aocc`.

## Next step
Step 3: tests of `get_rank_table` (EC-8), including the golden `Aggr_*` ranks.

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

## Open questions / Blocked
- (none yet)
