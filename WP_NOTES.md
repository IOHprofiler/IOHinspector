# WP I18: DSC, reimplemented from the papers
Branch: wp/I18-dsc
Status: ready for review

## Done
- step 1: `metrics/dsc.py` (sample, ranking with pDSC, omnibus, post-hoc) and
  `tests/test_metrics/test_dsc.py`; the published DSCTool example is reproduced
  (22 rankings exactly, omnibus and post-hoc within 1e-9).

- step 2: `docs/definitions/dsc.md` "How iohinspector implements it" (what was
  built and the details the page left open), CHANGELOG, FUNCTION_REFERENCE.

## Next step
Controller: push, open the pull request, request dvermetten's review (D6).

## Tests
- `uv run pytest -q`: 980 passed, 3 skipped (Python 3.14, scipy 1.18.1).
- `tests/test_metrics/test_dsc.py` also passes on Python 3.10 with scipy 1.15.3
  (the `midrank=True` fallback) and on polars 2.1.0 (Docker).

## Decisions
- scipy is now a direct dependency (it was already installed through
  robustranking/statsmodels): ks_2samp, anderson_ksamp, wilcoxon, chi2, f, norm.
- AD statistic: `anderson_ksamp(variant="midrank")`, falling back to
  `midrank=True` for scipy < 1.17 (Python 3.10 gets scipy 1.15).
- Limiting AD distribution: Marsaglia & Marsaglia (2004) `ADinf`, within 2e-6.
- Monte Carlo pDSC: the upper median of the I p-values, so that a pair differs
  exactly when more than half of the iterations say so (also for even I).
- Omnibus message strings: the service's "The null hypothesis is rejected." and,
  by analogy, "... is not rejected.".
- n < 2 problems: `valid_methods` is empty and `get_dsc_omnibus` raises with the
  service's message; the ranking is still returned.
- Holm and Hochberg are a small local `p_adjust`; I15 (PR #43) adds `holm`.
  When both are merged, one of them should go (lean codebase).

## Open questions / Blocked
- none
