# WP I05: Reader: IOHexperimenter JSON + .dat
Branch: wp/I05-ioh-reader
Status: in progress

## Done
- step 1: defect tests and ioh 0.4.0 reader fixtures

## Next step
Step 2: write `src/iohinspector/io/ioh.py` (and `io/__init__.py` with the reader
table) so that the reader-level defect tests in `tests/io/test_ioh_defects.py`
pass; remove their `xfail` markers.

## Decisions
- Workers commit locally only (CLAUDE.md); the controller pushes and opens the PR.
- Fixtures: `python tests/generate_test_data.py --reader-fixtures` (ioh 0.4.0,
  seeded) writes `tests/io/data/ioh040/` (36 KB). The existing random
  `tests/test_data` logs are unchanged.
- Defect tests start as `xfail(strict=True)`; each fix removes its marker.
- "An assert fails when f = 0": the assert is `metrics.utils.get_sequence`
  (`assert min > 0` under log scale), reached from `aggregate_running_time`. That
  is target-grid code, decided by TG-3 in `docs/definitions/targets.md` and owned
  by I10. This WP makes the reader keep f = 0 rows exactly; the metric-level
  test stays `xfail(strict=True)` with a pointer to I10.

## Open questions / Blocked
- none
