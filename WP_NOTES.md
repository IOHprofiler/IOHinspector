# WP I05: Reader: IOHexperimenter JSON + .dat
Branch: wp/I05-ioh-reader
Status: in progress

## Done
- step 1: defect tests and ioh 0.4.0 reader fixtures
- steps 2 and 3: `iohinspector.io` (reader table `READERS`, `detect_format`,
  `read`, `ReadResult`, `ReaderError`) and `iohinspector.io.ioh` (`read_ioh`,
  `parse_dat`, `load_ioh_json`, `infer_triggers`), with the quirks recorded in the
  manifest; tests in `tests/io/test_ioh.py`

## Next step
Step 4: back `DataManager` (``data.py`` ``Dataset.from_json`` and
``Scenario.load`` for IOH sources) with `load_ioh_json` and `parse_dat`; remove the
manager-level `xfail` markers in `tests/io/test_ioh_defects.py`.

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

- Descriptors per run: `algorithm_name`, `algorithm_info` (origin algorithm);
  experiment attributes as strings (origin experiment, the source type);
  run attributes as floats (origin run: IOHexperimenter run attributes are
  doubles, also when written like integers); `best_evals` (int) and `best_x`
  (the JSON list as text) from `best`.
- Clashing names are renamed `<name>_<origin>` (descriptors) and `<name>_logged`
  (trace), recorded in `renamed_columns`. Example: experiment attribute `budget`
  becomes `budget_experiment`, a watched `x0` becomes `x0_logged`.
- A missing `.dat` is listed in `sources` with null sha256 and bytes (format.md
  allows null for logical sources), so its runs keep their `source_file`.
- The FlatFile separator bug cannot occur in Analyzer output (the Analyzer passes
  no meta data columns, in 0.3.22 and 0.4.0); the reader refuses the shape with a
  message naming the bug instead of guessing.
- Triggers are not in the JSON; `all` / `improvement` are inferred from the data
  and marked `params.inferred`.
- `trace` is held in memory as one DataFrame; streaming is I09's convert command.

## Open questions / Blocked
- none
