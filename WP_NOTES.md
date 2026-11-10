# WP I05: Reader: IOHexperimenter JSON + .dat
Branch: wp/I05-ioh-reader
Status: in progress

## Done
- step 1: defect tests and ioh 0.4.0 reader fixtures
- steps 2 and 3: `iohinspector.io` (reader table `READERS`, `detect_format`,
  `read`, `ReadResult`, `ReaderError`) and `iohinspector.io.ioh` (`read_ioh`,
  `parse_dat`, `load_ioh_json`, `infer_triggers`), with the quirks recorded in the
  manifest; tests in `tests/io/test_ioh.py`
- step 4: `DataManager` reads IOHexperimenter data through the new reader:
  `Dataset.from_json` uses `load_ioh_json` (quirk repairs; run attributes in the
  overview as Float64 columns), `Scenario.load` uses `parse_dat` (split on headers,
  no dedup, file:line errors) and `monotone_per_run` (best-so-far per run, final
  row kept). COCO and the `.parquet` sidecar path are unchanged (I07).

## Next step
Step 5: hypothesis tests in `tests/io/test_ioh_hypothesis.py`: generate logs with
a writer that mimics IOHexperimenter (ties, f = 0, maximisation, inf/nan, runs
without improvement, missing final rows), write them, read them back, compare.

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
- Existing tests changed numbers because they encoded the final-row defect:
  `tests/test_data.py` monotonic loads 69 -> 84 and 125 -> 150 rows (one final row
  per run: 15 and 25 runs), ECDF frame (66, 14) -> (68, 14).
- `try_eval` uses `ast.literal_eval` instead of `eval` on experiment attributes.
- `tests/io/conftest.py` restores the legacy process-global `data_id` counter after
  each test, because older tests select by `data_id` (General defect, not I05).
- `trace` is held in memory as one DataFrame; streaming is I09's convert command.

## Open questions / Blocked
- none
