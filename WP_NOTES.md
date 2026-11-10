# WP I05: Reader: IOHexperimenter JSON + .dat
Branch: wp/I05-ioh-reader
Status: ready for review

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
- step 5: `tests/io/test_ioh_hypothesis.py`: logs written like the Analyzer
  (unescaped strings, bare nan/inf, `%.10f` or `%.17g`, ties, f = 0,
  maximisation, `None` values, JSON/.dat run-count mismatches) read back and
  stored exactly

## Done when
- Every defect test in `tests/io/test_ioh_defects.py` passes, on ioh 0.4.0 and
  0.3.22 logs, except `test_runtime_summary_on_f_zero`, which stays
  `xfail(strict=True)`: the assert is in `metrics.utils.get_sequence`, target-grid
  code that I10 replaces under TG-3 (`docs/definitions/targets.md`).
- Rows read equal the data lines of every `.dat` file for the whole IOH corpus
  in the repository (`tests/test_data`, `tests/io/fixtures/ioh040`,
  `tests/io/fixtures/ioh0322`, `tests/golden/fixtures/ioh040_*`), and the
  values equal the source tokens: `tests/io/test_ioh.py::test_rows_read_equal_data_lines`
  and `test_values_equal_source_tokens`. The legacy `.info` and COCO parts of the
  corpus are I06 and I07.

## Next step
None. The controller pushes the branch and opens the pull request.

## Decisions
- Workers commit locally only (CLAUDE.md); the controller pushes and opens the PR.
- Fixtures: `python tests/generate_test_data.py --reader-fixtures` (seeded)
  writes `tests/io/fixtures/ioh040/` with ioh 0.4.0 installed and
  `tests/io/fixtures/ioh0322/` with ioh 0.3.22 (72 KB together). Not under
  `tests/io/data/`, which `.gitignore` (`data/`) hides. The defect tests run on
  both versions. The existing random `tests/test_data` logs are unchanged.
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
- For the controller / I08: `src/iohinspector/io/__init__.py` holds the reader
  table `READERS` with one line for `ioh`; I08 adds its lines there.
  Shared helpers are in `io/base.py` (`ReaderError`, `ReadResult`, `Reader`,
  number parsing, `source_path`, `Renamer`).
- Metric code is not changed, so no D6 review is needed for this PR.
