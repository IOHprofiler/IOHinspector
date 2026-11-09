# WP I04: Storage format v1 (revision for D17, step 5)
Branch: wp/I04-storage-format (draft PR IOHinspector#18)
Status: ready for review

Steps 1 to 4 are done and described in the PR. This revision adds step 5 of I04 (descriptors, decision D17) and fixes the Codacy findings on #18.

## Done
- step 5 part 1/3: `docs/format.md` rewritten for descriptors: `descriptors.parquet` section (columns, origins, types, rules, editing, grouping), manifest `descriptors` summary, `counts.descriptors_rows` and `updated` in the JSON Schema, run order without the algorithm, H2 questions revised.
- step 5 part 2/3: descriptors implemented: `format/descriptors.py` (`encode_descriptor`, `decode_descriptor`, `descriptor_frame`, checks, manifest summary, `descriptor_columns`, `update_descriptors`), writer (`descriptors=`), reader (`StoredDataset.descriptors`, `descriptor_keys`, `run_columns(by)`), validator. The `attributes` column is gone. Tests in `tests/test_format/test_descriptors.py`.
- step 5 part 3/3: Codacy findings. No `assert` left in library code (explicit checks, or restructured so none is needed). `_normalise` (CCN 30) split into `_check_chunk_columns`, `_conformed`, `_with_runs`; `check_runs`, `stream`, `runs_problems`, `validate_manifest`, `descriptor_problems`, `summary_problems`, `update_descriptors`, `write_dataset` and `_merge` split; largest CCN now 14. `new_manifest` takes optional fields as keyword arguments (5 parameters instead of 10, no longer shadows the `warnings` module); `_TableSpec` is a dataclass; `positions` and `descriptors` of `write_dataset` are keyword-only. pylint: private `_reject_constant` used across modules made public, imports moved to the top, lines over 100 characters wrapped (the schema in `docs/format.md` rewrapped to match).

## Tests
- `pytest -q`: 370 passed, 1 skipped on Python 3.12 + polars 2.1.0, Python 3.10 + polars 1.44.2, and Python 3.11 + polars 1.33.1.
- `IOHINSPECTOR_RUN_SLOW=1 pytest -q -s tests/test_format/test_memory.py`: 10,000,000 trace rows, peak RSS 789 MiB (polars 2.1.0, 14.4 s) and 771 MiB (polars 1.44.2, 13.0 s).
- `ruff check`, `ruff format`, `pyright src/iohinspector/format`: clean. Local `bandit -r src/iohinspector/format`: no findings.

## Next step
None. Waiting for review and for checkpoint H2.

## Decisions
- Descriptors are a long typed table: `type`, exact canonical text in `value`, and typed copies in `value_int` (null for integers outside int64), `value_float`, `value_bool`. Types: string, int, float, bool, json (lists and objects), null. Origins: algorithm, experiment, run, path, user.
- `(run_id, key)` is unique; keys may not reuse runs or fixed trace column names, except the mirrored pair. Rows are sorted by key, then run id.
- `algorithm_name` and `algorithm_info` stay as runs columns, mirrored by required descriptors; the writer makes them from the runs columns, and validation checks agreement.
- Run order is now suite, function, dimension, source file, run index: no descriptor is a sort key, so descriptor edits never touch the trace. Run ids were already independent of descriptors.
- `update_descriptors` rewrites descriptors.parquet, the manifest (`descriptors`, `counts.descriptors_rows`, `updated`) and, for the algorithm pair, runs.parquet; edited values get origin `user`; nothing is written if a check fails.
- `run_columns(by)` turns keys into columns: one type gives that dtype; integers outside int64 or several types give the exact text (or an error with `mixed="error"`).
- No `# nosec` comments: the repository does not use them.

## Open questions / Blocked
- H2: see `docs/format.md`, "Open questions for H2" (15 items). New with D17: 1 (descriptor design, replacing JSON-or-struct), 2 (algorithm columns kept as validated copies), 5 (reserved keys, one value per key). Changed: 4 (sort order without the algorithm), 6 (run id reviewed, unchanged), 7 (big descriptor integers as text), 8 (`algorithm_info` stays a string).
- Codacy will probably still flag: `assert` in tests (pytest style), `subprocess` in `test_memory.py` and `random` in `synthetic.py` (test-only, left as asked), pylint too-many-instance-attributes on three stateful classes, and too-many-arguments on `write_dataset` (its tables plus four options; grouping them would make calls less clear).
