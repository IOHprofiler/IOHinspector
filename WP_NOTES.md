# WP I04: Storage format v1 (revision for D17, step 5)
Branch: wp/I04-storage-format (draft PR IOHinspector#18)
Status: in progress

Steps 1 to 4 are done and described in the PR. This revision adds step 5 of I04: descriptors (decision D17).

## Done
- step 5 part 1/3: `docs/format.md` rewritten for descriptors: `descriptors.parquet` section (columns, origins, types, rules, editing, grouping), manifest `descriptors` summary, `counts.descriptors_rows` and `updated` in the JSON Schema, run order without the algorithm, H2 questions revised. The code still has the old schema, so `test_schema_in_spec_equals_shipped_schema` fails until part 2.

- step 5 part 2/3: descriptors implemented: `format/descriptors.py` (encode/decode, `descriptor_frame`, preparation and checks, manifest summary, `descriptor_columns`, `update_descriptors`), writer (`descriptors=`), reader (`StoredDataset.descriptors`, `run_columns(by)`), validator; `attributes` removed; run order no longer uses the algorithm. `tests/test_format/test_descriptors.py`. 216 passed, 1 skipped in `tests/test_format` on polars 2.1.0, 1.44.2 and 1.33.1.

## Next step
Step 5 part 3/3: the Codacy findings on #18 (asserts in library code, `_normalise` complexity, other complexity and parameter-count findings, pylint info items), rerun the memory test, finish these notes, then remove WP_NOTES.md in a final commit.

## Decisions
- Descriptors are a long typed table: `type`, exact canonical text in `value`, and typed copies in `value_int` (null for integers outside int64), `value_float`, `value_bool`. Types: string, int, float, bool, json (lists and objects), null.
- `algorithm_name` and `algorithm_info` stay as runs columns, mirrored by required descriptors; the writer makes them from the runs columns, and validation checks agreement.
- Run order is now suite, function, dimension, source file, run index: no descriptor is a sort key, so descriptor edits never touch the trace.
- `update_descriptors` rewrites descriptors.parquet, the manifest and (for the algorithm pair) runs.parquet; edited values get origin `user`.
- `run_columns(by)` turns keys into columns: one type gives that dtype; integers outside int64 or several types give the exact text (or an error with `mixed="error"`).

## Open questions / Blocked
- See `docs/format.md`, "Open questions for H2".
