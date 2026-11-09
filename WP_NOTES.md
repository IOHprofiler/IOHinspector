# WP I04: Storage format v1 (revision for D17, step 5)
Branch: wp/I04-storage-format (draft PR IOHinspector#18)
Status: in progress

Steps 1 to 4 are done and described in the PR. This revision adds step 5 of I04: descriptors (decision D17).

## Done
- step 5 part 1/3: `docs/format.md` rewritten for descriptors: `descriptors.parquet` section (columns, origins, types, rules, editing, grouping), manifest `descriptors` summary, `counts.descriptors_rows` and `updated` in the JSON Schema, run order without the algorithm, H2 questions revised. The code still has the old schema, so `test_schema_in_spec_equals_shipped_schema` fails until part 2.

## Next step
Step 5 part 2/3: implement descriptors in `src/iohinspector/format/` (schema constants, new `descriptors.py` with encoding, preparation, checks, summary, `descriptor_columns`, `update_descriptors`; writer, reader `run_columns`, validator), remove `attributes`, and add `tests/test_format/test_descriptors.py`.

## Decisions
- (filled in as the revision goes)

## Open questions / Blocked
- See `docs/format.md`, "Open questions for H2".
