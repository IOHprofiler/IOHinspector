# WP I04: Storage format v1
Branch: wp/I04-storage-format
Status: in progress

## Done
- step 1/4: `docs/format.md`, the format specification with the manifest JSON Schema and the open questions for H2.

## Next step
Step 2/4: implement `src/iohinspector/format/` (schemas, run ids, manifest, `write_dataset`, `scan_dataset`) with round-trip tests in `tests/test_format/`.

## Decisions
- The spec in `docs/format.md` is normative; the decisions that need the user's sign-off are its "Open questions for H2" section (copied below when the work is done).

## Open questions / Blocked
- See `docs/format.md`, "Open questions for H2".
