# WP I04: Storage format v1
Branch: wp/I04-storage-format
Status: in progress

## Done
- step 1/4: `docs/format.md`, the format specification with the manifest JSON Schema and the open questions for H2.
- step 2/4: `src/iohinspector/format/` (schemas, run ids, manifest, `write_dataset` with an external merge sort, `scan_dataset`), tests in `tests/test_format/` (ids, manifest, writer round trips and refusals).
- step 3/4: `validate_dataset` in `format/validate.py` (manifest, schemas, runs, referential integrity, copied columns, sort order, row numbering, positions, counts; streamed in batches). `write_dataset` now validates before it moves the dataset into place. Tests in `tests/test_format/test_validate.py`, one corruption per check.

## Next step
Step 4/4: `tests/test_format/test_reader.py` (unknown major refused by `scan_dataset`, manifest warnings, predicate pushdown on function and dimension), the 10-million-row memory test marked `slow`, run it once and record the peak RAM here; update the API table in `docs/format.md`; copy the H2 questions below and set Status to "ready for review".

## Decisions
- The spec in `docs/format.md` is normative; the decisions that need the user's sign-off are its "Open questions for H2" section (copied below when the work is done).
- New runtime dependency `jsonschema>=4` in `pyproject.toml` (one line): it validates manifests against the JSON Schema in the spec, so there is one source of truth instead of a hand-written validator.
- The JSON Schema is embedded in `format/manifest.py` (not a package-data file, so wheels need no setuptools config); a test keeps it identical to the block in `docs/format.md`.
- `write_dataset` sorts trace and positions rows with an external merge sort: buffers of about `memory_limit` bytes (default 256 MiB) are sorted and spilled to lz4 Arrow IPC files in the temporary dataset directory, and merged while the Parquet file is written. Input that is already in order lands in one spill file.
- zstd and row-group statistics are SHOULD rules in the spec: `validate_dataset` only warns when they are missing, because the data is still correct.
- Row groups: at most 2^20 rows (and 128 MiB), cut at the first (suite, function_id, dimension) change once the group holds 2^16 rows.
- The dataset is written to `.<name>.tmp-*` next to the target and renamed into place; `overwrite=True` only replaces a directory that holds a `manifest.json` (or is empty).

## Open questions / Blocked
- See `docs/format.md`, "Open questions for H2".
