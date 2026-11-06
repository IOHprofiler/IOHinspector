# WP I04: Storage format v1
Branch: wp/I04-storage-format
Status: ready for review

The format specification is `docs/format.md`. The code is `src/iohinspector/format/`. Checkpoint H2 (the user signs off the format) is still open: the decisions are listed under "Open questions" below and in `docs/format.md`.

## Done
- step 1/4: `docs/format.md`, the format specification with the manifest JSON Schema and the open questions for H2.
- step 2/4: `src/iohinspector/format/` (schemas, run ids, manifest, `write_dataset` with an external merge sort, `scan_dataset`), tests in `tests/test_format/` (ids, manifest, writer round trips and refusals).
- step 3/4: `validate_dataset` in `format/validate.py` (manifest, schemas, runs, referential integrity, copied columns, sort order, row numbering, positions, counts; streamed in batches). `write_dataset` validates before it moves the dataset into place. Tests in `tests/test_format/test_validate.py`, one corruption per check.
- step 4/4: `tests/test_format/test_reader.py` (unknown major version refused, newer minor read with a warning, manifest warnings, predicate pushdown on function and dimension) and `tests/test_format/test_memory.py` (10 million rows, marked `slow`). The spill path now sorts through an arg-sort with sliced gathers, which halved the peak memory.

## Tests
- `pytest -q` (whole suite, the slow test skipped): 319 passed, 1 skipped on Python 3.12 + polars 2.1.0, Python 3.10 + polars 1.44.2, and Python 3.11 + polars 1.33.1 (the minimum the project supports).
- `pytest -q tests/test_format`: 165 passed, 1 skipped on each of them.
- `IOHINSPECTOR_RUN_SLOW=1 pytest -q -s tests/test_format/test_memory.py`: 10,000,000 trace rows over 20,000 runs, written in scrambled order in 100,000-row chunks and then validated. Peak RSS of the writing process (measured with `ru_maxrss` in a child process): **781 MiB** on polars 2.1.0 (13.1 s), **765 MiB** on polars 1.44.2 (12.9 s), against the 2 GB limit. The machine has 32 cores; before the arg-sort change the peak was 1.39 GB.
- Predicate pushdown is tested directly: the test overwrites the data pages of every row group that cannot match `function_id == 3 & dimension == 5`, then checks that the filtered scan still returns the right rows and that a full scan fails. Passes on both polars majors.
- `ruff check` and `ruff format` (line length 88) are clean on the new files; `pyright src/iohinspector/format` reports 0 errors.

## Next step
None. Waiting for review and for checkpoint H2.

## Decisions
- `docs/format.md` is normative. Where it departs from the plan's "Storage format v1" section, it says so in its "Open questions for H2".
- New runtime dependency `jsonschema>=4` in `pyproject.toml` (one added line): it validates manifests against the JSON Schema in the spec, so there is one source of truth instead of a hand-written validator. The CI install picks it up through `pip install -e ".[dev]"`.
- The JSON Schema is embedded in `format/manifest.py` (not a package-data file, so wheels need no setuptools configuration). A test keeps it identical to the block in `docs/format.md`, and another checks the run id test vector in the spec.
- `write_dataset` sorts trace and positions rows with an external merge sort: buffers of about `memory_limit` bytes (default 256 MiB) are arg-sorted and spilled to lz4 Arrow IPC files in the temporary dataset directory, then merged while the Parquet file is written. Input that is already in order lands in one spill file; small inputs never touch the disk.
- Row groups: at most 2^20 rows (and 128 MiB), cut at the first (suite, function_id, dimension) change once the group holds 2^16 rows.
- zstd level 3 and row-group statistics are SHOULD rules in the spec: `validate_dataset` warns when they are missing, because the data is still correct.
- The dataset is written to `.<name>.tmp-*` next to the target and renamed into place; a failed write leaves nothing behind and keeps an existing dataset. `overwrite=True` only replaces a directory that holds a `manifest.json` (or is empty).
- `write_dataset` never trusts the order of a polars join; the validator keeps the file's row order through its join with an explicit row index.
- The slow test is skipped unless `IOHINSPECTOR_RUN_SLOW=1`; the marker and the skip live in `tests/test_format/conftest.py`, so `pyproject.toml` needs no pytest section. It does not touch `tests/golden/` or the Makefile.
- `iohinspector/__init__.py` is unchanged: the package is imported as `iohinspector.format`, because the top-level star imports would clash (`Dataset` already exists in `data.py`).

## Open questions / Blocked
Checkpoint H2. Each question gives the recommendation that this branch implements (details in `docs/format.md`, "Open questions for H2"):

1. `attributes` as JSON text or a struct? Recommendation: JSON text.
2. Copy `function_id` and `dimension` into the trace and positions, so filters skip row groups (the plan lists only `run_id`, `row`, `evaluations`, `raw_y`)? Recommendation: yes.
3. Run id: first 63 bits of SHA-256 over `ioh-run-v1`, dataset id, source path and 0-based run index, newline-separated, stored as int64? Recommendation: yes.
4. Sort order: `suite` first, then function, dimension, algorithm name and info, source file and run index? Recommendation: yes.
5. Signed int32/int64 only, instead of the current unsigned 16-bit ids? Recommendation: yes.
6. Empty string instead of null for an unknown `suite` or `algorithm_info`? Recommendation: yes.
7. `best_y` and `evals_used` as the source reports them, null otherwise, never computed from the trace? Recommendation: yes.
8. Version 1 rejects `n_objectives` other than 1 and any `y0...` column? Recommendation: yes.
9. Positions as wide `x0...x(D-1)` columns, null beyond each run's dimension? Recommendation: yes.
10. `source_rows_sha256` over the whitespace-normalised data lines as text, null for non-text sources? Recommendation: yes.
11. Minor versions only add; newer minor read with a warning, unknown major refused? Recommendation: yes.
12. Readers of sources without function ids (Nevergrad, custom CSV) assign int ids and keep the name in `function_name`? Recommendation: yes.
13. New runtime dependency `jsonschema>=4`? Recommendation: yes.
