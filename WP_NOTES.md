# WP I08: Readers: Nevergrad and custom CSV
Branch: wp/I08-csv-readers
Status: in progress

## Done
- step 1: shared streaming CSV converter, Nevergrad reader, download script, tests

## Next step
Custom CSV tests (mapping validation, errors, round trip), docs page, final checks.

## Decisions
- Shared engine `io/_csv.py`: two passes over the file (first: validate, build runs, infer column types; then one pass per table), memory grows with the number of runs.
- Nevergrad: one row = one run with one trace row (evaluations = budget, raw_y = loss). All other columns are descriptors (origin experiment), except loss/budget/dimension/optimizer_name which are stored as raw_y, evaluations, runs columns and algorithm_name. Function ids: sorted names, 1..n, in manifest.extra.function_ids. `_rescaled` suffix as IOHanalyzer.
- source_rows_sha256 for CSV: tokens as parsed by the csv module (quotes removed), joined by one space, per data line.
- Descriptor type is that of the whole CSV column (bool/int/float/string); empty cell = no descriptor.
- No runtime deps added. Reader table: `iohinspector/io/__init__.py` (new at this commit; I05 adds its line there too, expect a trivial merge conflict).

## Open questions / Blocked
- Nevergrad `error` column: kept as a descriptor, no judgement on failed runs.
