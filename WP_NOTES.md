# WP I25: Reader consolidation: one lean io package
Branch: wp/I25-io-consolidation
Status: ready for review

## Done
- step 0: extdata golden fixture converted to format v1 with the legacy reader; legacy reader, docs and tests removed
- steps 1-2, 6: one reader table and a shared base; archives opened in one place; readers stream (see Decisions)
- step 3: functions split below pylint/lizard limits (two class-level suppressions), shared `input_root`;
  duplicate tests merged (ioh/COCO defect tests that repeated test_ioh/test_coco; COCO archive tests moved
  to one convert test of unsafe and broken archives)
- step 4: convert outputs of 53 inputs byte-identical before and after (below)
- step 5: line counts (below)

## Line counts (step 5)
| | before | after |
|---|---:|---:|
| `src/iohinspector/io` | 5,023 (14 modules) | 3,251 (10 modules) |
| of which the legacy reader | 1,234 | 0 |
| `tests/io` | 3,199 | 2,309 |
| `src/iohinspector/convert.py` | 382 | 248 |
| `src/iohinspector/verify/_scan.py` | 246 | 241 |
| `tests/test_convert.py` + `tests/test_verify.py` | 548 | 576 |

Net against main: `src` +1,393 / -3,304 lines, `tests` (without fixtures) +130 / -893, `docs` +11 / -119.

## Next step
Controller: push, open the PR, wait for CI and Codacy. Then delete this file.

## Inventory (step 1)
Before: io/ had 5,023 lines in 14 modules. Duplicated jobs, now one implementation each:
- opening archives: `archive.open_tree` (COCO's own archive reading) and `convert.staged_input`
  (extract, then re-link by source path) -> `archive.staged`, used by `convert` for every reader.
- splitting a text file into runs at header lines, hashing each run's lines: `ioh._split_lines`,
  `_coco_dat._split` -> `base.split_lines` / `DataLines`.
- UTF-8 decoding with ReaderError, first-line sniffing: two copies each -> `base.read_text`, `base.first_line`.
- number tokens: `base.NUMBER`, `_csv._FLOAT/_INT`, `convert._typed/_BOOLS`, `_csv.parse_*`
  -> `base.NUMBER/INTEGER` + `_csv.parse_token` (tags use it too).
- building the tables (runs rows, descriptors, sources, warnings, empty-trace schema, runs input schema):
  `ioh._Builder/_result/_RUNS_INPUT`, `coco._Builder/_result/_RUNS_INPUT`, `_csv.runs_frame/_run_dtypes`
  -> `base.Tables` (runs use `RUNS_SCHEMA`).
- renaming reserved names: `base.Renamer` and `_csv._rename` -> `Renamer.rename`.
- "did you mean": `_csv._unknown`, `custom_csv._suggest` -> `_csv.suggest`.
- CSV source file choice and default dataset id: `convert._csv_file`, `_csv._dataset_id_from` -> `_csv._csv_file`
  (dataset id is required by every reader; `convert` defaults it).
- reader tables: `READERS` + `CONVERTERS` -> `READERS` only; every reader is `read(path, dataset_id, **options) -> ReadResult`.
- COCO detection: `is_coco_info` and `_is_coco` -> one `_is_coco`.
- ioh text columns: second parse of every file with a text column -> extra columns kept as text per file,
  cast to float64 at the end when numeric in every file.

## Decisions
- `tests/golden/fixtures/extdata` is now a format v1 dataset (converted before the reader was removed).
  `generate.R` reads IOHanalyzer's own `inst/extdata` (identical to the old fixture) via `system.file`.
  `tests/golden/test_extdata_dataset.py` (was `tests/io/test_legacy_golden.py`) ties the dataset to the goldens.
- A legacy `.info` source fails with "legacy IOHprofiler format is not supported" (raised by the COCO
  data-file check, so detection and `read_coco` both say it).
- The split-file-family verify test now uses `coco_old` (`.dat` + `.tdat`, two dimensions).
- Archives are opened by `convert` only (`archive.staged`), for every reader. `read_coco` no longer opens
  archives itself, so the COCO-only `archive` descriptor (outermost archive, origin `path`) is gone. Through
  `convert` it never appeared (convert staged archives before the reader), so convert outputs are unchanged.
- Streaming (step 6): `base.Tables` writes each file's trace/positions frame to a temporary Arrow file;
  `ReadResult.lazy_trace` / `lazy_positions` scan them and the writer streams them. `ReadResult.trace` and
  `.positions` collect for in-memory use. The CSV reader reads its file once (was three passes).
- API: `convert_nevergrad` / `convert_custom_csv` (write a dataset) became `read_nevergrad(source,
  dataset_id)` / `read_custom_csv(source, dataset_id, mapping)` (return a `ReadResult`), like `read_ioh`
  and `read_coco`; `iohinspector.convert.convert` writes. A CSV reader given a folder reads its only CSV
  file (was in `convert`). Unreleased API (I08/I09), so no deprecation shims.
- `--tag` values use the format's number rules (`parse_token`, shared with CSV): `1_000` or `infinity`
  are no longer accepted as int/float tags.
- COCO header key `archive` is no longer renamed to `archive_experiment` (the key was reserved for the
  removed descriptor).
- CSV data errors are `ReaderError` (was `FormatError`); one message changed in precedence: a bad `budget`
  token in Nevergrad files is reported as the run field (`budget: '10.5' is not an integer`).
- Memory (the `bbob` part of the COCO archive sample, 15 archives, 2.6 M trace rows, 2.4 M position rows):
  reading peaks at 438 MB (was 1,008 MB); the whole convert still peaks at about 1.55 GB, old and new,
  because `format.writer` dominates (memory_limit 256 MB per table, positions with up to 40 columns).
  The writer is outside this package; follow-up if convert must use less.
- Byte-identical check: a scratch script converts 53 inputs (all non-legacy fixtures, CSV samples,
  archives, tags, and a 24-archive sample of the COCO data archive) before and after.

## Open questions / Blocked
- none
