# WP I25: Reader consolidation: one lean io package
Branch: wp/I25-io-consolidation
Status: in progress

## Done
- step 0: extdata golden fixture converted to format v1 with the legacy reader; legacy reader, docs and tests removed

- step 1-2 (first pass): one reader table and a shared base; archives and streaming (see Decisions)

- step 3 (first pass): functions split below pylint/lizard limits (no new suppressions beyond two
  class-level ones), shared `input_root`; duplicate tests merged (ioh/COCO defect tests that repeated
  test_ioh/test_coco, COCO archive tests moved to one convert test of unsafe/broken archives)

## Next step
Final snapshot comparison, line counts, notes cleanup; then hand back to the controller.

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
- CSV data errors are `ReaderError` (was `FormatError`); one message changed in precedence: a bad `budget`
  token in Nevergrad files is reported as the run field (`budget: '10.5' is not an integer`).
- Memory (bbob sample of the COCO archive, 13 archives, 2.6 M trace rows, 2.4 M position rows):
  reading peaks at 438 MB (was 1,008 MB); the whole convert still peaks at about 1.55 GB, old and new,
  because `format.writer` dominates (memory_limit 256 MB per table, positions with up to 40 columns).
  The writer is outside this package; follow-up if convert must use less.
- Byte-identical check: a scratch script converts 53 inputs (all non-legacy fixtures, CSV samples,
  archives, tags, and a 24-archive sample of the COCO data archive) before and after.

## Open questions / Blocked
- none
