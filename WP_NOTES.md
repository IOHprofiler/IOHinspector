# WP I25: Reader consolidation: one lean io package
Branch: wp/I25-io-consolidation
Status: in progress

## Done
- step 0: extdata golden fixture converted to format v1 with the legacy reader; legacy reader, docs and tests removed

## Next step
Step 1/2: inventory and consolidate the io package (shared builder, table assembly, line splitting).

## Decisions
- `tests/golden/fixtures/extdata` is now a format v1 dataset (converted before the reader was removed).
  `generate.R` reads IOHanalyzer's own `inst/extdata` (identical to the old fixture) via `system.file`.
  `tests/golden/test_extdata_dataset.py` (was `tests/io/test_legacy_golden.py`) ties the dataset to the goldens.
- A legacy `.info` source fails with "legacy IOHprofiler format is not supported" (raised by the COCO
  data-file check, so detection and `read_coco` both say it).
- The split-file-family verify test now uses `coco_old` (`.dat` + `.tdat`, two dimensions).
- Byte-identical check: a scratch script converts 53 inputs (all non-legacy fixtures, CSV samples,
  archives, tags, and a 24-archive sample of the COCO data archive) before and after.

## Open questions / Blocked
- none
