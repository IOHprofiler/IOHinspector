# WP I09: Convert and verify commands
Branch: wp/I09-convert-verify
Status: ready for review

## Done
- step 1: `iohinspector.convert` and the `iohinspector convert` CLI (`iohinspector.cli`, `[project.scripts]`): format detection over `READERS` plus Nevergrad, custom CSV with `--mapping`, archives via `io/archive.py`, `--tag` and `--tags-from-path`.
- step 2: `iohinspector.verify` and `iohinspector verify`: an independent scanner that writes a JSON report and exits 1 on any mismatch. `docs/convert-verify.md` documents it.
- step 3: `scripts/verify_corpus.py`.
- step 4: mutation tests in `tests/test_verify.py`. Three source mutations (drop a row, change a digit, swap runs) run on each of 9 corpus inputs. Two dataset-side ones change a trace value and a stored hash. Each mutation must trip a per-run check, not only the file sha256.

## Corpus results (zero mismatches)
- In the repository: `tests/golden/fixtures/*` (COCO, old COCO, IOHanalyzer extdata, ioh 0.4.0 min/max (PBO)/constraints/positions), `tests/io/fixtures/ioh040/*` and `ioh0322/*`, `tests/test_data/*`, `tests/test_coco_data`, and the Nevergrad sample. 24 datasets, all OK.
- Downloaded, not committed: the COCO data archive sample from `scripts/coco_archive.py` (24 archives covering bbob 2009 to 2023, noisy, largescale, boxed and incomplete; about 190 MB, 44k runs, 5.6M rows) and Nevergrad `yatinybbob.csv.gz`. 25 datasets, all OK. The largest took 30 s, which went on the 6480 small files.

## Decisions
- `convert` joins `READERS` (detect, then read to a `ReadResult`) and `CONVERTERS` (the CSV converters, which write the dataset themselves). It detects with the `READERS` entries in order, then Nevergrad by its columns. Custom CSV is used only when a mapping is given. A new reader plugs in by its `READERS` line alone.
- Archives: `io.archive.open_tree` (from I07) extracts them. `convert` then lays the tree out under a temporary folder by source path, using hard links, so every reader reads a plain folder. Source paths come out as `<archive>/<member>`. The manifest's `archives` and `sources[].archive` come from the tree. Inputs without archives are read in place. No archive code of its own.
- Tags:
  - `--tag key[:type]=value` adds a descriptor with origin `user`.
  - `--tags-from-path '{key}/{key2:int}_*'` adds descriptors with origin `path`. It matches each run's `source_file`, either the whole path or its leading parts, and every run must match.
  - `algorithm_name` and `algorithm_info` taken from a path get origin `user`, because the format allows only `algorithm` or `user` for them.
  - Reserved keys are refused before reading.
  - Tags are applied after writing, with `update_descriptors(..., origin=None)`. That keyword is new in format/descriptors.py and keeps the records' own origins; its default is unchanged.
  - `updated` is not set for tags added at conversion. The options are recorded in `manifest.extra.convert`.
- `verify` shares no code with the readers or `convert`:
  - It has its own walker (`verify/_sources.py`, using the stdlib zip and tar modules) and its own line scanners keyed by the manifest `kind` (`verify/_scan.py`).
  - Per source run it checks the row count, the first and last evaluation, the best value, the line hash against `source_rows_sha256`, and an order-sensitive polars fingerprint of the (evaluations, value) pairs against the trace.
  - It checks each source's sha256 and size and `data_rows`, that every trace row belongs to a verified source run, and runs `validate_dataset`.
  - Legacy and COCO file families are compared per file through `logged_in`. A kind it does not know is listed as not verified, and its runs then fail as not belonging to a verified source run.
- Memory: the CSV converters stream. `READERS` return an in-memory `ReadResult`, so `convert` needs as much memory as the reader's tables for the input; the writer is bounded. Streaming those readers would be a reader change (I25 consolidates io/).

## Open questions
- From I06: no real IOHexperimenter 0.2 multi-file legacy archive is in the corpus. The `.dat`/`.cdat`/`.tdat`/`.idat` family path is verified only on logs from I06's synthetic writer (`tests/io/legacy_writer.py`). Whether the 0.2 logger omits run headers in `.idat`/`.tdat` is still untested on real data.
- The mutation tests need `tests/io/legacy_writer.py` (imported as `tests.io.legacy_writer`).
- Unlisted files in the input (not in the manifest) are reported in `unlisted_files` but do not fail `verify`. Readers already record the files they skip as manifest warnings.
