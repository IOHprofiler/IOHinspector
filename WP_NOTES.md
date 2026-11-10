# WP I09: Convert and verify commands
Branch: wp/I09-convert-verify
Status: in progress

## Done
- step 1: `iohinspector.convert` + `iohinspector convert` CLI (`iohinspector.cli`, `[project.scripts]`), tags, archives via `io/archive.py`

- step 2: `iohinspector.verify` + `iohinspector verify`, docs/convert-verify.md, tests/test_verify.py
- step 3: scripts/verify_corpus.py (tested in tests/test_verify.py)
- step 4: mutation tests in tests/test_verify.py (drop row, change digit, swap runs x 9 corpus inputs; changed trace value; changed stored hash)

## Next step
Run the downloaded corpus (COCO archive sample via scripts/coco_archive.py, Nevergrad yatinybbob) through scripts/verify_corpus.py; record results; final notes.

## Decisions
- `convert` joins `READERS` (detect/read -> ReadResult) and `CONVERTERS` (CSV, write themselves). Detection: READERS in order, then Nevergrad by its columns. Custom CSV only with `mapping`. New readers plug in by their READERS line alone.
- Archives: `io.archive.open_tree` (I07) extracts; convert lays the tree out under a temp folder by source path (hard links) so every reader reads a plain folder and source paths come out as `<archive>/<member>`. The manifest gets `archives` and `sources[].archive` from the tree. Inputs without archives are read in place.
- Tags: `--tag key[:type]=value` (origin `user`), `--tags-from-path '{key}/{key2:int}_*'` (origin `path`), matched against each run's `source_file` (whole path or leading parts). Every run must match. `algorithm_name`/`algorithm_info` from a path get origin `user` (the format allows only algorithm/user for them). Applied after writing with `update_descriptors(..., origin=None)` (new keyword, keeps the records' origins); `updated` is not set for convert-time tags. Options in `manifest.extra.convert`.
- `verify` shares no code with the readers or convert: own walker (`verify/_sources.py`, stdlib zip/tar), own line scanners per manifest `kind` (`verify/_scan.py`). Per source run: rows, first/last evaluation, best value, line hash vs `source_rows_sha256`, and an order-sensitive fingerprint of (evaluations, value) pairs vs the trace. Merged file families (legacy, COCO) are compared via `logged_in`.
- Memory: CSV converters stream; READERS return in-memory `ReadResult`s, so convert's memory is that of the reader (the writer itself is bounded).

## Open questions / Blocked
- none yet
