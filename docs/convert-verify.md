# Convert and verify

Two commands turn benchmark logs into datasets of the [storage format](format.md) and check the result against the logs. Both are installed as `iohinspector`, and are also available from Python (`iohinspector.convert.convert`, `iohinspector.verify.verify`).

```text
iohinspector convert SOURCE DATASET [--format F] [--mapping FILE] [--dataset-id ID]
                     [--tag KEY[:TYPE]=VALUE ...] [--tags-from-path PATTERN] [--overwrite]
iohinspector verify  SOURCE DATASET [--report FILE]
```

## convert

`SOURCE` is a file, a folder, or an archive (`.zip`, `.tar`, `.tar.gz`/`.tgz`, `.tar.bz2`/`.tbz2`, `.tar.xz`/`.txz`). Archives inside folders and inside other archives are opened too, up to four levels deep. They are extracted into a temporary folder; the input is never changed. A file inside an archive keeps the source path `<archive path>/<member path>`, for example `algA.zip/data_f1/IOHprofiler_f1_DIM5.dat`. The manifest lists each archive under `archives`, with its sha256, and each source file names its innermost archive.

The format is detected unless `--format` names it:

| Format | Input | Detected |
|---|---|---|
| `ioh` | IOHexperimenter JSON + `.dat` | yes |
| `coco` | COCO `.info` + `.dat`/`.tdat`/`.rdat` | yes |
| `nevergrad-csv` | one Nevergrad CSV file (`.csv` or `.csv.gz`) | yes, by its columns |
| `custom-csv` | one CSV file with a column mapping ([CSV readers](readers-csv.md)) | only with `--mapping` |

Readers are tried in the order of `iohinspector.io.READERS`, so a new reader is picked up by its line in that table. `--dataset-id` defaults to the input's name, with characters outside `A-Z a-z 0-9 . _ -` replaced by `_`. The dataset directory must not exist unless `--overwrite` is given.

### Descriptors

Every reader stores all the attributes its source has as descriptors (D17). Two options add more, to every run:

- `--tag key=value` adds a string; `--tag key:int=value` (or `float`, `bool`) gives it a type. The origin is `user`. Repeat the option for several tags.
- `--tags-from-path PATTERN` takes descriptors from each run's source path, with the origin `path`. In the pattern, `{key}` captures text within one path part, `{key:int}` (or `float`, `bool`) also types it, and `*` matches any text within one part. The pattern matches the whole source path or its leading parts. For example, `{algorithm_name}/{budget_factor:int}x*` on `cma/100xD/data_f1/IOHprofiler_f1_DIM5.dat` gives `algorithm_name = "cma"` and `budget_factor = 100`. Every run's path must match.

A tag replaces a descriptor of the same key that the source gave. `algorithm_name` and `algorithm_info` taken from paths get the origin `user`, because the format allows only `algorithm` or `user` for them. Keys that the format reserves (such as `budget` or `dimension`) are refused. The options are recorded in `manifest.extra.convert`.

### Memory

The CSV converters stream their file. The other readers build their tables in memory and hand them to the writer, which sorts and writes with bounded memory; so the memory needed is that of the reader's tables for the input.

## verify

`verify` checks a dataset against the input it was converted from. It is an independent scanner: it walks the input, opens archives and splits data files into runs with its own code, and shares none with the readers. It prints a summary, writes a JSON report (default `DATASET.verify.json`), and exits with 1 on any mismatch (2 when the input or dataset cannot be read).

It checks:

- `validate_dataset` on the dataset;
- the sha256 and size of every source file and archive in the manifest; a listed file that is missing, or a file the manifest records as missing that is now present, is a mismatch;
- each data file's line count against `sources[].data_rows`;
- per run of every data file: the number of data rows, the first and last evaluation, and the best value (smallest or largest, by the run's `maximization`; NaN ignored);
- per run, the hash of its data lines against `runs.source_rows_sha256` (the rule in [format.md](format.md#runsparquet): tokens split at the separator, joined by one space, one `\n`-terminated line per row);
- per run, an order-sensitive fingerprint of every (evaluations, value) pair against the trace, so a change on either side is found;
- that every trace row belongs to a verified source run, and that dataset runs beyond a file's runs have no hash.

How the scanner reads each file kind (`sources[].kind`):

| Kind | Run starts at | Evaluations | Value |
|---|---|---|---|
| `ioh-dat` | a line whose first token is not a number | column `evaluations` | column `raw_y` |
| `coco-dat`, `-tdat`, `-rdat` | a `%` line naming the columns (`% f evaluations ...`, `% function evaluation ...`); other `%` lines are comments | column 1 | column 3 (best Δf) |
| `nevergrad-csv` | every row | `budget` | `loss` |
| `custom-csv` | a new value of the mapping's `run_by` columns (every row without them) | the mapping's `evaluations` | the mapping's `value` |

Metadata files (`ioh-json`, `coco-info`) are checked by their sha256 only. A kind that is not in the table is listed as not verified, and its runs' trace rows are then reported as not belonging to a verified source run.

COCO loggers write one run set to several files, which the reader merges (`logged_in` names the files that hold a row). A data file's rows in the trace are those whose `logged_in` names its kind (its file name suffix), or all rows of its own runs when `logged_in` is null. Each file is compared on its own; the line hash is compared for the run's own `source_file` only.

Files in the input that the manifest does not list are reported (`unlisted_files`) but do not fail the check: readers record the files they skip as manifest warnings.

## Checking a corpus

`scripts/verify_corpus.py FOLDER` converts and verifies every entry of a folder (each sub-folder, archive or file is one dataset) and prints one line per dataset. It exits with 1 when any dataset fails.
