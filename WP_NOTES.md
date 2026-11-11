# WP I27: Verify in bounded memory
Branch: wp/I27-verify-memory
Status: in progress

## Done
- step 1: trace side in batches of whole runs, source side in chunks, validate without pre-buffering, memory test

## Next step
Wait for CI and Codacy on the draft pull request; fix any findings.

## Decisions
- Trace: read `trace.parquet` with pyarrow in batches of 2^18 rows; the rows of the last run of a batch move to the next batch, so every run is aggregated whole. No global sort and no per-row `source_file` join over the whole trace; the per-run results are sorted by (run_id, label) at the end, as before, so reports and their order are unchanged.
- `pre_buffer=False` for the trace in verify and for every table in `validate_dataset`: with pyarrow's default pre-buffering, reading batches still held most of the file (580 MB against 140 MB for 48M rows).
- Source: rows move to polars in chunks of 2^16 rows, and text files are read line by line from the bytes (no decoded copy, no list of lines). Typed arrays did not help: on macOS every large allocation of a new size stays resident after it is freed, so per-file buffers of the file's own size grow the peak with the number of files. Small chunks of one size are reused.
- `_file_aggregates` maps rows to run ids with `gather` instead of a join and a sort.

## Open questions / Blocked
- none
