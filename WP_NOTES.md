# WP I23: Upload client
Branch: wp/I23-upload-client
Status: ready for review

## Done
- `iohinspector upload` (src/iohinspector/upload.py, cli.py), tests (tests/test_upload.py), README section, CHANGELOG.

## Next step
Controller: push and open the draft pull request.

## Decisions
- What is uploaded: the original files (a file or archive as is; a folder is zipped to `<folder>.zip` next to it and the zip is reused on later runs). The server converts and verifies again and keeps the original in raw/. The local convert and verify only guards against a failing upload; it is skipped when a state file exists (resume) or with `--no-check`.
- Resuming: `<archive>.upload.json` holds the upload id and size. A rerun asks the server (GET) for `received` and continues; a 404 starts a new upload; a 409 on a chunk re-reads `received`. The file is removed when done.
- Confirm happens in the same command (title defaults to the input's name, visibility private).
- stdlib urllib only; no new dependency. Token never printed.

## Open questions / Blocked
- Not tried against a real server (no network in tests).
