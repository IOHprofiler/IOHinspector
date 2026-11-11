"""Convert and verify every dataset of a corpus folder; one line per dataset.

    python scripts/verify_corpus.py FOLDER [FOLDER ...] [--out DIR] [--mapping FILE]

Each entry of a FOLDER (a sub-folder, an archive or a file) is one dataset. It is
converted into DIR (default: a temporary folder), then verified against the entry.
Exits with 1 when any dataset fails to convert or verify. Custom CSV files are
converted only with ``--mapping``; without it they are skipped as unknown input.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import warnings
from pathlib import Path

from iohinspector.convert import convert, default_dataset_id
from iohinspector.verify import verify


def check(entry: Path, out: Path, mapping: str | None) -> tuple[bool, str]:
    """Convert and verify one entry; returns (ok, the line to print)."""
    dataset = out / default_dataset_id(entry)
    start = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            manifest = convert(entry, dataset, mapping=mapping, overwrite=True)
            report = verify(entry, dataset)
    # One bad entry must not stop the corpus run; it is reported on its line.
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        return False, f"ERROR    {entry.name}: {type(exc).__name__}: {exc}"
    seconds = time.perf_counter() - start
    state = "OK" if report.ok else f"FAIL({len(report.mismatches)})"
    line = (
        f"{state:<8} {entry.name}: {manifest['reader']['name']}, "
        f"{report.files} data file(s), {report.runs} run(s), {report.rows} row(s), "
        f"{seconds:.1f} s"
    )
    if not report.ok:
        line += "\n" + "\n".join("    " + s for s in report.summary(5).split("\n")[1:])
        report.write(out / f"{dataset.name}.verify.json")
    return report.ok, line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, help="where to write the datasets")
    parser.add_argument("--mapping", help="column mapping for custom CSV files")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="verify-corpus-") as tmp:
        out = args.out or Path(tmp)
        out.mkdir(parents=True, exist_ok=True)
        failed = 0
        for folder in args.folders:
            for entry in sorted(folder.iterdir()):
                if entry.name.startswith("."):
                    continue
                ok, line = check(entry, out, args.mapping)
                failed += not ok
                print(line, flush=True)
    print(f"{failed} dataset(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
