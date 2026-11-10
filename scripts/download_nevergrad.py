#!/usr/bin/env python3
"""Download Nevergrad result files (``allxps``) for the reader tests.

    python scripts/download_nevergrad.py [--out DIR] [--list] [NAME ...]

Each NAME is an experiment such as ``yatinybbob``; the file
``https://dl.fbaipublicfiles.com/nevergrad/allxps/NAME/data.csv.gz`` is saved as
``DIR/NAME.csv.gz``. The default is ``yatinybbob`` (35 kB) into
``tests/io/data/downloaded``, which git ignores. ``--list`` prints the names that the
index page offers. The tests use the files when they are present.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

BASE = "https://dl.fbaipublicfiles.com/nevergrad/allxps"
DEFAULT_OUT = (
    Path(__file__).resolve().parents[1] / "tests" / "io" / "data" / "downloaded"
)


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:  # nosec B310
        return response.read()


def experiment_names() -> list[str]:
    page = _get(f"{BASE}/list.html").decode("utf-8", "replace")
    return re.findall(
        r'href="' + re.escape(BASE) + r'/([^/"]+)/data\.csv\.gz"',
        page,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("names", nargs="*", default=["yatinybbob"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--list", action="store_true", help="print the available names")
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(experiment_names()))
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            parser.error(f"invalid experiment name {name!r}")
        target = args.out / f"{name}.csv.gz"
        target.write_bytes(_get(f"{BASE}/{name}/data.csv.gz"))
        print(f"{target} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
