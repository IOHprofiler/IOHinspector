"""Download a sample of the COCO data archive, for the COCO reader's archive tests.

The COCO data archive (https://numbbo.github.io/data-archive) lists its files in
``coco_archive_definition.txt``: a Python list of ``(path, sha256, kB)`` tuples.
This script picks a sample from it, downloads the archives into a cache folder
and checks their sha256. Nothing it downloads is committed.

Default sample: for each suite the reader covers (``bbob``, ``bbob-noisy``,
``bbob-largescale``, ``bbob-boxed``) and each year folder, the smallest archive.
That covers every year, and with it every ``.info`` and ``.dat`` variant the
loggers wrote over the years.

Usage::

    python scripts/coco_archive.py [--cache DIR] [--per-folder N] [--list]

The cache defaults to ``$COCO_ARCHIVE_CACHE`` or ``.cache/coco-archive`` in the
repository; ``tests/io/test_coco_archive.py`` reads the same folder and skips when
it is empty.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

URLS = (
    "https://numbbo.github.io/data-archive/data-archive",
    "https://numbbo.github.io/gforge/data-archive",
)
DEFINITION = "coco_archive_definition.txt"
SUITES = ("bbob", "bbob-noisy", "bbob-largescale", "bbob-boxed")
REPO = Path(__file__).resolve().parent.parent


def default_cache() -> Path:
    env = os.environ.get("COCO_ARCHIVE_CACHE")
    return Path(env) if env else REPO / ".cache" / "coco-archive"


def _fetch(url: str) -> bytes:
    if not url.startswith("https://"):
        raise ValueError(f"refusing a non-https URL: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "coco-archive-sample"})
    with urllib.request.urlopen(request, timeout=120) as response:  # nosec B310
        return response.read()


def definition() -> tuple[str, list[tuple[str, str, int]]]:
    """The archive's base URL and its list of ``(path, sha256, kB)``."""
    errors = []
    for base in URLS:
        try:
            text = _fetch(f"{base}/{DEFINITION}").decode("utf-8")
        except OSError as exc:
            errors.append(f"{base}: {exc}")
            continue
        return base, [tuple(e) for e in ast.literal_eval(text)]
    raise SystemExit("could not read the archive definition:\n" + "\n".join(errors))


def sample(
    entries: list[tuple[str, str, int]], per_folder: int = 1
) -> list[tuple[str, str, int]]:
    """The ``per_folder`` smallest archives of each suite and year folder."""
    folders: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for entry in entries:
        path = entry[0]
        if path.split("/", 1)[0] in SUITES:
            folders[path.rsplit("/", 1)[0]].append(entry)
    out = []
    for folder in sorted(folders):
        out += sorted(folders[folder], key=lambda e: (e[2], e[0]))[:per_folder]
    return out


def download(base: str, entry: tuple[str, str, int], cache: Path) -> Path:
    path, sha256, _ = entry
    target = cache / path
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == sha256:
        return target
    data = _fetch(f"{base}/{path}")
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise SystemExit(
            f"{path}: sha256 {digest} differs from the definition {sha256}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--cache", type=Path, default=default_cache())
    parser.add_argument("--per-folder", type=int, default=1)
    parser.add_argument("--list", action="store_true", help="print the sample only")
    args = parser.parse_args(argv)
    base, entries = definition()
    chosen = sample(entries, args.per_folder)
    total = sum(e[2] for e in chosen)
    print(f"{len(chosen)} archives, about {total / 1024:.0f} MB", file=sys.stderr)
    if args.list:
        for entry in chosen:
            print(f"{entry[2]:>8} kB  {entry[0]}")
        return 0
    args.cache.mkdir(parents=True, exist_ok=True)
    for entry in chosen:
        print(f"{entry[0]}", file=sys.stderr)
        download(base, entry, args.cache)
    listing = [{"path": p, "sha256": h, "kB": s} for p, h, s in chosen]
    (args.cache / "sample.json").write_text(json.dumps(listing, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
