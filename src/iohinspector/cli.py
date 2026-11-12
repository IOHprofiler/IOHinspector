"""The ``iohinspector`` command: ``convert``, ``verify`` and ``upload``.

    iohinspector convert SOURCE DATASET [--format F] [--mapping M] [--dataset-id ID]
                         [--tag KEY=VALUE ...] [--tags-from-path PATTERN] [--overwrite]
    iohinspector verify SOURCE DATASET [--report FILE]
    iohinspector upload SOURCE [--token T] [--url U] [--title T] [--visibility V]
                        [--tag KEY=VALUE ...] [--rename OLD=NEW ...] [--no-check]

``verify`` exits with 1 when the dataset differs from its source, and both exit
with 2 when the input cannot be read; ``upload`` also exits with 2 when the
server refuses.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

from .convert import ConvertError, convert
from .format import FormatError, ManifestError
from .io import READERS, ReaderError
from .io.custom_csv import MappingError
from .upload import DEFAULT_URL, UploadError, upload
from .verify import verify

_ERRORS = (
    ConvertError,
    FormatError,
    ManifestError,
    MappingError,
    ReaderError,
    UploadError,
    FileNotFoundError,
    FileExistsError,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iohinspector", description="Convert and verify benchmark datasets."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    conv = sub.add_parser("convert", help="convert logs to dataset format v1")
    conv.add_argument("source", help="a file, folder, zip or tar archive")
    conv.add_argument("dataset", help="the dataset folder to write")
    conv.add_argument(
        "--format",
        choices=list(READERS),
        help="the input format (default: detected)",
    )
    conv.add_argument("--mapping", help="column mapping (JSON file) for custom-csv")
    conv.add_argument("--dataset-id", help="default: the input's name")
    conv.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="KEY[:TYPE]=VALUE",
        help="a descriptor for every run; TYPE is str (default), int, float or bool",
    )
    conv.add_argument(
        "--tags-from-path",
        metavar="PATTERN",
        help="descriptors from source paths, such as '{algorithm_name}/{budget:int}_*'",
    )
    conv.add_argument("--overwrite", action="store_true", help="replace DATASET")
    ver = sub.add_parser("verify", help="check a dataset against its source files")
    ver.add_argument("source", help="the input the dataset was converted from")
    ver.add_argument("dataset", help="the dataset folder")
    ver.add_argument(
        "--report", help="the JSON report to write (default: DATASET.verify.json)"
    )
    up = sub.add_parser("upload", help="upload a dataset to the website")
    up.add_argument("source", help="a file, folder (zipped first) or archive")
    up.add_argument("--token", help="default: $IOH_TOKEN (create one on /account)")
    up.add_argument("--url", help=f"default: $IOH_URL or {DEFAULT_URL}")
    up.add_argument("--title", help="default: the input's name")
    up.add_argument(
        "--visibility", choices=["private", "unlisted", "public"], default="private"
    )
    up.add_argument("--tag", action="append", default=[], metavar="KEY[:TYPE]=VALUE")
    up.add_argument("--rename", action="append", default=[], metavar="OLD=NEW")
    up.add_argument("--compare-by", action="append", help="default: algorithm_name")
    up.add_argument("--format", choices=list(READERS), help="default: detected")
    up.add_argument("--mapping", help="column mapping (JSON file) for custom-csv")
    up.add_argument("--no-check", action="store_true", help="skip the local check")
    return parser


def _upload(args: argparse.Namespace) -> int:
    token = args.token or os.environ.get("IOH_TOKEN")
    if not token:
        raise UploadError("no token: pass --token or set IOH_TOKEN")
    mapping = json.loads(Path(args.mapping).read_text()) if args.mapping else None
    dataset_id = upload(
        args.source,
        token=token,
        url=args.url or os.environ.get("IOH_URL") or DEFAULT_URL,
        title=args.title,
        visibility=args.visibility,
        tags=args.tag,
        renames=dict(item.split("=", 1) for item in args.rename),
        compare_by=args.compare_by or ["algorithm_name"],
        format=args.format,
        mapping=mapping,
        check=not args.no_check,
    )
    print(f"uploaded: dataset {dataset_id}")
    return 0


def _convert(args: argparse.Namespace) -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the manifest's warnings are printed below
        manifest = convert(
            args.source,
            args.dataset,
            format=args.format,
            mapping=args.mapping,
            dataset_id=args.dataset_id,
            tags=args.tag,
            tags_from_path=args.tags_from_path,
            overwrite=args.overwrite,
        )
    counts = manifest["counts"]
    print(
        f"{args.dataset}: {manifest['reader']['name']}, {counts['runs']} run(s), "
        f"{counts['trace_rows']} trace row(s), "
        f"{len(manifest['sources'])} source file(s)"
    )
    for item in manifest.get("warnings", []):
        print(f"  warning {item['code']}: {item.get('source', '')} {item['message']}")
    return 0


def _verify(args: argparse.Namespace) -> int:
    report = verify(args.source, args.dataset)
    path = Path(args.report or f"{str(args.dataset).rstrip('/')}.verify.json")
    report.write(path)
    print(report.summary())
    print(f"report: {path}")
    return 0 if report.ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; returns the exit code."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "convert":
            return _convert(args)
        if args.command == "upload":
            return _upload(args)
        return _verify(args)
    except _ERRORS as exc:
        print(f"iohinspector {args.command}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
