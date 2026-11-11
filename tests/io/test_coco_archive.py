"""The COCO reader on a sample of the COCO data archive (one archive per suite and
year), downloaded by ``scripts/coco_archive.py``. Nothing of it is committed.

Opt in with ``COCO_ARCHIVE_TESTS=1``; the archives are read from
``$COCO_ARCHIVE_CACHE`` or ``.cache/coco-archive``. Every data file of every
archive is counted here with ``tarfile``/``zipfile`` and plain line scanning,
independently of the reader, and checked against what the reader stored.
"""

from __future__ import annotations

import io
import json
import os
import re
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

import polars as pl
import pytest

from iohinspector.convert import convert
from iohinspector.format import scan_dataset

REPO = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("COCO_ARCHIVE_CACHE", REPO / ".cache" / "coco-archive"))
SAMPLE = CACHE / "sample.json"
ENABLED = os.environ.get("COCO_ARCHIVE_TESTS") == "1" and SAMPLE.is_file()
ARCHIVES = [e["path"] for e in json.loads(SAMPLE.read_text())] if ENABLED else []
HEADER = re.compile(rb"^%\s*(f evaluations|function evaluation)")

pytestmark = pytest.mark.skipif(
    not ENABLED, reason="set COCO_ARCHIVE_TESTS=1 after scripts/coco_archive.py"
)


def members(data: bytes, prefix: str) -> dict[str, bytes]:
    """Data files of an archive (and of archives inside it) by source path."""
    out: dict[str, bytes] = {}
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            items = [(i.filename, zf.read(i)) for i in zf.infolist() if not i.is_dir()]
    else:
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            items = [
                (m.name, tf.extractfile(m).read())
                for m in tf.getmembers()
                if m.isfile()
            ]
    for name, content in items:
        path = f"{prefix}/{name.removeprefix('./')}"
        if name.endswith((".zip", ".tgz", ".tar.gz", ".tar")):
            out.update(members(content, path))
        elif name.endswith((".dat", ".tdat", ".rdat")):
            out[path] = content
    return out


def count(content: bytes) -> tuple[int, int]:
    """(header lines, data lines) of a COCO data file."""
    headers = rows = 0
    for line in content.split(b"\n"):
        if not line.strip():
            continue
        if line.startswith(b"%"):
            headers += bool(HEADER.match(line))
        else:
            rows += 1
    return headers, rows


@pytest.mark.parametrize("archive", ARCHIVES)
def test_archive_reads_losslessly(archive, tmp_path):
    path = CACHE / archive
    manifest = convert(path, tmp_path / "ds", dataset_id="archive-sample")
    dataset = scan_dataset(tmp_path / "ds", warn=False)
    runs_df, trace = dataset.runs.collect(), dataset.trace.collect()
    files = members(path.read_bytes(), path.name)
    sources = {s["path"]: s for s in manifest["sources"]}
    read = {p: c for p, c in files.items() if p in sources}
    assert read, "no data file read"
    runs = Counter(runs_df["source_file"])
    flagged = _rows_per_file(runs_df, trace)
    for name, content in read.items():
        headers, rows = count(content)
        assert sources[name]["data_rows"] == rows, name
        assert flagged[name] == rows, name
        if name in runs:  # the run's primary file
            assert runs[name] >= headers, name
    unread = sorted(set(files) - set(sources))
    warned = {w.get("source") for w in manifest.get("warnings", [])}
    assert set(unread) <= warned


def _rows_per_file(runs: pl.DataFrame, trace: pl.DataFrame) -> Counter:
    """Trace rows per data file, from ``logged_in`` (or, for stems with one data
    file, the run's source file) and the stem of the run's source file."""
    trace = trace.join(runs.select("run_id", "source_file"), on="run_id")
    own = pl.col("source_file").str.extract(r"\.([a-z]+)$")
    logged = pl.col("logged_in") if "logged_in" in trace.columns else pl.lit(None)
    kinds = (
        trace.select(
            stem=pl.col("source_file").str.replace(r"\.[a-z]+$", ""),
            kind=pl.coalesce(logged, own).str.split("+"),
        )
        .explode("kind")
        .group_by("stem", "kind")
        .len()
    )
    return Counter({f"{stem}.{kind}": n for stem, kind, n in kinds.iter_rows()})
