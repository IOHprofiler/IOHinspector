"""Image snapshots: each gallery figure against its baseline in ``docs/figures/``.
After an intended change, rewrite the baselines with
``python -m tests.test_figures.sample`` and look at the difference.

Layout moves by a few pixels between matplotlib minor versions, so the snapshots
run only on the minor version that wrote the baselines (it is in their PNG
metadata); regenerate them when matplotlib moves on.
"""

import re

import matplotlib
import pytest
from matplotlib.testing.compare import compare_images
from PIL import Image

from iohinspector.figures import render

from .sample import GALLERY, gallery_runs, gallery_specs

TOLERANCE = 2  # RMS over 0-255 pixel values; allows small rendering changes


def minor(version: str) -> str:
    return ".".join(re.findall(r"\d+", version)[:2])


@pytest.mark.parametrize("name, spec", gallery_specs().items())
def test_snapshot(tmp_path, name, spec):
    with Image.open(GALLERY / f"{name}.png") as baseline:
        written_by = minor(baseline.text["Software"])  # type: ignore[attr-defined]
    if written_by != minor(matplotlib.__version__):
        pytest.skip(f"baseline from matplotlib {written_by}; regenerate it")
    actual = tmp_path / f"{name}.png"
    actual.write_bytes(render(spec, gallery_runs(spec)))
    assert compare_images(GALLERY / f"{name}.png", actual, tol=TOLERANCE) is None
