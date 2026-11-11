import io
import json
import os
import traceback
import warnings
from contextlib import redirect_stdout

import pytest

BASE_DIR = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))
EXAMPLES_DIR = os.path.normpath(os.path.join(BASE_DIR, "examples"))
NOTEBOOKS = [f for f in os.listdir(EXAMPLES_DIR) if f.endswith("ipynb")]
GB = globals()
LC = locals()


def iter_notebook(filename):
    with open(filename) as fp:
        nb = json.load(fp)

    for i, cell in enumerate(nb["cells"], 1):
        if cell["cell_type"] == "code":
            source = "".join(
                line for line in cell["source"] if not line.startswith("%")
            )
            yield i, source


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda f: f.split(".")[0])
def test_notebook(notebook):
    notebook = os.path.join(EXAMPLES_DIR, notebook)
    assert os.path.isfile(notebook)
    os.chdir(os.path.dirname(notebook))
    for i, block in iter_notebook(notebook):
        with io.StringIO() as buf, redirect_stdout(buf):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    exec(block, GB, LC)  # noqa: S102
            except Exception as e:  # noqa: BLE001
                pytest.fail(
                    f"failed in {notebook} cell {i}.\n\n"
                    f"Reason:\n{e}.\n\nBlock:\n{block}"
                )


def test_python_readme():
    fname = os.path.join(BASE_DIR, "README.md")
    assert os.path.isfile(fname)
    os.chdir(os.path.join(BASE_DIR, "tests"))
    with open(fname) as f:
        data = f.read().split("```")
        with io.StringIO() as buf, redirect_stdout(buf):
            for i, x in enumerate(data):
                if x.startswith("python"):
                    block = x[6:].strip()
                    if "help" not in block:
                        try:
                            exec(block, GB, LC)  # noqa: S102
                        except Exception as e:  # noqa: BLE001
                            pytest.fail(
                                f"failed in cell {i}. Reasion {e}.\n"
                                f"{traceback.format_exc()}"
                            )
