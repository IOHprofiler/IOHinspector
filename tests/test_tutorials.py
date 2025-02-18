import os
import traceback
import json
import sys
import shutil
import unittest
import io
from contextlib import redirect_stdout
from functools import partial
import warnings

BASE_DIR = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))
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


def test_notebook_runner(self, notebook):
    assert os.path.isfile(notebook)
    os.chdir(os.path.dirname(notebook))
    for i, block in iter_notebook(notebook):
        with io.StringIO() as buf, redirect_stdout(buf):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    exec(block, GB, LC)
            except Exception as e:
                assert (
                    False
                ), f"failed in {notebook} cell {i}.\n\nReason:\n{e}.\n\nBlock:\n{block}"


class MetaTest(type):
    def __new__(cls, name, bases, dct):
        instance = super().__new__(cls, name, bases, dct)
        dirname = os.path.normpath(os.path.join(BASE_DIR, "examples"))
        for f in filter(lambda x: x.endswith("ipynb"), os.listdir(dirname)):
            fname, *_ = os.path.basename(f).split(".")
            notebook = os.path.join(dirname, f)
            setattr(
                instance,
                f"test_notebook_{fname}",
                partial(test_notebook_runner, instance, notebook),
            )
        return instance


class TestExamples(unittest.TestCase, metaclass=MetaTest):

    """Examples test"""

    @unittest.skipUnless(sys.version_info.minor >= 7, "python version > 3.7")
    def test_python_readme(self):
        try:
            fname = os.path.join(BASE_DIR, "README.md")
            self.assertTrue(os.path.isfile(fname))
            os.chdir(os.path.join(BASE_DIR, "tests"))
            with open(fname) as f:
                data = f.read().split("```")
                with io.StringIO() as buf, redirect_stdout(buf):
                    for i, x in enumerate(data):
                        if x.startswith("python"):
                            block = x[6:].strip()
                            if not "help" in block:
                                try:
                                    exec(block, GB, LC)
                                except Exception as e:
                                    raise Exception(
                                        f"failed in cell {i}. Reasion {e}.\n{traceback.format_exc()}"
                                    )
        except:
            raise
        finally:
            pass

if __name__ == "__main__":
    unittest.main()