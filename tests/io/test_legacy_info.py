"""The legacy IOHprofiler ``.info`` parser (WP I06, step 1)."""

from __future__ import annotations

import pytest

from iohinspector.io import ReaderError
from iohinspector.io._legacy_info import parse_header, parse_info_text
from iohinspector.io._legacy_meta import FunctionIds, block_meta, header_value

V029 = (
    'suite = "PBO", funcId = 1, funcName = "OneMax", DIM = 16, maximization = "T", '
    'algId = "ea", algInfo = "a, b", budget = 100, seed = "42", rate = 0.5, '
    'dynamicAttribute = "s|t"\n'
    "%\n"
    "data_f1_OneMax/IOHprofiler_f1_DIM16.dat, 1:100|16;1;0.5, 2:90|15;2;inf\n"
)


def test_header_quotes_and_types():
    items = parse_header(
        "suite = 'PBO', funcId = 1, x = \"a = b, c\", y = -1.5e3, z = abc", "f", 1
    )
    assert [(i.key, i.value, i.quoted) for i in items] == [
        ("suite", "PBO", True),
        ("funcId", "1", False),
        ("x", "a = b, c", True),
        ("y", "-1.5e3", False),
        ("z", "abc", False),
    ]
    assert [header_value(i) for i in items] == ["PBO", 1, "a = b, c", -1500.0, "abc"]


def test_header_errors_name_file_and_line():
    with pytest.raises(ReaderError, match=r"^a.info:3: cannot read the header"):
        parse_header("suite = 'PBO', funcId", "a.info", 3)
    with pytest.raises(ReaderError, match="more than once"):
        parse_header("a = 1, a = 2", "a.info", 1)


def test_block_runs_and_dynamic_attributes():
    (block,) = parse_info_text(V029, "x.info")
    assert block.datafile == "data_f1_OneMax/IOHprofiler_f1_DIM16.dat"
    assert block.comment == "%"
    assert [(r.instance, r.evals, r.best_y) for r in block.runs] == [
        (1, 100, 16.0),
        (2, 90, 15.0),
    ]
    assert block.runs[1].attributes == ["2", "inf"]
    meta = block_meta(block, FunctionIds.of([block]))
    assert (meta.function_id, meta.function_name, meta.dimension) == (1, "OneMax", 16)
    assert meta.maximization is True
    assert (meta.algorithm_name, meta.algorithm_info) == ("ea", "a, b")
    assert meta.run_attributes == ["s", "t"]
    assert meta.experiment == [
        ("budget", 100, "experiment"),
        ("seed", "42", "experiment"),
        ("rate", 0.5, "experiment"),
    ]


def test_blocks_blank_lines_backslashes_and_cut_off_entries():
    text = (
        "funcId = 1, DIM = 5, algId = 'A'\n%\nd\\f.dat, 1:10|1.0, 2:0|0, ,\n"
        "\n\n"
        "funcId = 1, DIM = 10, algId = 'A'\n% my comment\r\nd\\g.dat\n"
    )
    first, second = parse_info_text(text, "x.info")
    assert first.datafile == "d/f.dat" and len(first.runs) == 1
    assert first.empty_runs == ["2:0|0"] and first.incomplete
    assert second.line == 6 and second.datafile == "d/g.dat" and second.runs == []
    meta = block_meta(second, FunctionIds.of([first, second]))
    assert ("comment", "my comment", "experiment") in meta.experiment


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("funcId = 1, DIM = 5\n%\n", r"x.info:1: a block needs three lines"),
        ("funcId = 1, DIM = 5\n%\nf.dat, 1-10|1\n", r"x.info:3: run entry '1-10\|1'"),
        (
            "funcId = 1, DIM = 5\n%\nf.dat, 1:10|abc\n",
            r"x.info:3: .*'abc' is not a number",
        ),
        (
            "funcId = 1, DIM = 5, dynamicAttribute = 'a'\n%\nf.dat, 1:10|1\n",
            r"x.info:3: .*0 attribute value\(s\)",
        ),
        ("\n\n", r"x.info: no blocks found"),
    ],
)
def test_malformed_info(text, message):
    with pytest.raises(ReaderError, match=message):
        parse_info_text(text, "x.info")


@pytest.mark.parametrize(
    ("header", "message"),
    [
        ("DIM = 5", "no funcId"),
        ("funcId = 1", "no DIM"),
        ("funcId = 1, DIM = 0", "not a positive integer"),
        ("funcId = 1, DIM = 5, maximization = 'yes'", "is not T, F, TRUE or FALSE"),
    ],
)
def test_bad_header_values(header, message):
    (block,) = parse_info_text(f"{header}\n%\nf.dat\n", "x.info")
    with pytest.raises(ReaderError, match=f"x.info:1: .*{message}"):
        block_meta(block, FunctionIds.of([block]))


def test_direction_follows_iohanalyzer_when_missing():
    blocks = parse_info_text(
        "suite = 'bbob', funcId = 1, DIM = 5\n%\nf.dat\n"
        "suite = 'PBO', funcId = 1, DIM = 5, Maximization = FALSE\n%\ng.dat\n"
        "funcId = 1, DIM = 5\n%\nh.dat\n",
        "x.info",
    )
    metas = [block_meta(b, FunctionIds.of(blocks)) for b in blocks]
    assert [m.maximization for m in metas] == [False, False, True]
    inferred = [
        [c for c, _ in m.warnings if c == "maximization_inferred"] for m in metas
    ]
    assert [len(w) for w in inferred] == [1, 0, 1]
    assert metas[2].suite == ""


def test_function_ids_for_names():
    blocks = parse_info_text(
        "funcId = blubb, DIM = 5\n%\nf.dat\n"
        "funcId = 7, DIM = 5\n%\ng.dat\n"
        "funcId = bla, funcName = 'Bla', DIM = 5\n%\nh.dat\n",
        "x.info",
    )
    ids = FunctionIds.of(blocks)
    assert ids.assigned == {"bla": 8, "blubb": 9}
    metas = [block_meta(b, ids) for b in blocks]
    assert [(m.function_id, m.function_name) for m in metas] == [
        (9, "blubb"),
        (7, None),
        (8, "Bla"),
    ]
    assert ("funcId", "bla", "experiment") in metas[2].experiment
