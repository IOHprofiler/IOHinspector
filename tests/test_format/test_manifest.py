import copy
import hashlib
import json
import re
import warnings
from pathlib import Path

import pytest

from iohinspector.format import (
    MANIFEST_SCHEMA,
    FormatWarning,
    ManifestError,
    UnsupportedFormatVersion,
    file_entry,
    new_manifest,
    read_manifest,
    validate_manifest,
    write_manifest,
)
from iohinspector.format.ids import source_path_problem

FORMAT_MD = Path(__file__).resolve().parents[2] / "docs" / "format.md"


def _manifest(**kw):
    m = new_manifest(
        "ds-1",
        "test-reader",
        "1.2",
        sources=[{"path": "a/b.dat", "sha256": "f" * 64, "bytes": 10}],
    )
    m.update(kw)
    return m


def test_schema_in_spec_equals_shipped_schema():
    section = FORMAT_MD.read_text(encoding="utf-8").split("### JSON Schema", 1)[1]
    block = re.search(r"```json\n(.*?)```", section, re.DOTALL).group(1)
    assert json.loads(block) == MANIFEST_SCHEMA


def test_schema_path_pattern_agrees_with_python_check():
    pattern = re.compile(MANIFEST_SCHEMA["$defs"]["path"]["pattern"])
    negative = re.compile(MANIFEST_SCHEMA["$defs"]["path"]["not"]["pattern"])
    for path in [
        "a",
        "a/b.dat",
        "a.zip/x y/ü.dat",
        "..a",
        "",
        "/a",
        "a//b",
        "a/",
        ".",
        "./a",
        "a/..",
        "a\\b",
        "a\tb",
    ]:
        schema_ok = (
            bool(path)
            and pattern.search(path) is not None
            and negative.search(path) is None
        )
        assert schema_ok == (source_path_problem(path) is None), path


def test_new_manifest_is_valid():
    m = _manifest()
    assert validate_manifest(m) == []
    assert m["format"] == "iohinspector-dataset" and m["format_version"] == "1.0"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", m["created"])
    assert m["writer"]["name"] == "iohinspector"


def test_write_then_read(tmp_path):
    m = _manifest(
        warnings=[
            {
                "code": "limited_precision",
                "message": "values below 5e-11 are 0",
                "source": "a/b.dat",
            }
        ]
    )
    m["sources"][0].update(
        kind="ioh-dat",
        data_rows=5,
        triggers=[
            {"kind": "improvement"},
            {"kind": "fixed_budget", "params": {"budgets": [10, 100]}},
        ],
        precision={"notation": "fixed", "digits": 10},
        logger={"name": "IOHexperimenter", "version": "0.3.18"},
    )
    write_manifest(tmp_path, m)
    text = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    assert text.endswith("\n") and text.index('"counts"') < text.index(
        '"dataset_id"'
    )  # sorted keys
    assert read_manifest(tmp_path) == m


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda m: m.pop("dataset_id"), "dataset_id"),
        (lambda m: m.update(dataset_id="bad id"), "dataset_id"),
        (lambda m: m.update(created="2026-11-06 12:00"), "created"),
        (lambda m: m["sources"][0].update(sha256="ABC"), "sources/0/sha256"),
        (lambda m: m["sources"][0].update(path="/abs"), "sources/0/path"),
        (lambda m: m["sources"][0].update(path="a/../b"), "sources/0/path"),
        (
            lambda m: m["sources"][0].update(triggers=[{"kind": "sometimes"}]),
            "triggers",
        ),
        (
            lambda m: m["sources"][0].update(
                precision={"notation": "fixed", "digits": -1}
            ),
            "precision",
        ),
        (lambda m: m.update(unknown_field=1), "unknown_field"),
        (lambda m: m.update(variable_types=["real"]), "variable_types"),
        (lambda m: m["counts"].update(runs=-1), "counts"),
        (lambda m: m["sources"].append(dict(m["sources"][0])), "duplicate source path"),
        (lambda m: m["sources"][0].update(archive="x.zip"), "not in 'archives'"),
        (
            lambda m: m.update(warnings=[{"code": "Bad Code", "message": ""}]),
            "warnings",
        ),
    ],
)
def test_invalid_manifests_are_reported(change, message):
    m = _manifest()
    change(m)
    problems = validate_manifest(m)
    assert problems and any(message in p for p in problems), problems
    with pytest.raises(ManifestError):
        write_manifest(Path("/nonexistent-dir-not-written"), m)


def test_logical_sources_and_archives_are_valid():
    m = _manifest(archives=[{"path": "x.zip", "sha256": "a" * 64, "bytes": 3}])
    m["sources"] = [
        {"path": "x.zip/a.dat", "sha256": "b" * 64, "bytes": 1, "archive": "x.zip"},
        {"path": "logger-session", "sha256": None, "bytes": None},
    ]
    assert validate_manifest(m) == []


def test_unknown_major_version_is_refused(tmp_path):
    m = _manifest()
    m["format_version"] = "2.0"
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(UnsupportedFormatVersion, match="2.0"):
        read_manifest(tmp_path)
    assert "not supported" in validate_manifest(m)[0]


@pytest.mark.parametrize("version", ["1", "v1.0", "1.0.0", 1.0, None])
def test_malformed_version_is_refused(tmp_path, version):
    m = _manifest(format_version=version)
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(UnsupportedFormatVersion):
        read_manifest(tmp_path)


def test_other_format_name_is_refused(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(_manifest(format="something-else"))
    )
    with pytest.raises(UnsupportedFormatVersion, match="not an iohinspector dataset"):
        read_manifest(tmp_path)


def test_newer_minor_version_is_read_with_a_warning(tmp_path):
    m = _manifest(format_version="1.7", future_field={"x": 1})
    m["sources"][0]["future_source_field"] = True
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    with pytest.warns(FormatWarning, match="1.7"):
        assert read_manifest(tmp_path)["future_field"] == {"x": 1}
    assert validate_manifest(m) == []
    strict = copy.deepcopy(m)
    strict["format_version"] = "1.0"
    assert validate_manifest(strict)


def test_current_version_reads_without_warning(tmp_path):
    write_manifest(tmp_path, _manifest())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        read_manifest(tmp_path)


def test_nan_and_bad_json_are_refused(tmp_path):
    (tmp_path / "manifest.json").write_text('{"format": NaN}')
    with pytest.raises(ManifestError):
        read_manifest(tmp_path)
    (tmp_path / "manifest.json").write_text("[1, 2]")
    with pytest.raises(ManifestError):
        read_manifest(tmp_path)


def test_file_entry_hashes_the_file(tmp_path):
    f = tmp_path / "x.dat"
    f.write_bytes(b"evaluations raw_y\n1 2.5\n")
    entry = file_entry(f, "x.dat", kind="ioh-dat")
    expected = hashlib.sha256(f.read_bytes()).hexdigest()
    assert entry == {
        "path": "x.dat",
        "sha256": expected,
        "bytes": 24,
        "kind": "ioh-dat",
    }
