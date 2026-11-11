"""Property tests: logs written like IOHexperimenter does, read back exactly.

A small writer mimics the Analyzer: the JSON is formatted without escaping and with
bare ``nan`` / ``inf``, and the ``.dat`` values use ``%.10f`` (or ``%.17g`` when
the JSON says so). The generated runs have ties, f = 0, maximisation, ``inf`` and
``nan``, runs without improvement, missing values (``None``) and runs whose final
row is not logged. Reading them must give every row, unchanged, and writing the
dataset must store the same rows.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from iohinspector.format import scan_dataset
from iohinspector.io.ioh import read_ioh

TEXT = st.text(alphabet='aZ0 \\"_-.', max_size=8)
VALUES = st.one_of(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
    st.sampled_from([0.0, -0.0, 1e-12, 4.9e-11, math.inf, -math.inf, math.nan]),
)


@dataclass
class Log:
    algorithm: str
    info: str
    maximization: bool
    full_precision: bool
    sigma: bool
    experiment: list[tuple[str, str]]
    #: per run: rows of (evaluations, raw_y, sigma or None)
    runs: list[list[tuple[int, float, float | None]]]
    json_runs: int
    seeds: list[float]


@st.composite
def run_rows(draw):
    n = draw(st.integers(min_value=1, max_value=6))
    steps = draw(
        st.lists(st.integers(min_value=0, max_value=5), min_size=n, max_size=n)
    )
    first = draw(st.integers(min_value=1, max_value=3))
    evals, rows = first, []
    for i in range(n):
        evals += steps[i] if i else 0
        y = draw(st.one_of(VALUES, st.just(rows[-1][1]) if rows else VALUES))
        s = draw(st.one_of(st.none(), st.floats(0, 10, allow_nan=False)))
        rows.append((evals, y, s))
    return rows


@st.composite
def logs(draw):
    runs = draw(st.lists(run_rows(), min_size=0, max_size=4))
    keys = draw(
        st.lists(st.sampled_from(["seed", "budget", "note", "dimension"]), unique=True)
    )
    return Log(
        algorithm=draw(TEXT.filter(bool)),
        info=draw(TEXT),
        maximization=draw(st.booleans()),
        full_precision=draw(st.booleans()),
        sigma=draw(st.booleans()),
        experiment=[(k, draw(TEXT)) for k in keys],
        runs=runs,
        json_runs=draw(st.integers(min_value=0, max_value=len(runs) + 1)),
        seeds=[
            draw(st.floats(allow_nan=True, allow_infinity=True))
            for _ in range(len(runs) + 1)
        ],
    )


def fmt_number(x: float) -> str:
    """fmt's ``{}`` for a double in the JSON: shortest text, bare inf and nan."""
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return repr(int(x)) if x.is_integer() and abs(x) < 1e16 else repr(x)


def dat_value(x: float, full: bool) -> str:
    return f"{x:.17g}" if full else f"{x:.10f}"


def write(log: Log, folder: Path) -> Path:
    data = folder / "data_f1_F"
    data.mkdir(parents=True)
    names = ["evaluations", "raw_y"] + (["sigma"] if log.sigma else [])
    lines = []
    for run in log.runs:
        lines.append(" ".join(names))
        for e, y, s in run:
            tokens = [str(e), dat_value(y, log.full_precision)]
            if log.sigma:
                tokens.append("None" if s is None else dat_value(s, log.full_precision))
            lines.append(" ".join(tokens))
    (data / "IOHprofiler_f1_DIM3.dat").write_text(
        "".join(f"{line}\n" for line in lines)
    )
    json_runs = []
    for i in range(log.json_runs):
        json_runs.append(
            f'{{"instance": {i + 1}, "evals": 10, "best": {{"evals": 1, "y": '
            f'{fmt_number(log.seeds[i])}, "x": [0, 1, 2]}}, "seed": {fmt_number(log.seeds[i])}}}'
        )
    experiment = ", ".join(f'{{"{k}": "{v}"}}' for k, v in log.experiment)
    attributes = ", ".join(f'"{n}"' for n in names)
    text = (
        '{\n\t"version": "0.4.0", \n\t"suite": "S", \n\t"function_id": 1, '
        '\n\t"function_name": "F", '
        f'\n\t"maximization": {str(log.maximization).lower()}, '
        + ('\n\t"precision": "%.17g", ' if log.full_precision else "")
        + f'\n\t"algorithm": {{"name": "{log.algorithm}", "info": "{log.info}"}},'
        + (f'\n\t"experiment_attributes": [{experiment}],' if log.experiment else "")
        + '\n\t"run_attributes": ["seed"],'
        + f'\n\t"attributes": [{attributes}],'
        + '\n\t"scenarios": [\n\t\t{"dimension": 3,'
        + '\n\t\t"path": "data_f1_F/IOHprofiler_f1_DIM3.dat",'
        + '\n\t\t"runs": [\n\t\t\t'
        + ",\n\t\t\t".join(json_runs)
        + "\n\t\t]}\n\t]\n}\n"
    )
    (folder / "IOHprofiler_f1_F.json").write_text(text)
    return folder


def expected(log: Log) -> list[list[tuple]]:
    out = []
    for run in log.runs:
        rows = []
        for e, y, s in run:
            row = (e, float(dat_value(y, log.full_precision)))
            if log.sigma and any(log.runs):
                row += (None if s is None else float(dat_value(s, log.full_precision)),)
            rows.append(row)
        out.append(rows)
    return out


def same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return a == b and math.copysign(1, a) == math.copysign(1, b)
    return a == b


def rows_by_run(
    runs: pl.DataFrame, trace: pl.DataFrame, columns: list[str]
) -> list[list[tuple]]:
    out = []
    for rid in runs.sort("run_index")["run_id"]:
        part = trace.filter(pl.col("run_id") == rid).sort("row")
        out.append(part.select(columns).rows())
    return out


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(logs())
def test_generated_logs_read_back_exactly(log):
    with tempfile.TemporaryDirectory() as tmp:
        folder = write(log, Path(tmp) / "src")
        result = read_ioh(folder, "hyp")
        has_rows = any(log.runs)
        columns = ["evaluations", "raw_y"] + (
            ["sigma"] if log.sigma and has_rows else []
        )
        want = expected(log)
        n_runs = max(len(log.runs), log.json_runs)
        assert result.runs.height == n_runs
        got = rows_by_run(result.runs, result.trace, columns)
        want += [[] for _ in range(n_runs - len(want))]
        assert len(got) == len(want)
        for g_run, w_run in zip(got, want):
            assert len(g_run) == len(w_run)
            for g, w in zip(g_run, w_run):
                assert all(same(x, y) for x, y in zip(g, w)), (g, w)
        d = result.descriptors
        names = dict(zip(d["key"], d["value"]))
        assert d.height == 0 or names["algorithm_name"] == log.algorithm
        assert d.height == 0 or names["algorithm_info"] == log.info
        for key, value in log.experiment if n_runs else []:
            # runs columns are reserved, so these experiment attributes are renamed
            key = f"{key}_experiment" if key in ("dimension", "budget") else key
            assert names[key] == value
        assert result.runs["maximization"].to_list() == [log.maximization] * n_runs

        result.write(Path(tmp) / "ds")
        stored = scan_dataset(Path(tmp) / "ds")
        stored_runs = stored.runs.collect()
        stored_rows = rows_by_run(stored_runs, stored.trace.collect(), columns)
        assert len(stored_rows) == len(got)
        for s_run, g_run in zip(stored_rows, got):
            assert all(same(x, y) for s, g in zip(s_run, g_run) for x, y in zip(s, g))
