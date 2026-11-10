import ast
import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from .align import turbo_align
from .io.ioh import IohJson, IohScenario, load_ioh_json, parse_dat
import re

METADATA_SCHEMA = [
    ("data_id", pl.UInt64),
    ("algorithm_name", pl.String),
    ("algorithm_info", pl.String),
    ("suite", pl.String),
    ("function_name", pl.String),
    ("function_id", pl.UInt16),
    ("dimension", pl.UInt16),
    ("instance", pl.UInt16),
    ("run_id", pl.UInt32),
    ("evals", pl.UInt64),
    ("best_y", pl.Float64),
]


def check_keys(data: dict, required_keys: list[str]):
    for key in required_keys:
        if key not in data:
            raise ValueError(
                f"data dict doesn't contain ioh format required key: {key}"
            )


def try_eval(value: str):
    """The Python literal ``value`` spells, or ``value`` itself."""
    try:
        return ast.literal_eval(value)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return value


def get_polars_type(value):
    if isinstance(value, bool):
        return pl.Boolean
    if isinstance(value, int):
        return pl.Int64
    if isinstance(value, float):
        return pl.Float64
    if isinstance(value, str):
        return pl.String

    warnings.warn(f"{type(value)} is not mapped to polars dtype", UserWarning)
    return pl.Object


@dataclass
class Function:
    id: int
    name: str
    maximization: bool


@dataclass
class Algorithm:
    name: str
    info: str


@dataclass
class Solution:
    evals: int
    x: np.ndarray = field(repr=None)
    y: float


@dataclass
class Run:
    data_id: int
    id: int
    instance: int
    evals: int
    best: Solution
    #: Run attributes from the JSON (IOHexperimenter writes them as doubles).
    attributes: dict = field(default_factory=dict)

    __lookup__ = {}
    __current_id__ = 1

    @staticmethod
    def hash(key: str):
        if value := Run.__lookup__.get(key):
            return value
        Run.__lookup__[key] = Run.__current_id__
        Run.__current_id__ += 1
        return Run.__lookup__[key]


@dataclass
class Scenario:
    dimension: int
    data_file: str
    runs: list[Run]
    #: The JSON's logged properties; ``.dat`` columns after them are positions.
    attributes: list[str] | None = None

    @staticmethod
    def from_ioh(scen: IohScenario, meta: IohJson) -> "Scenario":
        """A scenario of a parsed JSON file (``iohinspector.io.ioh``).

        Runs are matched to the ``.dat`` file's runs in order; runs that only the
        ``.dat`` file has get no instance and no evaluation count.
        """
        path = os.path.join(os.path.dirname(meta.path), scen.path)
        if not os.path.isfile(path) and not os.path.isfile(
            path.replace(".dat", ".parquet")
        ):
            raise FileNotFoundError(f"{path} is not found")
        n_dat = count_dat_runs(path) if os.path.isfile(path) else 0
        runs = []
        for index in range(max(n_dat, len(scen.runs))):
            info = scen.runs[index] if index < len(scen.runs) else None
            runs.append(
                Run(
                    Run.hash(f"{path}_{index + 1}"),
                    index + 1,
                    None if info is None else info.instance,
                    None if info is None else info.evals,
                    best=Solution(
                        None if info is None else info.best_evals,
                        None if info is None else np.array(info.best_x),
                        None if info is None else info.best_y,
                    ),
                    attributes={} if info is None else dict(info.attributes),
                )
            )
        return Scenario(scen.dimension, path, runs, meta.attributes)

    @staticmethod
    def from_dict(data: dict, dirname: str):
        """Constructs a Scenario object from a dictionary
        (output of json.load from ioh compatible file)
        """

        required_keys = (
            "dimension",
            "path",
            "runs",
        )
        check_keys(data, required_keys)

        data["path"] = os.path.join(dirname, data["path"])
        parquet_path = data["path"].replace(".dat", ".parquet")
        if not os.path.isfile(data["path"]) and not os.path.isfile(parquet_path):
            raise FileNotFoundError(f"{data['path']} is not found")

        return Scenario(
            data["dimension"],
            data["path"],
            [
                Run(
                    Run.hash(f"{data['path']}_{run_id}"),
                    run_id,
                    run["instance"],
                    run["evals"],
                    best=Solution(**run["best"]),
                )
                for run_id, run in enumerate(data["runs"], 1)
            ],
        )

    def scan_ioh(self, header: list[str]):
        parquet_file = self.data_file.replace(".dat", ".parquet")
        if os.path.isfile(parquet_file):
            try:
                lf = pl.scan_parquet(parquet_file)
                # Fail fast so we can fall back here if the parquet is corrupt or incompatible.
                if not set(header).issubset(set(lf.collect_schema().names())):
                    raise ValueError(
                        f"Parquet file is missing required columns: {header}"
                    )
                lf.limit(1).collect()
                return lf
            except Exception as e:
                warnings.warn(
                    f"Failed to read {parquet_file} as parquet file: {e}. Falling back to reading the original data file."
                )
        if not os.path.isfile(self.data_file):
            raise FileNotFoundError(
                f"No data file found for {self.data_file!r}: neither .parquet nor .dat exists."
            )
        return pl.scan_csv(
            self.data_file,
            separator=" ",
            decimal_comma=True,
            schema={header[0]: pl.Float64, **dict.fromkeys(header[1:], pl.Float64)},
            ignore_errors=True,
        )


    def scan_coco(self, header: list[str]):
        return pl.scan_csv(
            self.data_file,
            has_header=False,
            comment_prefix="%",
            separator=" ",
            decimal_comma=True,
            schema={header[0]: pl.Float64, **dict.fromkeys(header[1:], pl.Float64)},
            ignore_errors=True,
            truncate_ragged_lines=True,
        )

    def extract_header(self, is_coco: bool = True) -> list[str]:
        if not is_coco:
            parquet_file = self.data_file.replace(".dat", ".parquet")
            if not os.path.isfile(self.data_file) and os.path.isfile(parquet_file):
                return pl.scan_parquet(parquet_file).collect_schema().names()
            with open(self.data_file) as f:
                return next(f).strip().split()
        with open(self.data_file) as f:
            header = process_header(next(f))
            nextline = next(f).strip().split()
            if len(nextline) > len(header):
                for i in range(self.dimension):
                    header.append(f"x{i}")
            return header

    def load(
        self, monotonic=False, maximize=True, x_values=None, is_coco: bool = False
    ) -> pl.DataFrame:
        """Loads the data file stored at self.data_file to a pd.DataFrame

        An IOHexperimenter ``.dat`` file is read with ``iohinspector.io.ioh``:
        runs are split on header lines and every row is kept. ``monotonic`` then
        keeps, per run, the rows where the best-so-far value changes plus the
        run's final row.
        """
        if not is_coco and not os.path.isfile(self.data_file.replace(".dat", ".parquet")):
            dt = self.load_dat()
            if monotonic or x_values is not None:
                dt = monotone_per_run(dt, maximize)
            if x_values is not None:
                dt = turbo_align(dt, x_values)
            return dt

        key_lookup = dict([(r.id, r.data_id) for r in self.runs])
        header = self.extract_header(is_coco)

        dt = (
            (self.scan_coco(header) if is_coco else self.scan_ioh(header))
            .with_columns(
                pl.col("evaluations").cast(pl.UInt64),
                run_id=(pl.col("evaluations") == 1).cum_sum(),
            )
            .drop_nulls()
            .filter(pl.col("run_id").is_in([r.id for r in self.runs]))
            .with_columns(
                data_id=pl.col("run_id").map_elements(
                    key_lookup.__getitem__, return_dtype=pl.UInt64
                )
            )
        )

        if monotonic or x_values is not None:
            if maximize:
                dt = dt.with_columns(pl.col("raw_y").cum_max().over("run_id"))
            else:
                dt = dt.with_columns(pl.col("raw_y").cum_min().over("run_id"))

            dt = dt.filter(pl.col("raw_y").diff().fill_null(1.0).abs() > 0.0)

        dt = dt.collect()

        if x_values is not None:
            dt = turbo_align(dt, x_values)

        return dt


    def load_dat(self) -> pl.DataFrame:
        """Every row of the IOHexperimenter ``.dat`` file of the selected runs.

        Columns: ``evaluations``, ``raw_y``, the other logged properties, the
        positions ``x0`` ... when logged (``position_x0`` ... when a property is
        named like one of them), ``run_id`` (1-based, in file order) and ``data_id``.
        """
        dat = parse_dat(
            self.data_file, dimension=self.dimension, attributes=self.attributes
        )
        lookup = {r.id: r.data_id for r in self.runs}
        df = dat.trace
        if dat.positions is not None:
            taken = set(df.columns)
            xs = [c for c in dat.positions.columns if c not in ("run_index", "row")]
            pos = dat.positions
            if taken.intersection(xs):
                pos = pos.rename({c: f"position_{c}" for c in xs})
            df = df.join(pos, on=["run_index", "row"], how="left")
        df = df.with_columns(
            pl.col("evaluations").cast(pl.UInt64),
            run_id=(pl.col("run_index") + 1).cast(pl.UInt32),
        ).filter(pl.col("run_id").is_in(list(lookup)))
        return df.with_columns(
            data_id=pl.col("run_id").replace_strict(lookup, return_dtype=pl.UInt64)
        ).drop("run_index", "row")


def count_dat_runs(path: str) -> int:
    """The number of runs in an IOHexperimenter ``.dat`` file: its header lines."""
    count = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            first = line.lstrip()[:1]
            if first and not (first.isdigit() or first in "+-."):
                count += 1
    return count


def monotone_per_run(dt: pl.DataFrame, maximize: bool) -> pl.DataFrame:
    """Best-so-far ``raw_y`` per run; keeps the rows where it changes and the final row."""
    best = pl.col("raw_y").cum_max() if maximize else pl.col("raw_y").cum_min()
    dt = dt.with_columns(raw_y=best.over("run_id"))
    changed = (pl.col("raw_y") != pl.col("raw_y").shift(1)).over("run_id").fill_null(True)
    last = (pl.int_range(pl.len()) == pl.len() - 1).over("run_id")
    return dt.filter(changed | last)


@dataclass
class Dataset:
    source: str
    file: str
    version: str
    suite: str
    function: Function
    algorithm: Algorithm
    experiment_attributes: list[tuple[str, str]]
    data_attributes: list[str]
    scenarios: list[Scenario]

    @staticmethod
    def from_json(json_file: str):
        """Construct a dataset object from a json file"""

        if not os.path.isfile(json_file):
            raise FileNotFoundError(f"{json_file} not found")
        try:
            return Dataset.from_ioh(load_ioh_json(json_file))
        except (ValueError, OSError) as e:
            warnings.warn(f"Failed to load {json_file}: {e}")
            return None

    @staticmethod
    def from_ioh(meta: IohJson) -> "Dataset":
        """A dataset from a parsed JSON file (``iohinspector.io.ioh.load_ioh_json``)."""
        names = {name for name, _ in METADATA_SCHEMA}
        experiment_attributes = []
        for name, value in meta.experiment_attributes:
            if name in names:
                warnings.warn(
                    f"Experiment attribute '{name}' is already present in the "
                    "metadata schema. It will be renamed to avoid conflicts."
                )
                name = f"{name}_exp_attr"
            experiment_attributes.append((name, value))
        return Dataset(
            "ioh",
            str(meta.path),
            meta.version,
            meta.suite,
            Function(meta.function_id, meta.function_name, meta.maximization),
            Algorithm(meta.algorithm_name, meta.algorithm_info),
            experiment_attributes or None,
            meta.attributes or [],
            [Scenario.from_ioh(scen, meta) for scen in meta.scenarios],
        )

    @property
    def overview(self) -> pl.DataFrame:
        meta_data = [
            self.algorithm.name,
            self.algorithm.info,
            self.suite,
            self.function.name,
            self.function.id,
        ]
        if self.experiment_attributes:
            exattr_names, exattr_values = zip(*self.experiment_attributes)
            exattr_values = list(map(try_eval, exattr_values))
            exattr_schema = [
                (name, get_polars_type(value))
                for name, value in zip(exattr_names, exattr_values)
            ]
        else:
            exattr_values = []
            exattr_schema = []

        runattr_names = list(
            dict.fromkeys(
                name for scen in self.scenarios for run in scen.runs for name in run.attributes
            )
        )
        taken = {name for name, _ in METADATA_SCHEMA + exattr_schema}
        runattr_schema = [
            (f"{name}_run_attr" if name in taken else name, pl.Float64)
            for name in runattr_names
        ]

        records = []
        for scen in self.scenarios:
            for run in scen.runs:
                records.append(
                    [run.data_id]
                    + meta_data
                    + [scen.dimension, run.instance, run.id, run.evals, run.best.y]
                    + exattr_values
                    + [run.attributes.get(name) for name in runattr_names]
                )
        return pl.DataFrame(
            records,
            schema=METADATA_SCHEMA + exattr_schema + runattr_schema,
            orient="row",
        )

    @staticmethod
    def from_dict(data: dict, filepath: str):
        """Constructs a Dataset object from a dictionary
        (output of json.load from ioh compatible file)
        """

        required_keys = (
            "version",
            "suite",
            "function_id",
            "function_name",
            "maximization",
            "algorithm",
            # "experiment_attributes",
            "attributes",
            "scenarios",
        )
        check_keys(data, required_keys)

        if "experiment_attributes" in data:
            experiment_attributes = [
                tuple(x.items())[0] for x in data["experiment_attributes"]
            ]
            metadata_col_names = {col_name for col_name, _ in METADATA_SCHEMA}
            for i, (name, value) in enumerate(experiment_attributes):
                if name in metadata_col_names:
                    warnings.warn(
                        f"Experiment attribute '{name}' is already present in the metadata schema. It will be renamed to avoid conflicts."
                    )
                    experiment_attributes[i] = (f"{name}_exp_attr", value)
        else:
            experiment_attributes = None

        return Dataset(
            "ioh",
            filepath,
            data["version"],
            data["suite"],
            Function(data["function_id"], data["function_name"], data["maximization"]),
            Algorithm(
                data["algorithm"]["name"],
                data["algorithm"]["info"],
            ),
            experiment_attributes,
            data["attributes"],
            [
                Scenario.from_dict(scen, os.path.dirname(filepath))
                for scen in data["scenarios"]
            ],
        )

    @staticmethod
    def from_coco_info(coco_info_file: str):
        """Construct a dataset object from a json file"""

        if not os.path.isfile(coco_info_file):
            raise FileNotFoundError(f"{coco_info_file} not found")
        try:
            with open(coco_info_file, "r") as f:
                data = f.read()
                if len(data.strip()) == 0:
                    warnings.warn(
                        f"{coco_info_file} is empty, cannot parse COCO text format"
                    )
                    return None
                return Dataset.from_coco_text(data, coco_info_file)
        except Exception as e:
            warnings.warn(f"Failed to parse {coco_info_file} as COCO text format: {e}")
            return None

    @staticmethod
    def from_coco_text(coco_text: str, filepath: str):
        pattern_block = re.compile(
            r"^(?:suite\s*=\s*'(?P<suite>[^']+)',\s*)?"
            r"funcId\s*=\s*(?P<funcId>\d+),\s*"
            r"DIM\s*=\s*(?P<DIM>\d+),\s*"
            r"Precision\s*=\s*(?P<precision>[0-9.eE+-]+),\s*"
            r"algId\s*=\s*'(?P<algId>[^']+)'"
            r"(?:,\s*coco_version\s*=\s*'(?P<coco_version>[^']*)')?"
            r"(?:,\s*logger\s*=\s*'(?P<logger>[^']*)')?"
            r"(?:,\s*data_format\s*=\s*'(?P<data_format>[^']*)')?"
            r"(?:\n%[^\n]*)?"
            r"(?:\n(?P<filename>[^\s]+),\s*(?P<runs>.+?)(?=suite\s*=|funcId\s*=|\Z))?",
            re.DOTALL | re.MULTILINE,
        )

        pattern_run = re.compile(r"(\d+):(\d+)\|([-+eE0-9.]+)")

        scenarios = []
        algorithms = set()
        function_ids = set()
        suites = set()
        for match in pattern_block.finditer(coco_text):
            metadata = match.groupdict()
            dim = int(metadata["DIM"])
            metadata["filename"] = metadata["filename"].replace("\\", "/")
            if metadata["suite"]:
                suites.add(metadata["suite"])
            runs_data = pattern_run.findall(metadata.pop("runs"))
            runs = []
            for run_id, run in enumerate(runs_data):
                _, evals, y = run
                solution = Solution(
                    evals=int(evals), x=np.array([None] * dim), y=float(y)
                )
                run = Run(
                    data_id=Run.hash(
                        f"{os.path.join(os.path.dirname(filepath), metadata['filename'])}_{run_id+1}"
                    ),
                    id=run_id + 1,
                    instance=0,  # Instance is not provided in the COCO text format
                    evals=int(evals),
                    best=solution,
                )
                runs.append(run)

            scenario = Scenario(
                dimension=int(metadata["DIM"]),
                data_file=os.path.join(os.path.dirname(filepath), metadata["filename"]),
                runs=runs,
            )
            scenarios.append(scenario)

            algorithms.add(metadata["algId"])
            function_ids.add(int(metadata["funcId"]))
        if len(algorithms) != 1:
            raise ValueError("Multiple algorithms found in COCO text, expected one.")
        if len(function_ids) != 1:
            raise ValueError("Multiple function ids found in COCO text, expected one.")
        algorithm = Algorithm(
            name=algorithms.pop(),
            info="algorithm_info",  # Assuming no additional info in COCO text
        )
        function = Function(
            id=function_ids.pop(),
            name=None,
            maximization=False,  # Assuming minimization, adjust as needed
        )

        # Assuming no experiment attributes and data attributes in COCO text
        experiment_attributes = []

        # Read header from the first scenario's data file, if available
        if scenarios and os.path.isfile(scenarios[0].data_file):
            with open(scenarios[0].data_file, "r") as f:
                first_line = next(f)
            data_attributes = process_header(first_line)

        else:
            data_attributes = []

        return Dataset(
            source="coco",
            file=filepath,
            version="0.0.1",
            suite="unknown_suite" if not suites else suites.pop(),
            function=function,
            algorithm=algorithm,
            experiment_attributes=experiment_attributes,
            data_attributes=data_attributes,
            scenarios=scenarios,
        )


def process_header(line: str):

    header = line.strip().replace("%", "").split("|")
    header = [re.sub(r"\s*\(.*?\)", "", h).strip() for h in header if h.strip()]
    header = [h for h in header if not h.startswith("x")]

    raw_y_headers = ["best noise-free fitness"]
    for i, _ in enumerate(header):
        if any(raw_y_header in header[i] for raw_y_header in raw_y_headers):
            header[i] = "raw_y"
            break

    evaluations_headers = ["function evaluation", "f evaluations"]
    for i, _ in enumerate(header):
        if any(
            evaluations_header in header[i]
            for evaluations_header in evaluations_headers
        ):
            header[i] = "evaluations"
            break

    return header
