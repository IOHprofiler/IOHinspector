import os
import json
import warnings
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from .align import turbo_align
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
    try:
        return eval(value)
    except:
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
        if not os.path.isfile(data["path"]):
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

    def load(self, monotonic=False, maximize=True, x_values = None) -> pl.DataFrame:
        """Loads the data file stored at self.data_file to a pd.DataFrame"""

        with open(self.data_file) as f:
            header = next(f).strip().split()
        key_lookup = dict([(r.id, r.data_id) for r in self.runs])
        dt = (
            pl.scan_csv(
                self.data_file,
                separator=" ",
                decimal_comma=True,
                schema={header[0]: pl.Float64, **dict.fromkeys(header[1:], pl.Float64)},
                ignore_errors=True,
                
            )
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

    def load_coco(self, monotonic=False, maximize=True, x_values=None) -> pl.DataFrame:
        """Loads the data file stored at self.data_file to a pd.DataFrame"""

        with open(self.data_file) as f:
            header = process_header(next(f))
            nextline = next(f).strip().split()
        if(len(nextline) > len(header)):
            for i in range(self.dimension):
                header.append(f"x{i}")
        key_lookup = dict([(r.id, r.data_id) for r in self.runs])
        dt = (
            pl.scan_csv(
                self.data_file,
                has_header=False,
                comment_prefix="%",
                separator=" ",
                decimal_comma=True,
                schema={header[0]: pl.Float64, **dict.fromkeys(header[1:], pl.Float64)},
                ignore_errors=True,
                truncate_ragged_lines=True
            )
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
            with open(json_file) as f:
                data = json.load(f)
                return Dataset.from_dict(data, json_file)
        except Exception:
            return None

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

        records = []
        for scen in self.scenarios:
            for run in scen.runs:
                records.append(
                    [run.data_id]
                    + meta_data
                    + [scen.dimension, run.instance, run.id, run.evals, run.best.y]
                    + exattr_values
                )
        return pl.DataFrame(records, schema=METADATA_SCHEMA + exattr_schema, orient="row") 

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
            experiment_attributes = [tuple(x.items())[0] for x in data["experiment_attributes"]]
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
                    warnings.warn(f"{coco_info_file} is empty, cannot parse COCO text format")
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
            re.DOTALL | re.MULTILINE
        )

        pattern_run = re.compile(r"(\d+):(\d+)\|([-+eE0-9.]+)")
        
        scenarios = []
        algorithms = set()
        function_ids = set()
        suites = set()
        for match in pattern_block.finditer(coco_text):
            metadata = match.groupdict()
            dim = int(metadata['DIM'])
            metadata['filename'] = metadata['filename'].replace("\\", "/")
            if metadata['suite']:
                suites.add(metadata['suite'])
            runs_data = pattern_run.findall(metadata.pop('runs'))
            runs = []
            for run_id, run in enumerate(runs_data):
                _, evals, y = run
                solution = Solution(
                    evals=int(evals),
                    x=np.array([None] * dim),
                    y=float(y)
                )
                run = Run(
                    data_id=Run.hash(
                        f"{os.path.join(os.path.dirname(filepath), metadata['filename'])}_{run_id+1}"
                    ),
                    id=run_id+1,
                    instance=0,  # Instance is not provided in the COCO text format
                    evals=int(evals),
                    best=solution
                )
                runs.append(run)

            scenario = Scenario(
                dimension=int(metadata['DIM']),
                data_file=os.path.join(os.path.dirname(filepath), metadata['filename']),
                runs=runs
            )
            scenarios.append(scenario)

            algorithms.add(metadata['algId'])
            function_ids.add(int(metadata['funcId']))
        if len(algorithms) != 1:
            raise ValueError("Multiple algorithms found in COCO text, expected one.")
        if len(function_ids) != 1:
            raise ValueError("Multiple function ids found in COCO text, expected one.")
        algorithm = Algorithm(
            name=algorithms.pop(),
            info="algorithm_info"  # Assuming no additional info in COCO text
        )
        function = Function(
            id=function_ids.pop(),
            name=None,
            maximization=False  # Assuming minimization, adjust as needed
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
            scenarios=scenarios
        )

def process_header(line: str):

    header = line.strip().replace("%","").split("|")
    header = [re.sub(r"\s*\(.*?\)", "", h).strip() for h in header if h.strip()]
    header = [h for h in header if not h.startswith("x")]
    
    raw_y_headers = [
        "best noise-free fitness"
        ]
    for i, _ in enumerate(header):
        if any(raw_y_header in header[i] for raw_y_header in raw_y_headers):
            header[i] = "raw_y"
            break

    evaluations_headers = [
        "function evaluation",
        "f evaluations"
    ]
    for i, _ in enumerate(header):
        if any(evaluations_header in header[i] for evaluations_header in evaluations_headers):
            header[i] = "evaluations"
            break
    
    return header

