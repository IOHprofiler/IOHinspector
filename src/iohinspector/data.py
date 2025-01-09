import os
import json
import warnings
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from .align import turbo_align

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


@dataclass
class Dataset:
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

        with open(json_file) as f:
            data = json.load(f)
            return Dataset.from_dict(data, json_file)

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
