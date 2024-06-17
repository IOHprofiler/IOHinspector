import os
import json
from dataclasses import dataclass, field

import numpy as np
import polars as pl


def check_keys(data: dict, required_keys: list[str]):
    for key in required_keys:
        if key not in data:
            raise ValueError(
                f"data dict doesn't contain ioh format required key: {key}"
            )


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
    instance: int
    evals: int
    best: Solution


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
                Run(run["instance"], run["evals"], best=Solution(**run["best"]))
                for run in data["runs"]
            ],
        )

    def load(self) -> pl.DataFrame:
        """Loads the data file stored at self.data_file to a pd.DataFrame"""
        with open(self.data_file) as f:
            header = next(f).strip().split()

        dt = (
            pl.read_csv(
                self.data_file,
                separator=" ",
                decimal_comma=True,
                schema={header[0]: pl.Float64, **dict.fromkeys(header[1:], pl.Float64)},
                ignore_errors=True,
            )
            .drop_nulls()
            .with_columns(
                pl.col("evaluations").cast(pl.UInt64),
                run_id=(pl.col("evaluations") == 1).cum_sum(),
            )
        )
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
            "experiment_attributes",
            "attributes",
            "scenarios",
        )
        check_keys(data, required_keys)

        return Dataset(
            filepath,
            data["version"],
            data["suite"],
            Function(data["function_id"], data["function_name"], data["maximization"]),
            Algorithm(
                data["algorithm"]["name"],
                data["algorithm"]["info"],
            ),
            [tuple(x.items()) for x in data["experiment_attributes"]],
            data["attributes"],
            [
                Scenario.from_dict(scen, os.path.dirname(filepath))
                for scen in data["scenarios"]
            ],
        )

    def load_scenario(self, dimension: int):
        pass
