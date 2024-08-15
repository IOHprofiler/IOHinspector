import os
import warnings
from dataclasses import dataclass, field
from copy import deepcopy

import polars as pl

from .data import Dataset, Function, Algorithm, METADATA_SCHEMA


@dataclass
class DataManager:
    data_sets: list[Dataset] = field(default_factory=list, repr=None)
    overview: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema=METADATA_SCHEMA)
    )

    def __post_init__(self):
        for data_set in self.data_sets:
            self.overview = self.overview.extend(data_set.overview)

    def add_folder(self, folder_name: str):
        """Add a folder with ioh generated data"""

        if not os.path.isdir(folder_name):
            raise FileNotFoundError(f"{folder_name} not found")

        json_files = [
            fname
            for f in os.listdir(folder_name)
            if os.path.isfile(fname := os.path.join(folder_name, f))
            and f.endswith(".json")
        ]
        if not any(json_files):
            raise FileNotFoundError(f"{folder_name} does not contain any json files")

        for json_file in json_files:
            self.add_json(json_file)

    def add_json(self, json_file: str):
        """Add a single json file with ioh generated data"""

        if any((d.file == json_file) for d in self.data_sets):
            warnings.warn(
                f"{json_file} is already loaded. Skipping file", RuntimeWarning
            )
            return
        data_set = Dataset.from_json(json_file)
        self.add_data_set(data_set)

    def add_data_set(self, data_set: Dataset):
        """Only use this to add data sets"""
        self.data_sets.append(data_set)
        self.overview = self.overview.extend(data_set.overview)

    def __add__(self, other: "DataManager") -> "DataManager":
        # TODO: filter on overlap
        return DataManager(deepcopy(self.data_sets) + deepcopy(other.data_sets))

    def select(
        self,
        function_ids: list[int] = None,
        algorithms: list[str] = None,
        experiment_attributes: list[tuple[str, str]] = None,
        data_attributes: list[str] = None,
        dimensions: list[int] = None,
        instances: list[int] = None,
    ) -> "DataManager":
        selected_data_sets = deepcopy(self.data_sets)

        ## dataset filters
        if function_ids is not None:
            selected_data_sets = [
                x for x in selected_data_sets if x.function.id in function_ids
            ]

        if algorithms is not None:
            selected_data_sets = [
                x for x in selected_data_sets if x.algorithm.name in algorithms
            ]

        if experiment_attributes is not None:
            for attr in experiment_attributes:
                selected_data_sets = [
                    x for x in selected_data_sets if attr in x.experiment_attributes
                ]

        if data_attributes is not None:
            for attr in data_attributes:
                selected_data_sets = [
                    x for x in selected_data_sets if attr in x.data_attributes
                ]

        ## scenario_filters
        if dimensions is not None:
            for dset in selected_data_sets:
                dset.scenarios = [
                    scen for scen in dset.scenarios if scen.dimension in dimensions
                ]

        ## run filter
        if instances is not None:
            for dset in selected_data_sets:
                for scen in dset.scenarios:
                    scen.runs = [run for run in scen.runs if run.instance in instances]

        return DataManager(selected_data_sets)
    
    def select_indexes(self, idxs):
        return DataManager([self.data_sets[idx] for idx in idxs])

    @property
    def functions(self) -> tuple[Function]:
        return tuple([x.function for x in self.data_sets])

    @property
    def algorithms(self) -> tuple[Algorithm]:
        return tuple([x.algorithm for x in self.data_sets])

    @property
    def experiment_attributes(self) -> tuple[tuple[str, str]]:
        attrs = []
        for data_set in self.data_sets:
            for attr in data_set.experiment_attributes:
                if attr not in attrs:
                    attrs.append(attr)
        return tuple(attrs)

    @property
    def data_attributes(self) -> tuple[str]:
        attrs = []
        for data_set in self.data_sets:
            for attr in data_set.data_attributes:
                if attr not in attrs:
                    attrs.append(attr)
        return tuple(attrs)

    @property
    def dimensions(self) -> tuple[int]:
        dims = []
        for data_set in self.data_sets:
            for scen in data_set.scenarios:
                if scen.dim not in dims:
                    dims.append(scen.dim)
        return tuple(dims)

    @property
    def instances(self) -> tuple[int]:
        iids = []
        for data_set in self.data_sets:
            for scen in data_set.scenarios:
                for run in scen.runs:
                    if run.instance not in iids:
                        iids.append(run.instancem)
        return tuple(iids)

    @property
    def any(self):
        return len(self.data_sets) != 0

    def load(
        self,
        monotonic: bool = True,
        include_meta_data: bool = False,
    ) -> pl.DataFrame:
        if not self.any:
            return pl.DataFrame()
        
        data = []  
        for data_set in self.data_sets:
            for scen in data_set.scenarios:
                df = scen.load(monotonic, data_set.function.maximization)
                data.append(df)
                
        data = pl.concat(data)  
        
        if include_meta_data:
            data = self.overview.join(data, on=("data_id", "run_id"))   
        return data