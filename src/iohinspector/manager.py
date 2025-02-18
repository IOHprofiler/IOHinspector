import os
import warnings
from dataclasses import dataclass, field
from copy import deepcopy

import polars as pl

from .data import Dataset, Function, Algorithm, METADATA_SCHEMA
from glob import glob

@dataclass
class DataManager:
    data_sets: list[Dataset] = field(default_factory=list, repr=None)
    overview: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema=METADATA_SCHEMA)
    )

    def __post_init__(self):
        for data_set in self.data_sets:
            self.extend_overview(data_set)
            
    def add_folders(self, folders: list[str]):
        """Utility loop for adding multiple folders"""
        
        for folder in folders:
            self.add_folder(folder)

    def add_folder(self, folder_name: str):
        """Add a folder with ioh generated data"""

        if not os.path.isdir(folder_name):
            raise FileNotFoundError(f"{folder_name} not found")

        json_files = glob(f"{folder_name}/**/*.json", recursive = True)
        if not any(json_files):
            raise FileNotFoundError(f"{folder_name} does not contain any json files")

        datasets = [Dataset.from_json(json_file) for json_file in json_files]

        for ds in datasets:
            self.data_sets.append(ds)

        ds_overviews = pl.concat([ds.overview for ds in datasets], how='diagonal_relaxed')
        self.overview = pl.concat([ds_overviews, self.overview], how='diagonal_relaxed')

    def add_json(self, json_file: str):
        """Add a single json file with ioh generated data"""

        if any((d.file == json_file) for d in self.data_sets):
            warnings.warn(
                f"{json_file} is already loaded. Skipping file", RuntimeWarning
            )
            return
        data_set = Dataset.from_json(json_file)
        self.add_data_set(data_set)

    def extend_overview(self, data_set: Dataset):
        """ "Include a new data set in the manager"""
        columns = set(self.overview.schema.keys())
        ds_columns = set(data_set.overview.schema.keys())

        new_columns = list(ds_columns - columns)
        missing_columns = list(columns - ds_columns)

        ds_overview = data_set.overview.with_columns(
            *(
                pl.lit(None, dtype=self.overview.schema[name]).alias(name)
                for name in missing_columns
            )
        )
        self.overview = self.overview.with_columns(
            *(
                pl.lit(None, dtype=data_set.overview.schema[name]).alias(name)
                for name in new_columns
            )
        )
        self.overview = self.overview.extend(ds_overview.select(self.overview.columns))

    def add_data_set(self, data_set: Dataset):
        """Only use this to add data sets"""
        self.data_sets.append(data_set)
        self.extend_overview(data_set)

    def __add__(self, other: "DataManager") -> "DataManager":
        # TODO: filter on overlap
        return DataManager(deepcopy(self.data_sets) + deepcopy(other.data_sets))

    def select(
        self,
        data_ids: list[int] = None,
        function_ids: list[int] = None,
        algorithms: list[str] = None,
        experiment_attributes: list[tuple[str, str]] = None,
        data_attributes: list[str] = None,
        dimensions: list[int] = None,
        instances: list[int] = None,
    ) -> "DataManager":

        selected_data_sets = deepcopy(self.data_sets)

        if data_ids is not None:
            for dset in selected_data_sets:
                for scen in dset.scenarios:
                    scen.runs = [run for run in scen.runs if run.data_id in data_ids]

                dset.scenarios = [scen for scen in dset.scenarios if any(scen.runs)]
            return DataManager([x for x in selected_data_sets if any(x.scenarios)])

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
    def n_runs(self):
        return len(self.overview)

    @property
    def any(self):
        return len(self.data_sets) != 0

    def load(
        self,
        monotonic: bool = True,
        include_meta_data: bool = False,
        include_columns: list[str] = None,
        x_values: list[str] = None
    ) -> pl.DataFrame:
        if not self.any:
            return pl.DataFrame()

        data = []
        for data_set in self.data_sets:
            for scen in data_set.scenarios:
                df = scen.load(monotonic, data_set.function.maximization, x_values)
                data.append(df)

        data = pl.concat(data)

        if include_meta_data or include_columns is not None:
            if include_columns is None:
                include_columns = self.overview.columns

            for c in ("data_id", "run_id",):
                if c not in include_columns:
                    include_columns.append(c)
            
            data = self.overview.select(include_columns).join(
                data, on=("data_id", "run_id")
            )
        return data
