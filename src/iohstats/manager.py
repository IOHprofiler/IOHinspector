import os

import warnings
from dataclasses import dataclass, field
from copy import deepcopy

from .data import Dataset


@dataclass
class DataManager:
    data_sets: list[Dataset] = field(default_factory=list)

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

        self.data_sets.append(Dataset.from_json(json_file))

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
                dset.scenarios = [scen for scen in dset.scenarios if scen.dimension in dimensions]
                
        ## run filter
        if instances is not None:
            for dset in selected_data_sets:
                for scen in dset.scenarios:
                    scen.runs = [run for run in scen.runs if run.instance in instances]

        return DataManager(selected_data_sets)
    
    def any(self):
        return len(self.data_sets) != 0
    
    def load(self):
        breakpoint()
