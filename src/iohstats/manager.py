import os

import warnings
from dataclasses import dataclass, field

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


    def select(self, function_id: int, **kwargs):
        pass
