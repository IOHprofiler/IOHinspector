import os
import unittest
import warnings

import polars as pl

from iohstats import DataManager

from pprint import pprint

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.realpath(os.path.join(BASE_DIR, "data"))


class TestManager(unittest.TestCase):

    def setUp(self):
        self.data_dir, *_ = os.listdir(DATA_DIR)
        self.data_dir = os.path.join(DATA_DIR, self.data_dir)
        self.json_files = [
            fname
            for f in os.listdir(self.data_dir)
            if os.path.isfile((fname := os.path.join(self.data_dir, f)))
        ]

    def test_add_json(self):
        manager = DataManager()
        manager.add_json(self.json_files[0])
        data = manager.data_sets[0]
        df = data.scenarios[0].load()
        self.assertTrue(isinstance(df, pl.DataFrame))
        self.assertEqual(max(df["run_id"]), 15)
        self.assertEqual(min(df["run_id"]), 1)
        self.assertEqual(len(df), 659)

    def test_load_twice(self):
        manager = DataManager()
        manager.add_json(self.json_files[0])
        self.assertEqual(len(manager.data_sets), 1)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            manager.add_json(self.json_files[0])
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, RuntimeWarning))

        self.assertEqual(len(manager.data_sets), 1)


    def test_add_folder(self):
        manager = DataManager()
        manager.add_folder(self.data_dir)
        self.assertEqual(len(manager.data_sets), 2)       



if __name__ == "__main__":
    pass
