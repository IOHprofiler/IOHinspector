import os
import unittest
import warnings

import polars as pl

from iohinspector import DataManager, turbo_align, plot_ecdf

from pprint import pprint

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.realpath(os.path.join(BASE_DIR, "test_data"))


class TestManager(unittest.TestCase):

    def setUp(self):
        self.data_folders = [os.path.join(DATA_DIR, x) for x in sorted(os.listdir(DATA_DIR))]
        self.data_dir = self.data_folders[0]
        self.json_files = sorted(
            [
                fname
                for f in os.listdir(self.data_dir)
                if os.path.isfile((fname := os.path.join(self.data_dir, f)))
            ]
        )

    def test_add_json(self):
        manager = DataManager()
        manager.add_json(self.json_files[0])
        data = manager.data_sets[0]
        df = data.scenarios[0].load()
        self.assertTrue(isinstance(df, pl.DataFrame))
        self.assertEqual(max(df["run_id"]), 5)
        self.assertEqual(min(df["run_id"]), 1)
        self.assertEqual(len(df), 27)

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
        self.assertEqual(len(manager.data_sets), 1)

    def test_select(self):
        manager = DataManager()
        manager.add_folders(self.data_folders)
        
        def assert_shape(df, n, m = 4):
            self.assertEqual(df.shape[1], m)
            self.assertEqual(len(df), n)
            self.assertEqual(max(df["run_id"]), 5)
            self.assertEqual(min(df["run_id"]), 1)
            self.assertTrue(selection.any)
            

        selection = manager.select(instances=[1], function_ids=[1])
        df = selection.load(monotonic=False)
        assert_shape(df, 84)
        df = selection.load(monotonic=True)
        assert_shape(df, 69)
        df = selection.load(monotonic=True, include_meta_data=True)
        assert_shape(df, 69, 13)

        selection = manager.select(function_ids=[0])
        self.assertFalse(selection.any)
        df = selection.load()
        self.assertEqual(len(df), 0)

        selection1 = manager.select(instances=[1], function_ids=[1])
        selection2 = manager.select(instances=[1], function_ids=[2])
        selection = selection1 + selection2
        df = selection.load()
        assert_shape(df, 125)

    def test_align(self):
        manager = DataManager()
        manager.add_folders(self.data_folders)
        
        selection = manager.select(function_ids=[1], algorithms = ['algorithm_A', 'algorithm_B'])
        df = selection.load(monotonic=True, include_meta_data=True)
        
        evals = [1, 5, 10, 20, 50, 100]
        df = turbo_align(df, evals)
        self.assertTrue(set(df['evaluations'].unique()) == set(evals))
        self.assertEqual(len(df['data_id'].unique()) * len(evals), df.shape[0])
        
    def test_plot_ecdf(self):
        manager = DataManager()
        manager.add_folders(self.data_folders)
        
        selection = manager.select(function_ids=[1], algorithms = ['algorithm_A', 'algorithm_B'])
        df = selection.load(monotonic=True, include_meta_data=True)
        
        dt = plot_ecdf(df)
        self.assertEqual(dt.shape, (66, 14))
        

    def test_select_on_data_id(self):
        manager = DataManager()
        manager.add_folders(self.data_folders)

        selection = manager.select(data_ids=[20, 21, 22])
        self.assertEqual(selection.n_runs, 3)
        
    def test_load_subset_columns(self):
        manager = DataManager()
        manager.add_folders(self.data_folders)
        selection = manager.select([1]).load(include_columns=["function_id"])
        self.assertListEqual(selection.columns, ["function_id", "data_id", "run_id", "evaluations", "raw_y"])


if __name__ == "__main__":
    pass
