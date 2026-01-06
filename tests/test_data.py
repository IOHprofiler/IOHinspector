import os
import unittest
import warnings
import tempfile

from iohinspector.data import Dataset, Scenario, check_keys, process_header
import polars as pl

from iohinspector import DataManager, turbo_align, plot_ecdf

from pprint import pprint

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.realpath(os.path.join(BASE_DIR, "test_data"))
COCO_DATA_DIR = os.path.realpath(os.path.join(BASE_DIR, "test_coco_data"))

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
        
        ax, dt = plot_ecdf(df)
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

    def test_process_header_handles_raw_y_and_evaluations(self):
  
        line = "% function evaluation | best noise-free fitness | x1 | x2"
        header = process_header(line)
        self.assertIn("raw_y", header)
        self.assertIn("evaluations", header)
        self.assertNotIn("x1", header)
        self.assertNotIn("x2", header)

    def test_check_keys_raises_on_missing(self):
        with self.assertRaises(ValueError):
            check_keys({"a": 1}, ["a", "b"])


    def test_scenario_from_dict_file_not_found(self):
        data = {
            "dimension": 2,
            "path": "not_a_file.csv",
            "runs": []
        }
        with self.assertRaises(FileNotFoundError):
            Scenario.from_dict(data, "")

    def test_dataset_from_json_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            Dataset.from_json("not_a_file.json")

    def test_dataset_from_json_returns_none_on_invalid(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            f = os.path.join(tmp_dir, "invalid.json")
            with open(f, 'w') as file:
                file.write("{invalid json")
            self.assertIsNone(Dataset.from_json(f))

    def test_dataset_from_coco_info_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            Dataset.from_coco_info("not_a_file.txt")

    def test_dataset_from_coco_info_empty_warns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            f = os.path.join(tmp_dir, "empty.txt")
            with open(f, 'w') as file:
                file.write("")
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ds = Dataset.from_coco_info(f)
                self.assertIsNone(ds)
                self.assertTrue(any("empty" in str(wi.message) for wi in w))

    def test_dataset_from_coco_text_parses_minimal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            coco_text = (
                "suite = 'test_suite', funcId = 1, DIM = 2, Precision = 1e-8, algId = 'alg'\n"
                "datafile.dat, 1:10|0.5"
            )
            datafile = os.path.join(tmp_dir, "datafile.dat")
            with open(datafile, 'w') as f:
                f.write("function evaluation | best noise-free fitness\n1 0.5\n")
            
            info_file = os.path.join(tmp_dir, "info.txt")
            ds = Dataset.from_coco_text(coco_text, info_file)
            self.assertEqual(ds.suite, "test_suite")
            self.assertEqual(ds.function.id, 1)
            self.assertEqual(ds.algorithm.name, "alg")
            self.assertEqual(len(ds.scenarios), 1)
            self.assertEqual(ds.scenarios[0].dimension, 2)
            self.assertTrue(os.path.isfile(ds.scenarios[0].data_file))

    def test_dataset_from_coco_text_multiple_algorithms_raises(self):
        coco_text = (
            "funcId = 1, DIM = 2, Precision = 1e-8, algId = 'alg1'\n"
            "file1.dat, 1:10|0.5\n"
            "funcId = 1, DIM = 2, Precision = 1e-8, algId = 'alg2'\n"
            "file2.dat, 1:10|0.5"
        )
        with self.assertRaises(ValueError):
            Dataset.from_coco_text(coco_text, "info.txt")

    def test_dataset_from_coco_text_multiple_function_ids_raises(self):
        coco_text = (
            "funcId = 1, DIM = 2, Precision = 1e-8, algId = 'alg'\n"
            "file1.dat, 1:10|0.5\n"
            "funcId = 2, DIM = 2, Precision = 1e-8, algId = 'alg'\n"
            "file2.dat, 1:10|0.5"
        )
        with self.assertRaises(ValueError):
            Dataset.from_coco_text(coco_text, "info.txt")

    def test_coco_info_integration(self):
        manager = DataManager()
        manager.add_folder(COCO_DATA_DIR)
        df = manager.load(True, True)
        self.assertIn("raw_y", df.columns)
        self.assertIn("evaluations", df.columns)
        self.assertIn("function_id", df.columns)
        self.assertIn("algorithm_info", df.columns)
        self.assertEqual(df["algorithm_name"].unique().to_list(), ["BFGS-scipy-2019"])
        self.assertEqual(df["suite"].unique().to_list(), ["bbob"])
        self.assertEqual(df["function_id"].unique().to_list(), [1])
        self.assertEqual(df["dimension"].unique().to_list(), [2])

if __name__ == "__main__":
    unittest.main()
