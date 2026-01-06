import unittest
import polars as pl
import os
import tempfile
from typing import List
from iohinspector.manager import DataManager
from iohinspector.data import Dataset, Function, Algorithm, METADATA_SCHEMA

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.realpath(os.path.join(BASE_DIR, "test_data"))
COCO_DATA_DIR = os.path.realpath(os.path.join(BASE_DIR, "test_coco_data"))

class TestDataManager(unittest.TestCase):
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

    def test_add_folder_file_not_found(self):
        m = DataManager()
        with self.assertRaises(FileNotFoundError):
            m.add_folder("nonexistent_folder")

    def test_add_folder_no_json_or_coco(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m = DataManager()
            with self.assertRaises(FileNotFoundError):
                m.add_folder(tmpdir)

    def test_add_folder_json(self):
        manager = DataManager()
        manager.add_folder(self.data_dir)
        self.assertEqual(len(manager.functions), 1)
        self.assertEqual(len(manager.algorithms), 1)
        self.assertEqual(len(manager.data_sets), 1)

    def test_add_folder_coco(self):
        manager = DataManager()
        manager.add_folder(COCO_DATA_DIR)
        self.assertEqual(len(manager.functions), 1)
        self.assertEqual(len(manager.algorithms), 1)
        self.assertEqual(len(manager.data_sets), 1)

    def test_add_json(self):
        manager = DataManager()
        manager.add_json(self.json_files[0])
        self.assertEqual(len(manager.functions), 1)
        self.assertEqual(len(manager.algorithms), 1)
        self.assertEqual(len(manager.data_sets), 1)

    def test_add_coco_info(self):
        coco_file = os.path.join(COCO_DATA_DIR, "BFGS-scipy-2019_Varelas/BFGS-scipy-2019_bbob_Varelas_Dahito/minimize_on_bbob_budget100000xD/bbobexp_f1_i1.info")
        manager = DataManager()
        manager.add_coco_info(coco_file)
        self.assertEqual(len(manager.functions), 1)
        self.assertEqual(len(manager.algorithms), 1)
        self.assertEqual(len(manager.data_sets), 1)
    

    def test_add_coco_info_file_not_found(self):
        m = DataManager()
        with self.assertRaises(FileNotFoundError):
            m.add_coco_info("missing.info")

    def test_select_by_data_ids(self):
        m = DataManager()
        m.add_folder(self.data_dir)
        selected = m.select(data_ids=[1])
        self.assertTrue(selected.any)
        self.assertLessEqual(len(selected.data_sets), 2)

    def test_select_by_function_ids(self):
        m = DataManager()
        m.add_folder(self.data_dir)
        selected = m.select(function_ids=[1])
        self.assertEqual(len(selected.data_sets), 1)
        self.assertEqual(selected.data_sets[0].function.id, 1)

    def test_select_by_algorithms(self):
        m = DataManager()
        m.add_folder(self.data_dir)
        selected = m.select(algorithms=["algorithm_A"])
        self.assertEqual(len(selected.data_sets), 1)
        self.assertEqual(selected.data_sets[0].algorithm.name, "algorithm_A")

    def test_select_by_data_attributes(self):
        m = DataManager()
        m.add_folder(self.data_dir)
        selected = m.select(data_attributes=["evaluations", "raw_y"])
        self.assertEqual(len(selected.data_sets), 1)

   
    def test_select_by_dimensions(self):
        m = DataManager()
        m.add_folder(self.data_dir)
        selected = m.select(dimensions=[2])
        self.assertEqual(len(selected.data_sets), 1)

    def test_select_by_instances(self):
        m = DataManager()
        m.add_folder(self.data_dir)
        selected = m.select(instances=[1])
        self.assertEqual(len(selected.data_sets), 1)

    def test_select_indexes(self):
        m = DataManager()
        m.add_folder(self.data_dir)
        selected = m.select_indexes([0])
        self.assertEqual(len(selected.data_sets), 1)
        self.assertEqual(selected.data_sets[0].file.split("/")[-1], "IOHprofiler_f1_Sphere.json")

    def test_load(self):
        m = DataManager()
        m.add_folder(self.data_dir)
        df = m.load()
        self.assertIsInstance(df, pl.DataFrame)
        self.assertIn("raw_y", df.columns)
        df2 = m.load(include_meta_data=True)
        self.assertIsInstance(df2, pl.DataFrame)
        self.assertIn("algorithm_name", df2.columns)
        df3 = m.load(include_columns=["algorithm_name"])
        self.assertIn("algorithm_name", df3.columns)
        df4 = m.load(x_values=[1])
        self.assertIsInstance(df4, pl.DataFrame)

    def test_load_empty(self):
        m = DataManager()
        df = m.load()
        self.assertEqual(len(df), 0)

if __name__ == "__main__":
    unittest.main()