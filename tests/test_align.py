import unittest
import polars as pl
from iohinspector.align import align_data, turbo_align


class TestAlignData(unittest.TestCase):
        
    def test_align_data_minimization_long(self):
        df = pl.DataFrame({
            "data_id": [1, 1, 1, 2, 2, 2],
            "evaluations": [1, 2, 5, 1, 4, 5],
            "raw_y": [10, 8, 6, 20, 18, 16]
        })

        evals = [1, 2, 3, 4, 5]
        result = align_data(df, evals, group_cols=("data_id",), x_col="evaluations", y_col="raw_y", output="long", maximization=False, silence_warning=True)
        expected = pl.DataFrame({
            "evaluations": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "raw_y": [10, 8, 8, 8, 6, 20, 20, 20, 18, 16],
            "data_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        })
        result_sorted = result.sort(["data_id", "evaluations"])
        expected_sorted = expected.sort(["data_id", "evaluations"])
        self.assertEqual(result_sorted.to_dicts(), expected_sorted.to_dicts())

    def test_align_data_maximization_long(self):
        df = pl.DataFrame({
            "data_id": [1, 1, 1],
            "evaluations": [1, 2, 3],
            "raw_y": [5, 7, 6]
        })
        evals = [1, 2, 3]
        result = align_data(df, evals, group_cols=("data_id",), x_col="evaluations", y_col="raw_y", output="long", maximization=True, silence_warning=True)
        expected = pl.DataFrame({
            "evaluations": [1, 2, 3],
            "raw_y": [5, 7, 7],
            "data_id": [1, 1, 1]
        })
        result_sorted = result.sort(["data_id", "evaluations"])
        expected_sorted = expected.sort(["data_id", "evaluations"])
        self.assertEqual(result_sorted.to_dicts(), expected_sorted.to_dicts())

    def test_align_data_wide_output(self):
        df = pl.DataFrame({
            "data_id": [1, 1, 1, 2, 2, 2],
            "evaluations": [1, 2, 5, 1, 4, 5],
            "raw_y": [10, 8, 6, 20, 18, 16]
        })
        evals = [1, 2, 3, 4, 5]
       
        result = align_data(df, evals, group_cols=("data_id",), x_col="evaluations", y_col="raw_y", output="wide", maximization=False, silence_warning=True)
        # Should pivot to wide format
        self.assertIn("1", result.columns)
        self.assertIn("2", result.columns)
        self.assertIn("evaluations", result.columns)
        self.assertEqual(result.shape[0], 5)  # 3 evals


    def test_align_data_custom_group_col(self):
        df = pl.DataFrame({
            "exp_id": [1, 1, 2, 2],
            "evaluations": [1, 2, 1, 2],
            "raw_y": [5, 3, 7, 6]
        })
        evals = [1, 2]
        result = align_data(df, evals, group_cols=("exp_id",), x_col="evaluations", y_col="raw_y", output="long", maximization=False, silence_warning=True)
        self.assertTrue(set(result["exp_id"].to_list()) == {1, 2})

    def test_align_data_non_default_x_col(self):
        df = pl.DataFrame({
            "data_id": [1, 1, 1],
            "steps": [10, 20, 30],
            "score": [100, 90, 80]
        })
        evals = [10, 20, 30]
        result = align_data(df, evals, group_cols=("data_id",), x_col="steps", y_col="score", output="long", maximization=False, silence_warning=True)
        self.assertTrue(result["steps"].to_list() == [10, 20, 30])
        
class TestTurboAlignData(unittest.TestCase):  
    def test_turbo_align_minimization_long(self):
        df = pl.DataFrame({
            "data_id": [1, 1, 1, 2, 2, 2],
            "evaluations": [1, 2, 5, 1, 4, 5],
            "raw_y": [10, 8, 6, 20, 18, 16]
        })
        evals = [1, 2, 3, 4, 5]
        result = turbo_align(df, evals, x_col="evaluations", y_col="raw_y", output="long", maximization=False)
        expected = pl.DataFrame({
            "evaluations": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "data_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "raw_y": [10, 8, 8, 8, 6, 20, 20, 20, 18, 16]
        })
        result_sorted = result.sort(["data_id", "evaluations"])
        expected_sorted = expected.sort(["data_id", "evaluations"])
        self.assertEqual(result_sorted.to_dicts(), expected_sorted.to_dicts())


    def test_turbo_align_maximization_long(self):
        df = pl.DataFrame({
            "data_id": [1, 1, 1],
            "evaluations": [1, 2, 5],
            "raw_y": [5, 7, 8]
        })
        evals = [1, 2, 3, 4, 5]
        result = turbo_align(df, evals, x_col="evaluations", y_col="raw_y", output="long", maximization=True)
        expected = pl.DataFrame({
            "evaluations": [1, 2, 3, 4, 5],
            "data_id": [1, 1, 1, 1, 1],
            "raw_y": [5, 7, 7, 7, 8]
        })
        result_sorted = result.sort(["data_id", "evaluations"])
        expected_sorted = expected.sort(["data_id", "evaluations"])  
        self.assertEqual(result_sorted.to_dicts(), expected_sorted.to_dicts())

    def test_turbo_align_wide_output(self):
        df = pl.DataFrame({
            "data_id": [1, 1, 1, 2, 2, 2],
            "evaluations": [1, 2, 5, 1, 4, 5],
            "raw_y": [10, 8, 6, 20, 18, 16]
        })
        evals = [1, 2, 3, 4, 5]
        result = turbo_align(df, evals, x_col="evaluations", y_col="raw_y", output="wide", maximization=False)
        self.assertIn("1", result.columns)
        self.assertIn("2", result.columns)
        self.assertIn("evaluations", result.columns)
        self.assertEqual(result.shape[0], 5)

    def test_turbo_align_non_default_x_col(self):
        df = pl.DataFrame({
            "data_id": [1, 1, 1],
            "steps": [10, 20, 30],
            "score": [100, 90, 80]
        })
        evals = [10, 20, 30]
        result = turbo_align(df, evals, x_col="steps", y_col="score", output="long", maximization=False)
        self.assertTrue(result["steps"].to_list() == [10, 20, 30])

if __name__ == "__main__":
    unittest.main()