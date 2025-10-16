import unittest
import polars as pl
import numpy as np
from iohinspector.metrics.ecdf import get_data_ecdf
import iohinspector

class TestGetDataECDF(unittest.TestCase):
    def setUp(self):
        # Create a simple synthetic dataset
        self.df = pl.DataFrame({
            "evaluations": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "raw_y": [10, 8, 6, 4, 2, 18, 16, 14, 12, 10],
            "algorithm_name": ["algo1"] * 5 + ["algo2"] * 5,
            "data_id": [1] * 5 + [2] * 5,
        })

    def test_basic_ecdf(self):
        result = get_data_ecdf(self.df, scale_xlog=False, scale_ylog=False)
        algo1_eaf = result[result["algorithm_name"] == "algo1"]["eaf"].to_numpy()   
        algo1_eaf.sort()
        np.testing.assert_allclose(algo1_eaf, [0.5, 0.625, 0.75, 0.875, 1])

        algo2_eaf = result[result["algorithm_name"] == "algo2"]["eaf"].to_numpy()   
        algo2_eaf.sort()
        np.testing.assert_allclose(algo2_eaf, [0, 0.125, 0.25, 0.375, 0.5])

    def test_ecdf_with_custom_x_values(self):
        x_values = [2, 4]
        result = get_data_ecdf(self.df, x_values=x_values, scale_xlog=False, scale_ylog=False)
        algo1_eaf = result[result["algorithm_name"] == "algo1"]["eaf"].to_numpy()   
        algo1_eaf.sort()
        np.testing.assert_allclose(algo1_eaf, [2/3, 1])

        algo2_eaf = result[result["algorithm_name"] == "algo2"]["eaf"].to_numpy()   
        algo2_eaf.sort()
        np.testing.assert_allclose(algo2_eaf, [0, 1/3])

    def test_ecdf_with_maximization(self):
        result = get_data_ecdf(self.df, maximization=True)
        # eaf_raw_y should be between 0 and 1
        algo1_eaf = result[result["algorithm_name"] == "algo1"]["eaf"].to_numpy()
        # Assert that all values in algo1_eaf are 0 and the array is not empty
        np.testing.assert_allclose(algo1_eaf, [0, 0, 0, 0, 0])

        algo2_eaf = result[result["algorithm_name"] == "algo2"]["eaf"].to_numpy()  
        np.testing.assert_allclose(algo2_eaf, [1, 1, 1, 1, 1])
       

    def test_ecdf_with_custom_bounds(self):
        result = get_data_ecdf(self.df, y_min=0, y_max=100, scale_xlog=False, scale_ylog=False)
        algo1_eaf = result[result["algorithm_name"] == "algo1"]["eaf"].to_numpy()   
        algo1_eaf.sort()
        np.testing.assert_allclose(algo1_eaf, [90/100, 92/100, 94/100, 96/100, 98/100])

        algo2_eaf = result[result["algorithm_name"] == "algo2"]["eaf"].to_numpy()   
        algo2_eaf.sort()
        np.testing.assert_allclose(algo2_eaf, [82/100, 84/100, 86/100, 88/100, 90/100]) 

    def test_ecdf_with_x_min_x_max(self):
        result = get_data_ecdf(self.df, x_min=2, x_max=4, scale_xlog=False, scale_ylog=False)
        algo1_eaf = result[result["algorithm_name"] == "algo1"]["eaf"].to_numpy()   
        algo1_eaf.sort()
        np.testing.assert_allclose(algo1_eaf, [2/3, 5/6, 1])

        algo2_eaf = result[result["algorithm_name"] == "algo2"]["eaf"].to_numpy()   
        algo2_eaf.sort()
        np.testing.assert_allclose(algo2_eaf, [0, 1/6, 1/3])

    def test_basic_ecdf_turbo(self):
        result = get_data_ecdf(self.df, scale_xlog=False, scale_ylog=False, turbo=True)
        algo1_eaf = result[result["algorithm_name"] == "algo1"]["eaf"].to_numpy()   
        algo1_eaf.sort()
        np.testing.assert_allclose(algo1_eaf, [0.5, 0.625, 0.75, 0.875, 1])

        algo2_eaf = result[result["algorithm_name"] == "algo2"]["eaf"].to_numpy()   
        algo2_eaf.sort()
        np.testing.assert_allclose(algo2_eaf, [0, 0.125, 0.25, 0.375, 0.5])

if __name__ == "__main__":
    unittest.main()