import unittest
import polars as pl
import numpy as np
from iohinspector.plots import plot_attractor_network

class TestGetAttractorNetwork(unittest.TestCase):
    def test_basic(self):
        data = pl.DataFrame({
            "x1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "x2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],   
            "raw_y": [35, 33, 31, 29, 27, 23, 18, 16, 14, 12, 10, 9, 6],
            "evaluations": [1,42, 81,121,161,201,241,281,321,361,401,442,481],
            "data_id": [1]*13
        })
        plot_attractor_network(
            data, 
            coord_vars=["x1", "x2"], 
            fval_var="raw_y", 
            eval_var="evaluations",
            )
      



if __name__ == "__main__":
    unittest.main()