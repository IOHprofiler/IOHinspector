import unittest
import polars as pl
import numpy as np
from iohinspector.metrics import get_attractor_network

class TestGetAttractorNetwork(unittest.TestCase):
    def test_basic(self):
        data = pl.DataFrame({
            "x1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "x2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],   
            "raw_y": [35, 33, 31, 29, 27, 23, 18, 16, 14, 12, 10, 9, 6],
            "evaluations": [1,42, 81,121,161,201,241,281,321,361,401,442,481],
            "data_id": [1]*13
        })
        nodes, edges = get_attractor_network(
            data, 
            coord_vars=["x1", "x2"], 
            fval_var="raw_y", 
            eval_var="evaluations", 
            )
        # Check nodes DataFrame shape and content
        self.assertEqual(nodes.shape[1], 5)  # x1, x2, y, count, evals
        self.assertGreaterEqual(nodes.shape[0], 1)
        # Check that node coordinates and y values are as expected
        self.assertIn("x1", nodes.columns)
        self.assertIn("x2", nodes.columns)
        self.assertIn("y", nodes.columns)
        self.assertIn("count", nodes.columns)
        self.assertIn("evals", nodes.columns)
        # Check that the first node matches the first stagnation point
        self.assertEqual(nodes.iloc[0]["x1"], 0)
        self.assertEqual(nodes.iloc[0]["x2"], 0)
        self.assertEqual(nodes.iloc[0]["y"], 35)
        self.assertEqual(nodes.iloc[-1]["x1"], 10)
        self.assertEqual(nodes.iloc[-1]["x2"], 10)
        self.assertEqual(nodes.iloc[-1]["y"], 10)
        # Check that counts and evals are positive
        self.assertTrue((nodes["count"] > 0).all())
        self.assertTrue((nodes["evals"] > 0).all())

        # Check edges DataFrame shape and content
        self.assertEqual(edges.shape[1], 4)  # start, end, count, stag_length_avg
        self.assertTrue((edges["count"] > 0).all())
        self.assertTrue((edges["stag_length_avg"] > 0).all())
        # Check that start and end refer to valid node indices
        self.assertTrue(edges["start"].isin(nodes.index).all())
        self.assertTrue(edges["end"].isin(nodes.index).all())



if __name__ == "__main__":
    unittest.main()