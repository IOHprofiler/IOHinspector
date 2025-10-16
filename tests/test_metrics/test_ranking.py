import unittest
import numpy as np
import polars as pl
import pandas as pd
from iohinspector.metrics import get_tournament_ratings

class TestGetTournamentRatings(unittest.TestCase):
    def setUp(self):
        # Create a simple polars DataFrame for testing
        self.data = pl.DataFrame({
            "algorithm_name": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "function_name": ["f1", "f2", "f3", "f1", "f2", "f3", "f1", "f2", "f3"],
            "raw_y": [1.0, 2.0, 1.7, 1.5, 2.8, 2.1, 0.9, 0.5, 1.6]
        })

    def test_basic(self):
        result = get_tournament_ratings(self.data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("Rating", result.columns)
        self.assertIn("Deviation", result.columns)
        self.assertIn("algorithm_name", result.columns)
        self.assertEqual(len(result), 3)  # Three algorithms
        # Check that algorithms are ordered by rating: C, A, B
        sorted_algos = result.sort_values("Rating", ascending=False)["algorithm_name"].tolist()
        self.assertEqual(sorted_algos, ["C", "A", "B"])

    def test_basic_maximisation(self):
        result = get_tournament_ratings(self.data, maximization=True)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("Rating", result.columns)
        self.assertIn("Deviation", result.columns)
        self.assertIn("algorithm_name", result.columns)
        self.assertEqual(len(result), 3)  # Three algorithms
        # Check that algorithms are ordered by rating: C, A, B
        sorted_algos = result.sort_values("Rating", ascending=False)["algorithm_name"].tolist()
        self.assertEqual(sorted_algos, ["B", "A", "C"])


    def test_single_function(self):
        data = pl.DataFrame({
            "algorithm_name": ["A", "B"],
            "function_name": ["f1", "f1"],
            "raw_y": [1.0, 2.0]
        })
        result = get_tournament_ratings(data, nrounds=25)
        self.assertEqual(len(result), 2)
        self.assertTrue(set(result["algorithm_name"]) == {"A", "B"})


if __name__ == "__main__":
    unittest.main()