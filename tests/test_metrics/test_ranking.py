import unittest
import numpy as np
import polars as pl
import pandas as pd
from iohinspector.metrics import get_tournament_ratings, get_robustrank_over_time, get_robustrank_changes
from iohinspector.indicators import HyperVolume

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


class TestGetRobustRankOverTime(unittest.TestCase):
    def setUp(self):
        # Create simple polars DataFrame with different targets and ranks
        self.data = pl.DataFrame({
            "algorithm_name": ["A"] * 9 + ["B"] * 9 + ["C"] * 9,
            "evaluations": [1, 10, 100] * 9,
            "f1": [
            # A: best at eval 1, B: best at eval 10, A: best at eval 100 (for run 1)
            0.8, 1.5, 0.7,   # A, run 1
            1.0, 1.6, 0.9,   # A, run 2
            0.9, 1.4, 0.8,   # A, run 3

            1.0, 0.7, 1.2,   # B, run 1
            1.2, 0.8, 1.3,   # B, run 2
            1.1, 0.6, 1.1,   # B, run 3

            1.5, 1.5, 0.1,   # C, run 1
            1.6, 1.6, 0.2,   # C, run 2
            1.4, 1.4, 0.3    # C, run 3
            ],
            "f2": [
            1.0, 2.0, 0.9,   # A, run 1
            1.2, 2.1, 1.1,   # A, run 2
            1.1, 2.2, 1.0,   # A, run 3

            1.3, 0.8, 1.4,   # B, run 1
            1.5, 0.9, 1.5,   # B, run 2
            1.4, 0.7, 1.3,   # B, run 3

            2.0, 2.0, 0.1,   # C, run 1
            2.1, 2.1, 0.2,   # C, run 2
            1.9, 1.9, 0.3    # C, run 3
            ],
            "f3": [
            2.0, 3.0, 1.8,   # A, run 1
            2.2, 3.1, 2.0,   # A, run 2
            2.1, 3.2, 1.9,   # A, run 3

            2.3, 1.2, 2.4,   # B, run 1
            2.5, 1.3, 2.5,   # B, run 2
            2.4, 1.1, 2.3,   # B, run 3

            3.0, 3.0, 0.1,   # C, run 1
            3.1, 3.1, 0.3,   # C, run 2
            2.9, 2.9, 0.2    # C, run 3
            ],
            "data_id": [1]*3 + [2]*3 + [3]*3 + [4]*3 + [5]*3 + [6]*3 + [7]*3 + [8]*3 + [9]*3,
            "run_id": [1]*3 + [2]*3 + [3]*3 + [1]*3 + [2]*3 + [3]*3 + [1]*3 + [2]*3 + [3]*3,
        })

    def test_basic(self):
        evals = [1, 10, 100]
        comparison, benchmark = get_robustrank_over_time(
            self.data, 
            obj_vars=["f1","f2", "f3"], 
            evals=evals, 
            indicator=HyperVolume(reference_point=[5.0,5.0,5.0]),
            )


class TestGetRobustRankChanges(unittest.TestCase):
    def setUp(self):
        self.data = pl.DataFrame({
            "algorithm_name": ["A"] * 9 + ["B"] * 9 + ["C"] * 9,
            "evaluations": [1, 10, 100] * 9,
            "f1": [
            # A: best at eval 1, B: best at eval 10, A: best at eval 100 (for run 1)
            0.8, 1.5, 0.7,   # A, run 1
            1.0, 1.6, 0.9,   # A, run 2
            0.9, 1.4, 0.8,   # A, run 3

            1.0, 0.7, 1.2,   # B, run 1
            1.2, 0.8, 1.3,   # B, run 2
            1.1, 0.6, 1.1,   # B, run 3

            1.5, 1.5, 0.1,   # C, run 1
            1.6, 1.6, 0.2,   # C, run 2
            1.4, 1.4, 0.3    # C, run 3
            ],
            "f2": [
            1.0, 2.0, 0.9,   # A, run 1
            1.2, 2.1, 1.1,   # A, run 2
            1.1, 2.2, 1.0,   # A, run 3

            1.3, 0.8, 1.4,   # B, run 1
            1.5, 0.9, 1.5,   # B, run 2
            1.4, 0.7, 1.3,   # B, run 3

            2.0, 2.0, 0.1,   # C, run 1
            2.1, 2.1, 0.2,   # C, run 2
            1.9, 1.9, 0.3    # C, run 3
            ],
            "f3": [
            2.0, 3.0, 1.8,   # A, run 1
            2.2, 3.1, 2.0,   # A, run 2
            2.1, 3.2, 1.9,   # A, run 3

            2.3, 1.2, 2.4,   # B, run 1
            2.5, 1.3, 2.5,   # B, run 2
            2.4, 1.1, 2.3,   # B, run 3

            3.0, 3.0, 0.1,   # C, run 1
            3.1, 3.1, 0.3,   # C, run 2
            2.9, 2.9, 0.2    # C, run 3
            ],
            "data_id": [1]*3 + [2]*3 + [3]*3 + [4]*3 + [5]*3 + [6]*3 + [7]*3 + [8]*3 + [9]*3,
            "run_id": [1]*3 + [2]*3 + [3]*3 + [1]*3 + [2]*3 + [3]*3 + [1]*3 + [2]*3 + [3]*3,
        })

    def test_basic(self):
        evals = [1, 10, 100]
        result = get_robustrank_changes(
            self.data,
            obj_vars=["f1","f2", "f3"], 
            evals=evals, 
            indicator=HyperVolume(reference_point=[5.0,5.0,5.0]),

            )
        for eval in evals:
            self.assertIn(str(eval), result.keys())



if __name__ == "__main__":
    unittest.main()