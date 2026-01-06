import unittest
import polars as pl
import matplotlib
from iohinspector.plots import plot_robustrank_over_time,plot_tournament_ranking, plot_robustrank_changes
from iohinspector.indicators import HyperVolume

matplotlib.use("Agg")  # Use non-interactive backend for tests
import matplotlib.pyplot as plt


class TestPlotTournamentRanking(unittest.TestCase):
    def setUp(self):
        self.data = pl.DataFrame({
            "algorithm_name": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "function_name": ["f1", "f2", "f3", "f1", "f2", "f3", "f1", "f2", "f3"],
            "raw_y": [1.0, 2.0, 1.7, 1.5, 2.8, 2.1, 0.9, 0.5, 1.6]
        })
    def test_basic_call_returns_axes_and_data(self):
        ax, dt = plot_tournament_ranking(self.data)
        self.assertIsNotNone(ax)
        self.assertIsNotNone(dt)

class TestPlotRobustRankOverTime(unittest.TestCase):
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
            "function_id": [1]*9 + [1]*9 + [1]*9
        })

    def test_basic_call_returns_axes_and_data(self):
        evals = [1, 10, 100]
        axs, comparison, benchmark = plot_robustrank_over_time(
            self.data,
            obj_vars=["f1", "f2", "f3"],
            evals=evals,
            indicator=HyperVolume(reference_point=[5.0, 5.0, 5.0]),
        )
        self.assertIsNotNone(axs)
        self.assertIsNotNone(comparison)
        self.assertIsNotNone(benchmark)


class TestPlotRobustRankChanges(unittest.TestCase):
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
            "function_id": [1]*9 + [1]*9 + [1]*9
        })

    def test_basic_call_returns_axes_and_data(self):
        evals = [1, 10, 100]
        ax, dt = plot_robustrank_changes(
            self.data, 
            obj_vars=["f1","f2", "f3"], 
            evals=evals, 
            indicator=HyperVolume(reference_point=[5.0, 5.0, 5.0]),
        )
        self.assertIsNotNone(ax)
        self.assertIsNotNone(dt)


if __name__ == "__main__":
    unittest.main()
