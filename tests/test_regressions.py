"""Regression cases for the documented sorting and shortest-path domains."""
import unittest

import numpy as np

from neural_compiler import TransformerCircuitBackend
from stage2_transformer_benchmark import build_shortest_path_ir, build_sorting_ir


class CircuitRegressions(unittest.TestCase):
    def test_sort_repeated_integer_values(self):
        model = TransformerCircuitBackend.compile(build_sorting_ir(8), d_model=64, n_layers=6, n_heads=4)
        for values in ([1, 1, 2, 3, 4, 5, 6, 7], [3] * 8, [2, -1, 2, -1, 0, 0, 3, 3]):
            x = np.array(values, dtype=float)
            actual = model.run({"x": x, "pos_oh": np.eye(8)}, seq_len=8)
            np.testing.assert_array_equal(np.round(actual), np.sort(x))

    def test_reverse_pivot_chain_shortest_path(self):
        n = 5
        weights = np.full((n, n), 100.0)
        np.fill_diagonal(weights, 0.0)
        weights[4, 0] = weights[3, 4] = weights[2, 3] = weights[1, 2] = 1.0
        model = TransformerCircuitBackend.compile(build_shortest_path_ir(n), d_model=64, n_layers=20, n_heads=4)
        actual = model.run({"node_oh": np.eye(n), "edge_w": weights,
                            "dist_0": np.array([0., 100., 100., 100., 100.])}, seq_len=n)
        np.testing.assert_array_equal(np.round(actual), [0, 4, 3, 2, 1])


if __name__ == "__main__":
    unittest.main()
