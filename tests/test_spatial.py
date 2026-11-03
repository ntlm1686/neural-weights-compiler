"""Every spatial primitive must agree with its Python predicate on random binary grids."""
import unittest
import numpy as np
from neural_compiler.spatial import SpatialProgram, CapacityError


def grids(n, density, seed=0):
    rng = np.random.RandomState(seed)
    return (rng.rand(n, 10, 10, 6) < density).astype(np.float32)


class SpatialPrimitiveTest(unittest.TestCase):
    def check(self, sp, specs, density=0.04, n=4000):
        """specs: unit name -> predicate(grid) -> bool. Compares sign of the unit's activation."""
        mlp = sp.compile()
        X = grids(n, density)
        h1, h2, _ = mlp.inspect_activations(X.reshape(n, -1))
        for name, pred in specs.items():
            layer, i = (h1, mlp.unit_map["h1"][name]) if name in mlp.unit_map["h1"] else (h2, mlp.unit_map["h2"][name])
            want = np.array([pred(g) for g in X])
            got = layer[:, i] > 0
            self.assertTrue((want == got).all(), f"{name}: {(want != got).sum()} mismatches")
            self.assertGreater(np.abs(layer[:, i]).min(), 0.99, f"{name}: not saturated")

    def test_layer1(self):
        sp = SpatialProgram()
        sp.row_any("e3", 1, 3)
        sp.col_any("p7", 0, 7, rows=range(1, 9))
        sp.any_("diver5", 2, [(5, c) for c in range(10)], exclude_ch=1)
        sp.count_ge("two_enemies", 1, sp.all_cells(), 2)
        sp.pair_at("pair", 0, (4, 6), 1, (4, 5))
        self.check(sp, {
            "e3": lambda g: g[3, :, 1].any(),
            "p7": lambda g: g[1:9, 7, 0].any(),
            "diver5": lambda g: (g[5, :, 2] * (1 - g[5, :, 1])).any(),
            "two_enemies": lambda g: g[:, :, 1].sum() >= 2,
            "pair": lambda g: g[4, 6, 0] and g[4, 5, 1],
        })

    def test_layer2(self):
        sp = SpatialProgram()
        a, b, c = sp.row_any("a", 0, 2), sp.row_any("b", 1, 2), sp.row_any("c", 2, 2)
        sp.and_("ab_not_c", pos=[a, b], neg=[c])
        sp.or_("a_or_notb", pos=[a], neg=[b])
        self.check(sp, {
            "ab_not_c": lambda g: g[2, :, 0].any() and g[2, :, 1].any() and not g[2, :, 2].any(),
            "a_or_notb": lambda g: g[2, :, 0].any() or not g[2, :, 1].any(),
        }, density=0.08)

    def test_without_approx_fails_on_cancellation(self):
        """Documents the one-unit approximation's failure: an exclude-only cell cancels a valid cell."""
        sp = SpatialProgram()
        sp.any_without_approx("diver5", 2, [(5, c) for c in range(10)], 1)
        mlp = sp.compile()
        g = np.zeros((10, 10, 6), np.float32)
        g[5, 2, 2] = 1          # a diver
        g[5, 7, 1] = 1          # an enemy bullet elsewhere in the row
        h1, _, _ = mlp.inspect_activations(g.reshape(1, -1))
        self.assertLess(h1[0, mlp.unit_map["h1"]["diver5"]], 0)  # wrong: reports no diver

    def test_adjacent_pinned_column(self):
        sp = SpatialProgram()
        sp.adjacent("enemy_left", 0, 1, 0, -1, anchors=[(r, 6) for r in range(1, 9)])
        self.check(sp, {"enemy_left": lambda g: any(g[r, 6, 0] and g[r, 5, 1] for r in range(1, 9))})

    def test_adjacent_whole_grid_reports_cost(self):
        sp = SpatialProgram()
        with self.assertRaises(CapacityError) as ctx:
            sp.adjacent("enemy_left", 0, 1, 0, -1)
        self.assertIn("needs 90 H1", str(ctx.exception))

    def test_priority(self):
        sp = SpatialProgram()
        lo = sp.copy("lo", sp.row_any("x", 0, 0))
        hi = sp.copy("hi", sp.row_any("y", 1, 0))
        sp.rule(lo, 3, level=0); sp.rule(hi, 2, level=1); sp.default(0)
        mlp = sp.compile()
        X = grids(3000, 0.06, seed=3)
        logits = mlp.forward(X.reshape(3000, -1))
        lo_on, hi_on = X[:, 0, :, 0].any(1), X[:, 0, :, 1].any(1)
        want = np.where(hi_on, 2, np.where(lo_on, 3, 0))
        self.assertTrue((logits.argmax(1) == want).all())
        top2 = np.sort(logits, 1)[:, -2:]
        self.assertGreater((top2[:, 1] - top2[:, 0]).min(), 5.0)


if __name__ == "__main__":
    unittest.main()
