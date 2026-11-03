"""Exhaustive checks of the exact H1 bits and of the H2 player-row gate.

1. Every single-entity row configuration (row, type, x, direction) against the Python predicates of A/E/D/G.
2. Every player row: the binary code bits B[i].
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from solution import build_policy, ROWS, ENEMY, GOLD, TRAIL, PLAYER


def run():
    m = build_policy()
    u = m.unit_map["h1"]
    bad = 0; minabs = 9; n = 0
    for r in ROWS:
        for typ in (ENEMY, GOLD):
            for x in range(10):
                for right in (True, False):
                    g = np.zeros((10, 10, 6)); g[5 if r != 5 else 6, 5, PLAYER] = 1
                    g[r, x, typ] = 1
                    back = x - 1 if right else x + 1
                    if 0 <= back <= 9: g[r, back, TRAIL] = 1
                    h1 = m.inspect_activations(g.reshape(1, -1))[0][0]
                    appr = (x < 5 and right) or (x > 5 and not right) or x == 5
                    dist = abs(x - 5)
                    want = {"A": typ == ENEMY and appr, "E": typ == ENEMY and appr and dist <= 2,
                            "D": typ == ENEMY and appr and dist <= 1, "G": typ == GOLD and appr}
                    for (name, row), j in u.items():
                        if name == "B":
                            continue
                        w = want[name] if row == r else False
                        bad += (h1[j] > 0) != w; minabs = min(minabs, abs(h1[j])); n += 1
    for r in ROWS:
        g = np.zeros((10, 10, 6)); g[r, 5, PLAYER] = 1
        h1 = m.inspect_activations(g.reshape(1, -1))[0][0]
        for i in range(3):
            bad += (h1[u[("B", i)]] > 0) != bool(((r - 1) >> i) & 1); n += 1
    print("checked", n, "mismatches", bad, "min |h1|", round(minabs, 5))


if __name__ == "__main__":
    run()


def run_h2(n=20000, seed=0):
    """Feed random consistent H1 bit patterns (+-1) straight into layer 2 and compare every MOVE / ESC unit with
    its Boolean specification (written independently below)."""
    from solution import INNER, EDGE
    m = build_policy()
    u = m.unit_map["h1"]
    rng = np.random.RandomState(seed)
    bad = 0; minabs = 9
    for _ in range(n):
        bits = {}
        pr = rng.randint(1, 9)
        for i in range(3):
            bits[("B", i)] = bool(((pr - 1) >> i) & 1)
        st = {}
        for r in ROWS:   # per-row state: 0 none, 1 enemy far-approaching, 2 enemy dist 2, 3 enemy dist<=1, 4 gold
            st[r] = rng.randint(0, 5)
            if r in INNER:
                bits[("A", r)] = st[r] in (1, 2, 3); bits[("G", r)] = st[r] == 4
            bits[("E", r)] = st[r] in (2, 3); bits[("D", r)] = st[r] == 3
        h = np.zeros(m.W2.shape[0])
        for key, j in u.items():
            h[j] = 1.0 if bits[key] else -1.0
        h2 = np.tanh(h @ m.W2 + m.b2)
        g = lambda name, r: bits.get((name, r), False)
        k = 0
        for r in ROWS:                          # MOVE units, same enumeration order as solution.py
            for d in (-1, 1):
                t = r + d
                if t not in ROWS or t in EDGE:
                    continue
                gates = r == pr and not g("E", t) and not g("A", t) and not g("G", r) and not g("E", t + d)
                gold = sum((1 if (r2 - r) * d > 0 else -1) for r2 in ROWS if r2 != r and g("G", r2))
                want = gates and (g("E", r) or g("A", r) or r in EDGE or g("G", t) or gold >= 1)
                bad += (h2[k] > 0) != want; minabs = min(minabs, abs(h2[k])); k += 1
        for r in ROWS:                          # ESC units
            for d in (-1, 1):
                t = r + d
                if t not in ROWS:
                    continue
                want = r == pr and ((g("E", r) and not g("E", t)) or (g("D", r) and g("E", t) and not g("D", t)))
                bad += (h2[k] > 0) != want; minabs = min(minabs, abs(h2[k])); k += 1
    print("H2 random patterns", n, "mismatches", bad, "min |h2|", round(minabs, 5))


if __name__ == "__main__":
    run_h2()
