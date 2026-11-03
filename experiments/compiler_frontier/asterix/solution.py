"""Asterix policy built from explicit weight formulas (no training).  See NOTES.md."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
from primitives import (ENEMY_K1, ENEMY_K2, APPROACH_ANY, GOLD_ANY, h1_row_detector, h1_any,  # noqa: E402
                        h2_threshold, make_mlp)

PLAYER, ENEMY, GOLD, TRAIL = 0, 1, 2, 4          # observation channels (asterix_obs)
NOOP, LEFT, UP, RIGHT, DOWN, FIRE = range(6)       # env action_map ['n','l','u','r','d','f']
ROWS = range(1, 9)                                 # rows the player / entities can occupy
EDGE = (1, 8)
INNER = range(2, 8)
# The player never moves horizontally (left/right/fire logits are -60), so it stays pinned to column 5 and every
# row's threat can be read relative to column 5 with one H1 unit per row.


def build_policy():
    H1, H2 = 32, 32
    W1 = np.zeros((600, H1)); b1 = np.zeros(H1)
    W2 = np.zeros((H1, H2)); b2 = np.zeros(H2)
    W3 = np.zeros((H2, 6)); b3 = np.zeros(6)
    u = {}
    j = 0
    # ---------------- H1: exact bits
    # Player row in binary (3 units): B[i] = player in some row r with bit i of (r-1) set.
    for i in range(3):
        h1_any(W1, b1, j, [(r, c, PLAYER) for r in ROWS if ((r - 1) >> i) & 1 for c in range(10)])
        u[("B", i)] = j; j += 1
    for r in INNER:  # A[r]: an enemy approaches col 5 (any distance) or is on it  -> row will turn dangerous
        h1_row_detector(W1, b1, j, r, ENEMY, TRAIL, APPROACH_ANY); u[("A", r)] = j; j += 1
    for r in ROWS:   # E[r]: an enemy approaches col 5 within 2 cells or is on it -> row is (about to be) deadly
        h1_row_detector(W1, b1, j, r, ENEMY, TRAIL, ENEMY_K2); u[("E", r)] = j; j += 1
    for r in ROWS:   # D[r]: an enemy is adjacent to col 5 and approaching, or on it -> entering it now kills
        h1_row_detector(W1, b1, j, r, ENEMY, TRAIL, ENEMY_K1); u[("D", r)] = j; j += 1
    for r in INNER:  # G[r]: a gold approaches col 5 (any distance) or is on it   -> row is a safe, paying haven
        h1_row_detector(W1, b1, j, r, GOLD, TRAIL, GOLD_ANY); u[("G", r)] = j; j += 1
    n_h1 = j

    # Every H2 unit is one linear threshold over H1 bits.  The player-row gate uses the binary code: the sum
    # sum_i (+M if bit i of (r-1) else -M) * B[i] equals M*popcount(r-1) iff the player is in row r and is at
    # least M lower otherwise, so "player in row r" costs no H1 unit per row.
    def new_unit(gate_row, M):
        terms = {}
        def add(name, row, c):
            if (name, row) in u:                    # A/G of edge rows are not computed (never used there)
                key = u[(name, row)]
                terms[key] = terms.get(key, 0) + c
        pc = 0
        for i in range(3):
            if ((gate_row - 1) >> i) & 1:
                add("B", i, M); pc += 1
            else:
                add("B", i, -M)
        return terms, add, M * pc            # gate satisfied <=> its terms contribute exactly M*pc

    # ---- MOVE[r, d] (12 units): a move to a *good* target t = r + d, i.e. an inner row (2..7) with no
    # approaching enemy at all (from an edge row, any row-2/7 target that is not deadly counts).
    # Hard gates (weight M): P[r], !E[t], !A[t] (inner targets), !G[r], !E[t+d] (the row beyond the target: a
    # 10% sticky repeat of the move would carry the player there next step).  Then fires iff S >= 1 with
    #   S = 20*E[r] + 20*A[r] + 20*[r is an edge row] + 10*G[t] + sum_{r2 != r} (+-1)*G[r2]
    # i.e. escape (E[r]), pre-empt (A[r]) or leave an edge row always; otherwise only to move towards gold
    # (gold in t, or more golds on that side).
    M = 120
    k = 0
    moves = {UP: [], DOWN: []}
    emerg = {UP: [], DOWN: []}
    for r in ROWS:
        for d, act in ((-1, UP), (+1, DOWN)):
            t = r + d
            if t not in ROWS or t in EDGE:
                continue
            terms, add, gate = new_unit(r, M)
            add("E", t, -M); add("A", t, -M); add("G", r, -M)
            add("E", r, 20); add("A", r, 20); add("G", t, 10)
            for r2 in ROWS:
                if r2 != r:
                    add("G", r2, 1 if (r2 - r) * d > 0 else -1)
            if t + d in ROWS:
                add("E", t + d, -M)
            edge = 20 if r in EDGE else 0
            h2_threshold(W2, b2, k, terms, gate - edge)
            moves[act].append(k); k += 1
    # ---- ESC[r, d] (14 units): an *acceptable* escape.  Gate P[r]; fires iff E[r] + D[r] - E[t] - D[t] >= 1:
    #   E[r] & !E[t]               (own enemy within 2, target has none within 2), or
    #   D[r] & E[t] & !D[t]        (staying kills; the target's enemy is exactly 2 away, so entering survives)
    # and stays silent when own and target threats are equally close (D implies E, so this is exhaustive).
    for r in ROWS:
        for d, act in ((-1, UP), (+1, DOWN)):
            t = r + d
            if t not in ROWS:
                continue
            terms, add, gate = new_unit(r, 10)
            add("E", r, 1); add("D", r, 1); add("E", t, -1); add("D", t, -1)
            h2_threshold(W2, b2, k, terms, gate)
            emerg[act].append(k); k += 1
    n_h2 = k

    # ---------------- output (in H2 bits; bit = (h+1)/2, coefficient c -> W3 += c/2, b3 += c/2):
    #   noop = -24*[any ESC];  move = -20 + 40*[MOVE] + 24*[ESC];  left/right/fire = -60 (never chosen)
    def out(k, act, c):
        W3[k, act] += c / 2.0; b3[act] += c / 2.0
    for act in (UP, DOWN):
        b3[act] = -20.0
        for k in moves[act]:
            out(k, act, 40.0)
        for k in emerg[act]:
            out(k, act, 24.0); out(k, NOOP, -24.0)
    b3[LEFT] = b3[RIGHT] = b3[FIRE] = -60.0
    return make_mlp(W1, b1, W2, b2, W3, b3, {"n_h1": n_h1, "n_h2": n_h2, "h1": dict(u)})


if __name__ == "__main__":
    m = build_policy()
    print("H1 used", m.unit_map["n_h1"], "H2 used", m.unit_map["n_h2"])
