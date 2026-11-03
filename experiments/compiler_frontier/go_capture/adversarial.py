"""Adversarial tests: long snakes, spirals, huge groups, random self-avoiding white walks.
Run: OPENBLAS_NUM_THREADS=1 python3 experiments/compiler_frontier/go_capture/adversarial.py [n_layers]
     GO_SOL=solution_logdepth python3 .../adversarial.py [n_squarings]"""
import importlib
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
S = importlib.import_module(os.environ.get('GO_SOL', 'solution'))
from check import _go_groups, go_play  # noqa: E402


def legal(b, m):
    if b[m] != 0:
        return False
    for color in (1, -1):
        if any(not libs for _, libs in _go_groups(b, color)):
            return False
    nb = go_play(b, m)
    own = [libs for st, libs in _go_groups(nb, 1) if m in st][0]
    return bool(own)


def white_radius(b, m):
    """Max over white stones (after the move) of 1 + BFS distance (inside the white group) to a stone
    adjacent to a liberty = number of propagation iterations this case needs."""
    b = b.copy(); b[m] = 1
    dist = {}
    q = deque()
    for r in range(9):
        for c in range(9):
            if b[r, c] == -1 and any(0 <= r + dr < 9 and 0 <= c + dc < 9 and b[r + dr, c + dc] == 0
                                     for dr, dc in S.DIRS):
                dist[(r, c)] = 1; q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in S.DIRS:
            y, x = r + dr, c + dc
            if 0 <= y < 9 and 0 <= x < 9 and b[y, x] == -1 and (y, x) not in dist:
                dist[(y, x)] = dist[(r, c)] + 1; q.append((y, x))
    return max(dist.values(), default=0)


def build_snake(length_target, rng, capture):
    """Random self-avoiding white walk; surround it by black; leave exactly one white liberty at the
    walk's far end; then the move either fills that liberty (capture) or is elsewhere (no capture).
    Black groups get liberties by leaving empty points that are not adjacent to white."""
    for _ in range(2000):
        b = np.zeros((9, 9), int)
        path = [(rng.randint(9), rng.randint(9))]
        S_ = {path[0]}
        while len(path) < length_target:
            r, c = path[-1]
            nbrs = [(r + dr, c + dc) for dr, dc in S.DIRS if 0 <= r + dr < 9 and 0 <= c + dc < 9]
            # snake-like: new cell must touch only the previous cell among path cells (induced path)
            cand = [p for p in nbrs if p not in S_ and
                    sum((p[0] + dr, p[1] + dc) in S_ for dr, dc in S.DIRS) == 1]
            if not cand:
                break
            p = cand[rng.randint(len(cand))]
            path.append(p); S_.add(p)
        if len(path) < length_target:
            continue
        for p in path:
            b[p] = -1
        tail_lib = None
        r, c = path[-1]
        for dr, dc in S.DIRS:
            y, x = r + dr, c + dc
            if 0 <= y < 9 and 0 <= x < 9 and (y, x) not in S_:
                tail_lib = (y, x); break
        if tail_lib is None:
            continue
        for y in range(9):
            for x in range(9):
                if b[y, x] == 0 and (y, x) != tail_lib:
                    b[y, x] = 1
        # black liberties: empty points not adjacent to white
        for y in range(9):
            for x in range(9):
                if b[y, x] == 1 and not any(0 <= y + dr < 9 and 0 <= x + dc < 9 and b[y + dr, x + dc] == -1
                                            for dr, dc in S.DIRS) and rng.rand() < 0.3:
                    b[y, x] = 0
        if capture:
            m = tail_lib
        else:
            emp = [tuple(e) for e in np.argwhere(b == 0) if tuple(e) != tail_lib]
            if not emp:
                continue
            m = emp[rng.randint(len(emp))]
        if legal(b, m):
            return b, m
    return None


def serpentine_cases():
    b = np.zeros((9, 9), int)
    for r in (0, 2, 4, 6, 8):
        b[r, :] = -1
    for i, r in enumerate((1, 3, 5, 7)):
        b[r, :] = 1
        b[r, 8 if i % 2 == 0 else 0] = -1
    b[8, 7] = 0; b[8, 8] = 0
    return [("serpentine47_nocap", b, (8, 8)), ("serpentine47_cap", b, (8, 7))]


def all_white(capture):
    b = -np.ones((9, 9), int)
    b[4, 4] = 0
    if capture:
        return b, (4, 4)
    b[4, 5] = 0  # second liberty (shared), black keeps a liberty too
    return b, (4, 4)


def run_case(model, b, m):
    regs, L = S.encode((b, m))
    raw = model.run(regs, L)
    pred = S.decode(raw)
    ok = np.array_equal(pred, go_play(b, m))
    dev = float(np.max(np.abs(raw[:81] - np.rint(raw[:81]))))
    return ok, dev


def main():
    if hasattr(S, "N_SQ"):  # log-depth variant: argument = number of squaring layers
        if len(sys.argv) > 1:
            S.N_SQ = int(sys.argv[1]); S.N_LAYERS = 2 + S.N_SQ
        print(f"log-depth variant: {S.N_SQ} squarings -> R covers in-group distance {2 ** (S.N_SQ - 1)}")
    elif len(sys.argv) > 1:
        S.N_LAYERS = int(sys.argv[1])
    n_layers = S.N_LAYERS
    model = S.build_model()
    rng = np.random.RandomState(12345)
    cases = []
    for cap in (True, False):
        cases.append(("all_white_80", ) + all_white(cap))
    for L in (10, 20, 30, 35, 40, 45, 50):
        for cap in (True, False):
            for k in range(6):
                c = build_snake(L, rng, cap)
                if c is not None:
                    cases.append((f"snake{L}_{'cap' if cap else 'nocap'}",) + c)
    # random dense boards with big white groups (p_white high)
    for _ in range(100):
        b = rng.choice([-1, 0, 1], size=(9, 9), p=[0.6, 0.1, 0.3])
        for color in (1, -1):
            for st, libs in _go_groups(b, color):
                if not libs:
                    for s in st: b[s] = 0
        emp = [tuple(e) for e in np.argwhere(b == 0)]
        lg = [m for m in emp if legal(b, m)]
        if lg:
            cases.append(("dense_white",) + (b, lg[rng.randint(len(lg))]))
    # Stress cases outside the legal-input distribution (some black groups have no liberty; the network's
    # white-liberty propagation does not look at black, and go_play only removes white): a 47-stone
    # serpentine induced white snake whose only liberty is at its tail -> needs 47 iterations.
    for name, b, m in serpentine_cases():
        ok, dev = run_case(model, b, m)
        rad = white_radius(b, m)
        print(f"  stress {name}: {'OK' if ok else 'FAIL'} (needs {rad} iters, has {n_layers}) dev={dev:.1e}")
    n_ok, worst_dev, max_rad, fails = 0, 0.0, 0, []
    by_kind = {}
    for name, b, m in cases:
        assert legal(b, m), name
        ok, dev = run_case(model, b, m)
        rad = white_radius(b, m)
        max_rad = max(max_rad, rad)
        worst_dev = max(worst_dev, dev)
        n_ok += ok
        k = by_kind.setdefault(name.split("_")[0], [0, 0, 0]); k[0] += ok; k[1] += 1; k[2] = max(k[2], rad)
        if not ok:
            fails.append((name, rad))
    print(f"n_layers={n_layers}: {n_ok}/{len(cases)} correct; max iterations needed={max_rad}; "
          f"max |raw-round(raw)|={worst_dev:.2e}")
    for k, (a, t, r) in by_kind.items():
        print(f"  {k:12s} {a}/{t}  (max needed iters {r})")
    if fails:
        print("  failures (name, needed iters):", fails[:10])


if __name__ == "__main__":
    main()
