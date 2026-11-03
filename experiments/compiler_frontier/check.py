"""Held-out checker for the compiler-frontier tasks.

Usage: python experiments/compiler_frontier/check.py <task> [path/to/solution.py]

Game tasks: solution.build_policy() -> ExecutableMLP with shape 600 -> <=32 -> <=32 -> 6, tanh.
Transformer tasks: solution.build_model() -> ExecutableTransformer,
                   solution.encode(example) -> (reg_inputs: dict, seq_len: int),
                   solution.decode(raw_output) -> answer.
encode may only place raw input symbols (fixed per-symbol codes / one-hots and position codes) into registers;
decode may only round, threshold or argmax the output register. Both are reviewed by hand.
Authors develop with --dev; final scores are recomputed by the evaluator on seeds chosen after submission.
"""
import importlib.util
import itertools
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from neural_compiler import ExecutableMLP, ExecutableTransformer  # noqa: E402

EVAL_SEED = 20261103
N_CASES = 500
LIMITS = {"d_model": 512, "n_layers": 48, "n_heads": 16}
HARD = {"sudoku", "go_capture", "program", "multiply", "chess_mate1", "chess_check", "poker", "mahjong"}
HARD_LIMITS = {"d_model": 1024, "n_layers": 96, "n_heads": 16}


# ------------------------------------------------------------------------------------------ task generators
def gen_dyck(rng):
    n = 16
    if rng.rand() < 0.5:  # balanced: random walk conditioned to stay >= 0 and end at 0
        while True:
            s, depth = [], 0
            for i in range(n):
                remaining = n - i
                if depth == 0:
                    c = "("
                elif depth >= remaining:
                    c = ")"
                else:
                    c = "(" if rng.rand() < 0.5 else ")"
                depth += 1 if c == "(" else -1
                s.append(c)
            if depth == 0:
                return "".join(s)
    if rng.rand() < 0.5:  # equal counts but not balanced (hard negatives)
        while True:
            s = list("(" * 8 + ")" * 8)
            rng.shuffle(s)
            s = "".join(s)
            if not dyck_answer(s):
                return s
    return "".join(rng.choice(["(", ")"], n))


def dyck_answer(s):
    d = 0
    for c in s:
        d += 1 if c == "(" else -1
        if d < 0:
            return False
    return d == 0


def gen_life(rng):
    return (rng.rand(8, 8) < rng.uniform(0.2, 0.5)).astype(int)


def life_answer(g):
    n = sum(np.roll(np.roll(g, dr, 0), dc, 1) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))
    return ((n == 3) | ((g == 1) & (n == 2))).astype(int)


def gen_lcs(rng):
    return (rng.randint(0, 4, 6), rng.randint(0, 4, 6))


def lcs_answer(ab):
    a, b = ab
    L = np.zeros((7, 7), int)
    for i in range(6):
        for j in range(6):
            L[i + 1, j + 1] = L[i, j] + 1 if a[i] == b[j] else max(L[i, j + 1], L[i + 1, j])
    return int(L[6, 6])


LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def ttt_winner(b):
    for i, j, k in LINES:
        if b[i] != 0 and b[i] == b[j] == b[k]:
            return b[i]
    return 0


@lru_cache(maxsize=None)
def ttt_value(b):
    """Minimax value for the side to move (+1 win, 0 draw, -1 loss). Board: 1 = X, -1 = O, 0 = empty; X moves first."""
    mover = 1 if b.count(1) == b.count(-1) else -1
    best = -2
    for i in range(9):
        if b[i] == 0:
            nb = b[:i] + (mover,) + b[i + 1:]
            w = ttt_winner(nb)
            v = 1 if w == mover else (0 if 0 not in nb else -ttt_value(nb))
            best = max(best, v)
    return best


@lru_cache(maxsize=1)
def ttt_positions():
    out = set()
    def walk(b):
        if ttt_winner(b) or 0 not in b:
            return
        out.add(b)
        mover = 1 if b.count(1) == b.count(-1) else -1
        for i in range(9):
            if b[i] == 0:
                walk(b[:i] + (mover,) + b[i + 1:])
    walk((0,) * 9)
    return sorted(out)


def gen_ttt(rng):
    P = ttt_positions()
    return P[rng.randint(len(P))]


# ---- hard tier ------------------------------------------------------------------------------------------
def _sudoku_solutions(g, limit=2):
    """Count solutions (up to `limit`) by backtracking on the most constrained cell. g: 9x9 int, 0 = empty."""
    g = g.copy()
    count = [0]

    def cands(r, c):
        used = set(g[r]) | set(g[:, c]) | set(g[r // 3 * 3:r // 3 * 3 + 3, c // 3 * 3:c // 3 * 3 + 3].ravel())
        return [v for v in range(1, 10) if v not in used]

    sol = [None]

    def rec():
        if count[0] >= limit:
            return
        best = None
        for r in range(9):
            for c in range(9):
                if g[r, c] == 0:
                    cs = cands(r, c)
                    if best is None or len(cs) < len(best[2]):
                        best = (r, c, cs)
        if best is None:
            count[0] += 1
            sol[0] = g.copy()
            return
        r, c, cs = best
        for v in cs:
            g[r, c] = v
            rec()
            g[r, c] = 0
    rec()
    return count[0], sol[0]


def gen_sudoku(rng):
    g = np.zeros((9, 9), int)
    base = np.array([[(3 * (r % 3) + r // 3 + c) % 9 + 1 for c in range(9)] for r in range(9)])
    # random valid full grid: permute digits, rows within bands, bands, cols within stacks, stacks, transpose
    perm = rng.permutation(9) + 1
    g = perm[base - 1]
    rows = np.concatenate([b * 3 + rng.permutation(3) for b in rng.permutation(3)])
    cols = np.concatenate([b * 3 + rng.permutation(3) for b in rng.permutation(3)])
    g = g[rows][:, cols]
    if rng.rand() < 0.5:
        g = g.T.copy()
    puzzle = g.copy()
    target_blanks = rng.randint(40, 55)
    for idx in rng.permutation(81):
        if (puzzle == 0).sum() >= target_blanks:
            break
        r, c = divmod(idx, 9)
        v, puzzle[r, c] = puzzle[r, c], 0
        if _sudoku_solutions(puzzle)[0] != 1:
            puzzle[r, c] = v
    return puzzle


def sudoku_answer(p):
    return _sudoku_solutions(p, limit=1)[1]


def _go_groups(board, color):
    """Return list of (stones, liberties) sets for `color` on a 9x9 board (1 black, -1 white, 0 empty)."""
    seen, out = set(), []
    for r in range(9):
        for c in range(9):
            if board[r, c] == color and (r, c) not in seen:
                stack, stones, libs = [(r, c)], set(), set()
                seen.add((r, c))
                while stack:
                    y, x = stack.pop()
                    stones.add((y, x))
                    for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                        if 0 <= ny < 9 and 0 <= nx < 9:
                            if board[ny, nx] == 0:
                                libs.add((ny, nx))
                            elif board[ny, nx] == color and (ny, nx) not in seen:
                                seen.add((ny, nx)); stack.append((ny, nx))
                out.append((stones, libs))
    return out


def go_play(board, move):
    """Black (1) plays at `move`; remove white groups left without liberties. Returns the new board."""
    b = board.copy()
    b[move] = 1
    for stones, libs in _go_groups(b, -1):
        if not libs:
            for s in stones:
                b[s] = 0
    return b


def gen_go(rng):
    """Random legal position (no group without liberties) and a non-suicidal black move.
    About half of the cases are chosen among capturing moves when one exists."""
    while True:
        b = rng.choice([-1, 0, 1], size=(9, 9), p=[0.33, 0.34, 0.33] if rng.rand() < 0.5 else [0.4, 0.3, 0.3])
        for color in (1, -1):
            for stones, libs in _go_groups(b, color):
                if not libs:
                    for s in stones:
                        b[s] = 0
        empties = [tuple(e) for e in np.argwhere(b == 0)]
        legal, capturing = [], []
        for m in empties:
            nb = go_play(b, m)
            own = [libs for stones, libs in _go_groups(nb, 1) if m in stones][0]
            if own:
                legal.append(m)
                if (nb == -1).sum() < (b == -1).sum():
                    capturing.append(m)
        if not legal:
            continue
        pool = capturing if (capturing and rng.rand() < 0.5) else legal
        return (b, pool[rng.randint(len(pool))])


def go_answer(ex):
    return go_play(*ex)


# Tiny register machine: 4 registers (mod 16), 8 instructions, executed for exactly 12 steps (HALT stays put).
# Instruction (op, a, b): 0 LOADI ra <- b | 1 ADD ra <- ra + rb | 2 SUB ra <- ra - rb | 3 JNZ if ra != 0: pc <- b
#                         4 DEC ra <- ra - 1 | 5 HALT.  All arithmetic mod 16. pc wraps mod 8.
def gen_prog(rng):
    prog = []
    for _ in range(8):
        op = rng.choice(6, p=[0.25, 0.2, 0.15, 0.2, 0.15, 0.05])
        prog.append((int(op), int(rng.randint(4)), int(rng.randint(16) if op == 0 else rng.randint(8) if op == 3 else rng.randint(4))))
    return tuple(prog)


def prog_answer(prog):
    r, pc = [0, 0, 0, 0], 0
    for _ in range(12):
        op, a, b = prog[pc]
        if op == 5:
            break
        if op == 0:
            r[a] = b
        elif op == 1:
            r[a] = (r[a] + r[b]) % 16
        elif op == 2:
            r[a] = (r[a] - r[b]) % 16
        elif op == 4:
            r[a] = (r[a] - 1) % 16
        if op == 3 and r[a] != 0:
            pc = b
        else:
            pc = (pc + 1) % 8
    return tuple(r)


def gen_mul(rng):
    return (int(rng.randint(100000, 1000000)), int(rng.randint(100000, 1000000)))


# ---- card / board games -----------------------------------------------------------------------------------
PIECE_CODE = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6}


def _chess_example(board):
    import chess
    arr = np.zeros((8, 8), int)  # arr[rank][file], rank 0 = rank 1 (white's back rank)
    for sq, pc in board.piece_map().items():
        arr[chess.square_rank(sq), chess.square_file(sq)] = PIECE_CODE[pc.symbol().upper()] * (1 if pc.color else -1)
    return {"board": arr, "turn": 1 if board.turn else -1,
            "castling": (board.has_kingside_castling_rights(True), board.has_queenside_castling_rights(True),
                         board.has_kingside_castling_rights(False), board.has_queenside_castling_rights(False)),
            "ep": -1 if board.ep_square is None else int(board.ep_square), "fen": board.fen()}


def _random_chess_position(rng):
    import chess
    b = chess.Board()
    for _ in range(rng.randint(10, 120)):
        moves = list(b.legal_moves)
        if not moves or b.is_game_over():
            break
        b.push(moves[rng.randint(len(moves))])
    return b


def _has_mate1(b):
    for m in list(b.legal_moves):
        b.push(m)
        mate = b.is_checkmate()
        b.pop()
        if mate:
            return True
    return False


def gen_chess_mate1(rng):
    want = rng.rand() < 0.5
    while True:
        b = _random_chess_position(rng)
        if not b.is_game_over() and _has_mate1(b) == want:
            return _chess_example(b)


def chess_mate1_answer(ex):
    import chess
    return _has_mate1(chess.Board(ex["fen"]))


def gen_chess_check(rng):
    want = rng.rand() < 0.5
    while True:
        b = _random_chess_position(rng)
        if not b.is_game_over() and b.is_check() == want:
            return _chess_example(b)


def chess_check_answer(ex):
    import chess
    return chess.Board(ex["fen"]).is_check()


def _hand5_rank(cards):
    """cards: 5 (rank 2..14, suit 0..3). Returns a comparable tuple."""
    ranks = sorted((r for r, _ in cards), reverse=True)
    flush = len({s for _, s in cards}) == 1
    uniq = sorted(set(ranks), reverse=True)
    straight_high = 0
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight_high = uniq[0]
        elif uniq == [14, 5, 4, 3, 2]:
            straight_high = 5
    counts = sorted(((ranks.count(r), r) for r in uniq), reverse=True)
    shape = [c for c, _ in counts]
    order = [r for _, r in counts]
    if straight_high and flush:
        return (8, straight_high)
    if shape == [4, 1]:
        return (7, *order)
    if shape == [3, 2]:
        return (6, *order)
    if flush:
        return (5, *ranks)
    if straight_high:
        return (4, straight_high)
    if shape == [3, 1, 1]:
        return (3, *order)
    if shape == [2, 2, 1]:
        return (2, *order)
    if shape == [2, 1, 1, 1]:
        return (1, *order)
    return (0, *ranks)


def _hand7_rank(cards):
    return max(_hand5_rank(c) for c in itertools.combinations(cards, 5))


def gen_poker(rng):
    deck = [(r, s) for r in range(2, 15) for s in range(4)]
    idx = rng.permutation(52)[:9]
    cards = [deck[i] for i in idx]
    return {"hole_a": cards[0:2], "hole_b": cards[2:4], "board": cards[4:9]}


def poker_answer(ex):
    a = _hand7_rank(ex["hole_a"] + ex["board"])
    b = _hand7_rank(ex["hole_b"] + ex["board"])
    return int(a > b) - int(a < b)


# Mahjong: 34 tile types: 0-8 characters 1-9, 9-17 dots, 18-26 bamboo, 27-33 honours. Standard complete hand only:
# four melds (pung = three identical, or chow = three consecutive numbers in one suit; honours cannot chow) + one pair.
def _mj_complete(counts):
    counts = list(counts)

    def melds(c):
        i = next((k for k in range(34) if c[k]), None)
        if i is None:
            return True
        if c[i] >= 3:
            c[i] -= 3
            ok = melds(c)
            c[i] += 3
            if ok:
                return True
        if i < 27 and i % 9 <= 6 and c[i + 1] and c[i + 2]:
            c[i] -= 1; c[i + 1] -= 1; c[i + 2] -= 1
            ok = melds(c)
            c[i] += 1; c[i + 1] += 1; c[i + 2] += 1
            if ok:
                return True
        return False

    for p in range(34):
        if counts[p] >= 2:
            counts[p] -= 2
            ok = melds(counts)
            counts[p] += 2
            if ok:
                return True
    return False


def gen_mahjong(rng):
    while True:
        counts = [0] * 34
        p = rng.randint(34)
        counts[p] += 2
        for _ in range(4):
            if rng.rand() < 0.5:
                t = rng.randint(34)
                counts[t] += 3
            else:
                s, n = rng.randint(3), rng.randint(7)
                for k in range(3):
                    counts[s * 9 + n + k] += 1
        if max(counts) > 4:
            continue
        tiles = [t for t in range(34) for _ in range(counts[t])]
        if rng.rand() < 0.5:  # near miss: replace one tile
            i = rng.randint(14)
            new = rng.randint(34)
            if new == tiles[i] or tiles.count(new) >= 4:
                continue
            tiles[i] = new
            if rng.rand() < 0.5 and _mj_complete(np.bincount(tiles, minlength=34)):
                continue
        return tuple(sorted(tiles))


def mahjong_answer(tiles):
    return _mj_complete(np.bincount(list(tiles), minlength=34))


TASKS = {
    "chess_mate1": (gen_chess_mate1, chess_mate1_answer, lambda p, a: bool(p) == a),
    "chess_check": (gen_chess_check, chess_check_answer, lambda p, a: bool(p) == a),
    "poker": (gen_poker, poker_answer, lambda p, a: int(p) == a),
    "mahjong": (gen_mahjong, mahjong_answer, lambda p, a: bool(p) == a),
    "sudoku": (gen_sudoku, sudoku_answer, lambda p, a: np.array_equal(np.asarray(p).reshape(9, 9), a)),
    "go_capture": (gen_go, go_answer, lambda p, a: np.array_equal(np.asarray(p).reshape(9, 9), a)),
    "program": (gen_prog, prog_answer, lambda p, a: tuple(int(v) for v in p) == a),
    "multiply": (gen_mul, lambda ab: ab[0] * ab[1], lambda p, a: int(p) == a),
    "dyck": (gen_dyck, dyck_answer, lambda p, a: bool(p) == a),
    "life": (gen_life, life_answer, lambda p, a: np.array_equal(np.asarray(p).reshape(8, 8), a)),
    "lcs": (gen_lcs, lcs_answer, lambda p, a: int(p) == a),
    "tictactoe": (gen_ttt, ttt_value, lambda p, a: int(p) == a),
}


# ------------------------------------------------------------------------------------------ checking
def load(path):
    spec = importlib.util.spec_from_file_location("solution", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_game(task, mod, episodes=200, seed=0):
    from minatar_envs import SeaquestEnv, AsterixEnv, seaquest_obs, asterix_obs
    from stage1_mlp_benchmark import evaluate_game
    mlp = mod.build_policy()
    assert isinstance(mlp, ExecutableMLP), "build_policy must return ExecutableMLP"
    assert mlp.activation == "tanh" and mlp.W1.shape[0] == 600 and mlp.W3.shape[1] == 6
    assert mlp.W1.shape[1] <= 32 and mlp.W2.shape[1] <= 32, "hidden widths must be <= 32"
    env_cls, obs_fn = {"seaquest": (SeaquestEnv, seaquest_obs), "asterix": (AsterixEnv, asterix_obs)}[task]
    score, se, length = evaluate_game(env_cls, obs_fn, mlp, episodes, seed)
    return dict(task=task, score=round(score, 2), se=round(se, 2), length=round(length, 1),
                nonzero=mlp.nonzero_params, total=mlp.total_params, episodes=episodes, seed=seed)


def check_transformer(task, mod, seed=EVAL_SEED):
    gen, answer, same = TASKS[task]
    model = mod.build_model()
    assert isinstance(model, ExecutableTransformer), "build_model must return ExecutableTransformer"
    for k, lim in (HARD_LIMITS if task in HARD else LIMITS).items():
        assert getattr(model, k) <= lim, f"{k}={getattr(model, k)} exceeds {lim}"
    rng = np.random.RandomState(seed)
    correct, fails = 0, []
    for _ in range(N_CASES):
        ex = gen(rng)
        regs, seq_len = mod.encode(ex)
        pred = mod.decode(model.run(regs, seq_len))
        ok = same(pred, answer(ex))
        correct += ok
        if not ok and len(fails) < 3:
            fails.append(repr(ex))
    return dict(task=task, correct=correct, total=N_CASES, nonzero=model.nonzero_params, total_params=model.total_params,
                d_model=model.d_model, n_layers=model.n_layers, n_heads=model.n_heads, example_failures=fails)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("solution", nargs="?")
    ap.add_argument("--dev", action="store_true", help="development seeds (authors use this; final evaluation uses unseen seeds)")
    ap.add_argument("--seed", type=int, default=None, help="evaluator only")
    a = ap.parse_args()
    path = Path(a.solution) if a.solution else Path(__file__).parent / a.task / "solution.py"
    mod = load(path)
    if a.task in ("seaquest", "asterix"):
        res = check_game(a.task, mod, episodes=50 if a.dev else 200, seed=100 if a.dev else (a.seed or 0))
    else:
        res = check_transformer(a.task, mod, seed=7 if a.dev else (a.seed or EVAL_SEED))
    print(json.dumps(res))
