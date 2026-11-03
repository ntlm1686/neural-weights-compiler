"""Chess arena for a compiled policy network.

The submitted network sees the raw position once per move (one forward pass, no search outside the network) and
outputs a score for every (from_square, to_square) pair. The arena masks illegal moves (standard action masking) and
plays the highest-scoring legal move; promotions default to a queen. It also records how often the network's
unmasked top-scoring move was legal.

solution.py must define:
  build_model()               -> ExecutableTransformer or ExecutableMLP
  encode(position: dict)      -> (reg_inputs, seq_len) for transformers, or a flat input vector for MLPs
                                 (raw placement only; the dict has board/turn/castling/ep, see check._chess_example)
  decode(raw_output)          -> array reshapeable to (64, 64): score[from_sq, to_sq], square = rank*8 + file
                                 (read / reshape only)

Opponents: random mover, greedy capturer (1-ply material) and a depth-2 alpha-beta material searcher.
Usage: python arena.py [solution.py] [--games N] [--dev]
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import chess
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2]))
from check import _chess_example  # noqa: E402

VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def material(b, color):
    return sum(VALUE[p.piece_type] * (1 if p.color == color else -1) for p in b.piece_map().values())


class NetPlayer:
    def __init__(self, mod):
        self.mod, self.model = mod, mod.build_model()
        self.legal_top = self.moves = 0
        self.secs = 0.0

    def __call__(self, b, rng):
        t = time.time()
        enc = self.mod.encode(_chess_example(b))
        raw = self.model.run(*enc) if isinstance(enc, tuple) else self.model.forward(np.asarray(enc))
        score = np.asarray(self.mod.decode(raw), dtype=float).reshape(64, 64)
        self.secs += time.time() - t
        legal = list(b.legal_moves)
        fr, to = np.unravel_index(np.argmax(score), score.shape)
        self.moves += 1
        self.legal_top += any(m.from_square == fr and m.to_square == to for m in legal)
        best = max(legal, key=lambda m: (score[m.from_square, m.to_square],
                                         m.promotion in (None, chess.QUEEN)))
        return best


def random_player(b, rng):
    moves = list(b.legal_moves)
    return moves[rng.randint(len(moves))]


def greedy_player(b, rng):
    moves = list(b.legal_moves)
    rng.shuffle(moves)
    def gain(m):
        if b.is_capture(m):
            victim = b.piece_at(m.to_square)
            return VALUE[victim.piece_type] if victim else 1
        return 0
    for m in moves:  # take a mate in one if available
        b.push(m); mate = b.is_checkmate(); b.pop()
        if mate:
            return m
    return max(moves, key=gain)


def _ab(b, depth, alpha, beta, color):
    if b.is_checkmate():
        return -1000 if b.turn == color else 1000
    if depth == 0 or b.is_game_over():
        return material(b, color)
    maximizing = b.turn == color
    best = -10 ** 9 if maximizing else 10 ** 9
    for m in b.legal_moves:
        b.push(m); v = _ab(b, depth - 1, alpha, beta, color); b.pop()
        if maximizing:
            best = max(best, v); alpha = max(alpha, v)
        else:
            best = min(best, v); beta = min(beta, v)
        if beta <= alpha:
            break
    return best


def minimax2_player(b, rng):
    moves = list(b.legal_moves)
    rng.shuffle(moves)
    color = b.turn
    best, best_v = moves[0], -10 ** 9
    for m in moves:
        b.push(m); v = _ab(b, 1, -10 ** 9, 10 ** 9, color); b.pop()
        if v > best_v:
            best, best_v = m, v
    return best


OPPONENTS = {"random": random_player, "greedy": greedy_player, "minimax2": minimax2_player}


def play(white, black, rng, opening_plies=2, max_plies=200):
    b = chess.Board()
    for _ in range(opening_plies):  # random opening for diversity
        moves = list(b.legal_moves)
        b.push(moves[rng.randint(len(moves))])
    while not b.is_game_over() and b.ply() < max_plies:
        mover = white if b.turn else black
        b.push(mover(b, rng))
    if b.is_checkmate():
        return 1 if not b.turn else -1, "mate"
    if b.is_game_over():
        return 0, "draw"
    d = material(b, chess.WHITE)  # adjudicate unfinished games by material
    return (1 if d >= 3 else -1 if d <= -3 else 0), "adjudicated"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("solution", nargs="?", default=str(HERE / "solution.py"))
    ap.add_argument("--games", type=int, default=40, help="games per opponent (half as each colour)")
    ap.add_argument("--dev", action="store_true", help="development seed")
    ap.add_argument("--seed", type=int, default=None, help="evaluator only")
    ap.add_argument("--opponents", default="random,greedy,minimax2")
    a = ap.parse_args()
    seed = 11 if a.dev else (a.seed if a.seed is not None else 20261103)
    spec = importlib.util.spec_from_file_location("solution", a.solution)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    net = NetPlayer(mod)
    out = {"nonzero": int(net.model.nonzero_params), "total_params": int(net.model.total_params)}
    for name in a.opponents.split(","):
        opp = OPPONENTS[name]
        rng = np.random.RandomState(seed)
        res = {"win": 0, "draw": 0, "loss": 0, "mates_by_net": 0, "adjudicated": 0}
        for g in range(a.games):
            net_white = g % 2 == 0
            r, how = play(net, opp, rng) if net_white else play(opp, net, rng)
            r = r if net_white else -r
            res["win" if r > 0 else "loss" if r < 0 else "draw"] += 1
            res["mates_by_net"] += int(r > 0 and how == "mate")
            res["adjudicated"] += int(how == "adjudicated")
        res["score"] = round((res["win"] + 0.5 * res["draw"]) / a.games, 3)
        out[name] = res
        print(name, json.dumps(res), flush=True)
    out["top_move_legal_rate"] = round(net.legal_top / max(1, net.moves), 3)
    out["sec_per_move"] = round(net.secs / max(1, net.moves), 4)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
