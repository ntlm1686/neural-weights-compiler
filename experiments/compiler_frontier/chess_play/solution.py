"""Chess policy network with hand-designed weights.  All chess logic lives in the weights (see chess_net.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chess_net  # noqa: E402

NTOK = chess_net.NTOK


def build_model():
    return chess_net.build()


def encode(position):
    """Raw placement only: per-square piece one-hot into the token that owns the square, token position code,
    side to move one-hot, castling flags."""
    board = np.asarray(position["board"])
    B = np.zeros((NTOK, 768))
    for s in range(64):
        code = int(board[s // 8, s % 8])
        if code != 0:
            B[s // 2, s * 12 + (code - 1 if code > 0 else 6 + (-code - 1))] = 1.0
    turn = np.tile([1.0, 0.0] if position["turn"] == 1 else [0.0, 1.0], (NTOK, 1))
    castle = np.tile(np.array(position["castling"], dtype=float), (NTOK, 1))
    return {"B": B, "tok": np.eye(NTOK), "turn": turn, "castle": castle}, NTOK


def decode(raw_output):
    return np.asarray(raw_output).reshape(64, 64)
