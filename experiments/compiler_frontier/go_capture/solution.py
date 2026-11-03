"""Go move with captures (9x9) compiled into a Transformer by explicit weight formulas.

Algorithm (all inside model.run):
  board' = board with Black at the move point; empty' = empty - move (exact, linear).
  alive_0 = 0
  alive_{k+1}[p] = white[p] AND ( sum over the 4 neighbours q of (empty'[q] + alive_k[q]) >= 1 )
  After K >= 80 iterations alive = white stones whose group has a liberty (monotone BFS; a white
  group has <= 80 stones so its geodesic radius is <= 79 < K).
  out = black + move - alive   (captured white stones become 0).
One iteration = one Transformer layer: 4 attention heads (one per direction) hard-attend to the
exact neighbour (or to a null blank token off-board) and add its (empty' + alive) into `nsum`;
the MLP turns nsum into a clean 0/1 gated by white, replaces `alive`, and clears `nsum`.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from neural_compiler import ExecutableTransformer  # noqa: E402

N = 9
P = N * N              # board positions 0..80
SEQ = P + 1            # + one blank "null" token (index 81) attended to when a neighbour is off-board
D_MODEL = 80
N_HEADS = 4
D_HEAD = D_MODEL // N_HEADS   # 20 (needs >= 2*9+1)
N_LAYERS = 81          # = number of propagation iterations (>= 80 needed for any group on 9x9)
D_MLP = 8
A = 60.0               # attention sharpness: neighbour score 2A, null 1.5A, others <= A  (margin 0.5A = 30)
GATE_M = 10.0          # gate constant for "white AND ..."

# residual-stream layout
REGS = {}
_c = 0
for name, w in [("const_one", 1), ("state", 3), ("move", 1), ("null", 1), ("row", N), ("col", N),
                ("alive", 1), ("nsum", 1), ("out", 1)]:
    REGS[name] = slice(_c, _c + w)
    _c += w
assert _c <= D_MODEL
EMPTY, BLACK, WHITE = REGS["state"].start, REGS["state"].start + 1, REGS["state"].start + 2
MOVE, NULL = REGS["move"].start, REGS["null"].start
ALIVE, NSUM, OUT, ONE = REGS["alive"].start, REGS["nsum"].start, REGS["out"].start, REGS["const_one"].start
ROW0, COL0 = REGS["row"].start, REGS["col"].start
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _zeros():
    w = {}
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            w[f"L{l}.H{h}.W_Q"] = np.zeros((D_MODEL, D_HEAD))
            w[f"L{l}.H{h}.b_Q"] = np.zeros(D_HEAD)
            w[f"L{l}.H{h}.W_K"] = np.zeros((D_MODEL, D_HEAD))
            w[f"L{l}.H{h}.b_K"] = np.zeros(D_HEAD)
            w[f"L{l}.H{h}.W_V"] = np.zeros((D_MODEL, D_HEAD))
            w[f"L{l}.H{h}.b_V"] = np.zeros(D_HEAD)
            w[f"L{l}.H{h}.W_O"] = np.zeros((D_HEAD, D_MODEL))
        w[f"L{l}.b_O"] = np.zeros(D_MODEL)
        w[f"L{l}.W_fc"] = np.zeros((D_MODEL, D_MLP))
        w[f"L{l}.b_fc"] = np.zeros(D_MLP)
        w[f"L{l}.W_proj"] = np.zeros((D_MLP, D_MODEL))
        w[f"L{l}.b_proj"] = np.zeros(D_MODEL)
    return w


def neighbour_read_head(w, l, h, dr, dc):
    """Primitive: hard attention from (r,c) to (r+dr,c+dc); off-board -> null token (value 0).
    score/sqrt(d) = A*[row match] + A*[col match] + 1.5A*[key is null].
    Value = empty' + alive = empty - move + alive, written (added) to nsum."""
    s = np.sqrt(D_HEAD) * A
    WQ, bQ, WK = w[f"L{l}.H{h}.W_Q"], w[f"L{l}.H{h}.b_Q"], w[f"L{l}.H{h}.W_K"]
    for r in range(N):
        if 0 <= r + dr < N:
            WQ[ROW0 + r, r + dr] = s          # query asks for row r+dr
        if 0 <= r + dc < N:
            WQ[COL0 + r, N + r + dc] = s      # query asks for column c+dc
        WK[ROW0 + r, r] = 1.0
        WK[COL0 + r, N + r] = 1.0
    bQ[2 * N] = 1.5 * s
    WK[NULL, 2 * N] = 1.0
    WV, WO = w[f"L{l}.H{h}.W_V"], w[f"L{l}.H{h}.W_O"]
    WV[EMPTY, 0] = 1.0
    WV[MOVE, 0] = -1.0
    WV[ALIVE, 0] = 1.0
    WO[0, NSUM] = 1.0


def alive_update_mlp(w, l, last):
    """Primitive: alive <- white AND clamp01(2*(nsum-0.25)); nsum <- 0; (last) out <- black+move-alive.
    Dead zone: nsum <= 0.25 -> 0 exactly, nsum >= 0.75 -> 1 exactly (true nsum is an integer)."""
    Wf, bf, Wp = w[f"L{l}.W_fc"], w[f"L{l}.b_fc"], w[f"L{l}.W_proj"]
    # u0 - u1 = gated clamp
    for u, extra in ((0, 0.0), (1, -1.0)):
        Wf[NSUM, u] = 2.0
        Wf[WHITE, u] = GATE_M
        bf[u] = -0.5 - GATE_M + extra
    Wp[0, ALIVE] += 1.0
    Wp[1, ALIVE] -= 1.0
    # u2: remove old alive (old alive in {0,1})
    Wf[ALIVE, 2] = 1.0
    Wp[2, ALIVE] = -1.0
    # u3,u4: clear nsum exactly
    Wf[NSUM, 3] = 1.0
    Wp[3, NSUM] = -1.0
    Wf[NSUM, 4] = -1.0
    Wp[4, NSUM] = 1.0
    if last:
        Wf[BLACK, 5] = 1.0
        Wf[MOVE, 5] = 1.0
        Wp[5, OUT] = 1.0
        Wp[0, OUT] -= 1.0
        Wp[1, OUT] += 1.0


def build_model():
    w = _zeros()
    for l in range(N_LAYERS):
        for h, (dr, dc) in enumerate(DIRS):
            neighbour_read_head(w, l, h, dr, dc)
        alive_update_mlp(w, l, last=(l == N_LAYERS - 1))
    return ExecutableTransformer(D_MODEL, N_LAYERS, N_HEADS, D_MLP, w, {}, dict(REGS), "out", "register")


def encode(example):
    board, move = example
    board = np.asarray(board).reshape(N, N)
    state = np.zeros((SEQ, 3))
    flat = board.ravel()
    state[:P, 0] = flat == 0
    state[:P, 1] = flat == 1
    state[:P, 2] = flat == -1
    mv = np.zeros(SEQ)
    mv[move[0] * N + move[1]] = 1.0
    null = np.zeros(SEQ)
    null[P] = 1.0
    row = np.zeros((SEQ, N))
    col = np.zeros((SEQ, N))
    for i in range(P):
        row[i, i // N] = 1.0
        col[i, i % N] = 1.0
    return {"state": state, "move": mv, "null": null, "row": row, "col": col}, SEQ


def decode(raw):
    return np.rint(np.asarray(raw)[:P]).astype(int).reshape(N, N)
