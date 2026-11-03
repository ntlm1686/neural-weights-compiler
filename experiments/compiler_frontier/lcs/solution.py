"""LCS length of two length-6 strings over {0,1,2,3}, computed by a hand-constructed Transformer.

Layout: one sequence position per DP cell (i, j), i, j in 0..5 (position p = 6*i + j), plus one
"null" position (p = 36) whose DP value stays 0 and serves as the out-of-range neighbour.
Every layer performs one parallel (Jacobi) relaxation of the LCS recurrence at all cells;
cell (i, j) is exact after i + j + 1 relaxations, so 11 relaxations suffice (layers 1..11),
layer 0's MLP computes the match bit eq(i, j) = [a_i == b_j]. All weights are written from
explicit formulas below (no training, no fitting).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from neural_compiler import ExecutableTransformer  # noqa: E402

N = 6                 # string length
V = 4                 # alphabet size
NCELL = N * N         # 36 DP cells
NPOS = NCELL + 1      # + null position
NULL = NCELL
N_STEPS = 2 * N - 1   # 11 anti-diagonals -> 11 relaxations
N_LAYERS = 1 + N_STEPS
N_HEADS = 3
D_HEAD = 64
D_MODEL = N_HEADS * D_HEAD   # 192
D_MLP = 8
QK_GAIN = 400.0       # score gap = QK_GAIN / sqrt(D_HEAD) = 50  -> leak <= 36 * e^-50
BIG = 20.0            # gating constant (> max DP value 6 + 1)

# ---------------------------------------------------------------- residual register allocation
_regs = {}
_off = 0


def _reg(name, width):
    global _off
    _regs[name] = slice(_off, _off + width)
    _off += width


_reg("tok_a", V)        # one-hot of a[i] at cell (i, j)   (zeros at null position)
_reg("tok_b", V)        # one-hot of b[j] at cell (i, j)
_reg("cell", NPOS)      # one-hot position id (fixed function of position index)
_reg("eq", 1)           # match bit, written by layer-0 MLP
_reg("L", 1)            # current DP estimate (output register)
for _l in range(1, N_LAYERS):
    for _n in ("up", "left", "diag"):
        _reg(f"{_n}{_l}", 1)   # neighbour values fetched by attention in layer _l
assert _off <= D_MODEL


def _neighbour(p, kind):
    """Target position of the 'up' (i-1,j), 'left' (i,j-1) or 'diag' (i-1,j-1) neighbour; NULL if out of range."""
    if p == NULL:
        return NULL
    i, j = divmod(p, N)
    di, dj = {"up": (1, 0), "left": (0, 1), "diag": (1, 1)}[kind]
    i, j = i - di, j - dj
    return NULL if (i < 0 or j < 0) else N * i + j


def build_model():
    W = {}
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            for nm, shp in (("W_Q", (D_MODEL, D_HEAD)), ("W_K", (D_MODEL, D_HEAD)), ("W_V", (D_MODEL, D_HEAD)),
                            ("W_O", (D_HEAD, D_MODEL))):
                W[f"L{l}.H{h}.{nm}"] = np.zeros(shp)
            for nm in ("b_Q", "b_K", "b_V"):
                W[f"L{l}.H{h}.{nm}"] = np.zeros(D_HEAD)
        W[f"L{l}.b_O"] = np.zeros(D_MODEL)
        W[f"L{l}.W_fc"] = np.zeros((D_MODEL, D_MLP))
        W[f"L{l}.b_fc"] = np.zeros(D_MLP)
        W[f"L{l}.W_proj"] = np.zeros((D_MLP, D_MODEL))
        W[f"L{l}.b_proj"] = np.zeros(D_MODEL)

    ta, tb, cell = _regs["tok_a"].start, _regs["tok_b"].start, _regs["cell"].start
    eq, L = _regs["eq"].start, _regs["L"].start

    # ---- layer 0 MLP: eq = sum_c ReLU(a_c + b_c - 1)   (exact for one-hots)
    for c in range(V):
        W["L0.W_fc"][ta + c, c] = 1.0
        W["L0.W_fc"][tb + c, c] = 1.0
        W["L0.b_fc"][c] = -1.0
        W["L0.W_proj"][c, eq] = 1.0

    # ---- layers 1..11: one relaxation each
    for l in range(1, N_LAYERS):
        out = {}
        for h, kind in enumerate(("up", "left", "diag")):
            # Q at position p = QK_GAIN * onehot(neighbour(p));  K at position q = onehot(q)
            for p in range(NPOS):
                W[f"L{l}.H{h}.W_Q"][cell + p, _neighbour(p, kind)] = QK_GAIN
                W[f"L{l}.H{h}.W_K"][cell + p, p] = 1.0
            W[f"L{l}.H{h}.W_V"][L, 0] = 1.0                       # value = DP estimate
            out[kind] = _regs[f"{kind}{l}"].start
            W[f"L{l}.H{h}.W_O"][0, out[kind]] = 1.0                 # -> fresh scratch register
        up, left, diag = out["up"], out["left"], out["diag"]
        fc, bfc, pj = W[f"L{l}.W_fc"], W[f"L{l}.b_fc"], W[f"L{l}.W_proj"]
        # new = (1-eq)*max(up,left) + eq*(diag+1)
        #     = ReLU(up - BIG*eq) + ReLU(left - up - BIG*eq) + ReLU(diag + 1 - BIG + BIG*eq)
        fc[up, 0] = 1.0; fc[eq, 0] = -BIG                                   # noqa: E702
        fc[left, 1] = 1.0; fc[up, 1] = -1.0; fc[eq, 1] = -BIG               # noqa: E702
        fc[diag, 2] = 1.0; fc[eq, 2] = BIG; bfc[2] = 1.0 - BIG              # noqa: E702
        fc[L, 3] = 1.0                                                      # ReLU(L_old) = L_old (L >= 0)
        pj[0, L] = pj[1, L] = pj[2, L] = 1.0
        pj[3, L] = -1.0                                                     # overwrite: L <- new
    return ExecutableTransformer(D_MODEL, N_LAYERS, N_HEADS, D_MLP, W, {}, dict(_regs), "L", "register")


def encode(example):
    a, b = example
    A = np.zeros((NPOS, V))
    B = np.zeros((NPOS, V))
    for p in range(NCELL):
        i, j = divmod(p, N)
        A[p, int(a[i])] = 1.0   # raw symbol a[i] placed at cell (i, j)
        B[p, int(b[j])] = 1.0   # raw symbol b[j] placed at cell (i, j)
    return {"tok_a": A, "tok_b": B, "cell": np.eye(NPOS)}, NPOS


def decode(raw_output):
    return int(np.rint(raw_output[NCELL - 1]))   # DP value at cell (5, 5)
