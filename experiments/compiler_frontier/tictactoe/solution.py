"""Tic-tac-toe minimax value by a compiled (untrained) Transformer.

Algorithm: exhaustive retrograde / minimax search over every continuation of the input board, executed
inside the network.  Token <-> set A of cells the ROOT MOVER adds (|A| = 0..5, plus one sink token).
Inside each token, residual dims indexed by subsets B of the 9 cells (|B| = 0..4) hold the value of the
search node (A, B) = root board + A (mover stones) + B (opponent stones):
  * MO(A,B), |B| = |A|   (mover to move)    stored in class |A| dims, "g" encoding
  * OP(A,B), |B| = |A|-1 (opponent to move) stored in class |A|-1 dims, "f" encoding
Mover moves change A  -> attention (token attends uniformly to its children tokens A+c).
Opponent moves change B -> fixed linear map inside the token (MLP).
max / min over children of values in {-1,0,1} are computed exactly by averaging / summing an
encoding and thresholding.  See NOTES.md.
"""
import sys
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from neural_compiler import ExecutableTransformer  # noqa: E402

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

D_MODEL, N_HEADS = 512, 4
D_HEAD = D_MODEL // N_HEADS
EPS = 0.02          # encoding of a draw inside f / g
MPEN = 1e5          # gating penalty
S_ATT = 40.0        # attention sharpness

# ------------------------------------------------------------------ tokens (fixed function of index)
TOKENS = [frozenset(s) for a in range(6) for s in combinations(range(9), a)]
SINK = len(TOKENS)
SEQ_LEN = SINK + 1

# B-subset classes: class k = all k-subsets of the 9 cells, lexicographic
CLASS = [[frozenset(s) for s in combinations(range(9), k)] for k in range(5)]
CLASS_OFF = [sum(comb(9, j) for j in range(k)) for k in range(6)]  # 0,1,10,46,130,256
CLASS_IDX = [{b: i for i, b in enumerate(c)} for c in CLASS]

# ------------------------------------------------------------------ residual registers
_REGS = [("const_one", 1), ("iscell", 1), ("A", 9), ("lvl", 7), ("xin", 9), ("oin", 9),
         ("X", 9), ("O", 9), ("M", 9), ("R", 9), ("nroh", 10), ("aocc", 1), ("nml", 1),
         ("V", CLASS_OFF[5]), ("NL", comb(9, 4)), ("OUT", 1)]
REG = {}
_c = 0
for _n, _w in _REGS:
    REG[_n] = slice(_c, _c + _w)
    _c += _w
assert _c <= D_MODEL


def r(name, i=0):
    return REG[name].start + i


def V(k, b):
    return REG["V"].start + CLASS_OFF[k] + CLASS_IDX[k][b]


# ------------------------------------------------------------------ MLP layer builder
class MLPLayer:
    def __init__(self):
        self.units = []  # (in: {dim: coef}, bias, out: {dim: coef})

    def unit(self, inp, bias, out):
        self.units.append((inp, bias, out))

    def step(self, inp, bias, slope, theta, out_coef, out_dim, gate=None):
        """out_dim += out_coef * clamp01(slope*(x - theta)) where x = inp.x + bias; gate = penalty terms."""
        gate = gate or {}
        for extra, sign in ((0.0, 1.0), (-1.0, -1.0)):
            d = {k: slope * v for k, v in inp.items()}
            for k, v in gate.items():
                d[k] = d.get(k, 0.0) - MPEN * v
            gb = -MPEN * gate.get("_bias", 0.0)
            d.pop("_bias", None)
            self.unit(d, slope * (bias - theta) + extra + gb, {out_dim: sign * out_coef})


def gate_level(level, extra=None):
    """penalty dict that is 0 iff token level == level (plus extra penalty terms, all >= 0)."""
    g = {r("const_one"): 1.0, r("lvl", level): -1.0}
    for k, v in (extra or {}).items():
        g[k] = g.get(k, 0.0) + v
    return g


def pv_terms(b):
    """terms of the 'invalid' count: sum_{c in B} (occ_c + A_c) + aocc  (0 iff state (A,B) valid)."""
    t = {r("aocc"): 1.0}
    for c in b:
        for reg in ("X", "O", "A"):
            t[r(reg, c)] = t.get(r(reg, c), 0.0) + 1.0
    return t


def add(d1, d2, s=1.0):
    d = dict(d1)
    for k, v in d2.items():
        d[k] = d.get(k, 0.0) + s * v
    return d


# ------------------------------------------------------------------ model
def build_model():
    layers = []  # list of (attn_heads: list of dict, mlp: MLPLayer)

    def new_layer():
        layers.append(([], MLPLayer()))
        return layers[-1]

    # ---- L0: broadcast the 9 cells (stored at tokens 0..8) to every token; derive mover/opp stones
    heads, mlp = new_layer()
    heads.append(dict(kind="bcast"))
    sX = {r("X", c): 1.0 for c in range(9)}
    sO = {r("O", c): 1.0 for c in range(9)}
    # diff = sum X - sum O in {0 (X to move), 1 (O to move)}
    for c in range(9):
        mvx = add({r("X", c): 1.0}, add(sX, sO, -1.0), -1.0)        # X_c - diff
        mvo = add({r("O", c): 1.0}, add(sX, sO, -1.0), 1.0)         # O_c + diff - 1
        mlp.unit(mvx, 0.0, {r("M", c): 1.0})
        mlp.unit(mvo, -1.0, {r("M", c): 1.0})
        rpo = add({r("O", c): 1.0}, add(sX, sO, -1.0), -1.0)
        rpx = add({r("X", c): 1.0}, add(sX, sO, -1.0), 1.0)
        mlp.unit(rpo, 0.0, {r("R", c): 1.0})
        mlp.unit(rpx, -1.0, {r("R", c): 1.0})
    n_occ = add(sX, sO)
    for j in range(10):  # one-hot of number of stones (hat functions on integers)
        mlp.unit(n_occ, -j + 1.0, {r("nroh", j): 1.0})
        mlp.unit(n_occ, -j, {r("nroh", j): -2.0})
        mlp.unit(n_occ, -j - 1.0, {r("nroh", j): 1.0})

    # ---- L1: aocc = |A & occupied|, nml = # lines completed by mover stones (root mover + A)
    heads, mlp = new_layer()
    for c in range(9):
        mlp.unit({r("A", c): 1.0, r("X", c): 1.0, r("O", c): 1.0}, -1.0, {r("aocc"): 1.0})
    for L in LINES:
        inp = {}
        for c in L:
            inp[r("A", c)] = 1.0
            inp[r("M", c)] = 1.0
        mlp.unit(inp, -2.0, {r("nml"): 1.0})

    def op_step(mlp, a):
        """OP(A,B) for |A| = a, |B| = a-1 -> V class a-1 (f encoding, 0 if invalid)."""
        full = {r("nroh", 10 - 2 * a): 1.0} if 0 <= 10 - 2 * a <= 9 else {}
        for b in CLASS[a - 1]:
            out = V(a - 1, b)
            pv = pv_terms(b)
            # valid term: +1
            mlp.unit(add({}, add(pv, gate_level(a)), -1.0), 1.0, {out: 1.0})
            s = {}
            if a <= 4:
                for c in range(9):
                    if c not in b:
                        s[V(a, b | {c})] = 1.0
            gate = add(gate_level(a, pv), {r("nml"): 1.0})
            mlp.step(add(s, full), 0.0, 500.0, 0.005, -(1 - EPS), out, gate)   # exists child <= 0 (or full)
            if s:
                mlp.step(s, 0.0, 10.0, 0.5, -EPS, out, gate)                     # exists child == -1

    # ---- L2: OP for |A| = 5
    heads, mlp = new_layer()
    op_step(mlp, 5)

    for k in range(4, -1, -1):
        # attention: V class k of level-k token += mean over children (A+c) of their V class k (= f(OP))
        heads, mlp = new_layer()
        heads.append(dict(kind="children", k=k))
        for b in CLASS[k]:  # NL[b] = # opponent lines in root-opp + B
            for L in LINES:
                inp = {r("R", c): 1.0 for c in L}
                m = len(set(L) & b)
                mlp.unit(inp, m - 2.0, {r("NL", CLASS_IDX[k][b]): 1.0})
        # MO(A,B), |A| = |B| = k -> g encoding
        heads, mlp = new_layer()
        full = {r("nroh", 9 - 2 * k): 1.0}
        for b in CLASS[k]:
            out = V(k, b)
            i = CLASS_IDX[k][b]
            raw = {out: 1.0}
            pv = pv_terms(b)
            mlp.unit(add({}, add(pv, gate_level(k)), -1.0), 1.0, {out: 1.0})
            gate = add(gate_level(k, pv), {r("NL", i): 1.0})
            mlp.step(add(raw, full), 0.0, 2000.0, 0.0005, -(1 - EPS), out, gate)  # some child >= 0 (or full)
            mlp.step(raw, 0.0, 50.0, 0.04, -EPS, out, gate)                       # some child == +1
            mlp.unit(add(raw, gate_level(k), -MPEN), 0.0, {out: -1.0})              # clear raw mean
            mlp.unit({r("NL", i): 1.0}, 0.0, {r("NL", i): -1.0})                   # clear scratch
        if k >= 1:
            heads, mlp = new_layer()
            op_step(mlp, k)

    # ---- output: value for the side to move from g = V[class 0] at token 0
    heads, mlp = new_layer()
    g = {V(0, frozenset()): 1.0}
    mlp.step(g, 0.0, 500.0, 0.005, -1.0, r("OUT"))
    mlp.step(g, 0.0, 10.0, 0.3, -1.0, r("OUT"))
    mlp.unit({r("const_one"): 1.0}, 0.0, {r("OUT"): 1.0})

    return _assemble(layers)


def _assemble(layers):
    nl = len(layers)
    d_mlp = max(len(m.units) for _, m in layers)
    W = {}
    sq = np.sqrt(D_HEAD)
    for l, (heads, mlp) in enumerate(layers):
        for h in range(N_HEADS):
            for nm, shp in (("W_Q", (D_MODEL, D_HEAD)), ("W_K", (D_MODEL, D_HEAD)), ("W_V", (D_MODEL, D_HEAD)),
                            ("W_O", (D_HEAD, D_MODEL))):
                W[f"L{l}.H{h}.{nm}"] = np.zeros(shp)
            for nm in ("b_Q", "b_K", "b_V"):
                W[f"L{l}.H{h}.{nm}"] = np.zeros(D_HEAD)
        W[f"L{l}.b_O"] = np.zeros(D_MODEL)
        for h, hd in enumerate(heads):
            WQ, WK, WV, WO = (W[f"L{l}.H{h}.{n}"] for n in ("W_Q", "W_K", "W_V", "W_O"))
            alpha = S_ATT * sq
            if hd["kind"] == "bcast":
                # score = S * iscell(key); value = cell codes laid out per cell; mean over 9 cells * 9
                WQ[r("const_one"), 0] = alpha
                WK[r("iscell"), 0] = 1.0
                for c in range(9):
                    WV[r("xin", c), c] = 1.0
                    WV[r("oin", c), 9 + c] = 1.0
                    WO[c, r("X", c)] = 9.0
                    WO[9 + c, r("O", c)] = 9.0
            else:
                k = hd["k"]
                # score = S * (|A & A'| + [lvl(A') == lvl(A)+1] - lvl(A) - 1): 0 iff A' child of A, else <= -1
                for c in range(9):
                    WQ[r("A", c), c] = alpha
                    WK[r("A", c), c] = 1.0
                for lv in range(7):
                    if lv + 1 < 7:
                        WQ[r("lvl", lv), 9 + lv + 1] = alpha
                    WK[r("lvl", lv), 9 + lv] = 1.0
                    WQ[r("lvl", lv), 16] = -alpha * (lv + 1)
                WK[r("const_one"), 16] = 1.0
                for j in range(comb(9, k)):
                    WV[r("V", CLASS_OFF[k] + j), j] = 1.0
                    WO[j, r("V", CLASS_OFF[k] + j)] = 1.0
        Wfc = np.zeros((D_MODEL, d_mlp))
        bfc = np.zeros(d_mlp)
        Wp = np.zeros((d_mlp, D_MODEL))
        for u, (inp, bias, out) in enumerate(mlp.units):
            for dim, cf in inp.items():
                Wfc[dim, u] += cf
            bfc[u] = bias
            for dim, cf in out.items():
                Wp[u, dim] += cf
        W[f"L{l}.W_fc"], W[f"L{l}.b_fc"], W[f"L{l}.W_proj"] = Wfc, bfc, Wp
        W[f"L{l}.b_proj"] = np.zeros(D_MODEL)
    return ExecutableTransformer(d_model=D_MODEL, n_layers=nl, n_heads=N_HEADS, d_mlp=d_mlp, weights=W,
                                 causal_masks={}, register_slices=dict(REG), output_register="OUT",
                                 output_mode="first_pos_register")


# ------------------------------------------------------------------ position codes (fixed per index)
def _position_codes():
    A = np.zeros((SEQ_LEN, 9))
    lvl = np.zeros((SEQ_LEN, 7))
    for t, s in enumerate(TOKENS):
        for c in s:
            A[t, c] = 1.0
        lvl[t, len(s)] = 1.0
    A[SINK, :] = 1.0
    lvl[SINK, 6] = 1.0
    iscell = np.zeros(SEQ_LEN)
    iscell[:9] = 1.0
    return A, lvl, iscell


_A, _LVL, _ISCELL = _position_codes()


def encode(example):
    """Raw cell one-hots: cell c's [is X] / [is O] written at token c, slot c. Plus position codes."""
    xin = np.zeros((SEQ_LEN, 9))
    oin = np.zeros((SEQ_LEN, 9))
    for c in range(9):
        xin[c, c] = 1.0 if example[c] == 1 else 0.0
        oin[c, c] = 1.0 if example[c] == -1 else 0.0
    return {"A": _A, "lvl": _LVL, "iscell": _ISCELL, "xin": xin, "oin": oin}, SEQ_LEN


def decode(raw_output):
    return int(np.round(float(raw_output)))
