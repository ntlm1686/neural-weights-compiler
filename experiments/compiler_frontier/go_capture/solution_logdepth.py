"""Variant: log-depth group connectivity by repeated squaring of a per-position reachability row.

R0[p] = white[p] * (onehot(p) + sum_{white neighbours q} onehot(q))          (81-wide row per position)
R_{k+1}[p] = white[p] * OR_{q in R_k[p]} R_k[q]   (one attention head: Q = R_k[p], K = onehot(q),
                                                    uniform average of V = R_k[q] over the support, then
                                                    an MLP dead-zone threshold at ~1/81)
R_k covers in-group distance <= 2^k, so k = 7 suffices for any 9x9 group (diameter <= 79).
alive[p] = white[p] * OR_{q in R_7[p]} lib[q],  lib[q] = white[q] AND (an empty' neighbour).
Total 10 layers (vs 81 for the BFS version) at the cost of 81-wide registers (d_model 1024, d_head 128).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from neural_compiler import ExecutableTransformer  # noqa: E402

N = 9
P = N * N
SEQ = P + 1
D_MODEL = 1024
N_HEADS = 8
D_HEAD = D_MODEL // N_HEADS   # 128 >= 82
N_SQ = 8                      # squaring layers (use R_0..R_7)
N_LAYERS = 2 + N_SQ
D_MLP = 512
A = 30.0                      # attention sharpness (support score A vs 0)
LO, HI = 0.002, 0.010         # dead zone for "average > 0" (true positives are >= 1/81 = 0.0123)

REGS = {}
_c = 0
for name, w in [("const_one", 1), ("state", 3), ("move", 1), ("null", 1), ("row", N), ("col", N),
                ("pos", P), ("wp", P), ("nbr", P), ("R", P), ("avg", P), ("libc", 1), ("lib", 1),
                ("avgl", 1), ("alive", 1), ("out", 1)]:
    REGS[name] = slice(_c, _c + w)
    _c += w
assert _c <= D_MODEL
EMPTY, BLACK, WHITE = REGS["state"].start, REGS["state"].start + 1, REGS["state"].start + 2
r0 = {k: REGS[k].start for k in REGS}
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _zeros():
    w = {}
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            for nm, shp in (("W_Q", (D_MODEL, D_HEAD)), ("W_K", (D_MODEL, D_HEAD)), ("W_V", (D_MODEL, D_HEAD)),
                            ("W_O", (D_HEAD, D_MODEL)), ("b_Q", (D_HEAD,)), ("b_K", (D_HEAD,)), ("b_V", (D_HEAD,))):
                w[f"L{l}.H{h}.{nm}"] = np.zeros(shp)
        w[f"L{l}.b_O"] = np.zeros(D_MODEL)
        w[f"L{l}.W_fc"] = np.zeros((D_MODEL, D_MLP))
        w[f"L{l}.b_fc"] = np.zeros(D_MLP)
        w[f"L{l}.W_proj"] = np.zeros((D_MLP, D_MODEL))
        w[f"L{l}.b_proj"] = np.zeros(D_MODEL)
    return w


class MLP:
    """Tiny allocator of ReLU units in one layer's MLP."""
    def __init__(self, w, l):
        self.Wf, self.bf, self.Wp, self.u = w[f"L{l}.W_fc"], w[f"L{l}.b_fc"], w[f"L{l}.W_proj"], 0

    def unit(self, ins, bias, outs):
        u = self.u; self.u += 1
        assert u < D_MLP
        for i, c in ins.items():
            self.Wf[i, u] += c
        self.bf[u] = bias
        for o, c in outs.items():
            self.Wp[u, o] += c
        return u

    def gated_clamp(self, x, lo, hi, gate, M, outs):
        """outs += [gate=1] * clamp01((x-lo)/(hi-lo)); exactly 0 if x<=lo or gate=0 (needs slope*x_max < M)."""
        s = 1.0 / (hi - lo)
        self.unit({x: s, gate: M}, -s * lo - M, outs)
        self.unit({x: s, gate: M}, -s * lo - M - 1.0, {o: -c for o, c in outs.items()})

    def clear(self, x):
        self.unit({x: 1.0}, 0.0, {x: -1.0})
        self.unit({x: -1.0}, 0.0, {x: 1.0})

    def remove_binary(self, x):  # x in {0,1} exactly
        self.unit({x: 1.0}, 0.0, {x: -1.0})


def neighbour_head(w, l, h, dr, dc, vmap):
    """Hard attention to neighbour (r+dr, c+dc), off-board -> null token; V/O routes vmap {src: dst}."""
    s = np.sqrt(D_HEAD) * 60.0
    WQ, bQ, WK = w[f"L{l}.H{h}.W_Q"], w[f"L{l}.H{h}.b_Q"], w[f"L{l}.H{h}.W_K"]
    for r in range(N):
        if 0 <= r + dr < N:
            WQ[r0["row"] + r, r + dr] = s
        if 0 <= r + dc < N:
            WQ[r0["col"] + r, N + r + dc] = s
        WK[r0["row"] + r, r] = 1.0
        WK[r0["col"] + r, N + r] = 1.0
    bQ[2 * N] = 1.5 * s
    WK[r0["null"], 2 * N] = 1.0
    WV, WO = w[f"L{l}.H{h}.W_V"], w[f"L{l}.H{h}.W_O"]
    for i, (srcs, dst) in enumerate(vmap):
        for src, c in srcs.items():
            WV[src, i] = c
        WO[i, dst] = 1.0


def square_head(w, l, h):
    """Q = R[p] (81), K = onehot(q): attend uniformly over q in support(R[p]); V = [R[q], lib[q]]."""
    s = np.sqrt(D_HEAD) * A
    WQ, WK, WV, WO = (w[f"L{l}.H{h}.{n}"] for n in ("W_Q", "W_K", "W_V", "W_O"))
    for k in range(P):
        WQ[r0["R"] + k, k] = s
        WK[r0["pos"] + k, k] = 1.0
        WV[r0["R"] + k, k] = 1.0
        WO[k, r0["avg"] + k] = 1.0
    WV[r0["lib"], P] = 1.0
    WO[P, r0["avgl"]] = 1.0


def build_model():
    w = _zeros()
    # L0: wp = white * onehot(pos)
    m = MLP(w, 0)
    for k in range(P):
        m.unit({r0["pos"] + k: 1.0, WHITE: 1.0}, -1.0, {r0["wp"] + k: 1.0})
    # L1: nbr = sum of neighbours' wp; libc = number of empty' neighbours
    for h, (dr, dc) in enumerate(DIRS):
        vmap = [({r0["wp"] + k: 1.0}, r0["nbr"] + k) for k in range(P)]
        vmap.append(({EMPTY: 1.0, r0["move"]: -1.0}, r0["libc"]))
        neighbour_head(w, 1, h, dr, dc, vmap)
    m = MLP(w, 1)
    for k in range(P):  # R0 = white * (wp + nbr)  (each entry in {0,1})
        m.unit({r0["wp"] + k: 1.0, r0["nbr"] + k: 1.0, WHITE: 2.0}, -2.0, {r0["R"] + k: 1.0})
        m.clear(r0["nbr"] + k)
    m.gated_clamp(r0["libc"], 0.25, 0.75, WHITE, 10.0, {r0["lib"]: 1.0})
    m.clear(r0["libc"])
    # L2..: squaring; alive recomputed each time from the current R
    for i in range(N_SQ):
        l = 2 + i
        square_head(w, l, 0)
        m = MLP(w, l)
        for k in range(P):
            m.gated_clamp(r0["avg"] + k, LO, HI, WHITE, 200.0, {r0["R"] + k: 1.0})
            m.remove_binary(r0["R"] + k)
            m.clear(r0["avg"] + k)
        last = i == N_SQ - 1
        outs = {r0["alive"]: 1.0}
        if last:
            outs[r0["out"]] = -1.0
            m.unit({BLACK: 1.0, r0["move"]: 1.0}, 0.0, {r0["out"]: 1.0})
        m.gated_clamp(r0["avgl"], LO, HI, WHITE, 200.0, outs)
        m.remove_binary(r0["alive"])
        m.clear(r0["avgl"])
    return ExecutableTransformer(D_MODEL, N_LAYERS, N_HEADS, D_MLP, w, {}, dict(REGS), "out", "register")


def encode(example):
    board, move = example
    flat = np.asarray(board).reshape(-1)
    state = np.zeros((SEQ, 3))
    state[:P, 0] = flat == 0
    state[:P, 1] = flat == 1
    state[:P, 2] = flat == -1
    mv = np.zeros(SEQ); mv[move[0] * N + move[1]] = 1.0
    null = np.zeros(SEQ); null[P] = 1.0
    row = np.zeros((SEQ, N)); col = np.zeros((SEQ, N)); pos = np.zeros((SEQ, P))
    for i in range(P):
        row[i, i // N] = 1.0; col[i, i % N] = 1.0; pos[i, i] = 1.0
    return {"state": state, "move": mv, "null": null, "row": row, "col": col, "pos": pos}, SEQ


def decode(raw):
    return np.rint(np.asarray(raw)[:P]).astype(int).reshape(N, N)
