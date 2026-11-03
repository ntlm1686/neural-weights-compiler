"""Mahjong 14-tile completeness (4 melds + 1 pair) as a hand-constructed Transformer.

Sequence: 14 tile positions (tile one-hot) + 34 "pair-candidate" positions (blank; only a position one-hot).
Layer 0 attn : uniform attention over all 48 positions, V = tile one-hot, W_O scaled by 48 -> every position
               holds the 34-bin tile histogram c[0..33].
Layers 0..8 MLP: greedy left-to-right meld decomposition per suit (3 suits + honours in parallel),
               at position 14+p the candidate pair p is removed (read from the position one-hot).
               For tile i with suit offset j: r = c_i - 2[p==i] - k_{i-1} - k_{i-2};
               k_i = r mod 3 (chows started at i); penalty += ReLU(-r); for j>=7 penalty += k_i;
               honours: penalty += [r mod 3 != 0].
Layer 9 MLP : ok = ReLU(is_candidate - penalty).
Layer 10 attn: uniform mean of ok, scaled by 48 -> number of valid pair choices; MLP clamps to [0,1].
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from neural_compiler import ExecutableTransformer  # noqa: E402

NT, NC = 14, 34
SEQ = NT + NC
D_MODEL, N_LAYERS, N_HEADS, D_MLP = 256, 11, 1, 128

# residual registers
REG = {}
_cur = 0
for name, w in [("const_one", 1), ("tile", 34), ("pos", SEQ), ("cnt", 34), ("k", 34), ("pen", 1),
                ("ok", 1), ("nvalid", 1), ("out", 1)]:
    REG[name] = slice(_cur, _cur + w)
    _cur += w
assert _cur <= D_MODEL


def _col(name, i=0):
    return REG[name].start + i


def _empty_weights():
    W = {}
    dh = D_MODEL // N_HEADS
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            for n, shp in [("W_Q", (D_MODEL, dh)), ("b_Q", (dh,)), ("W_K", (D_MODEL, dh)), ("b_K", (dh,)),
                           ("W_V", (D_MODEL, dh)), ("b_V", (dh,)), ("W_O", (dh, D_MODEL))]:
                W[f"L{l}.H{h}.{n}"] = np.zeros(shp)
        W[f"L{l}.b_O"] = np.zeros(D_MODEL)
        W[f"L{l}.W_fc"] = np.zeros((D_MODEL, D_MLP))
        W[f"L{l}.b_fc"] = np.zeros(D_MLP)
        W[f"L{l}.W_proj"] = np.zeros((D_MLP, D_MODEL))
        W[f"L{l}.b_proj"] = np.zeros(D_MODEL)
    return W


class _MLP:
    """Unit allocator for one layer's MLP."""

    def __init__(self, W, l):
        self.fc, self.bfc, self.proj = W[f"L{l}.W_fc"], W[f"L{l}.b_fc"], W[f"L{l}.W_proj"]
        self.u = 0

    def unit(self, lin, bias, outs):
        """lin: {residual col: coeff}; outs: {residual col: coeff}. Adds outs * ReLU(lin.x + bias)."""
        u = self.u
        assert u < D_MLP
        for c, v in lin.items():
            self.fc[c, u] += v
        self.bfc[u] = bias
        for c, v in outs.items():
            self.proj[u, c] += v
        self.u += 1

    def hat(self, lin, bias, center, outs):
        """outs * hat(r - center) where r = lin.x + bias; equals outs at r==center, 0 at other integers."""
        for shift, m in ((1.0, 1.0), (0.0, -2.0), (-1.0, 1.0)):
            self.unit(lin, bias - center + shift, {c: v * m for c, v in outs.items()})


def build_model():
    W = _empty_weights()
    # ---- layer 0 attention: histogram (Q=K=0 -> uniform over all SEQ positions)
    WV, WO = W["L0.H0.W_V"], W["L0.H0.W_O"]
    for t in range(NC):
        WV[_col("tile", t), t] = 1.0
        WO[t, _col("cnt", t)] = float(SEQ)
    pen = _col("pen")
    # ---- layers 0..8 MLP: greedy decomposition step j
    for j in range(9):
        mlp = _MLP(W, j)
        tiles = [s * 9 + j for s in range(3)] + ([27 + h for h in range(7)] if j == 0 else [])
        for i in tiles:
            lin = {_col("cnt", i): 1.0, _col("pos", NT + i): -2.0}
            honour = i >= 27
            if not honour:
                if j >= 1:
                    lin[_col("k", i - 1)] = -1.0
                if j >= 2:
                    lin[_col("k", i - 2)] = -1.0
            # penalty for r < 0
            mlp.unit({c: -v for c, v in lin.items()}, 0.0, {pen: 1.0})
            # k_i = r mod 3 on r in {0..4}: values 1@1, 2@2, 1@4
            for center, val in ((1, 1.0), (2, 2.0), (4, 1.0)):
                outs = {}
                if honour:
                    outs[pen] = 1.0  # honours: any nonzero remainder fails
                else:
                    outs[_col("k", i)] = val
                    if j >= 7:
                        outs[pen] = val  # no chow can start at 8 or 9
                mlp.hat(lin, 0.0, center, outs)
    # ---- layer 9 MLP: ok = ReLU(is_candidate - penalty)
    mlp = _MLP(W, 9)
    lin = {_col("pos", NT + p): 1.0 for p in range(NC)}
    lin[pen] = -1.0
    mlp.unit(lin, 0.0, {_col("ok"): 1.0})
    # ---- layer 10 attention: count valid pair choices; MLP clamps to {0,1}
    W["L10.H0.W_V"][_col("ok"), 0] = 1.0
    W["L10.H0.W_O"][0, _col("nvalid")] = float(SEQ)
    mlp = _MLP(W, 10)
    mlp.unit({_col("nvalid"): 1.0}, 0.0, {_col("out"): 1.0})
    mlp.unit({_col("nvalid"): 1.0}, -1.0, {_col("out"): -1.0})
    return ExecutableTransformer(D_MODEL, N_LAYERS, N_HEADS, D_MLP, W, {}, dict(REG), "out", "last_pos_register")


def encode(example):
    tile = np.zeros((SEQ, NC))
    for n, t in enumerate(example):
        tile[n, t] = 1.0
    return {"tile": tile, "pos": np.eye(SEQ)}, SEQ


def decode(raw_output):
    return bool(raw_output > 0.5)
