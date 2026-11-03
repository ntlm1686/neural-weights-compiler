"""Dyck-1 (balanced parentheses, length 16) recognizer as a hand-constructed 2-layer Transformer.

All weights are written from explicit formulas (no training, no search). See NOTES.md.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from neural_compiler import ExecutableTransformer  # noqa: E402

N = 16
D_MODEL, N_LAYERS, N_HEADS, D_MLP = 16, 2, 2, 4
D_HEAD = D_MODEL // N_HEADS
# residual-stream registers (one dimension each)
REG = dict(const_one=0, x=1, first=2, m=3, p=4, bad=5, out=6)
K = 32.0  # step-function gain: margins are >= 1/32 in pre-activation units, so K*margin >= 1


def _empty_weights():
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


def uniform_prefix_mean(w, layer, head, src, dst, scale=1.0):
    """Primitive: causal head with Q=K=0 -> uniform attention over j<=i; writes scale*mean_{j<=i} X[j,src] into dst."""
    w[f"L{layer}.H{head}.W_V"][src, 0] = 1.0
    w[f"L{layer}.H{head}.W_O"][0, dst] = scale


def build_model():
    w = _empty_weights()
    masks = {(l, h): True for l in range(N_LAYERS) for h in range(N_HEADS)}
    r = REG

    # ---- Layer 0 attention: m_i = depth_i/(i+1), p_i = 1/(i+1)
    uniform_prefix_mean(w, 0, 0, r["x"], r["m"])
    uniform_prefix_mean(w, 0, 1, r["first"], r["p"])

    # ---- Layer 0 MLP: exact step functions on depth sign
    # a_i = K*(-m_i - p_i/2) = K*(-d_i - 1/2)/(i+1);  d_i <= -1 -> a >= K/(2*16) = 1;  d_i >= 0 -> a <= -1/2*K/16 < 0
    # b_i = K*( m_i - p_i/2) = K*( d_i - 1/2)/(i+1);  d_i >= +1 -> b >= 1;             d_i <= 0 -> b < 0
    Wfc, bfc, Wp, bp = w["L0.W_fc"], w["L0.b_fc"], w["L0.W_proj"], w["L0.b_proj"]
    for u, sgn, bias in [(0, -1, 0.0), (1, -1, -1.0), (2, +1, 0.0), (3, +1, -1.0)]:
        Wfc[r["m"], u] = sgn * K
        Wfc[r["p"], u] = -K / 2
        bfc[u] = bias
    neg = np.array([1.0, -1.0, 0.0, 0.0])   # step(a) = ReLU(a) - ReLU(a-1): 1 iff depth_i < 0
    pos = np.array([0.0, 0.0, 1.0, -1.0])   # step(b): 1 iff depth_i > 0
    Wp[:, r["bad"]] = neg
    # out_i = 0.5 - [depth_i != 0]  (only read at the last position)
    Wp[:, r["out"]] = -(neg + pos)
    bp[r["out"]] = 0.5

    # ---- Layer 1 attention: out_i -= (i+1) * mean_{j<=i} bad_j  ( = #violations at last position i=15)
    uniform_prefix_mean(w, 1, 0, r["bad"], r["out"], scale=-float(N))

    regs = {k: slice(v, v + 1) for k, v in REG.items()}
    return ExecutableTransformer(D_MODEL, N_LAYERS, N_HEADS, D_MLP, w, masks, regs, "out",
                                 output_mode="last_pos_register")


def encode(example):
    s = example
    x = np.array([1.0 if c == "(" else -1.0 for c in s])     # per-symbol code
    first = np.zeros(len(s)); first[0] = 1.0                  # position code: one-hot of index 0
    return {"x": x, "first": first}, len(s)


def decode(raw_output):
    return bool(raw_output > 0)
