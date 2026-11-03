"""Register-machine interpreter compiled directly into Transformer weights (no training).

Layout: 9 positions. Positions 0..7 hold instruction i (one-hot op/a/b) + position one-hot;
position 8 is a blank "machine" position (every position runs the same machine; we read pos 8).
Machine state (pc one-hot 8, registers one-hot 4x16) lives in the residual stream and is
updated additively. 1 init layer + 2 layers per executed step -> 25 layers for 12 steps.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from neural_compiler import ExecutableTransformer  # noqa: E402

N_STEPS = 12
N_INSTR = 8
SEQ = N_INSTR + 1

# ------------------------------------------------------------------ residual layout
_LAYOUT = [("const_one", 1), ("pos", SEQ), ("op", 6), ("a", 4), ("b", 16),
           ("pc", 8), ("R", 64), ("fop", 6), ("fa", 4), ("fb", 16), ("va", 16), ("vb", 16)]
SL, _c = {}, 0
for _n, _w in _LAYOUT:
    SL[_n] = slice(_c, _c + _w)
    _c += _w
D_MODEL = 192
N_HEADS = 1
D_HEAD = D_MODEL // N_HEADS
ATTN_GAP = 60.0  # score margin between the fetched instruction and every other position


def idx(name, k=0):
    return SL[name].start + k


def R(i, v):
    return idx("R", 16 * i + v)


class MLPBuilder:
    """Collects ReLU units: unit = ReLU(sum_j w_j x_j + bias), with output writes."""

    def __init__(self):
        self.units = []

    def unit(self, inputs, bias, outputs):
        self.units.append((inputs, bias, outputs))

    def linear_copy(self, src, dst, coef):
        # coef * x exactly, via ReLU(x) - ReLU(-x)
        self.unit({src: 1.0}, 0.0, {dst: coef})
        self.unit({src: -1.0}, 0.0, {dst: -coef})


def fetch_va_vb_mlp():
    m = MLPBuilder()
    for i in range(4):
        for v in range(16):
            # va[v] = OR_i (a==i AND R_i==v)   (exactly one i active)
            m.unit({idx("fa", i): 1.0, R(i, v): 1.0}, -1.0, {idx("va", v): 1.0})
            m.unit({idx("fb", i): 1.0, R(i, v): 1.0}, -1.0, {idx("vb", v): 1.0})
    return m


def update_mlp():
    m = MLPBuilder()
    fop = lambda k: idx("fop", k)
    for i in range(4):
        fa = idx("fa", i)
        for x in range(16):
            for y in range(16):
                m.unit({fa: 1, idx("va", x): 1, idx("vb", y): 1, fop(1): 1}, -3.0, {R(i, (x + y) % 16): 1.0})
                m.unit({fa: 1, idx("va", x): 1, idx("vb", y): 1, fop(2): 1}, -3.0, {R(i, (x - y) % 16): 1.0})
            m.unit({fa: 1, idx("va", x): 1, fop(4): 1}, -2.0, {R(i, (x - 1) % 16): 1.0})
        for v in range(16):
            m.unit({fa: 1, idx("fb", v): 1, fop(0): 1}, -2.0, {R(i, v): 1.0})
            # remove old value of the written register: w = op in {0,1,2,4}
            m.unit({fa: 1, R(i, v): 1, fop(0): 1, fop(1): 1, fop(2): 1, fop(4): 1}, -2.0, {R(i, v): -1.0})
    for p in range(8):
        pc, nxt = idx("pc", p), idx("pc", (p + 1) % 8)
        # JNZ taken (op3 and r[a] != 0): pc <- b
        m.unit({fop(3): 1, idx("fb", p): 1, idx("va", 0): -1}, -1.0, {idx("pc", p): 1.0})
        # sequential advance: op not in {JNZ, HALT}, or JNZ with r[a]==0
        m.unit({pc: 1, fop(3): -1, fop(5): -1}, 0.0, {nxt: 1.0})
        m.unit({pc: 1, fop(3): 1, idx("va", 0): 1}, -2.0, {nxt: 1.0})
        # HALT: pc stays
        m.unit({pc: 1, fop(5): 1}, -1.0, {pc: 1.0})
        # remove old pc
        m.linear_copy(pc, pc, -1.0)
    # clear per-step scratch registers
    for name in ("fop", "fa", "fb", "va", "vb"):
        for k in range(SL[name].stop - SL[name].start):
            m.linear_copy(idx(name, k), idx(name, k), -1.0)
    return m


def build_model():
    m_fetch, m_upd = fetch_va_vb_mlp(), update_mlp()
    d_mlp = max(len(m_fetch.units), len(m_upd.units))
    n_layers = 1 + 2 * N_STEPS
    W = {}
    for l in range(n_layers):
        for h in range(N_HEADS):
            for nm, shp in (("W_Q", (D_MODEL, D_HEAD)), ("W_K", (D_MODEL, D_HEAD)), ("W_V", (D_MODEL, D_HEAD)),
                            ("b_Q", (D_HEAD,)), ("b_K", (D_HEAD,)), ("b_V", (D_HEAD,)), ("W_O", (D_HEAD, D_MODEL))):
                W[f"L{l}.H{h}.{nm}"] = np.zeros(shp)
        W[f"L{l}.b_O"] = np.zeros(D_MODEL)
        W[f"L{l}.W_fc"] = np.zeros((D_MODEL, d_mlp))
        W[f"L{l}.b_fc"] = np.zeros(d_mlp)
        W[f"L{l}.W_proj"] = np.zeros((d_mlp, D_MODEL))
        W[f"L{l}.b_proj"] = np.zeros(D_MODEL)

    # layer 0: initial machine state pc=0, all registers = 0 (one-hot slot 0), via a constant bias
    W["L0.b_O"][idx("pc", 0)] = 1.0
    for i in range(4):
        W["L0.b_O"][R(i, 0)] = 1.0

    def put_mlp(l, m):
        for u, (ins, bias, outs) in enumerate(m.units):
            for j, w in ins.items():
                W[f"L{l}.W_fc"][j, u] += w
            W[f"L{l}.b_fc"][u] = bias
            for j, w in outs.items():
                W[f"L{l}.W_proj"][u, j] += w

    c = np.sqrt(ATTN_GAP * np.sqrt(D_HEAD))
    for s in range(N_STEPS):
        la, lb = 1 + 2 * s, 2 + 2 * s
        # fetch head: query = pc one-hot, key = instruction position one-hot
        for p in range(8):
            W[f"L{la}.H0.W_Q"][idx("pc", p), p] = c
            W[f"L{la}.H0.W_K"][idx("pos", p), p] = c
        for k, (src, dst) in enumerate([(idx("op", j), idx("fop", j)) for j in range(6)] +
                                       [(idx("a", j), idx("fa", j)) for j in range(4)] +
                                       [(idx("b", j), idx("fb", j)) for j in range(16)]):
            W[f"L{la}.H0.W_V"][src, k] = 1.0
            W[f"L{la}.H0.W_O"][k, dst] = 1.0
        put_mlp(la, m_fetch)
        put_mlp(lb, m_upd)

    return ExecutableTransformer(d_model=D_MODEL, n_layers=n_layers, n_heads=N_HEADS, d_mlp=d_mlp, weights=W,
                                 causal_masks={}, register_slices=dict(SL), output_register="R",
                                 output_mode="last_pos_register")


def encode(example):
    regs = {k: np.zeros((SEQ, SL[k].stop - SL[k].start)) for k in ("pos", "op", "a", "b")}
    for p in range(SEQ):
        regs["pos"][p, p] = 1.0
    for p, (op, a, b) in enumerate(example):
        regs["op"][p, op] = 1.0
        regs["a"][p, a] = 1.0
        regs["b"][p, b] = 1.0
    return regs, SEQ


def decode(raw):
    return tuple(int(np.argmax(raw[16 * i:16 * (i + 1)])) for i in range(4))
