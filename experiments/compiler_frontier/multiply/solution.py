"""Exact 6x6-digit decimal multiplication by a compiled (untrained) Transformer.

Layout (seq_len = 24): positions 0..5 hold digits of x (pos i = 10^i), 6..11 hold digits of y,
12..23 are blank output positions (pos 12+k receives digit k of x*y).

Layer 0  attn : 12 heads, head i attends (score gap 60) to position i and copies its digit into G[i]
                at every position.
Layer 0  MLP  : column sums  C[k] = sum_{i+j=k} x_i*y_j  via the exact integer product identity
                a*b = sum_{v=1..9} ReLU(b + 10a - 10v) - 10 * sum_{v=1..8} ReLU(a - v).
Layer 1-3 MLP : carry-save normalisation  C[k] -= 10*q_k, C[k+1] += q_k  with q_k = floor(C[k]/10)
                (sum of step functions).  Column bounds 486 -> 57 -> 14 -> 10.
Layer 4  MLP  : g_k = [C_k >= 10], p_k = [C_k == 9]    (C_k in 0..10, binary flags)
Layer 5  MLP  : carry_k = sum_{m<k} AND(g_m, p_{m+1..k-1})  (mutually exclusive terms),
                C[k] += carry_k - 10*carry_{k+1}   -> C[k] is now the k-th result digit.
Layer 6  MLP  : out = sum_k ReLU(C_k - 10*(1 - onehot_out_k))   (each output position picks its digit)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from neural_compiler import ExecutableTransformer  # noqa: E402

ND = 6                 # digits per operand (the task); build_model(nd) also works for other sizes
N_HEADS = 16


def _dims(nd):
    nc = 2 * nd
    seq = 2 * nd + nc
    d_model = 128 if nd == 6 else int(np.ceil((2 + seq + 2 * nd + 3 * nc + 1) / N_HEADS) * N_HEADS)
    n_gather = int(np.ceil(2 * nd / N_HEADS))          # attention layers needed to gather 2*nd digits
    return nc, seq, d_model, n_gather


def _alloc(ND, NC, SEQ, D_MODEL):
    regs, cur = {}, 0
    for name, w in [("const_one", 1), ("digit", 1), ("pos", SEQ), ("G", 2 * ND), ("C", NC),
                    ("g", NC), ("p", NC), ("out", 1)]:
        regs[name] = slice(cur, cur + w)
        cur += w
    assert cur <= D_MODEL
    return regs


class _MLP:
    """Helper that appends ReLU units h = ReLU(sum w_i x_i + b) with output coefficients."""

    def __init__(self, W, l):
        self.W, self.l, self.u = W, l, 0
        self.cap = W[f"L{l}.W_fc"].shape[1]

    def unit(self, inputs, bias, outputs):
        W_fc, b_fc, W_proj = self.W[f"L{self.l}.W_fc"], self.W[f"L{self.l}.b_fc"], self.W[f"L{self.l}.W_proj"]
        assert self.u < self.cap, "MLP width exceeded"
        for idx, w in inputs:
            W_fc[idx, self.u] += w
        b_fc[self.u] += bias
        for idx, w in outputs:
            W_proj[self.u, idx] += w
        self.u += 1

    def step(self, inputs, T, outputs):
        """Adds outputs * [x >= T] for integer-valued x = sum inputs (robust to |noise| < 0.25)."""
        # ReLU(2(x-T)+1.5) - ReLU(2(x-T)+0.5): 0 for x <= T-0.75, 1 for x >= T-0.25
        self.unit([(i, 2 * w) for i, w in inputs], -2 * T + 1.5, outputs)
        self.unit([(i, 2 * w) for i, w in inputs], -2 * T + 0.5, [(o, -c) for o, c in outputs])


def build_model(nd=ND):
    ND = nd
    NC, SEQ, D_MODEL, n_gather = _dims(nd)
    # MLP width: largest layer is the first normalisation step (2 units per threshold)
    D_MLP = 9 * nd * nd + 8 * nd + 20 * nd * nd     # generous; trimmed to the widest used layer below
    L0 = n_gather - 1                       # layer holding the product MLP
    # carry-save normalisation schedule: per-column upper bounds until every column is <= 10
    bounds = [[min(k + 1, NC - 1 - k, ND) * 81 for k in range(NC)]]
    while max(bounds[-1]) > 10:
        b, new = bounds[-1], [0] * NC
        for k in range(NC):
            new[k] += min(b[k], 9)
            if k + 1 < NC:
                new[k + 1] += b[k] // 10
        bounds.append(new)
    n_norm = len(bounds) - 1
    lg = L0 + 1 + n_norm                    # g/p layer
    N_LAYERS = lg + 3
    R = _alloc(ND, NC, SEQ, D_MODEL)
    dh = D_MODEL // N_HEADS
    W = {}
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            for n, s in [("W_Q", (D_MODEL, dh)), ("b_Q", (dh,)), ("W_K", (D_MODEL, dh)), ("b_K", (dh,)),
                         ("W_V", (D_MODEL, dh)), ("b_V", (dh,)), ("W_O", (dh, D_MODEL))]:
                W[f"L{l}.H{h}.{n}"] = np.zeros(s)
        W[f"L{l}.b_O"] = np.zeros(D_MODEL)
        W[f"L{l}.W_fc"] = np.zeros((D_MODEL, D_MLP))
        W[f"L{l}.b_fc"] = np.zeros(D_MLP)
        W[f"L{l}.W_proj"] = np.zeros((D_MLP, D_MODEL))
        W[f"L{l}.b_proj"] = np.zeros(D_MODEL)

    one, dig, pos = R["const_one"].start, R["digit"].start, R["pos"].start
    G, C, g, p, out = R["G"].start, R["C"].start, R["g"].start, R["p"].start, R["out"].start
    GAP = 60.0
    # ---- Layer 0 attention: head i gathers digit at position i (i < 12) into G[i]
    for i in range(2 * ND):
        la, h = divmod(i, N_HEADS)
        W[f"L{la}.H{h}.b_Q"][0] = np.sqrt(dh) * GAP     # score = GAP * pos_onehot[i] of key
        W[f"L{la}.H{h}.W_K"][pos + i, 0] = 1.0
        W[f"L{la}.H{h}.W_V"][dig, 0] = 1.0
        W[f"L{la}.H{h}.W_O"][0, G + i] = 1.0

    # ---- Layer 0 MLP: column sums of digit products
    m = _MLP(W, L0)
    for i in range(ND):
        a = G + i
        for j in range(ND):
            b = G + ND + j
            for v in range(1, 10):
                m.unit([(b, 1.0), (a, 10.0)], -10.0 * v, [(C + i + j, 1.0)])
        for v in range(1, 9):  # correction 5a(a-1) for each of the ND products that use x_i
            m.unit([(a, 1.0)], -float(v), [(C + i + j, -10.0) for j in range(ND)])

    # ---- Layers 1-3 MLP: carry-save normalisation  C[k] -= 10*q_k, C[k+1] += q_k
    for step in range(n_norm):
        m = _MLP(W, L0 + 1 + step)
        for k in range(NC):
            outs = [(C + k, -10.0)] + ([(C + k + 1, 1.0)] if k + 1 < NC else [])
            for t in range(1, bounds[step][k] // 10 + 1):
                m.step([(C + k, 1.0)], 10 * t, outs)

    # ---- Layer 4 MLP: generate / propagate flags
    m = _MLP(W, lg)
    for k in range(NC):
        m.step([(C + k, 1.0)], 10, [(g + k, 1.0), (p + k, -1.0)])
        m.step([(C + k, 1.0)], 9, [(p + k, 1.0)])

    # ---- Layer 5 MLP: carry lookahead, digit fix-up
    m = _MLP(W, lg + 1)
    for k in range(1, NC):
        for mm in range(k):
            # AND(g_m, p_{m+1..k-1}) = ReLU(g_m + sum p - (k-m-1))   (binary inputs)
            ins = [(g + mm, 1.0)] + [(p + j, 1.0) for j in range(mm + 1, k)]
            outs = [(C + k, 1.0)] + [(C + k - 1, -10.0)]
            m.unit(ins, -float(k - mm - 1), outs)

    # ---- Layer 6 MLP: each output position selects its digit
    m = _MLP(W, lg + 2)
    for k in range(NC):
        m.unit([(C + k, 1.0), (pos + 2 * ND + k, 10.0)], -10.0, [(out, 1.0)])

    used = max(int(np.max(np.nonzero(np.any(W[f"L{l}.W_proj"], axis=1))[0], initial=-1)) + 1
               for l in range(N_LAYERS))
    for l in range(N_LAYERS):
        W[f"L{l}.W_fc"] = W[f"L{l}.W_fc"][:, :used].copy()
        W[f"L{l}.b_fc"] = W[f"L{l}.b_fc"][:used].copy()
        W[f"L{l}.W_proj"] = W[f"L{l}.W_proj"][:used].copy()
    return ExecutableTransformer(D_MODEL, N_LAYERS, N_HEADS, used, W, {}, R, "out")


def _digits(n):
    return [int(c) for c in str(n)[::-1]]  # least-significant first


def encode(example, nd=ND):
    ND, (NC, SEQ, _, _) = nd, _dims(nd)
    x, y = example
    dx, dy = _digits(x), _digits(y)
    dx += [0] * (ND - len(dx)); dy += [0] * (ND - len(dy))
    assert len(dx) == ND and len(dy) == ND
    digit = np.zeros(SEQ)
    digit[:ND] = dx
    digit[ND:2 * ND] = dy
    return {"digit": digit, "pos": np.eye(SEQ)}, SEQ


def decode(raw_output, nd=ND):
    ds = [int(round(float(v))) for v in np.asarray(raw_output)[2 * nd:]]
    return int(sum(d * 10 ** k for k, d in enumerate(ds)))
