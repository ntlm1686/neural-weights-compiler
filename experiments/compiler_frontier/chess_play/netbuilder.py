"""Tiny weight-writing toolkit: named residual registers + explicitly written ReLU units / attention heads.

Every weight is written by an explicit formula (no training).  A "unit" is one hidden ReLU neuron of a layer's MLP:
    h = ReLU(sum_i w_i * X[i] + bias);   X[j] += sum_j p_j * h
"""
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from neural_compiler import ExecutableTransformer  # noqa: E402


class NetBuilder:
    def __init__(self, d_model=2048, n_heads=16, n_layers=1):
        self.d_model, self.n_heads, self.n_layers = d_model, n_heads, n_layers
        self.d_head = d_model // n_heads
        self.regs = {"const_one": slice(0, 1)}
        self.cursor = 1
        self.units = [[] for _ in range(n_layers)]      # per layer: (in_idx, in_w, bias, out_idx, out_w)
        self.heads = [dict() for _ in range(n_layers)]  # per layer: head -> (WQ, bQ, WK, WV, WO) sparse specs

    # ---------------------------------------------------------------- registers
    def reg(self, name, width, at=None):
        if at is None:
            at = self.cursor
            self.cursor += width
        if at + width > self.d_model or self.cursor > self.d_model:
            raise ValueError(f"residual overflow allocating {name}")
        self.regs[name] = slice(at, at + width)
        return at

    def idx(self, name, i=0):
        return self.regs[name].start + i

    # ---------------------------------------------------------------- MLP units
    def unit(self, layer, ins, bias, outs):
        """ins / outs: dict or list of (residual index, weight)."""
        ins = list(ins.items()) if isinstance(ins, dict) else list(ins)
        outs = list(outs.items()) if isinstance(outs, dict) else list(outs)
        self.units[layer].append((ins, float(bias), outs))

    # ---------------------------------------------------------------- attention heads
    def head(self, layer, h, WQ=(), bQ=(), WK=(), WV=(), WO=()):
        """Sparse entries: WQ/WK/WV: (resid_row, head_col, w); bQ: (head_col, w); WO: (head_row, resid_col, w)."""
        self.heads[layer][h] = (list(WQ), list(bQ), list(WK), list(WV), list(WO))

    # ---------------------------------------------------------------- build
    def build(self, output_register):
        D, dh = self.d_model, self.d_head
        W = {}
        for l in range(self.n_layers):
            for h in range(self.n_heads):
                WQ = np.zeros((D, dh)); bQ = np.zeros(dh); WK = np.zeros((D, dh)); bK = np.zeros(dh)
                WV = np.zeros((D, dh)); bV = np.zeros(dh); WO = np.zeros((dh, D))
                if h in self.heads[l]:
                    q, bq, k, v, o = self.heads[l][h]
                    for r, c, w in q: WQ[r, c] += w
                    for c, w in bq: bQ[c] += w
                    for r, c, w in k: WK[r, c] += w
                    for r, c, w in v: WV[r, c] += w
                    for r, c, w in o: WO[r, c] += w
                W[f"L{l}.H{h}.W_Q"], W[f"L{l}.H{h}.b_Q"], W[f"L{l}.H{h}.W_K"], W[f"L{l}.H{h}.b_K"] = WQ, bQ, WK, bK
                W[f"L{l}.H{h}.W_V"], W[f"L{l}.H{h}.b_V"], W[f"L{l}.H{h}.W_O"] = WV, bV, WO
            W[f"L{l}.b_O"] = np.zeros(D)
            n = max(1, len(self.units[l]))
            fc = np.zeros((D, n)); bfc = np.zeros(n); pj = np.zeros((n, D))
            for u, (ins, b, outs) in enumerate(self.units[l]):
                for i, w in ins: fc[i, u] += w
                bfc[u] = b
                for j, w in outs: pj[u, j] += w
            W[f"L{l}.W_fc"], W[f"L{l}.b_fc"], W[f"L{l}.W_proj"], W[f"L{l}.b_proj"] = fc, bfc, pj, np.zeros(D)
        d_mlp = max(len(u) for u in self.units)
        return ExecutableTransformer(d_model=D, n_layers=self.n_layers, n_heads=self.n_heads, d_mlp=d_mlp,
                                     weights=W, causal_masks={}, register_slices=dict(self.regs),
                                     output_register=output_register, output_mode="register")
