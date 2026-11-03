"""Extra weight-construction primitives for the Seaquest policy (all weights from explicit formulas).

Key primitive: `rel_rows` -- "is there an X in rows >= y+k (or <= y-k)", where y is the (unknown) row of the
player's submarine.  It is ONE tanh unit, built with lexicographic (base-K) row weights:

    S = sum_{cells (r,c) in zone} K^{pos(r)} * x[r,c,ch]  -  sum_{y} K^{pos(t(y))} * sub(y)     unit = [S > -0.5]

where sub(y) = (#ch0 cells in row y)/2 is exactly 1 for the submarine's row and 0 elsewhere, pos(r) = r (down) or
9-r (up) and t(y) = y+k (down) / y-k (up).  If some X lies in a row with pos >= pos(t) then S >= 0; otherwise
S <= (K-1) * (K^0 + ... + K^{pos(t)-1}) - K^{pos(t)} = -1.  So the unit is exact whenever every row holds at most
K-1 X-cells inside the zone (K=4 -> at most 3, which essentially always holds for 5-cell zones).

Two such units with k and k+1 (a monotone pair, unit_k+1 implies unit_k) give the exact indicator
"X in row y+k" = (u_k - u_{k+1})/2 in {0,1}, which a layer-2 linear threshold can consume directly.

Layer-2 units are linear threshold gates over 0/1 literals:  a single H1 unit u gives (u+1)/2, a monotone pair
gives (u_a - u_b)/2.  Output arbitration is strict priority with weights chosen so that a higher level always
beats the sum of all lower levels by >= 10 logits
(and the default action, bias 10, by >= 10).
"""
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_compiler.ir import MLPProgram  # noqa: E402
from neural_compiler.mlp_backend import MLPCompilerBackend  # noqa: E402

G1 = 10.0   # H1 gain on an integer slack with margin 0.5 -> |z| >= 5
G2 = 20.0   # H2 gain per unit of 0/1-literal slack (margin 0.5 -> |z| >= 10)
K = 4       # lexicographic base for rel_rows


class Policy:
    def __init__(self, h1=32, h2=32):
        self.prog = MLPProgram(600, h1, h2, 6, activation="tanh")
        self.h1_names: List[str] = []
        self.h2_names: List[str] = []
        self.rules: List[Tuple[str, int, int]] = []
        self.default_action = None

    @staticmethod
    def idx(r, c, ch):
        return (r * 10 + c) * 6 + ch

    # ---------------------------------------------------------------- layer 1
    def h1(self, name, weights: Dict[int, float], thr: float):
        assert len(self.h1_names) < self.prog.h1_dim, f"H1 overflow at {name}"
        self.prog.add_h1_threshold(name, weights, threshold=thr, gain=G1)
        self.h1_names.append(name)
        return name

    def any_cells(self, name, ch, cells):
        return self.h1(name, {self.idx(r, c, ch): 1.0 for r, c in cells}, 0.5)

    def none_cells(self, name, ch, cells):
        return self.h1(name, {self.idx(r, c, ch): -1.0 for r, c in cells}, -0.5)

    def rel_rows(self, name, ch, cols, k, direction, rows=range(1, 9), force_true=(), neg_ch=None, base=None):
        """True iff some cell (r, c), c in cols, r in rows, has `ch` set with r >= y+k ('down') or r <= y-k ('up').
        For sub rows y in `force_true` the unit is constant True (its sub-row weight is 0, so S >= 0).

        Signed variant (neg_ch given): each cell contributes x[ch] - x[neg_ch] (a digit in [-m, m]); with base
        K >= 2m+1 the sign of S is the sign of the most significant nonzero row digit, so the unit reads
        "the deepest (down) / highest (up) qualifying row with a nonzero digit has a positive digit".  This is an
        APPROXIMATION of "exists": a deeper row with more neg_ch than ch cells masks shallower positive rows."""
        Kb = base or K
        pos = (lambda r: r) if direction == "down" else (lambda r: 9 - r)
        w: Dict[int, float] = {}
        for r in rows:
            for c in cols:
                w[self.idx(r, c, ch)] = w.get(self.idx(r, c, ch), 0.0) + float(Kb ** pos(r))
                if neg_ch is not None:
                    w[self.idx(r, c, neg_ch)] = w.get(self.idx(r, c, neg_ch), 0.0) - float(Kb ** pos(r))
        top = max(pos(r) for r in rows) + 1        # weight exceeding any achievable target sum
        for y in range(9):
            t = y + k if direction == "down" else y - k
            p = min(max(pos(t), 0), top)
            sub_w = 0.0 if y in force_true else float(Kb ** p)
            if neg_ch is not None:
                # all rows enter S; rows with pos < p sum to at most (K^p - 1)/2 in magnitude, and a positive
                # digit pattern at pos >= p is worth >= K^p, so compare against K^p/2 (integer slack >= 0.5).
                sub_w /= 2.0
            for c in range(10):
                w[self.idx(y, c, 0)] = w.get(self.idx(y, c, 0), 0.0) - sub_w / 2.0
        return self.h1(name, w, 0.0 if neg_ch is not None else -0.5)

    def row_pair(self, name, ch, cols, k=0, direction="down", rows=range(1, 9), force_true=(), k2=None, **kw):
        """Monotone pair (u_k, u_k2), k2 = k+1 by default.  Literal ('pair', a, b) = (a - b)/2 is 1 iff an X lies in
        rows y+k .. y+k2-1 (down) AND none lies further away (rows >= y+k2).  Exact "in row y+k" only when no
        X lies beyond; it never reports a false positive."""
        a = self.rel_rows(name + "_a", ch, cols, k, direction, rows, force_true, **kw)
        b = self.rel_rows(name + "_b", ch, cols, k + 1 if k2 is None else k2, direction, rows, **kw)
        return ("pair", a, b)

    # ---------------------------------------------------------------- layer 2
    def gate(self, name, terms: Sequence[Tuple[float, Union[str, tuple]]], thr: float):
        """H2 unit: true iff sum coef * lit > thr, where lit in {0,1}. Use half-integer thr for margin 0.5."""
        assert len(self.h2_names) < self.prog.h2_dim, f"H2 overflow at {name}"
        w: Dict[str, float] = {}
        bias = -G2 * thr
        for coef, lit in terms:
            if isinstance(lit, tuple):          # pair literal: (a - b)/2
                _, a, b = lit
                w[a] = w.get(a, 0.0) + G2 * coef / 2
                w[b] = w.get(b, 0.0) - G2 * coef / 2
            else:                               # single literal: (u + 1)/2
                w[lit] = w.get(lit, 0.0) + G2 * coef / 2
                bias += G2 * coef / 2
        self.prog.add_h2_gate(name, w, bias=bias)
        self.h2_names.append(name)
        return name

    def AND(self, name, pos=(), neg=(), extra=()):
        """All `pos` literals 1 and all `neg` literals 0 (literals may be pairs)."""
        terms = [(1.0, p) for p in pos] + [(-1.0, n) for n in neg] + list(extra)
        return self.gate(name, terms, len(pos) - 0.5)

    # ---------------------------------------------------------------- output
    def rule(self, h2, action, level):
        self.rules.append((h2, action, level))

    def compile(self):
        levels = sorted(set(l for _, _, l in self.rules))
        # A firing rule at level L must beat any other action's logit built from lower levels; the largest such
        # logit is max_a (sum of lower-level rule weights for action a [+ default bias]).  Margin 20.
        weight = {}
        per_action = [10.0 if a == self.default_action else 0.0 for a in range(6)]
        for L in levels:
            weight[L] = 20.0 + max(per_action)
            for _, a, l in self.rules:
                if l == L:
                    per_action[a] += weight[L]
        bias = [0.0] * 6
        terms: Dict[int, Dict[str, float]] = {}
        for h2, a, L in self.rules:
            terms.setdefault(a, {})[h2] = terms.get(a, {}).get(h2, 0.0) + 0.5 * weight[L]
            bias[a] += 0.5 * weight[L]
        if self.default_action is not None:
            bias[self.default_action] += 10.0
        self.prog.output_bias = bias
        self.prog.output_rules = []
        for a in range(6):
            self.prog.set_output_action(a, terms.get(a, {}), bias=bias[a])
        return MLPCompilerBackend.compile(self.prog)
