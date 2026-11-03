"""
Grid-game primitives on top of `MLPProgram` (600 -> H1 -> H2 -> 6 tanh MLP over a one-hot 10x10x6 grid).

Every primitive is Boolean (+1 true / -1 false, up to tanh saturation) and exact on binary grid inputs:
  - Layer 1 (H1) reads the grid:      any_, row_any, col_any, count_ge, pair_at
  - Layer 2 (H2) combines H1 signals: and_, or_, copy
  - Output layer:                     arbitrate (strict priority between named H2 rules)
  - Relative position:                adjacent, which is exact but costs one H1 unit per anchor cell.

The builder tracks what each unit costs and raises `CapacityError` with the required and available counts
before a primitive overflows a layer, so an author learns that a design is too expensive rather than receiving a
silently wrong approximation.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .ir import MLPProgram
from .mlp_backend import MLPCompilerBackend, ExecutableMLP

Cell = Tuple[int, int]
G1 = 10.0  # H1 gain: integer input sums with a 0.5 margin give |z| >= 5, so |tanh| >= 0.9999
G2 = 5.0   # H2 gain: Boolean literals with a margin of 1 give |z| >= ~5


class CapacityError(ValueError):
    pass


class SpatialProgram:
    def __init__(self, rows: int = 10, cols: int = 10, channels: int = 6, h1_dim: int = 32, h2_dim: int = 32,
                 n_actions: int = 6):
        self.rows, self.cols, self.channels = rows, cols, channels
        self.prog = MLPProgram(rows * cols * channels, h1_dim, h2_dim, n_actions, activation="tanh")
        self.layer = {}             # unit name -> 1 or 2
        self.cost_log: List[Tuple[str, str, int, int]] = []  # (primitive, name, h1 units, h2 units)
        self._rules: List[Tuple[str, int, int]] = []
        self._default: Optional[int] = None

    # ------------------------------------------------------------------ helpers
    def idx(self, r: int, c: int, ch: int) -> int:
        return (r * self.cols + c) * self.channels + ch

    def all_cells(self) -> List[Cell]:
        return [(r, c) for r in range(self.rows) for c in range(self.cols)]

    def _need(self, primitive: str, h1: int = 0, h2: int = 0):
        free1 = self.prog.h1_dim - len(self.prog.h1_units)
        free2 = self.prog.h2_dim - len(self.prog.h2_units)
        if h1 > free1 or h2 > free2:
            raise CapacityError(
                f"{primitive} needs {h1} H1 and {h2} H2 units; {free1} H1 and {free2} H2 remain "
                f"(of {self.prog.h1_dim}/{self.prog.h2_dim}). Reduce the anchor region or redesign.")

    def _check_new(self, name: str):
        if name in self.layer:
            raise ValueError(f"duplicate unit name {name!r}")

    def _h1(self, primitive: str, name: str, weights: Dict[int, float], threshold: float):
        self._check_new(name)
        self._need(primitive, h1=1)
        self.prog.add_h1_threshold(name, weights, threshold=threshold, gain=G1)
        self.layer[name] = 1
        self.cost_log.append((primitive, name, 1, 0))
        return name

    def _lits(self, pos: Sequence[str], neg: Sequence[str]) -> Dict[str, float]:
        terms: Dict[str, float] = {}
        for n in list(pos) + list(neg):
            if self.layer.get(n) != 1:
                raise KeyError(f"{n!r} is not an H1 unit; H2 gates read H1 units only")
        for n in pos:
            terms[n] = terms.get(n, 0.0) + G2
        for n in neg:
            terms[n] = terms.get(n, 0.0) - G2
        return terms

    # ------------------------------------------------------------------ layer 1
    def any_(self, name: str, ch: int, cells: Iterable[Cell], exclude_ch: Optional[int] = None) -> str:
        """True iff some cell has `ch` set. With `exclude_ch`: some single cell has `ch` set AND `exclude_ch` unset.

        Without exclusion this is one H1 unit. With exclusion the exact form needs one H1 unit per cell plus an H2 OR,
        because a single linear sum lets an `exclude_ch`-only cell elsewhere cancel a valid cell (see any_without_approx).
        Note: with exclusion the returned unit is in layer 2.
        """
        cells = list(cells)
        if exclude_ch is None:
            return self._h1("any", name, {self.idx(r, c, ch): 1.0 for r, c in cells}, 0.5)
        self._need(f"any_without({name})", h1=len(cells), h2=1)
        self._check_new(name)
        parts = [self._h1("cell_without", f"{name}@{r},{c}",
                          {self.idx(r, c, ch): 1.0, self.idx(r, c, exclude_ch): -1.0}, 0.5) for r, c in cells]
        return self.or_(name, pos=parts)

    def any_without_approx(self, name: str, ch: int, cells: Iterable[Cell], exclude_ch: int) -> str:
        """One-unit approximation of any_(..., exclude_ch): sum(ch) - sum(exclude_ch) over the region > 0.5.

        Wrong whenever cells with `exclude_ch` but not `ch` coexist with valid cells (e.g. in Seaquest, an enemy
        bullet in the same row as a diver). Exact only if that combination cannot occur.
        """
        w: Dict[int, float] = {}
        for r, c in cells:
            w[self.idx(r, c, ch)] = 1.0
            w[self.idx(r, c, exclude_ch)] = -1.0
        return self._h1("any_without_approx", name, w, 0.5)

    def row_any(self, name: str, ch: int, r: int, cols: Optional[Iterable[int]] = None, **kw) -> str:
        return self.any_(name, ch, [(r, c) for c in (range(self.cols) if cols is None else cols)], **kw)

    def col_any(self, name: str, ch: int, c: int, rows: Optional[Iterable[int]] = None, **kw) -> str:
        return self.any_(name, ch, [(r, c) for r in (range(self.rows) if rows is None else rows)], **kw)

    def count_ge(self, name: str, ch: int, cells: Iterable[Cell], k: int) -> str:
        """True iff at least k of `cells` have `ch` set."""
        return self._h1("count_ge", name, {self.idx(r, c, ch): 1.0 for r, c in cells}, k - 0.5)

    def pair_at(self, name: str, a_ch: int, a: Cell, b_ch: int, b: Cell) -> str:
        """True iff cell a has a_ch AND cell b has b_ch (exact for binary inputs)."""
        w = {self.idx(*a, a_ch): 1.0}
        w[self.idx(*b, b_ch)] = w.get(self.idx(*b, b_ch), 0.0) + 1.0
        return self._h1("pair_at", name, w, 1.5)

    # ------------------------------------------------------------------ layer 2
    def and_(self, name: str, pos: Sequence[str] = (), neg: Sequence[str] = ()) -> str:
        """True iff every `pos` H1 unit is true and every `neg` H1 unit is false."""
        n = len(pos) + len(neg)
        self._check_new(name)
        self._need("and", h2=1)
        self.prog.add_h2_gate(name, self._lits(pos, neg), bias=-G2 * (n - 1))
        self.layer[name] = 2
        self.cost_log.append(("and", name, 0, 1))
        return name

    def or_(self, name: str, pos: Sequence[str] = (), neg: Sequence[str] = ()) -> str:
        n = len(pos) + len(neg)
        self._check_new(name)
        self._need("or", h2=1)
        self.prog.add_h2_gate(name, self._lits(pos, neg), bias=G2 * (n - 1))
        self.layer[name] = 2
        self.cost_log.append(("or", name, 0, 1))
        return name

    def copy(self, name: str, src: str) -> str:
        return self.and_(name, pos=[src])

    # ------------------------------------------------------------------ relative position
    def adjacent(self, name: str, a_ch: int, b_ch: int, dr: int, dc: int,
                 anchors: Optional[Iterable[Cell]] = None) -> str:
        """True iff some anchor cell p has a_ch and p + (dr, dc) has b_ch.

        Exact, but costs one H1 unit per in-bounds anchor plus one H2 OR. With the default anchors (whole grid)
        this needs ~90-100 H1 units; restricting where `a` can be (e.g. pinning the player's column) is what makes
        relative-position logic affordable.
        """
        anchors = list(self.all_cells() if anchors is None else anchors)
        pairs = [(p, (p[0] + dr, p[1] + dc)) for p in anchors
                 if 0 <= p[0] + dr < self.rows and 0 <= p[1] + dc < self.cols]
        self._need(f"adjacent({name})", h1=len(pairs), h2=1)
        self._check_new(name)
        parts = [self.pair_at(f"{name}@{p[0]},{p[1]}", a_ch, p, b_ch, q) for p, q in pairs]
        return self.or_(name, pos=parts)

    # ------------------------------------------------------------------ output
    def rule(self, h2_name: str, action: int, level: int):
        """When H2 unit `h2_name` is true, prefer `action`; higher `level` strictly overrides lower levels."""
        if self.layer.get(h2_name) != 2:
            raise KeyError(f"{h2_name!r} is not an H2 unit")
        self._rules.append((h2_name, action, level))

    def default(self, action: int):
        self._default = action

    def compile(self) -> ExecutableMLP:
        n = max(1, len(self._rules))
        biases = [0.0] * self.prog.output_dim
        terms: Dict[int, Dict[str, float]] = {}
        base = 12.0  # logit gap >= base/2 = 6 between a firing rule and the default (softmax odds ~400:1)
        for h2, action, level in self._rules:
            w = base * (n + 1) ** level  # exceeds the sum of all lower-level rules
            # 0.5 * w * (h + 1): contributes w when the rule fires and 0 otherwise.
            terms.setdefault(action, {})[h2] = terms.get(action, {}).get(h2, 0.0) + 0.5 * w
            biases[action] += 0.5 * w
        if self._default is not None:
            biases[self._default] += base / 2  # below any single firing rule
        self.prog.output_rules = []
        self.prog.output_bias = biases
        for a in range(self.prog.output_dim):
            self.prog.set_output_action(a, terms.get(a, {}), bias=biases[a])
        return MLPCompilerBackend.compile(self.prog)

    def cost_report(self) -> str:
        by: Dict[str, List[int]] = {}
        for prim, _, h1, h2 in self.cost_log:
            c = by.setdefault(prim, [0, 0, 0]); c[0] += 1; c[1] += h1; c[2] += h2
        lines = [f"H1 {len(self.prog.h1_units)}/{self.prog.h1_dim}, H2 {len(self.prog.h2_units)}/{self.prog.h2_dim}"]
        lines += [f"  {p:10s} x{c[0]:3d}  H1={c[1]:3d}  H2={c[2]:3d}" for p, c in sorted(by.items())]
        return "\n".join(lines)
