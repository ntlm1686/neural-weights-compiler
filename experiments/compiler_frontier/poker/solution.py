"""Texas hold'em showdown compiled into a Transformer (no training).

Layout: 9 positions, one per card. Registers per position: rank one-hot (13), suit one-hot (4),
owner one-hot (A hole / B hole / board). Every position computes the same thing; output read at position 0.
See NOTES.md for the per-layer algorithm.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from neural_compiler import ExecutableTransformer  # noqa: E402

NR, NS = 13, 4
KB = 2 ** 13          # kicker block (sum of 2^r over <=5 ranks < 2^13)
CB = 2 ** 26          # category base
GATE = 2 ** 30        # gate constant (> max category value 9*2^26)
PLAYERS = ("A", "B")


class Builder:
    def __init__(self):
        self.regs = {"const_one": (0, 1)}
        self.cursor = 1
        self.mlp = {}      # layer -> list of units (in_list, bias, out_list)
        self.attn = {}     # layer -> list of (src_reg, dst_reg, scale)

    def reg(self, name, width=1):
        self.regs[name] = (self.cursor, width)
        self.cursor += width
        return name

    def ix(self, name, i=0):
        s, w = self.regs[name]
        assert 0 <= i < w, (name, i)
        return s + i

    def unit(self, layer, inp, bias, out):
        """inp/out: list of ((reg, i), weight). hidden = ReLU(sum inp + bias); residual += hidden * out."""
        self.mlp.setdefault(layer, []).append((inp, bias, out))

    def build(self, n_layers, n_heads, out_reg):
        d_model = 64 * int(np.ceil(self.cursor / 64))
        while d_model % n_heads:
            d_model += 64
        d_head = d_model // n_heads
        d_mlp = max(len(u) for u in self.mlp.values())
        W = {}
        for l in range(n_layers):
            for h in range(n_heads):
                for k in ("Q", "K", "V"):
                    W[f"L{l}.H{h}.W_{k}"] = np.zeros((d_model, d_head))
                    W[f"L{l}.H{h}.b_{k}"] = np.zeros(d_head)
                W[f"L{l}.H{h}.W_O"] = np.zeros((d_head, d_model))
            W[f"L{l}.b_O"] = np.zeros(d_model)
            W[f"L{l}.W_fc"] = np.zeros((d_model, d_mlp))
            W[f"L{l}.b_fc"] = np.zeros(d_mlp)
            W[f"L{l}.W_proj"] = np.zeros((d_mlp, d_model))
            W[f"L{l}.b_proj"] = np.zeros(d_model)
            for h, (src, dst, scale) in enumerate(self.attn.get(l, [])):
                # Q = K = 0 -> uniform attention over all positions; V = src, O = scale * dst
                s0, w = self.regs[src]
                d0, w2 = self.regs[dst]
                assert w == w2 and w <= d_head
                for i in range(w):
                    W[f"L{l}.H{h}.W_V"][s0 + i, i] = 1.0
                    W[f"L{l}.H{h}.W_O"][i, d0 + i] = scale
            for j, (inp, bias, out) in enumerate(self.mlp.get(l, [])):
                for (r, i), w in inp:
                    W[f"L{l}.W_fc"][self.ix(r, i), j] += w
                W[f"L{l}.b_fc"][j] = bias
                for (r, i), w in out:
                    W[f"L{l}.W_proj"][j, self.ix(r, i)] += w
        regs = {k: slice(s, s + w) for k, (s, w) in self.regs.items()}
        return ExecutableTransformer(d_model, n_layers, n_heads, d_mlp, W, {}, regs, out_reg,
                                     output_mode="first_pos_register")


def straight_ranks(h):
    """h = 0..9: straight with high card index h+3 (h=0 is the wheel A-2-3-4-5)."""
    return [12, 0, 1, 2, 3] if h == 0 else list(range(h - 1, h + 4))


def step_units(b, layer, inp, k, out):
    """out += [s >= k] for an integer-valued linear form s (inp): ReLU(s-k+1) - ReLU(s-k)."""
    b.unit(layer, inp, -(k - 1), [(o, w) for o, w in out])
    b.unit(layer, inp, -k, [(o, -w) for o, w in out])


def topk_units(b, layer, bits, k, out_fn):
    """bits: list over index r of input forms (each binary). For each r, t_r = bits_r AND (#{r' > r: bits_r'} < k)
    computed in one layer as ReLU(b_r - a_r) - ReLU(-a_r), a_r = above_r - k + 1 (integer).
    out_fn(r) -> list of (target, weight) receiving t_r."""
    n = len(bits)
    for r in range(n):
        above = [term for rr in range(r + 1, n) for term in bits[rr]]
        a_pos = above                      # a = sum(above) - (k-1)
        b.unit(layer, bits[r] + [(t, -w) for t, w in a_pos], (k - 1), out_fn(r))
        b.unit(layer, [(t, -w) for t, w in a_pos], (k - 1), [(o, -w) for o, w in out_fn(r)])


def build_model():
    b = Builder()
    b.reg("rank", NR); b.reg("suit", NS); b.reg("owner", 3)
    b.reg("out")
    for P in PLAYERS:
        for name, w in (("x", 52), ("X", 52), ("p1", NR), ("p2", NR), ("p3", NR), ("p4", NR), ("fl", NS), ("y", NR),
                        ("st", 10), ("sf", 10), ("t2p", NR), ("t1p3", NR), ("i_p", 1), ("i_2p", 1), ("i_3", 1),
                        ("i_4", 1), ("i_st", 1), ("i_fh", 1), ("i_sf", 1), ("T", 9), ("S", 1)):
            b.reg(P + name, w)
    one = ("const_one", 0)
    owner_of = {"A": 0, "B": 1}
    for P in PLAYERS:
        R = lambda n, i=0: (P + n, i)  # noqa: E731
        # L0 MLP: per-card visible indicator x[r,s] = rank_r AND suit_s AND (owner == P or board)
        for r in range(NR):
            for s in range(NS):
                b.unit(0, [(("rank", r), 1), (("suit", s), 1), (("owner", owner_of[P]), 1), (("owner", 2), 1)], -2,
                       [(R("x", r * NS + s), 1)])
        # L1 attention: X = 9 * mean_positions(x) = counts over the 7 visible cards (set membership)
        b.attn.setdefault(1, []).append((P + "x", P + "X", 9.0))
        # L1 MLP: rank multiplicity steps and flush-suit indicator
        for r in range(NR):
            cnt = [(R("X", r * NS + s), 1) for s in range(NS)]
            for k in range(1, 5):
                step_units(b, 1, cnt, k, [(R(f"p{k}", r), 1)])
        for s in range(NS):
            step_units(b, 1, [(R("X", r * NS + s), 1) for r in range(NR)], 5, [(R("fl", s), 1)])
        # L2 MLP
        for r in range(NR):  # flush-suit cards y[r]
            for s in range(NS):
                b.unit(2, [(R("X", r * NS + s), 1), (R("fl", s), 1)], -1, [(R("y", r), 1)])
        for h in range(10):  # straights
            b.unit(2, [(R("p1", r), 1) for r in straight_ranks(h)], -4, [(R("st", h), 1)])
        step_units(b, 2, [(R("p2", r), 1) for r in range(NR)], 1, [(R("i_p"), 1)])
        step_units(b, 2, [(R("p2", r), 1) for r in range(NR)], 2, [(R("i_2p"), 1)])
        step_units(b, 2, [(R("p3", r), 1) for r in range(NR)], 1, [(R("i_3"), 1)])
        step_units(b, 2, [(R("p4", r), 1) for r in range(NR)], 1, [(R("i_4"), 1)])
        bits = lambda f: [f(r) for r in range(NR)]  # noqa: E731
        # high card: top-5 of p1
        topk_units(b, 2, bits(lambda r: [(R("p1", r), 1)]), 5, lambda r: [(R("T", 0), 2 ** r)])
        # pair kickers: top-3 of singles (p1 - p2)
        topk_units(b, 2, bits(lambda r: [(R("p1", r), 1), (R("p2", r), -1)]), 3, lambda r: [(R("T", 1), 2 ** r)])
        # trips kickers: top-2 of p1 - p3
        topk_units(b, 2, bits(lambda r: [(R("p1", r), 1), (R("p3", r), -1)]), 2, lambda r: [(R("T", 3), 2 ** r)])
        # two pair: top-2 pairs
        topk_units(b, 2, bits(lambda r: [(R("p2", r), 1)]), 2, lambda r: [(R("t2p", r), 1), (R("T", 2), KB * 2 ** r)])
        # quads kicker: top-1 of p1 - p4
        topk_units(b, 2, bits(lambda r: [(R("p1", r), 1), (R("p4", r), -1)]), 1, lambda r: [(R("T", 7), 2 ** r)])
        # full house trips rank: top-1 of p3
        topk_units(b, 2, bits(lambda r: [(R("p3", r), 1)]), 1, lambda r: [(R("t1p3", r), 1), (R("T", 6), KB * 2 ** r)])
        # L3 MLP
        for h in range(10):  # straight flushes
            b.unit(3, [(R("y", r), 1) for r in straight_ranks(h)], -4, [(R("sf", h), 1)])
        topk_units(b, 3, bits(lambda r: [(R("y", r), 1)]), 5, lambda r: [(R("T", 5), 2 ** r)])
        topk_units(b, 3, [[(R("st", h), 1)] for h in range(10)], 1, lambda h: [(R("T", 4), 2 ** h)])
        topk_units(b, 3, bits(lambda r: [(R("p1", r), 1), (R("t2p", r), -1)]), 1, lambda r: [(R("T", 2), 2 ** r)])
        topk_units(b, 3, bits(lambda r: [(R("p2", r), 1), (R("t1p3", r), -1)]), 1, lambda r: [(R("T", 6), 2 ** r)])
        step_units(b, 3, [(R("st", h), 1) for h in range(10)], 1, [(R("i_st"), 1)])
        b.unit(3, [(R("i_3"), 1), (R("i_2p"), 1)], -1, [(R("i_fh"), 1)])
        # L4 MLP: straight flush
        topk_units(b, 4, [[(R("sf", h), 1)] for h in range(10)], 1, lambda h: [(R("T", 8), 2 ** h)])
        step_units(b, 4, [(R("sf", h), 1) for h in range(10)], 1, [(R("i_sf"), 1)])
        # L5 MLP: S = value of the best category (gate: category present and no higher category present)
        ind = [None, R("i_p"), R("i_2p"), R("i_3"), R("i_st"), "flush", R("i_fh"), R("i_4"), R("i_sf")]

        def ind_terms(c, w):
            if c == 5:
                return [(R("fl", s), w) for s in range(NS)]
            return [(ind[c], w)]
        linear_extra = {1: [(R("p2", r), KB * r) for r in range(NR)],
                        3: [(R("p3", r), KB * r) for r in range(NR)],
                        7: [(R("p4", r), KB * r) for r in range(NR)]}
        for c in range(9):
            inp = [(R("T", c), 1)] + linear_extra.get(c, [])
            bias = c * CB
            if c > 0:
                inp += ind_terms(c, GATE)
                bias -= GATE
            for c2 in range(c + 1, 9):
                inp += ind_terms(c2, -GATE)
            b.unit(5, inp, bias, [(R("S"), 1)])
    # L6 MLP: out = clamp(S_A - S_B, -1, 1)
    d = [(("AS", 0), 1), (("BS", 0), -1)]
    nd = [(("AS", 0), -1), (("BS", 0), 1)]
    o = ("out", 0)
    b.unit(6, d, 0, [(o, 1)]); b.unit(6, d, -1, [(o, -1)])
    b.unit(6, nd, 0, [(o, -1)]); b.unit(6, nd, -1, [(o, 1)])
    return b.build(n_layers=7, n_heads=2, out_reg="out")


def encode(example):
    cards = list(example["hole_a"]) + list(example["hole_b"]) + list(example["board"])
    owners = [0, 0, 1, 1, 2, 2, 2, 2, 2]
    rank = np.zeros((9, NR)); suit = np.zeros((9, NS)); owner = np.zeros((9, 3))
    for i, ((r, s), o) in enumerate(zip(cards, owners)):
        rank[i, r - 2] = 1; suit[i, s] = 1; owner[i, o] = 1
    return {"rank": rank, "suit": suit, "owner": owner}, 9


def decode(raw_output):
    return int(np.round(float(raw_output)))
