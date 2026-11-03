"""Chess 'side to move is in check' as a hand-constructed 4-layer Transformer (no training).

Sequence: 64 square positions (index = 8*rank + file) + 1 sink position (index 64).
Layer 0 MLP : side-relative piece features (own king, enemy R/Q, B/Q, N, K, attacking pawns).
Layer 1 attn: 1 head broadcasts the own king's location codes to every position.
Layer 2 attn: 8 ray heads; each selects the NEAREST occupied square on one ray from the king
              (or the sink if the ray is empty) and reads "enemy slider of matching type".
Layer 3 attn: 4 contact heads (knight, king, pawn-for-white-to-move, pawn-for-black-to-move):
              each attends to {enemy piece of that type at an attacking offset} U {sink}.
Output register `att` = sum of the 12 head outputs (~0 or ~1 each); decode thresholds at 0.5.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from neural_compiler import ExecutableTransformer  # noqa: E402

D_MODEL, N_LAYERS, N_HEADS, D_MLP = 1024, 4, 8, 16
D_HEAD = D_MODEL // N_HEADS
SEQ = 65
SINK = 64
A = 2000.0   # condition weight (attention logit units)
S = 40.0     # distance weight (attention logit units per square)

REGS = [
    ("const_one", 1),
    ("pc", 12),       # piece one-hot: 0..5 white P,N,B,R,Q,K ; 6..11 black P,N,B,R,Q,K
    ("tw", 1),        # 1 if White to move else 0
    ("R", 8), ("F", 8), ("D", 15), ("AD", 15), ("SQ", 64),   # position one-hots (rank, file, r-f+7, r+f, square)
    ("Rgt", 8), ("Rlt", 8), ("Fgt", 8), ("Flt", 8),          # thermometers: [r>i], [r<i], [f>i], [f<i]
    ("sink", 1),
    # layer-0 features
    ("ownK", 1), ("eRQ", 1), ("eBQ", 1), ("eN", 1), ("eK", 1), ("bpA", 1), ("wpA", 1),
    # layer-1 broadcast of own king location
    ("kR", 8), ("kF", 8), ("kD", 15), ("kAD", 15), ("kSQ", 64),
    ("att", 1),
]


def _slices():
    sl, c = {}, 0
    for n, w in REGS:
        sl[n] = slice(c, c + w)
        c += w
    assert c <= D_MODEL
    return sl


SL = _slices()


def idx(reg, i=0):
    return SL[reg].start + i


def _empty():
    W = {}
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            W[f"L{l}.H{h}.W_Q"] = np.zeros((D_MODEL, D_HEAD))
            W[f"L{l}.H{h}.b_Q"] = np.zeros(D_HEAD)
            W[f"L{l}.H{h}.W_K"] = np.zeros((D_MODEL, D_HEAD))
            W[f"L{l}.H{h}.b_K"] = np.zeros(D_HEAD)
            W[f"L{l}.H{h}.W_V"] = np.zeros((D_MODEL, D_HEAD))
            W[f"L{l}.H{h}.b_V"] = np.zeros(D_HEAD)
            W[f"L{l}.H{h}.W_O"] = np.zeros((D_HEAD, D_MODEL))
        W[f"L{l}.b_O"] = np.zeros(D_MODEL)
        W[f"L{l}.W_fc"] = np.zeros((D_MODEL, D_MLP))
        W[f"L{l}.b_fc"] = np.zeros(D_MLP)
        W[f"L{l}.W_proj"] = np.zeros((D_MLP, D_MODEL))
        W[f"L{l}.b_proj"] = np.zeros(D_MODEL)
    return W


WP, WN, WB, WR, WQ, WK = range(6)
BP, BN, BB, BR, BQ, BK = range(6, 12)


def _layer0_mlp(W):
    """Side-relative features. For binary x,t: x*(1-t) = ReLU(x - t), x*t = ReLU(x + t - 1)."""
    fc, bfc, pj = W["L0.W_fc"], W["L0.b_fc"], W["L0.W_proj"]
    u = [0]

    def unit(pieces, mode, out):  # mode 'w': gate by White-to-move (tw), 'b': gate by Black-to-move (1-tw)
        k = u[0]
        for p in pieces:
            fc[idx("pc", p), k] = 1.0
        if mode == "w":
            fc[idx("tw"), k] = 1.0
            bfc[k] = -1.0
        else:
            fc[idx("tw"), k] = -1.0
        pj[k, idx(out)] = 1.0
        u[0] += 1

    unit([WK], "w", "ownK"); unit([BK], "b", "ownK")
    unit([BR, BQ], "w", "eRQ"); unit([WR, WQ], "b", "eRQ")
    unit([BB, BQ], "w", "eBQ"); unit([WB, WQ], "b", "eBQ")
    unit([BN], "w", "eN"); unit([WN], "b", "eN")
    unit([BK], "w", "eK"); unit([WK], "b", "eK")
    unit([BP], "w", "bpA")   # black pawn, White to move (attacks downward)
    unit([WP], "b", "wpA")   # white pawn, Black to move (attacks upward)
    return u[0]


def _layer1_broadcast(W):
    """Every position attends to the square holding the own king and copies its location codes."""
    sq = np.sqrt(D_HEAD)
    p = "L1.H0."
    W[p + "b_Q"][0] = sq
    W[p + "W_K"][idx("ownK"), 0] = A
    c = 0
    for src, dst in (("R", "kR"), ("F", "kF"), ("D", "kD"), ("AD", "kAD"), ("SQ", "kSQ")):
        w = SL[src].stop - SL[src].start
        for i in range(w):
            W[p + "W_V"][idx(src, i), c] = 1.0
            W[p + "W_O"][c, idx(dst, i)] = 1.0
            c += 1
    assert c <= D_HEAD


# ray heads: (line one-hot pair, thermometer, coordinate used for distance, sign, value feature)
RAYS = [
    ("F", "kF", "Rgt", "kR", "R", -1, "eRQ"),   # north: same file, r > kr, nearest = smallest r
    ("F", "kF", "Rlt", "kR", "R", +1, "eRQ"),   # south
    ("R", "kR", "Fgt", "kF", "F", -1, "eRQ"),   # east
    ("R", "kR", "Flt", "kF", "F", +1, "eRQ"),   # west
    ("D", "kD", "Rgt", "kR", "R", -1, "eBQ"),   # north-east (r-f const)
    ("D", "kD", "Rlt", "kR", "R", +1, "eBQ"),   # south-west
    ("AD", "kAD", "Rgt", "kR", "R", -1, "eBQ"),  # north-west (r+f const)
    ("AD", "kAD", "Rlt", "kR", "R", +1, "eBQ"),  # south-east
]


def _layer2_rays(W):
    """logit_j = A*[j on king's line] + A*[j on the correct side] + A*occ_j + sign*S*coord_j.
    Exactly the occupied squares of the ray reach ~3A; the nearest one wins by >= S.
    Sink logit 3A - 7.5S (sign -1) or 3A - 0.5S (sign +1): below every real candidate,
    above every non-candidate (<= 2A + 7S)."""
    sq = np.sqrt(D_HEAD)
    for h, (line, kline, thermo, kcoord, coord, sign, val) in enumerate(RAYS):
        p = f"L2.H{h}."
        WQm, WKm = W[p + "W_Q"], W[p + "W_K"]
        c = 0
        n = SL[line].stop - SL[line].start
        for i in range(n):           # same line
            WQm[idx(kline, i), c] = A * sq
            WKm[idx(line, i), c] = 1.0
            c += 1
        for i in range(8):           # correct side of king
            WQm[idx(kcoord, i), c] = A * sq
            WKm[idx(thermo, i), c] = 1.0
            c += 1
        W[p + "b_Q"][c] = sq          # constant query channel
        for pc in range(12):
            WKm[idx("pc", pc), c] = A
        for i in range(8):
            WKm[idx(coord, i), c] = sign * S * i
        WKm[idx("sink"), c] = 3 * A - (7.5 * S if sign < 0 else 0.5 * S)
        W[p + "W_V"][idx(val), 0] = 1.0
        W[p + "W_O"][0, idx("att")] = 1.0


def _offsets_matrix(offsets):
    M = np.zeros((64, 64))
    for s in range(64):
        r, f = divmod(s, 8)
        for dr, df in offsets:
            rr, ff = r + dr, f + df
            if 0 <= rr < 8 and 0 <= ff < 8:
                M[s, rr * 8 + ff] = 1.0
    return M


KNIGHT = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
KING = [(dr, df) for dr in (-1, 0, 1) for df in (-1, 0, 1) if (dr, df) != (0, 0)]
CONTACT = [
    (KNIGHT, "eN"),
    (KING, "eK"),
    ([(1, -1), (1, 1)], "bpA"),    # White king attacked by black pawn one rank above
    ([(-1, -1), (-1, 1)], "wpA"),  # Black king attacked by white pawn one rank below
]


def _layer3_contact(W):
    """logit_j = A*[j at attack offset from king] + A*feat_j ; sink 1.5A. Output ~1 iff some candidate."""
    sq = np.sqrt(D_HEAD)
    for h, (offs, feat) in enumerate(CONTACT):
        p = f"L3.H{h}."
        M = _offsets_matrix(offs)  # M[king_sq, attacker_sq]
        for ks in range(64):
            for js in range(64):
                if M[ks, js]:
                    W[p + "W_Q"][idx("kSQ", ks), js] = A * sq
        for js in range(64):
            W[p + "W_K"][idx("SQ", js), js] = 1.0
        W[p + "b_Q"][64] = sq
        W[p + "W_K"][idx(feat), 64] = A
        W[p + "W_K"][idx("sink"), 64] = 1.5 * A
        W[p + "W_V"][idx(feat), 0] = 1.0
        W[p + "W_O"][0, idx("att")] = 1.0


def build_model():
    W = _empty()
    _layer0_mlp(W)
    _layer1_broadcast(W)
    _layer2_rays(W)
    _layer3_contact(W)
    return ExecutableTransformer(D_MODEL, N_LAYERS, N_HEADS, D_MLP, W, {}, dict(SL), "att", "register")


def _position_codes():
    P = {k: np.zeros((SEQ, SL[k].stop - SL[k].start)) for k in
         ("R", "F", "D", "AD", "SQ", "Rgt", "Rlt", "Fgt", "Flt", "sink")}
    for s in range(64):
        r, f = divmod(s, 8)
        P["R"][s, r] = 1; P["F"][s, f] = 1
        P["D"][s, r - f + 7] = 1; P["AD"][s, r + f] = 1; P["SQ"][s, s] = 1
        for i in range(8):
            P["Rgt"][s, i] = r > i; P["Rlt"][s, i] = r < i
            P["Fgt"][s, i] = f > i; P["Flt"][s, i] = f < i
    P["sink"][SINK, 0] = 1
    return P


_POS = _position_codes()


def encode(example):
    board = np.asarray(example["board"])
    pc = np.zeros((SEQ, 12))
    for r in range(8):
        for f in range(8):
            v = int(board[r][f])
            if v > 0:
                pc[8 * r + f, v - 1] = 1
            elif v < 0:
                pc[8 * r + f, 6 + (-v) - 1] = 1
    regs = dict(_POS)
    regs["pc"] = pc
    regs["tw"] = np.full(SEQ, 1.0 if example["turn"] == 1 else 0.0)
    return regs, SEQ


def decode(raw_output):
    return bool(raw_output[SINK] > 0.5)
