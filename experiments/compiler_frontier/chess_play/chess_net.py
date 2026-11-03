"""Hand-written chess policy network (no training).  See NOTES.md for the design.

Sequence: 32 tokens; token k receives the raw one-hots of squares 2k, 2k+1 and outputs the 2x64 move scores of
from-squares 2k, 2k+1.  Everything else (board gather, relative colours, attack maps, SEE-lite risk, hanging
pieces, move scoring) is computed inside the network by explicitly written ReLU units.
"""
import numpy as np

from netbuilder import NetBuilder

P, N, B, R, Q, K = range(6)
VAL = [1, 3, 3, 5, 9, 0]
NTOK = 32
ORTH = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIAG = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
KNIGHT = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
KING = ORTH + DIAG


def sq(r, c):
    return r * 8 + c if 0 <= r < 8 and 0 <= c < 8 else None


def rc(s):
    return divmod(s, 8)


def ray(s, d):
    r, c = rc(s)
    out = []
    while True:
        r, c = r + d[0], c + d[1]
        t = sq(r, c)
        if t is None:
            return out
        out.append(t)


def offsets(s, offs):
    r, c = rc(s)
    return [t for t in (sq(r + a, c + b) for a, b in offs) if t is not None]


def centrality(s):
    r, c = rc(s)
    return -(abs(r - 3.5) + abs(c - 3.5))  # -7 (corner) .. -1 (centre)


# piece-square preferences (pawn units), hand-chosen
def pst(T, s, up):
    r, c = rc(s)
    if T == N:
        return 0.08 * centrality(s)
    if T == B:
        return 0.04 * centrality(s)
    if T == Q:
        return 0.01 * centrality(s)
    if T == P:
        adv = r if up else 7 - r
        centre_file = 0.06 if c in (3, 4) else 0.03 if c in (2, 5) else 0.0
        return 0.04 * adv + centre_file * min(adv, 3)
    return 0.0




def line_pairs(s, dirs):
    """(target, between-squares) for every square visible from s along dirs (geometry only)."""
    for d in dirs:
        line = ray(s, d)
        for j, a in enumerate(line):
            yield a, line[:j]


def build():
    L_ATT, L_IND, L_COV, L_TAC, L_OUT = 1, 2, 3, 4, 5
    nb = NetBuilder(d_model=2048, n_heads=16, n_layers=6)
    nb.reg("tok", NTOK)
    nb.reg("turn", 2)
    nb.reg("castle", 4)
    nb.reg("B", 768)     # input board -> (L0) relative board REL: slots 0..5 side to move, 6..11 opponent
    nb.reg("A", 448)     # L1 attack counts (6x64) -> L3 COV/BLK (4x64) -> L4 RISK (6x64) + H (64)
    nb.reg("J", 448)     # L2 indicators (7x64) -> L4 tactical bonus TB (5x64)
    nb.reg("C", 257)     # L1 CK (2x64), L2 FL, KCAP (2x64), MOBP (1); first 128 dims become OUT in L4
    nb.regs["OUT"] = slice(nb.regs["C"].start, nb.regs["C"].start + 128)

    REL = lambda s, k: nb.idx("B", s * 12 + k)
    TOK = lambda k: nb.idx("tok", k)
    WT, BT = nb.idx("turn", 0), nb.idx("turn", 1)
    cnt = {n: i for i, n in enumerate(["EP", "EN", "ER", "EQ", "EK", "OA"])}
    CNT = lambda name, s: nb.idx("A", cnt[name] * 64 + s)
    cvn = {n: i for i, n in enumerate(["COVO", "COVD", "BLKO", "BLKD"])}
    COV = lambda name, s: nb.idx("A", cvn[name] * 64 + s)
    rk = {n: i for i, n in enumerate(["Ppush", "Pcap", "M", "R", "Q", "Qpush"])}
    RISK = lambda name, s: nb.idx("A", rk[name] * 64 + s)
    Hh = lambda s: nb.idx("A", 384 + s)
    jn = {n: i for i, n in enumerate(["J1", "J3", "J5", "J9", "JA", "D1", "D2"])}
    J = lambda name, s: nb.idx("J", jn[name] * 64 + s)
    tbn = {N: 0, B: 1, R: 2, Q: 3, K: 4}
    TB = lambda T, s: nb.idx("J", tbn[T] * 64 + s)
    cn = {n: i for i, n in enumerate(["CKO", "CKD", "FL", "KCAP"])}
    Cc = lambda name, s: nb.idx("C", cn[name] * 64 + s)
    MOBP = nb.idx("C", 256)
    OUT = lambda f, t: nb.idx("OUT", (f % 2) * 64 + t)

    def occ(s, w=1.0):
        return [(REL(s, k), w) for k in range(12)]

    def occ_noking(s, w=1.0):  # occupancy with the enemy king transparent
        return [(REL(s, k), w) for k in range(12) if k != 6 + K]

    # ------------------------------------------------------------------ L0: gather board, make it relative
    for h in range(6):
        WV = [(nb.idx("B", 128 * h + i), i, 1.0) for i in range(128)]
        WO = [(i, nb.idx("B", 128 * h + i), float(NTOK - 1)) for i in range(128)]
        WQ = [(TOK(j), j, 60.0) for j in range(NTOK)]
        WK = [(TOK(j), j, -60.0) for j in range(NTOK)]   # self-exclusion: uniform mean over the other 31 tokens
        nb.head(0, h, WQ=WQ, WK=WK, WV=WV, WO=WO)
    for s in range(64):
        for q in range(12):  # black to move: swap colours so that slots 0..5 = side to move
            nb.unit(0, [(REL(s, q), 1.0), (BT, 1.0)], -1.0, [(REL(s, q), -1.0), (REL(s, (q + 6) % 12), 1.0)])

    # ------------------------------------------------------------------ L1: attack counts + king-visibility
    for s in range(64):
        r, c = rc(s)
        below = [t for t in (sq(r - 1, c - 1), sq(r - 1, c + 1)) if t is not None]  # white-pawn attackers of s
        above = [t for t in (sq(r + 1, c - 1), sq(r + 1, c + 1)) if t is not None]  # black-pawn attackers of s
        for srcs, gate, who in [(above, WT, "E"), (below, BT, "E"), (below, WT, "O"), (above, BT, "O")]:
            if srcs:
                k = 6 + P if who == "E" else P
                nb.unit(L_ATT, [(REL(t, k), 1.0) for t in srcs] + [(gate, 4.0)], -4.0,
                        [(CNT("EP" if who == "E" else "OA", s), 1.0)])
        kn = offsets(s, KNIGHT)
        nb.unit(L_ATT, [(REL(t, 6 + N), 1.0) for t in kn], 0.0, [(CNT("EN", s), 1.0)])
        nb.unit(L_ATT, [(REL(t, N), 1.0) for t in kn], 0.0, [(CNT("OA", s), 1.0)])
        kg = offsets(s, KING)
        nb.unit(L_ATT, [(REL(t, 6 + K), 1.0) for t in kg], 0.0, [(CNT("EK", s), 1.0)])
        nb.unit(L_ATT, [(REL(t, K), 1.0) for t in kg], 0.0, [(CNT("OA", s), 1.0)])
        for dirs, etype, ename, ckname in [(ORTH, R, "ER", "CKO"), (DIAG, B, "EN", "CKD")]:
            for a, betw in line_pairs(s, dirs):
                between = [x for b in betw for x in occ(b, -1.0)]
                nb.unit(L_ATT, [(REL(a, 6 + etype), 1.0)] + between, 0.0, [(CNT(ename, s), 1.0)])
                nb.unit(L_ATT, [(REL(a, 6 + Q), 1.0)] + between, 0.0, [(CNT("EQ", s), 1.0)])
                nb.unit(L_ATT, [(REL(a, etype), 1.0), (REL(a, Q), 1.0)] + between, 0.0, [(CNT("OA", s), 1.0)])
                # a slider of this line type standing on s would check the enemy king on a
                nb.unit(L_ATT, [(REL(a, 6 + K), 1.0)] + between, 0.0, [(Cc(ckname, s), 1.0)])

    # ------------------------------------------------------------------ L2: indicators, king flight squares
    for s in range(64):
        cum = []
        for name, jname in [("EP", "J1"), ("EN", "J3"), ("ER", "J5"), ("EQ", "J9"), ("EK", "JA")]:
            cum = cum + [(CNT(name, s), 1.0)]
            nb.unit(L_IND, cum, 0.0, [(J(jname, s), 1.0)])
            nb.unit(L_IND, cum, -1.0, [(J(jname, s), -1.0)])
        oa = [(CNT("OA", s), 1.0)]
        nb.unit(L_IND, oa, 0.0, [(J("D1", s), 1.0)])
        nb.unit(L_IND, oa, -1.0, [(J("D1", s), -1.0), (J("D2", s), 1.0)])
        nb.unit(L_IND, oa, -2.0, [(J("D2", s), -1.0)])
        for name in cnt:
            nb.unit(L_IND, [(CNT(name, s), 1.0)], 0.0, [(CNT(name, s), -1.0)])
        nearK = [(REL(y, 6 + K), 1.0) for y in offsets(s, KING)]
        # FL: s is a free flight square of the enemy king (adjacent, not attacked by us, not enemy-occupied)
        nb.unit(L_IND, nearK + oa_neg(oa) + [(REL(s, 6 + k), -1.0) for k in range(6)], 0.0, [(Cc("FL", s), 1.0)])
        # KCAP: enemy king adjacent to s and s defended by at most one of our pieces (the mover itself)
        two_near = [(i, 2 * w) for i, w in nearK]
        nb.unit(L_IND, two_near + oa_neg(oa), 0.0, [(Cc("KCAP", s), 1.0)])
        nb.unit(L_IND, two_near + oa_neg(oa), -1.0, [(Cc("KCAP", s), -1.0)])
        # mobile enemy pawns (single push square empty)
        r, c = rc(s)
        for dr, gate in [(-1, WT), (1, BT)]:
            t = sq(r + dr, c)
            if t is not None:
                nb.unit(L_IND, [(REL(s, 6 + P), 1.0), (gate, 1.0)] + occ(t, -1.0), -1.0, [(MOBP, 1.0)])

    # ------------------------------------------------------------------ L3: flight coverage and blockability from t
    for t in range(64):
        for dirs, cov, blk in [(ORTH, "COVO", "BLKO"), (DIAG, "COVD", "BLKD")]:
            for x, betw in line_pairs(t, dirs):
                nb.unit(L_COV, [(Cc("FL", x), 1.0)] + [y for b in betw for y in occ_noking(b, -1.0)], 0.0,
                        [(COV(cov, t), 1.0)])
                if betw:
                    nb.unit(L_COV, [(REL(x, 6 + K), 10.0)] + [y for b in betw for y in occ(b, -10.0)] +
                            [(J("J9", b), 1.0) for b in betw], -10.0, [(COV(blk, t), 1.0)])

    # ------------------------------------------------------------------ L4: SEE-lite risk, hanging values, mate/stalemate
    TERM2 = {1: {}, 3: {"J1": 2.0}, 5: {"J1": 2.0, "J3": 2.0}, 9: {"J1": 2.0, "J3": 2.0, "J5": 4.0}}
    totfl = [(Cc("FL", x), 1.0) for x in range(64)]
    enp = [(REL(s, 6 + k), 1.0) for s in range(64) for k in (N, B, R, Q)]
    for s in range(64):
        # clear COV/BLK (A[0:256]) and J, CK (C[0:128]) before reuse
        for name in cvn:
            nb.unit(L_TAC, [(COV(name, s), 1.0)], 0.0, [(COV(name, s), -1.0)])
        for name in jn:
            nb.unit(L_TAC, [(J(name, s), 1.0)], 0.0, [(J(name, s), -1.0)])
        for name in ("CKO", "CKD"):
            nb.unit(L_TAC, [(Cc(name, s), 1.0)], 0.0, [(Cc(name, s), -1.0)])
        for name, v, D in [("Ppush", 1, "D1"), ("Pcap", 1, "D2"), ("M", 3, "D2"), ("R", 5, "D2"),
                           ("Q", 9, "D2"), ("Qpush", 9, "D1")]:
            t2 = [(J(j, s), w) for j, w in TERM2[v].items()]
            if t2:
                nb.unit(L_TAC, t2, 0.0, [(RISK(name, s), 1.0)])
            nb.unit(L_TAC, [(J("JA", s), v), (J(D, s), -v)] + [(i, -w) for i, w in t2], 0.0,
                    [(RISK(name, s), 1.0)])
        for T in [P, N, B, R, Q]:
            v = VAL[T]
            t2 = [(J(j, s), w) for j, w in TERM2[v].items()]
            g = [(REL(s, T), 100.0)]
            if t2:
                nb.unit(L_TAC, t2 + g, -100.0, [(Hh(s), 1.0)])
            nb.unit(L_TAC, [(J("JA", s), v), (J("D1", s), -v)] + [(i, -w) for i, w in t2] + g, -100.0,
                    [(Hh(s), 1.0)])
        # tactical bonus TB(T, t): mate in one, stalemate avoidance, checks, king-net tightening
        kn = offsets(s, KNIGHT)
        for T in (N, B, R, Q):
            if T == N:
                check = [(REL(x, 6 + K), 1.0) for x in kn]
                cover = [(Cc("FL", x), 1.0) for x in kn]
                block = []
            else:
                lines = {B: ["D"], R: ["O"], Q: ["O", "D"]}[T]
                check = [(Cc("CK" + l, s), 1.0) for l in lines]
                cover = [(COV("COV" + l, s), 1.0) for l in lines]
                block = [(COV("BLK" + l, s), -1.0) for l in lines]
            flights = totfl + [(Cc("FL", s), -1.0), (Cc("KCAP", s), 1.0)] + [(i, -w) for i, w in cover]
            neg_fl = [(i, -w) for i, w in flights]
            nb.unit(L_TAC, check + neg_fl + [(J("J9", s), -2.0)] + block, 0.0, [(TB(T, s), 100.0)])
            nb.unit(L_TAC, [(i, -w) for i, w in check] + neg_fl + [(i, -w) for i, w in enp] +
                    [(REL(s, 6 + k), 1.0) for k in (N, B, R, Q)] + [(MOBP, -1.0)], 1.0, [(TB(T, s), -50.0)])
            nb.unit(L_TAC, check, 0.0, [(TB(T, s), 0.25)])
            nb.unit(L_TAC, cover, 0.0, [(TB(T, s), 0.05)])
        kcover = [(Cc("FL", x), 1.0) for x in offsets(s, KING)]
        kfl = totfl + [(Cc("FL", s), -1.0)] + [(i, -w) for i, w in kcover]
        nb.unit(L_TAC, [(i, -w) for i, w in kfl] + [(i, -w) for i, w in enp] + [(MOBP, -1.0)], 1.0,
                [(TB(K, s), -50.0)])
        nb.unit(L_TAC, [(J("JA", s), 1.0)], 0.0, [(TB(K, s), -200.0)])

    # ------------------------------------------------------------------ L5: move scores
    G, Cst = 200.0, 50.0
    victim = lambda t: [(REL(t, 6 + k), float(VAL[k])) for k in range(5)]
    own_at = lambda t: [(REL(t, k), -G) for k in range(6)]

    def move_unit(f, t, T, risk, extra_ins=(), extra_bias=0.0):
        ins = [(TOK(f // 2), G), (REL(f, T), G), (Hh(f), 1.0)] + victim(t) + own_at(t) + list(extra_ins)
        if risk is not None:
            ins.append((RISK(risk, t), -1.0))
        if T in tbn:
            ins.append((TB(T, t), 1.0))
        nb.unit(L_OUT, ins, Cst - 2 * G + extra_bias, [(OUT(f, t), 1.0)])

    for f in range(64):
        fr, fc = rc(f)
        for dirs, types in [(ORTH, [(R, "R"), (Q, "Q")]), (DIAG, [(B, "M"), (Q, "Q")])]:
            for t, betw in line_pairs(f, dirs):
                between = [x for b in betw for x in occ(b, -G)]
                for T, rname in types:
                    move_unit(f, t, T, rname, between, pst(T, t, True) - pst(T, f, True))
        for t in offsets(f, KNIGHT):
            move_unit(f, t, N, "M", (), pst(N, t, True) - pst(N, f, True))
        for t in offsets(f, KING):
            move_unit(f, t, K, None, (), -0.3)
        for up, gate in [(True, WT), (False, BT)]:
            dr = 1 if up else -1
            last = 7 if up else 0
            start = 1 if up else 6
            t1 = sq(fr + dr, fc)
            if t1 is None or fr == (0 if up else 7):
                continue
            promo = rc(t1)[0] == last
            pb = (8.0 if promo else 0.0) + pst(P, t1, up) - pst(P, f, up)
            move_unit(f, t1, P, "Qpush" if promo else "Ppush", [(gate, G)] + occ(t1, -G), pb - G)
            if fr == start:
                t2 = sq(fr + 2 * dr, fc)
                move_unit(f, t2, P, "Ppush", [(gate, G)] + occ(t1, -G) + occ(t2, -G),
                          pst(P, t2, up) - pst(P, f, up) - G)
            for dc in (-1, 1):
                tc = sq(fr + dr, fc + dc)
                if tc is None:
                    continue
                enemy_here = [(REL(tc, 6 + k), G) for k in range(6)]
                move_unit(f, tc, P, "Q" if promo else "Pcap", [(gate, G)] + enemy_here,
                          (8.0 if promo else 0.0) + pst(P, tc, up) - pst(P, f, up) - 2 * G)
    for gate, ci, f, t, empty in [(WT, 0, 4, 6, [5, 6]), (WT, 1, 4, 2, [1, 2, 3]),
                                  (BT, 2, 60, 62, [61, 62]), (BT, 3, 60, 58, [57, 58, 59])]:
        ins = [(gate, G), (nb.idx("castle", ci), G)] + [x for e in empty for x in occ(e, -G)] + \
              [(TB(K, x), 1.0) for x in (f, (f + t) // 2)]   # TB(K, .) carries -200 * attacked
        move_unit(f, t, K, None, ins, 0.6 - 2 * G)

    return nb.build("OUT")


def oa_neg(oa):
    return [(i, -w) for i, w in oa]
