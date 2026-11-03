"""MinAtar Seaquest policy built by construction (no training).  See NOTES.md for the design.

Actions: 0 n, 1 l, 2 u, 3 r, 4 d, 5 f.
Obs channels: 0 player sub (front|back), 1 enemy|enemy-bullet, 2 enemy|diver, 3 friendly bullet,
              4 trail|player-back, 5 oxygen/diver gauges (row 9).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from primitives import Policy  # noqa: E402

N, L, U, R, D, F = range(6)
# Column zones of the player's row (the sub always occupies cols 4-5: x=4 facing left or x=5 facing right).
LEFT, RIGHT = range(0, 5), range(5, 10)          # aim zones
IMM_L, IMM_R = range(3, 5), range(5, 7)          # immediate: dangerous whatever I face (bullets, unshot enemies)
NEAR = range(3, 7)                               # cols checked in rows above / below before a vertical move


def build_policy():
    p = Policy()
    # ------------------------------------------------ layer 1 (32 units; pair = 2 units)
    WL = p.row_pair("wl", 1, LEFT)             # enemy/bullet in my row, left  (pair: "and none in lower rows")
    WR = p.row_pair("wr", 1, RIGHT)
    ILd = p.row_pair("ild", 1, IMM_L)
    ILu = p.row_pair("ilu", 1, IMM_L, direction="up")   # same, "and none in higher rows"
    IRd = p.row_pair("ird", 1, IMM_R)
    IRu = p.row_pair("iru", 1, IMM_R, direction="up")
    # facing left <=> my back is at col 5 <=> ch4 at (y,5)  (pair: misses if a trail sits in col 5 in a lower row)
    FD = p.row_pair("face_d", 4, [5], rows=range(0, 9))
    # vertical safety: unit _a = danger near my column in ANY row above/below (exact, conservative).
    # Window "danger in the next two rows" = union of two pairs from opposite families:
    #   DU1: rows y-2..y-1 and none in rows <= y-3      DU2: rows y-2..y-1 and none in rows >= y
    # it only misses when danger lies both beyond the window and on my side of it.  Row 8 = "danger below".
    DU1 = p.row_pair("du1", 1, NEAR, k=1, k2=3, direction="up")
    DU2 = p.row_pair("du2", 1, NEAR, k=-2, k2=0, direction="down")
    DD1 = p.row_pair("dd1", 1, NEAR, k=1, k2=3, direction="down", force_true=(8,))
    DD2 = p.row_pair("dd2", 1, NEAR, k=-2, k2=0, direction="up")
    DUc, DDc = DU1[1], DD1[1]
    DU = [(1, DU1), (1, DU2)]
    DD = [(1, DD1), (1, DD2)]
    B2 = p.rel_rows("t_below", 2, range(10), 1, "down")                  # enemy/diver in rows > y
    A2 = p.rel_rows("t_above", 2, range(10), 1, "up", rows=range(2, 9))  # ... in rows 2..y-1 (never seek row 1)
    # divers (ch2 - ch1: diver +1, enemy 0, enemy bullet -1), signed lexicographic digits, base 5
    DB = p.rel_rows("d_below", 2, range(10), 1, "down", neg_ch=1, base=5)
    DA = p.rel_rows("d_above", 2, range(10), 1, "up", rows=range(2, 9), neg_ch=1, base=5)
    # turning moves me one cell toward that side (two with a sticky repeat): only turn when nothing is at distance 2
    # (cols 2 / 7) in my row.  Conservative single unit: anything at cols 2/7 in my row OR any lower row.
    CZ = p.rel_rows("col27_ge0", 1, [2, 7], 0, "down")
    OL = p.none_cells("oxy_low", 5, [(9, 2)])               # oxygen < 60
    HD = p.any_cells("have_diver", 5, [(9, 8)])             # >= 1 diver, or oxygen >= 180
    R1 = p.any_cells("row1", 0, [(1, c) for c in range(10)])
    X3 = p.any_cells("col3", 0, [(r, 3) for r in range(10)])  # drifted left
    X6 = p.any_cells("col6", 0, [(r, 6) for r in range(10)])  # drifted right

    # ------------------------------------------------ layer 2 + priorities (0/1 literal arithmetic)
    S = [(1, FD)]
    neg = lambda terms: [(-a, l) for a, l in terms]
    mul = lambda terms, c: [(c * a, l) for a, l in terms]
    TL = [(1, ILd), (1, ILu)]                  # immediate threat (bullet or enemy about to reach my front)
    TR = [(1, IRd), (1, IRu)]
    p.rule(p.gate("surface", [(1, OL), (1, HD), (-2, DUc)], 1.5), U, 11)
    p.rule(p.gate("surface_window", [(1, OL), (1, HD)] + mul(DU, -2), 1.5), U, 10)
    for lvl, act, tag, block in [(9, U, "up_c", [(1, DUc)]), (8, D, "dn_c", [(1, DDc)]), (7, U, "up_w", DU),
                                 (6, D, "dn_w", DD)]:
        edge = [(-3, R1)] if act == U else []
        p.rule(p.gate(f"dodge_{tag}_l", TL + mul(block, -3) + edge, 0.5), act, lvl)
        p.rule(p.gate(f"dodge_{tag}_r", TR + mul(block, -3) + edge, 0.5), act, lvl)
    # both vertical escapes blocked: run horizontally away from a one-sided threat (I outrun enemies)
    p.rule(p.gate("flee_r", TL + mul(TR, -2), 0.5), R, 4)
    p.rule(p.gate("flee_l", TR + mul(TL, -2), 0.5), L, 4)
    p.rule(p.gate("center_r", [(1, X3)], 0.5), R, 3)
    p.rule(p.gate("center_l", [(1, X6)], 0.5), L, 3)
    p.rule(p.gate("fire_l", [(2, WL)] + S, 2.5), F, 2)
    p.rule(p.gate("fire_r", [(2, WR)] + neg(S), 1.5), F, 2)
    p.rule(p.gate("turn_l", [(2, WL), (-2, WR), (-2, CZ)] + neg(S), 1.5), L, 2)
    p.rule(p.gate("turn_r", [(2, WR), (-3, WL), (-3, CZ)] + S, 2.5), R, 2)
    # need a diver: none visible and oxygen < 180
    p.rule(p.gate("div_dn", [(1, DB), (-1, HD)] + mul(DD, -2), 0.5), D, 1)
    p.rule(p.gate("div_up", [(1, DA), (-1, HD), (-1, DB)] + mul(DU, -2), 0.5), U, 1)
    p.rule(p.gate("seek_dn", [(2, B2), (-4, DDc), (-1, OL), (-1, HD)], 0.5), D, 0)   # not while I want to surface
    p.rule(p.gate("seek_up", [(1, A2), (-1, B2), (-2, DUc)], 0.5), U, 0)
    p.default_action = F
    return p.compile()


if __name__ == "__main__":
    m = build_policy()
    print("H1 used", sum(abs(m.W1).sum(0) > 0), "H2 used", sum(abs(m.W2).sum(0) > 0))
