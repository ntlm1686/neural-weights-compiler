"""Pure-array reference spec of the circuit the network implements (used for design/debug only).
Canonical frame: side to move = S = positive pieces, moving up (rank increasing)."""
import numpy as np
DIRS = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]  # (drank,dfile); 0-3 orth, 4-7 diag
KN = [(1,2),(2,1),(-1,2),(-2,1),(1,-2),(2,-1),(-1,-2),(-2,-1)]
KG = DIRS
P,N,B,R,Q,K = 1,2,3,4,5,6

def canon(ex):
    b = np.array(ex["board"]); t = ex["turn"]
    if t == 1:
        return b.copy(), ex["ep"]
    ep = ex["ep"]
    if ep >= 0:
        ep = (7 - ep // 8) * 8 + ep % 8
    return -b[::-1].copy(), ep

def on(r,f): return 0<=r<8 and 0<=f<8

def fb(b, r, f, d, transparent=None):
    dr, df = d; r += dr; f += df
    while on(r,f):
        if b[r,f] != 0 and (r,f) != transparent:
            return b[r,f]
        r += dr; f += df
    return 0

def ray_squares(r,f,d):
    out=[]; dr,df=d; r+=dr; f+=df
    while on(r,f): out.append((r,f)); r+=dr; f+=df
    return out

def attacked_by(b, r, f, sign, transparent=None, nopk=False):
    """is (r,f) attacked by side `sign` (+1 S, -1 O). S pawns attack up, O pawns down."""
    for dr,df in KN:
        if on(r+dr,f+df) and b[r+dr,f+df] == sign*N: return True
    for dr,df in KG:
        if not nopk and on(r+dr,f+df) and b[r+dr,f+df] == sign*K: return True
    pr = r - sign
    for df in (-1,1):
        if not nopk and on(pr,f+df) and b[pr,f+df] == sign*P: return True
    for i,d in enumerate(DIRS):
        x = fb(b,r,f,d,transparent)
        if x == sign*Q or (i<4 and x == sign*R) or (i>=4 and x == sign*B): return True
    return False

def mate1(ex, verbose=False):
    b, ep = canon(ex)
    kr, kf = map(int, np.argwhere(b == -K)[0])
    found = False
    for r in range(8):
        for f in range(8):
            if b[r,f] > 0: continue
            # candidate (type, checks?) list at t=(r,f)
            cands = []
            for i,d in enumerate(DIRS):
                x = fb(b,r,f,d)
                for ptype in ([R,Q] if i<4 else [B,Q]):
                    if x == ptype: cands.append(ptype)
            for dr,df in KN:
                if on(r+dr,f+df) and b[r+dr,f+df]==N: cands.append(N)
            pawn=[]
            if b[r,f]==0 and on(r-1,f) and b[r-1,f]==P: pawn.append(1)
            if r==3 and b[r,f]==0 and b[r-1,f]==0 and b[r-2,f]==P: pawn.append(1)
            if (b[r,f]<0 or ep==r*8+f):
                for df in (-1,1):
                    if on(r-1,f+df) and b[r-1,f+df]==P: pawn.append(1)
            if pawn:
                cands += ([Q, N] if r==7 else [P])
            for ptype in set(cands):
                if gives_check(b, r, f, ptype, kr, kf) and approx_mate(b, r, f, ptype, kr, kf):
                    if verbose: print("mate", ptype, r, f)
                    found = True
    return found

def piece_attacks(b, r, f, ptype, er, ef, transparent):
    """does a piece of type ptype (S) on (r,f) attack (er,ef), board b (with Ko transparent)"""
    dr, df = er-r, ef-f
    if ptype == N: return (dr,df) in KN
    if ptype == P: return dr == 1 and abs(df) == 1
    if ptype == K: return max(abs(dr),abs(df)) == 1
    for i,d in enumerate(DIRS):
        if (ptype==R and i>=4) or (ptype==B and i<4): continue
        rr, ff = r+d[0], f+d[1]
        while on(rr,ff):
            if (rr,ff)==(er,ef): return True
            if b[rr,ff]!=0 and (rr,ff)!=transparent: break
            rr+=d[0]; ff+=d[1]
    return False

def gives_check(b, r, f, ptype, kr, kf):
    return piece_attacks(b, r, f, ptype, kr, kf, None)

def approx_mate(b, r, f, ptype, kr, kf):
    tr = (kr, kf)
    # escapes
    for dr,df in KG:
        er, ef = kr+dr, kf+df
        if not on(er,ef) or b[er,ef] < 0: continue
        if (er,ef) == (r,f):
            # king captures checker unless defended (by another S piece)
            if not attacked_by(b, er, ef, +1, tr) or _only_mover(b, r, f): return False
            continue
        if attacked_by(b, er, ef, +1, tr) or piece_attacks(b, r, f, ptype, er, ef, tr): continue
        return False
    # capture checker by O non-king piece
    bb = b.copy(); bb[kr,kf] = 0
    if attacked_by(bb, r, f, -1): return False
    # block
    if ptype in (B,R,Q):
        dr = np.sign(kr-r); df = np.sign(kf-f)
        rr, ff = r+dr, f+df
        while (rr,ff) != (kr,kf):
            if can_O_reach(bb, rr, ff): return False
            rr+=dr; ff+=df
    return True

def _only_mover(b, r, f):
    return False  # placeholder (defender count refinement later)

def can_O_reach(b, r, f):
    # non-pawn O pieces attacking empty square (r,f), or O pawn push to it (O pawns move down)
    if attacked_by(b, r, f, -1, nopk=True): return True
    if on(r+1,f) and b[r+1,f] == -P: return True
    if r == 4 and b[r+1,f] == 0 and b[r+2,f] == -P: return True
    return False
