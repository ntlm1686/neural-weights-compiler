import sys, numpy as np
sys.path.insert(0, '..')
from check import gen_sudoku, sudoku_answer
R = np.repeat(np.arange(9), 9); Cc = np.tile(np.arange(9), 9); B = (R//3)*3 + Cc//3
RS = R*3 + Cc//3; CS = Cc*3 + R//3
M = {k: (U[:, None] == U[None, :]).astype(float) for k, U in [('r', R), ('c', Cc), ('b', B), ('rs', RS), ('cs', CS)]}
def step(A, E, locked=True):
    asg = A.sum(1) > 0
    pc = (M['r'] + M['c'] + M['b']) @ A
    C = np.where(asg[:, None], A, (pc == 0) & (E == 0)).astype(float)
    n = C.sum(1)
    cnt = {k: M[k] @ C for k in M}
    cond = (n[:, None] == 1) | (cnt['r'] == 1) | (cnt['c'] == 1) | (cnt['b'] == 1)
    new = A + (~asg)[:, None] * C * cond
    if locked:
        fbr = C * (cnt['b'] == cnt['rs']); fbc = C * (cnt['b'] == cnt['cs'])
        frb = C * (cnt['r'] == cnt['rs']); fcb = C * (cnt['c'] == cnt['cs'])
        e = (M['r'] - M['rs']) @ fbr + (M['c'] - M['cs']) @ fbc + (M['b'] - M['rs']) @ frb + (M['b'] - M['cs']) @ fcb
        E = np.maximum(E, (e > 0) * (~asg)[:, None])
    contra = ((n == 0).any()) or (cnt['r'] == 0).any() or (cnt['c'] == 0).any() or (cnt['b'] == 0).any() or ((M['r']+M['c']+M['b']) @ A * A > 3).any()
    return np.minimum(new, 1), E, C, contra
def solve(p, phases, R1, R2, locked=True):
    A = np.zeros((81, 9)); g = p.ravel(); A[g > 0, g[g > 0]-1] = 1; E = np.zeros((81, 9))
    for _ in range(R1): A, E, C, _ = step(A, E, locked)
    for ph in range(phases):
        A, E, C, _ = step(A, E, locked)
        asg = A.sum(1) > 0
        biv = np.where((~asg) & (C.sum(1) == 2))[0]
        if len(biv) == 0: continue
        i = biv[0]; a, b = np.where(C[i])[0]
        st = []
        for v in (a, b):
            A2 = A.copy(); A2[i, v] = 1; E2 = E.copy(); bad = False
            for _ in range(R2):
                A2, E2, _, c = step(A2, E2, locked); bad |= c
            st.append((A2, E2, bad))
        if st[0][2] and not st[1][2]: A, E = st[1][0], st[1][1]
        elif st[1][2] and not st[0][2]: A, E = st[0][0], st[0][1]
    return A
D=np.load('/tmp/claude-1000/-home-tiger-weight-space/fb7e2958-01d9-4b15-8172-0c84c7e1c143/scratchpad/puz'+sys.argv[1]+'.npz'); N = 250; ph, R1, R2 = map(int, sys.argv[2:5]); locked = int(sys.argv[5])
ok = 0
for k in range(N):
    p = D['P'][k]; a = D['S'][k]
    A = solve(p, ph, R1, R2, locked)
    ok += (A.sum(1) > 0).all() and np.array_equal(A.argmax(1).reshape(9, 9)+1, a)
print(ok, N)
