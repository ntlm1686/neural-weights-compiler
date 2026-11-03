import sys, numpy as np
sys.path.insert(0, '..')
from check import gen_sudoku, sudoku_answer
R = np.repeat(np.arange(9), 9); Cc = np.tile(np.arange(9), 9); B = (R//3)*3 + Cc//3
P = [(R == R[i]) | (Cc == Cc[i]) | (B == B[i]) for i in range(81)]
units = [R, Cc, B]
def run(p, rounds, locked=False):
    A = np.zeros((81, 9)); g = p.ravel()
    A[g > 0, g[g > 0]-1] = 1
    for t in range(rounds):
        asg = A.sum(1) > 0
        C = A.copy()
        for i in range(81):
            if not asg[i]:
                pc = A[P[i]].sum(0)
                C[i] = (pc == 0)
        n = C.sum(1)
        new = A.copy()
        for i in range(81):
            if asg[i]: continue
            cond = (n[i] == 1) | np.any([C[U == U[i]].sum(0) == 1 for U in units], axis=0)
            new[i] += C[i] * cond
        if np.array_equal(new, A): return A, t
        A = new
    return A, rounds
rng = np.random.RandomState(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
ok = 0; rs = []
N = 200
for k in range(N):
    p = gen_sudoku(rng); a = sudoku_answer(p)
    A, t = run(p, 100)
    pred = A.argmax(1).reshape(9, 9)+1
    good = (A.sum(1) > 0).all() and np.array_equal(pred, a)
    ok += good
    if good: rs.append(t)
print(ok, N, max(rs), np.percentile(rs, [50, 90, 99]))
