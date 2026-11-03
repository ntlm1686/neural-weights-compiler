import sys, numpy as np
sys.path.insert(0, '..')
from check import gen_sudoku, sudoku_answer
R = np.repeat(np.arange(9), 9); Cc = np.tile(np.arange(9), 9); B = (R//3)*3 + Cc//3
RS = R*3 + Cc//3; CS = Cc*3 + R//3
P = [(R == R[i]) | (Cc == Cc[i]) | (B == B[i]) for i in range(81)]
units = [R, Cc, B]
def agg(X, U):  # per cell sum over its unit
    return np.array([X[U == U[i]].sum(0) for i in range(81)])
def run(p, rounds, locked=True):
    A = np.zeros((81, 9)); g = p.ravel()
    A[g > 0, g[g > 0]-1] = 1
    E = np.zeros((81, 9))
    for t in range(rounds):
        asg = A.sum(1) > 0
        pc = agg(A, R) + agg(A, Cc) + agg(A, B)
        C = np.where(asg[:, None], A, (pc == 0) & (E == 0)).astype(float)
        n = C.sum(1)
        cnt = {k: agg(C, U) for k, U in [('r', R), ('c', Cc), ('b', B), ('rs', RS), ('cs', CS)]}
        cond = (n[:, None] == 1) | (cnt['r'] == 1) | (cnt['c'] == 1) | (cnt['b'] == 1)
        new = A + (~asg)[:, None] * C * cond
        if locked:
            fbr = C * (cnt['b'] == cnt['rs']); fbc = C * (cnt['b'] == cnt['cs'])
            frb = C * (cnt['r'] == cnt['rs']); fcb = C * (cnt['c'] == cnt['cs'])
            e = (agg(fbr, R) - agg(fbr, RS)) + (agg(fbc, Cc) - agg(fbc, CS)) + (agg(frb, B) - agg(frb, RS)) + (agg(fcb, B) - agg(fcb, CS))
            E = np.maximum(E, (e > 0) * (~asg)[:, None])
        if np.array_equal(new, A) and not (locked and (e > 0).any() and t < rounds): 
            pass
        A = new
    return A
rng = np.random.RandomState(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
ok = 0; N = 200; rounds = int(sys.argv[2])
for k in range(N):
    p = gen_sudoku(rng); a = sudoku_answer(p)
    A = run(p, rounds)
    pred = A.argmax(1).reshape(9, 9)+1
    ok += (A.sum(1) > 0).all() and np.array_equal(pred, a)
print(ok, N)
