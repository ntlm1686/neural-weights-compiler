import sys, numpy as np
sys.path.insert(0, '.')
import solution as S
M = S.build_model()
def run(x, y):
    raw = M.run(*S.encode((x, y)))
    return S.decode(raw), np.max(np.abs(raw[12:] - np.round(raw[12:])))
cases = [(999999,999999),(100000,100000),(999999,100001),(111111,999999),(142857,700000),(333334,300000),
         (999999,100000),(500000,200000),(909091,110000),(990001,101010),(999001,1001*999//1 if False else 999001)]
# products near powers of ten -> long carry chains
for x in range(100000, 1000000, 1):
    if 10**11 % x == 0 or 10**12 % x == 0: pass
rng = np.random.RandomState(123)
for _ in range(3000): cases.append((int(rng.randint(100000,1000000)), int(rng.randint(100000,1000000))))
# carry chains: x*y = 10^k +- small, e.g. divisors
for x in range(100000, 1000000):
    for t in (10**11, 10**11+1, 10**11-1, 10**12-1):
        if t % x == 0 and 100000 <= t//x <= 999999: cases.append((x, t//x))
bad, worst = 0, 0
for x, y in cases:
    r, e = run(x, y); worst = max(worst, e)
    if r != x*y: bad += 1; print('FAIL', x, y, r, x*y)
print(len(cases), 'cases', bad, 'fails, max |out - round(out)| =', worst)
