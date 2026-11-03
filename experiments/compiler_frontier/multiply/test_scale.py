import sys, numpy as np, time
sys.path.insert(0, '.')
import solution as S
for nd in (6, 10, 20):
    M = S.build_model(nd); rng = np.random.RandomState(nd); bad = 0; worst = 0
    cases = [(10**nd - 1, 10**nd - 1), (10**(nd-1), 10**(nd-1))] + \
            [(int(''.join(map(str, rng.randint(1, 10, nd)))), int(''.join(map(str, rng.randint(0, 10, nd))))) for _ in range(300)]
    t = time.time()
    for x, y in cases:
        raw = M.run(*S.encode((x, y), nd)); worst = max(worst, np.abs(raw[2*nd:] - np.round(raw[2*nd:])).max())
        bad += S.decode(raw, nd) != x * y
    print(f"nd={nd}: d_model={M.d_model} layers={M.n_layers} heads={M.n_heads} d_mlp={M.d_mlp} nonzero={M.nonzero_params} "
          f"fails={bad}/{len(cases)} max_dev={worst:.2e} ({time.time()-t:.1f}s)")
