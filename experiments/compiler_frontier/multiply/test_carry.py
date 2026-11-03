import sys, numpy as np, copy
sys.path.insert(0, '.')
import solution as S
M = S.build_model(); R = M.register_slices
sub = copy.copy(M); sub.weights = dict(M.weights)
# run only layers 4..6 by shifting: build a model view with layers 4,5,6 renamed to 0,1,2
W = {}
for new, old in enumerate((4, 5, 6)):
    for k, v in M.weights.items():
        if k.startswith(f"L{old}."): W[f"L{new}." + k.split('.', 1)[1]] = v
from neural_compiler import ExecutableTransformer
T = ExecutableTransformer(M.d_model, 3, M.n_heads, M.d_mlp, W, {}, R, "out")
rng = np.random.RandomState(0); bad = 0; n = 0
for _ in range(20000):
    c = rng.choice([9, 9, 9, 10, 10, 0, 5], size=12); c[11] = rng.randint(0, 9) if rng.rand()<.5 else 9
    val = sum(int(c[k]) * 10**k for k in range(12))
    if val >= 10**12: continue
    X = M.encode_inputs({"pos": np.eye(24)}, 24); X[:, R["C"]] = c
    o = T.forward_residual(X)[12:, R["out"].start]
    got = sum(int(round(o[k])) * 10**k for k in range(12)); n += 1
    bad += got != val
print(n, 'vectors', bad, 'fails')
