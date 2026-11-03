"""Debug script: compare against python-chess on extra random positions and report output margins."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check  # noqa
import solution as sol
m = sol.build_model()
rng = np.random.RandomState(12345)
bad, lo1, hi0, n = 0, 9, -9, int(sys.argv[1]) if len(sys.argv) > 1 else 1000
for _ in range(n):
    ex = check.gen_chess_check(rng)
    out = m.run(*sol.encode(ex))
    a = check.chess_check_answer(ex)
    v = out[sol.SINK]
    assert np.allclose(out, v, atol=1e-6)
    if a: lo1 = min(lo1, v)
    else: hi0 = max(hi0, v)
    bad += (v > 0.5) != a
print("n", n, "wrong", bad, "min positive output", lo1, "max negative output", hi0)
