"""M1: the differentiable copy reproduces the compiled forward pass; measure tanh saturation."""
import json
import numpy as np, torch
from common import load_qwen, TEACHERS, collect, MaskedMLP, HERE

out = {}
for game in ("breakout", "seaquest"):
    for name, mlp in (("qwen", load_qwen(game)), ("argon", TEACHERS[game]())):
        X = collect(mlp, game, 3000, seed=7)
        m = MaskedMLP(mlp, "sparse")
        with torch.no_grad():
            ref = np.stack([mlp.forward(x) for x in X])
            got = m(torch.tensor(X)).numpy()
            exp = np.stack([m.export().forward(x) for x in X[:500]])
            z1, z2 = (z.numpy() for z in m.preacts(torch.tensor(X)))
        used1 = (np.abs(mlp.W1).sum(0) + np.abs(mlp.b1)) > 0
        used2 = (np.abs(mlp.W2).sum(0) + np.abs(mlp.b2)) > 0
        z1, z2 = z1[:, used1], z2[:, used2]
        r = {"max_abs_diff_torch": float(np.abs(ref - got).max()),
             "max_abs_diff_export": float(np.abs(ref[:500] - exp).max()),
             "h1_frac_sat(|z|>3)": float((np.abs(z1) > 3).mean()),
             "h2_frac_sat(|z|>3)": float((np.abs(z2) > 3).mean()),
             "h1_units_never_unsat": int(((np.abs(z1) > 3).all(0)).sum()), "h1_used": int(used1.sum()),
             "h2_units_never_unsat": int(((np.abs(z2) > 3).all(0)).sum()), "h2_used": int(used2.sum()),
             "h1_median|z|": float(np.median(np.abs(z1))), "h2_median|z|": float(np.median(np.abs(z2)))}
        out[f"{game}/{name}"] = r
        print(game, name, json.dumps(r), flush=True)
(HERE / "results" / "m1.json").write_text(json.dumps(out, indent=2))
