"""M2/M3: behaviour cloning from the Argon teacher into the Qwen skeleton under different trainable groups."""
import argparse, json, time
from multiprocessing import Pool
import numpy as np, torch
from common import load_qwen, TEACHERS, collect, MaskedMLP, behaviour_clone, evaluate, HERE

SIZES = (1000, 5000, 20000)
VARIANTS = [("glue", "exact"), ("glue", "surrogate"), ("sparse", "exact"), ("sparse", "surrogate"),
            ("holes", "exact"), ("holes", "surrogate"), ("dense", "surrogate"), ("random", "exact")]
DATA = {}


def job(cfg):
    torch.set_num_threads(1)
    game, n, variant, grad, episodes = cfg
    X, T, Xv, Tv = DATA[game]
    extra = 8 if (game == "seaquest" and variant in ("holes", "random")) else 0
    t0 = time.time()
    model = MaskedMLP(load_qwen(game), variant, extra=extra, seed=0)
    if grad == "surrogate":
        model.set_surrogate(X[:n])
    fit = behaviour_clone(model, X[:n], T[:n], val=(Xv, Tv))
    score, se, length = evaluate(model.export(), game, episodes)
    r = dict(game=game, n=n, variant=variant, grad=grad, extra=extra, trainable=model.n_trainable(),
             score=score, se=se, length=length, secs=round(time.time() - t0), **fit)
    print(json.dumps(r), flush=True)
    return r


def ref(cfg):
    game, who, episodes = cfg
    mlp = load_qwen(game) if who == "qwen" else TEACHERS[game]()
    X, T, Xv, Tv = DATA[game]
    agree = float((mlp.forward(Xv).argmax(-1) == Tv.argmax(-1)).mean())
    score, se, length = evaluate(mlp, game, episodes)
    r = dict(game=game, variant=who + "_untrained", score=score, se=se, length=length, val_agree=agree)
    print(json.dumps(r), flush=True)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--games", default="breakout,seaquest"); args = ap.parse_args()
    games = args.games.split(",")
    for g in games:
        teacher = TEACHERS[g]()
        X = collect(teacher, g, max(SIZES), seed=1000)
        Xv = collect(teacher, g, 2000, seed=2000)
        DATA[g] = (X, teacher.forward(X).astype(np.float32), Xv, teacher.forward(Xv).astype(np.float32))
        print(g, "collected", X.shape, flush=True)
    cfgs = [(g, n, v, gr, args.episodes) for g in games for n in SIZES for v, gr in VARIANTS]
    with Pool(22) as pool:
        refs = pool.map(ref, [(g, w, args.episodes) for g in games for w in ("qwen", "argon")])
        res = pool.map(job, cfgs, chunksize=1)
    (HERE / "results" / "m23_bc.json").write_text(json.dumps(refs + res, indent=2))
