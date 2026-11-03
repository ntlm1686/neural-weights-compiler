"""M4: no teacher. Antithetic evolution strategies on game reward, compiled Qwen skeleton vs random init.

Only parameters selected by the variant's mask are perturbed. Noise for each tensor is scaled by the RMS of its
nonzero initial entries (relative perturbation), matching the relative learning rate used in behaviour cloning.
Environment steps are counted so variants are compared at equal interaction budgets.
"""
import argparse, json, time
from multiprocessing import Pool
import numpy as np, torch
from common import load_qwen, MaskedMLP, GAMES, evaluate, HERE
from neural_compiler import ExecutableMLP

NAMES = ("W1", "b1", "W2", "b2", "W3", "b3", "g1", "g2")


def rollout(args):
    flat, spec, game, seed, episodes = args
    params = unflatten(flat, spec)
    mlp = to_mlp(params)
    env_cls, obs_fn = GAMES[game]
    rng = np.random.RandomState(seed)
    env = env_cls(); env.random = np.random.RandomState(seed)
    total, steps = 0.0, 0
    for _ in range(episodes):
        env.reset(); last, t = 0, 0
        while not env.terminal and t < 10000:
            logits = mlp.forward(obs_fn(env.state()))
            p = np.exp(logits - logits.max()); p /= p.sum()
            a = rng.choice(6, p=p)
            if env.random.rand() < 0.1:
                a = last
            last = a
            r, _ = env.act(a); total += r; t += 1
        steps += t
    return total / episodes, steps


def to_mlp(p):
    s1, s2 = np.exp(p["g1"]), np.exp(p["g2"])
    return ExecutableMLP(W1=p["W1"] * s1, b1=p["b1"] * s1, W2=p["W2"] * s2, b2=p["b2"] * s2, W3=p["W3"], b3=p["b3"])


def unflatten(flat, spec):
    base, idx = spec
    out = {k: v.copy() for k, v in base.items()}
    off = 0
    for k in NAMES:
        n = len(idx[k])
        out[k].reshape(-1)[idx[k]] = flat[off:off + n]
        off += n
    return out


def run(game, variant, gens, pop, sigma, lr, episodes, seed, pool, free_scale=0.1):
    extra = 8 if (game == "seaquest" and variant in ("holes", "random")) else 0
    m = MaskedMLP(load_qwen(game), variant, extra=extra, seed=seed)
    base = {k: getattr(m, k).detach().numpy().astype(np.float64) for k in NAMES}
    idx = {k: np.flatnonzero(m.masks[k].numpy().reshape(-1)) for k in NAMES}
    theta = np.concatenate([base[k].reshape(-1)[idx[k]] for k in NAMES])
    # Per-entry relative scale: |w0| for nonzero compiled entries, fixed `free_scale` for zero entries
    # (hidden "hole" weights, log-gains). Random init perturbs every entry at scale 1 (its weights are O(1/sqrt(n))).
    w0 = np.concatenate([base[k].reshape(-1)[idx[k]] for k in NAMES])
    gmask = np.concatenate([np.full(len(idx[k]), k in ("g1", "g2")) for k in NAMES])
    scale = np.where(w0 != 0, np.abs(w0), free_scale)
    scale[gmask] = 1.0
    if variant == "random":
        scale[:] = 1.0
    spec = (base, idx)
    rng = np.random.RandomState(seed)
    adam_m, adam_v = np.zeros_like(theta), np.zeros_like(theta)
    total_steps, log = 0, []
    for g in range(gens):
        eps = rng.randn(pop // 2, theta.size)
        cands = [theta + sigma * scale * e for e in eps] + [theta - sigma * scale * e for e in eps]
        res = pool.map(rollout, [(c, spec, game, 10_000 + g * pop + i, episodes) for i, c in enumerate(cands)])
        f = np.array([r for r, _ in res]); total_steps += sum(s for _, s in res)
        ranks = np.empty(pop); ranks[f.argsort()] = np.arange(pop); ranks = ranks / (pop - 1) - 0.5
        grad = (ranks[:pop // 2] - ranks[pop // 2:]) @ eps / (pop * sigma)
        adam_m = 0.9 * adam_m + 0.1 * grad; adam_v = 0.999 * adam_v + 0.001 * grad ** 2
        mh, vh = adam_m / (1 - 0.9 ** (g + 1)), adam_v / (1 - 0.999 ** (g + 1))
        theta = theta + lr * scale * mh / (np.sqrt(vh) + 1e-8)
        if g % 25 == 24 or g == gens - 1:
            score, se, _ = evaluate(to_mlp(unflatten(theta, spec)), game, 100 if g == gens - 1 else 30, seed=0)
            rec = dict(game=game, variant=variant, seed=seed, gen=g + 1, env_steps=total_steps, pop_mean=float(f.mean()),
                       eval=score, eval_se=se, n_params=int(theta.size))
            log.append(rec); print(json.dumps(rec), flush=True)
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="breakout,seaquest")
    ap.add_argument("--variants", default="glue,sparse,holes,random")
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--pop", type=int, default=44)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="m4_es.json")
    args = ap.parse_args()
    torch.set_num_threads(1)
    out = []
    with Pool(22) as pool:
        for game in args.games.split(","):
            for v in args.variants.split(","):
                for sd in map(int, args.seeds.split(",")):
                    t0 = time.time()
                    out += run(game, v, args.gens, args.pop, args.sigma, args.lr, args.episodes, sd, pool)
                    print(game, v, sd, "done in", round(time.time() - t0), "s", flush=True)
    (HERE / "results" / args.out).write_text(json.dumps(out, indent=2))
