"""Reference ceilings for MinAtar Asterix: Monte-Carlo planners with full simulator access.

fair   : clones the full env state (including timers the network cannot see) but re-seeds the clone's RNG,
         so future spawns are unknown; value = mean return of random rollouts.
oracle : clones the RNG too, so it sees the actual future spawns; value = best rollout.
Both mirror the evaluation protocol: 10% sticky actions drawn from env.random, 10,000-step cap.
"""
import copy, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from minatar_envs import AsterixEnv

ACTIONS = [0, 1, 2, 3, 4]
H, N, DEATH = 10, 16, 50.0


def rollout(env, a0, last, rng, fair):
    e = copy.deepcopy(env)
    if fair:
        e.random = np.random.RandomState(rng.randint(2 ** 31))
    total, a = 0.0, a0
    for t in range(H):
        if t > 0:
            a = ACTIONS[rng.randint(len(ACTIONS))]
        if e.random.rand() < 0.1:
            a = last
        r, term = e.act(a)
        total += r
        last = a
        if term:
            return total - DEATH * (H - t) / H - DEATH
    return total


def plan(env, last, rng, fair):
    best, best_v = 0, -1e9
    for a in ACTIONS:
        vals = [rollout(env, a, last, rng, fair) for _ in range(N)]
        v = np.mean(vals) if fair else np.max(vals)
        if v > best_v:
            best, best_v = a, v
    return best


def episode(args):
    mode, seed = args
    env = AsterixEnv()
    env.random = np.random.RandomState(seed)
    env.reset()
    rng = np.random.RandomState(10_000 + seed)
    last, ret, steps = 0, 0, 0
    while not env.terminal and steps < 10000:
        a = plan(env, last, rng, mode == "fair")
        if env.random.rand() < 0.1:
            a = last
        last = a
        r, _ = env.act(a)
        ret += r
        steps += 1
    return mode, seed, ret, steps


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    jobs = [(m, s) for m in ("fair", "oracle") for s in range(n)]
    t0 = time.time()
    with Pool(22) as p:
        res = p.map(episode, jobs, chunksize=1)
    out = {}
    for m in ("fair", "oracle"):
        r = np.array([x[2] for x in res if x[0] == m]); L = np.array([x[3] for x in res if x[0] == m])
        out[m] = dict(mean=round(float(r.mean()), 2), se=round(float(r.std() / np.sqrt(len(r))), 2),
                      min=int(r.min()), max=int(r.max()), mean_len=round(float(L.mean()), 1),
                      capped=int((L >= 10000).sum()), episodes=len(r))
        print(m, json.dumps(out[m]), flush=True)
    print("secs", round(time.time() - t0))
    Path(__file__).with_name("asterix_ceiling.json").write_text(json.dumps(out, indent=2))
