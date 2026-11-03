"""Shared utilities: load compiled policies, a differentiable copy with per-group masks,
teacher data collection, behaviour cloning and evaluation through the repository's evaluator."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from minatar_envs import BreakoutEnv, SeaquestEnv, AsterixEnv, breakout_obs, seaquest_obs, asterix_obs  # noqa: E402
from neural_compiler import ExecutableMLP  # noqa: E402
from stage1_mlp_benchmark import evaluate_game, compile_breakout_mlp, compile_seaquest_mlp, compile_asterix_mlp  # noqa: E402

GAMES = {
    "breakout": (BreakoutEnv, breakout_obs),
    "seaquest": (SeaquestEnv, seaquest_obs),
    "asterix": (AsterixEnv, asterix_obs),
}
TEACHERS = {"breakout": compile_breakout_mlp, "seaquest": compile_seaquest_mlp, "asterix": compile_asterix_mlp}


def load_qwen(game):
    path = HERE / "policies" / f"qwen38_thinking112k_compiler_{game}.py"
    spec = importlib.util.spec_from_file_location(f"qwen_{game}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_policy()


def evaluate(mlp, game, episodes=50, seed=0):
    env_cls, obs_fn = GAMES[game]
    return evaluate_game(env_cls, obs_fn, mlp, episodes, seed)


def collect(policy, game, n_states, seed=1000):
    """Roll out `policy` with the same sampling/sticky-action rules as evaluate_game; record observations."""
    env_cls, obs_fn = GAMES[game]
    rng = np.random.RandomState(seed)
    env = env_cls()
    env.random = np.random.RandomState(seed)
    xs = []
    while len(xs) < n_states:
        env.reset()
        last, steps = 0, 0
        while not env.terminal and steps < 10000 and len(xs) < n_states:
            x = obs_fn(env.state())
            xs.append(x)
            logits = policy.forward(x)
            p = np.exp(logits - logits.max()); p /= p.sum()
            a = rng.choice(6, p=p)
            if env.random.rand() < 0.1:
                a = last
            last = a
            env.act(a)
            steps += 1
    return np.asarray(xs, dtype=np.float32)


class SurrogateTanh(torch.autograd.Function):
    """Exact tanh forward; backward uses sech^2(z / tau) so saturated units still receive gradient."""

    @staticmethod
    def forward(ctx, z, tau):
        ctx.save_for_backward(z, tau)
        return torch.tanh(z)

    @staticmethod
    def backward(ctx, grad):
        z, tau = ctx.saved_tensors
        return grad * (1 - torch.tanh(z / tau) ** 2), None


class MaskedMLP(torch.nn.Module):
    """tanh MLP with per-unit log-gains; gradients are multiplied by fixed 0/1 masks.

    Variants:
      glue   - biases, per-unit gains and the whole output layer (W3, b3)
      sparse - glue + the existing nonzero entries of W1 and W2 (skeleton connectivity fixed)
      holes  - sparse + all weights of unused hidden units (optionally `extra` added units)
      dense  - every parameter, compiled initialisation
      random - every parameter, random initialisation (same width as the holes variant)
    """

    def __init__(self, mlp: ExecutableMLP, variant: str, extra: int = 0, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        W1, b1, W2, b2, W3, b3 = (np.asarray(t, dtype=np.float32) for t in
                                  (mlp.W1, mlp.b1, mlp.W2, mlp.b2, mlp.W3, mlp.b3))
        used1 = (np.abs(W1).sum(0) + np.abs(b1)) > 0
        used2 = (np.abs(W2).sum(0) + np.abs(b2)) > 0
        if extra:
            W1 = np.pad(W1, ((0, 0), (0, extra))); b1 = np.pad(b1, (0, extra))
            W2 = np.pad(W2, ((0, extra), (0, extra))); b2 = np.pad(b2, (0, extra))
            W3 = np.pad(W3, ((0, extra), (0, 0)))
            used1 = np.pad(used1, (0, extra)); used2 = np.pad(used2, (0, extra))
        t = lambda a: torch.tensor(a)
        if variant == "random":
            h1, h2 = W1.shape[1], W2.shape[1]
            W1 = (torch.randn(W1.shape, generator=g) / np.sqrt(600)).numpy()
            W2 = (torch.randn(W2.shape, generator=g) / np.sqrt(h1)).numpy()
            W3 = (torch.randn(W3.shape, generator=g) / np.sqrt(h2)).numpy()
            b1, b2, b3 = np.zeros_like(b1), np.zeros_like(b2), np.zeros_like(b3)
        if variant == "holes":
            # small random init for unused units so they can break symmetry
            free1, free2 = ~used1, ~used2
            W1[:, free1] = (torch.randn((600, free1.sum()), generator=g) * 0.05).numpy()
            W2[:, free2] = (torch.randn((W2.shape[0], free2.sum()), generator=g) * 0.05).numpy()
        self.W1, self.b1 = torch.nn.Parameter(t(W1)), torch.nn.Parameter(t(b1))
        self.W2, self.b2 = torch.nn.Parameter(t(W2)), torch.nn.Parameter(t(b2))
        self.W3, self.b3 = torch.nn.Parameter(t(W3)), torch.nn.Parameter(t(b3))
        self.g1 = torch.nn.Parameter(torch.zeros(W1.shape[1]))
        self.g2 = torch.nn.Parameter(torch.zeros(W2.shape[1]))

        ones = lambda p: torch.ones_like(p)
        zeros = lambda p: torch.zeros_like(p)
        if variant in ("dense", "random"):
            m = {n: ones(p) for n, p in self.named_parameters()}
        else:
            m = {"b1": ones(self.b1), "b2": ones(self.b2), "W3": ones(self.W3), "b3": ones(self.b3),
                 "g1": ones(self.g1), "g2": ones(self.g2), "W1": zeros(self.W1), "W2": zeros(self.W2)}
            if variant in ("sparse", "holes"):
                m["W1"] = (self.W1.detach() != 0).float()
                m["W2"] = (self.W2.detach() != 0).float()
            if variant == "holes":
                f1, f2 = torch.tensor(~used1), torch.tensor(~used2)
                m["W1"][:, f1] = 1.0
                m["W2"][:, f2] = 1.0
                m["W2"][f1, :] = 1.0
        self.masks = m
        self.tau1 = self.tau2 = None
        for n, p in self.named_parameters():
            p.register_hook(lambda grad, n=n: grad * self.masks[n])

    def n_trainable(self):
        return int(sum(m.sum().item() for m in self.masks.values()))

    def preacts(self, x):
        z1 = (x @ self.W1 + self.b1) * self.g1.exp()
        h1 = torch.tanh(z1)
        z2 = (h1 @ self.W2 + self.b2) * self.g2.exp()
        return z1, z2

    def set_surrogate(self, X):
        """Per-unit tau = max(1, median |pre-activation|) on the training states."""
        with torch.no_grad():
            z1, z2 = self.preacts(torch.as_tensor(X))
            self.tau1 = z1.abs().median(0).values.clamp(min=1.0)
            self.tau2 = z2.abs().median(0).values.clamp(min=1.0)

    def _act(self, z, tau):
        return torch.tanh(z) if tau is None else SurrogateTanh.apply(z, tau)

    def forward(self, x):
        h1 = self._act((x @ self.W1 + self.b1) * self.g1.exp(), self.tau1)
        h2 = self._act((h1 @ self.W2 + self.b2) * self.g2.exp(), self.tau2)
        return h2 @ self.W3 + self.b3

    def export(self) -> ExecutableMLP:
        with torch.no_grad():
            s1, s2 = self.g1.exp(), self.g2.exp()
            f = lambda a: a.detach().cpu().numpy().astype(np.float32)
            return ExecutableMLP(W1=f(self.W1 * s1), b1=f(self.b1 * s1), W2=f(self.W2 * s2), b2=f(self.b2 * s2),
                                 W3=f(self.W3), b3=f(self.b3), activation="tanh")


def behaviour_clone(model, X, teacher_logits, steps=1500, lr=3e-3, val=None):
    """Distil teacher action distributions (soft targets) with full-batch Adam."""
    X, T = torch.tensor(X), torch.softmax(torch.tensor(teacher_logits), -1)
    # Relative learning rate: compiled tensors live at very different scales (|b1| ~ 50 vs log-gain ~ 0),
    # so each tensor's step size is scaled by the RMS of its initially nonzero entries.
    groups = []
    for _, p in model.named_parameters():
        nz = p.detach()[p.detach() != 0]
        scale = float(nz.pow(2).mean().sqrt()) if nz.numel() else 0.0
        groups.append({"params": [p], "lr": lr * max(1.0, scale)})
    opt = torch.optim.Adam(groups)
    for _ in range(steps):
        opt.zero_grad()
        loss = -(T * torch.log_softmax(model(X), -1)).sum(-1).mean()
        loss.backward()
        opt.step()
    out = {"train_ce": float(loss.detach())}
    if val is not None:
        Xv, Tv = torch.tensor(val[0]), torch.tensor(val[1])
        with torch.no_grad():
            out["val_agree"] = float((model(Xv).argmax(-1) == Tv.argmax(-1)).float().mean())
    return out
