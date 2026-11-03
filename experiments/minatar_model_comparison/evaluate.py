"""Evaluate submitted Breakout policies without giving evaluation feedback to authors."""
import argparse
import ast
import importlib.util
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from minatar_envs import BreakoutEnv, SeaquestEnv, AsterixEnv, breakout_obs, seaquest_obs, asterix_obs
from neural_compiler import ExecutableMLP
from stage1_mlp_benchmark import evaluate_game


class CallableAdapter:
    """Adapt a submitted callable policy to the benchmark's forward interface."""
    def __init__(self, model):
        self.model = model

    def forward(self, x):
        return self.model(x)

    @property
    def nonzero_params(self):
        return sum(np.count_nonzero(x) for x in self.model.weights + self.model.biases)


def load_edits(path):
    text = path.read_text(encoding="utf-8")
    blocks = re.findall(r"```edits\s*\n(.*?)```", text, re.S)
    if not blocks:
        raise ValueError("No edits block")
    weights = {name: np.zeros(shape, dtype=np.float32) for name, shape in (
        ("W1", (600, 32)), ("b1", (32,)), ("W2", (32, 32)),
        ("b2", (32,)), ("W3", (32, 6)), ("b3", (6,)))}
    # The submitted block contains only literal indexed assignments. Reject other syntax.
    for statement in ast.parse(blocks[-1]).body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            raise ValueError("Only literal indexed assignments are accepted")
        target = statement.targets[0]
        if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name) or target.value.id not in weights:
            raise ValueError("Invalid assignment target")
        def index(node):
            if isinstance(node, ast.Tuple):
                return tuple(index(item) for item in node.elts)
            if isinstance(node, ast.Slice):
                return slice(*(ast.literal_eval(part) if part is not None else None
                               for part in (node.lower, node.upper, node.step)))
            return ast.literal_eval(node)
        idx = index(target.slice)
        weights[target.value.id][idx] = ast.literal_eval(statement.value)
    return ExecutableMLP(**weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path)
    parser.add_argument("--game", choices=("breakout", "seaquest", "asterix"), default="breakout")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.policy.suffix in (".txt", ".md"):
        model = load_edits(args.policy)
    else:
        # Some generated programs use the short names from the supplied source snippets.
        import neural_compiler.ir as ir
        import neural_compiler.mlp_backend as backend
        sys.modules.setdefault("ir", ir)
        sys.modules.setdefault("backend", backend)
        spec = importlib.util.spec_from_file_location("submitted_policy", args.policy)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.build_policy()
        if not hasattr(model, "forward") and callable(model):
            model = CallableAdapter(model)
    env_cls, obs_fn = {"breakout": (BreakoutEnv, breakout_obs),
                       "seaquest": (SeaquestEnv, seaquest_obs),
                       "asterix": (AsterixEnv, asterix_obs)}[args.game]
    score, stderr, length = evaluate_game(env_cls, obs_fn, model, args.episodes, args.seed)
    print(f"{args.policy}: score={score:.2f}, se={stderr:.2f}, length={length:.1f}, "
          f"nonzero={model.nonzero_params}, episodes={args.episodes}, seed={args.seed}")


if __name__ == "__main__":
    main()
