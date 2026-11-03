"""
Stage 1 Deterministic Weight Backend for MLP Architectures.
Compiles an `MLPProgram` (Neural IR) into exact floating-point weight tensors
(W1, b1, W2, b2, W3, b3) and provides an executable wrapper.
"""

from typing import Dict, Tuple
import numpy as np
from .ir import MLPProgram


class ExecutableMLP:
    """Directly executable neural network loaded with compiled weights."""

    def __init__(
        self,
        W1: np.ndarray,
        b1: np.ndarray,
        W2: np.ndarray,
        b2: np.ndarray,
        W3: np.ndarray,
        b3: np.ndarray,
        activation: str = "tanh",
        unit_map: Dict = None,
    ):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.W3 = W3
        self.b3 = b3
        self.activation = activation
        self.unit_map = unit_map or {}

    def _act(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return np.tanh(z)
        elif self.activation == "relu":
            return np.maximum(0.0, z)
        raise ValueError(f"Unsupported activation: {self.activation}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        h1 = self._act(x @ self.W1 + self.b1)
        h2 = self._act(h1 @ self.W2 + self.b2)
        logits = h2 @ self.W3 + self.b3
        return logits

    def inspect_activations(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h1 = self._act(x @ self.W1 + self.b1)
        h2 = self._act(h1 @ self.W2 + self.b2)
        logits = h2 @ self.W3 + self.b3
        return h1, h2, logits

    @property
    def nonzero_params(self) -> int:
        return int(sum(np.count_nonzero(t) for t in [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]))

    @property
    def total_params(self) -> int:
        return int(sum(t.size for t in [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]))

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "W3": self.W3,
            "b3": self.b3,
        }


class MLPCompilerBackend:
    """
    Deterministic backend that maps symbolic `MLPProgram` IR into fixed MLP weight tensors.
    Handles register (hidden unit) allocation, gain scaling, and threshold biases.
    """

    @staticmethod
    def compile(prog: MLPProgram) -> ExecutableMLP:
        if len(prog.h1_units) > prog.h1_dim:
            raise ValueError(f"Program uses {len(prog.h1_units)} H1 units, exceeding h1_dim={prog.h1_dim}")
        if len(prog.h2_units) > prog.h2_dim:
            raise ValueError(f"Program uses {len(prog.h2_units)} H2 units, exceeding h2_dim={prog.h2_dim}")

        W1 = np.zeros((prog.input_dim, prog.h1_dim), dtype=np.float32)
        b1 = np.zeros((prog.h1_dim,), dtype=np.float32)
        W2 = np.zeros((prog.h1_dim, prog.h2_dim), dtype=np.float32)
        b2 = np.zeros((prog.h2_dim,), dtype=np.float32)
        W3 = np.zeros((prog.h2_dim, prog.output_dim), dtype=np.float32)
        b3 = np.zeros((prog.output_dim,), dtype=np.float32)

        h1_name_to_idx: Dict[str, int] = {}
        for u_idx, unit in enumerate(prog.h1_units):
            name = unit["name"]
            if name in h1_name_to_idx:
                raise ValueError(f"Duplicate H1 unit name: {name}")
            h1_name_to_idx[name] = u_idx

            utype = unit.get("type", "THRESHOLD")
            if utype == "THRESHOLD":
                gain = float(unit.get("gain", 10.0))
                thresh = float(unit.get("threshold", 0.0))
                for in_idx, w in unit["input_weights"].items():
                    W1[int(in_idx), u_idx] += gain * float(w)
                b1[u_idx] = -gain * thresh
            elif utype == "RAW":
                for in_idx, w in unit["input_weights"].items():
                    W1[int(in_idx), u_idx] += float(w)
                b1[u_idx] = float(unit.get("bias", 0.0))
            else:
                raise ValueError(f"Unknown H1 unit type: {utype}")

        h2_name_to_idx: Dict[str, int] = {}
        for v_idx, unit in enumerate(prog.h2_units):
            name = unit["name"]
            if name in h2_name_to_idx:
                raise ValueError(f"Duplicate H2 unit name: {name}")
            h2_name_to_idx[name] = v_idx

            for h1_name, w in unit["h1_terms"].items():
                if h1_name not in h1_name_to_idx:
                    raise KeyError(f"H2 unit '{name}' references unknown H1 unit '{h1_name}'")
                W2[h1_name_to_idx[h1_name], v_idx] += float(w)
            b2[v_idx] = float(unit.get("bias", 0.0))

        if prog.output_bias is not None:
            b3[:] = np.array(prog.output_bias, dtype=np.float32)

        for rule in prog.output_rules:
            act_idx = int(rule["action_idx"])
            for h2_name, w in rule["h2_terms"].items():
                if h2_name not in h2_name_to_idx:
                    raise KeyError(f"Output rule for action {act_idx} references unknown H2 unit '{h2_name}'")
                W3[h2_name_to_idx[h2_name], act_idx] += float(w)
            if "bias" in rule:
                b3[act_idx] = float(rule["bias"])

        return ExecutableMLP(
            W1=W1,
            b1=b1,
            W2=W2,
            b2=b2,
            W3=W3,
            b3=b3,
            activation=prog.activation,
            unit_map={"h1": h1_name_to_idx, "h2": h2_name_to_idx},
        )
