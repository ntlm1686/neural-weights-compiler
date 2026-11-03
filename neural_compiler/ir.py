"""
Intermediate Representation (Neural IR) for Stage 1 (MLP) and Stage 2 (Transformer Circuits).
Provides a clean declarative DSL (and JSON-serializable spec) that decouples semantic algorithm
design from target architecture dimensions and raw floating-point weight matrices.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class MLPOp:
    """Single symbolic circuit operation in an MLP layer."""
    op_type: str  # "LINEAR_THRESHOLD", "AND", "OR", "MUX", "LINEAR_COMBINATION", "LOOKUP_1D"
    name: str
    layer: int  # 1 or 2
    params: Dict


@dataclass
class MLPProgram:
    """
    Stage 1 Neural IR for fixed-architecture MLPs (e.g., 600 -> H1 -> H2 -> Out).
    Instead of manually indexing raw weight rows/columns, the LLM specifies named symbolic
    signals in Layer 1, logical/routing gates in Layer 2, and action scoring rules in Output.
    """
    input_dim: int
    h1_dim: int
    h2_dim: int
    output_dim: int
    activation: str = "tanh"
    h1_units: List[Dict] = field(default_factory=list)
    h2_units: List[Dict] = field(default_factory=list)
    output_rules: List[Dict] = field(default_factory=list)
    output_bias: Optional[List[float]] = None

    def add_spatial_coord_extractor(
        self,
        name: str,
        grid_shape: Tuple[int, int, int],
        channel: int,
        col_weights: List[float],
        row_filter: Optional[List[int]] = None,
    ) -> Dict[int, float]:
        """Helper returning an input_index -> weight map for a grid channel."""
        rows, cols, chs = grid_shape
        mapping: Dict[int, float] = {}
        active_rows = row_filter if row_filter is not None else list(range(rows))
        for r in active_rows:
            for c in range(cols):
                idx = (r * cols + c) * chs + channel
                if col_weights[c] != 0:
                    mapping[idx] = mapping.get(idx, 0.0) + float(col_weights[c])
        return mapping

    def add_h1_threshold(
        self,
        name: str,
        input_weights: Dict[int, float],
        threshold: float = 0.0,
        gain: float = 10.0,
    ) -> str:
        """
        Adds a Layer-1 comparator unit:
            h1[u] = tanh(gain * (sum_i w_i x_i - threshold))
        Outputs ~ +1 when sum_i w_i x_i > threshold, and ~ -1 when < threshold.
        """
        self.h1_units.append({
            "name": name,
            "type": "THRESHOLD",
            "input_weights": {int(k): float(v) for k, v in input_weights.items()},
            "threshold": float(threshold),
            "gain": float(gain),
        })
        return name

    def add_h1_and_gate(
        self,
        name: str,
        group_a_indices: List[int],
        group_b_indices: List[int],
        gain: float = 20.0,
    ) -> str:
        """
        Adds a Layer-1 logical AND detector over two one-hot/binary indicator groups:
            h1[u] = tanh(gain * I(group_a) + gain * I(group_b) - 1.5 * gain)
        Outputs +1 iff both group_a and group_b have an active input.
        """
        iw: Dict[int, float] = {}
        for idx in group_a_indices:
            iw[int(idx)] = iw.get(int(idx), 0.0) + float(gain)
        for idx in group_b_indices:
            iw[int(idx)] = iw.get(int(idx), 0.0) + float(gain)
        self.h1_units.append({
            "name": name,
            "type": "RAW",
            "input_weights": iw,
            "bias": -1.5 * float(gain),
        })
        return name

    def add_h2_gate(
        self,
        name: str,
        h1_terms: Dict[str, float],
        bias: float = 0.0,
    ) -> str:
        """
        Adds a Layer-2 unit taking named Layer-1 signals:
            h2[v] = tanh(sum_u w_u * h1[u] + bias)
        """
        self.h2_units.append({
            "name": name,
            "h1_terms": dict(h1_terms),
            "bias": float(bias),
        })
        return name

    def add_h2_muxPass(
        self,
        name: str,
        signal_h1: str,
        gate_h1: str,
        active_when_gate_positive: bool = True,
        gain: float = 15.0,
    ) -> str:
        """
        Passes `signal_h1` in {-1, +1} when `gate_h1` matches `active_when_gate_positive`,
        otherwise clamps output to -1.
        """
        gate_w = float(gain) if active_when_gate_positive else -float(gain)
        return self.add_h2_gate(
            name=name,
            h1_terms={signal_h1: float(gain), gate_h1: gate_w},
            bias=-float(gain),
        )

    def set_output_action(
        self,
        action_idx: int,
        h2_terms: Dict[str, float],
        bias: float = 0.0,
    ):
        self.output_rules.append({
            "action_idx": int(action_idx),
            "h2_terms": dict(h2_terms),
            "bias": float(bias),
        })


@dataclass
class TransformerProgram:
    """
    Stage 2 Neural IR for compiling symbolic algorithms into fixed Transformer architectures:
        Transformer(d_model, n_layers, n_heads, d_mlp, max_seq_len)
    Supports automatic Residual Stream Subspace Allocation (named registers mapped to orthogonal
    slices of the residual stream) so the SAME Neural IR program compiles to ANY sufficiently wide
    Transformer architecture (Architecture Transfer).
    """
    name: str
    registers: Dict[str, int] = field(default_factory=dict)  # register_name -> subspace width
    ops: List[Dict] = field(default_factory=list)
    output_register: str = "out"
    output_mode: str = "register"  # "register", "last_pos_register", "scalar_register"

    def declare_register(self, name: str, dim: int = 1):
        """Allocates a named register of width `dim` in the residual stream."""
        self.registers[name] = int(dim)

    def add_attention_op(
        self,
        layer: int,
        head: int,
        op_type: str,
        query_reg: str,
        key_reg: str,
        val_reg: str,
        out_reg: str,
        scale: float = 50.0,
        causal: bool = False,
        params: Optional[Dict] = None,
    ):
        """
        High-level Transformer Attention Circuit primitives:
          - "SCORE_MAX_BROADCAST": Q=1, K=score_reg, V=val_reg -> broadcasts V of max-score token to all positions
          - "MATCH_SUM" / "COUNT_MATCH": Q=query_reg, K=key_reg (dot-product or equality), V=val_reg
          - "UNIFORM_MEAN": Q=0, K=0, V=val_reg -> computes prefix mean (if causal=True) or global mean (if causal=False)
          - "ADJACENCY_PROPAGATE": Q=reach_reg, K=edge_reg, V=reach_reg -> 1-step graph reachability / BFS relaxation
          - "MIN_PLUS_RELAX": Tropical / shortest-path step via softmin attention
          - "RANK_COMPARE": For sorting / ranking via pairwise comparison attention
        """
        self.ops.append({
            "block": "attn",
            "layer": int(layer),
            "head": int(head),
            "op_type": op_type,
            "query_reg": query_reg,
            "key_reg": key_reg,
            "val_reg": val_reg,
            "out_reg": out_reg,
            "scale": float(scale),
            "causal": bool(causal),
            "params": params or {},
        })

    def add_mlp_op(
        self,
        layer: int,
        op_type: str,
        in_regs: List[str],
        out_reg: str,
        params: Optional[Dict] = None,
    ):
        """
        High-level Transformer ReLU-MLP Circuit primitives:
          - "GREATER_THAN": out_reg = step(in_regs[0] - in_regs[1] - margin)
          - "EQUAL_INDICATOR": out_reg = triangle_hat(in_regs[0] - in_regs[1], eps)
          - "RELU_LINEAR": out_reg = W2 * ReLU(W1 * in + b1) + b2
          - "CLAMP_01": out_reg = ReLU(x) - ReLU(x - 1)  (saturates positive values to 1.0)
          - "SELECT_BY_INDEX": routes vector slot matching target index/rank
          - "PAIRWISE_MIN_MAX": computes min(a, b) or max(a, b) using ReLU identities:
                                max(a, b) = a + ReLU(b - a), min(a, b) = a - ReLU(a - b)
        """
        self.ops.append({
            "block": "mlp",
            "layer": int(layer),
            "op_type": op_type,
            "in_regs": list(in_regs),
            "out_reg": out_reg,
            "params": params or {},
        })

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "registers": self.registers,
            "ops": self.ops,
            "output_register": self.output_register,
            "output_mode": self.output_mode,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "TransformerProgram":
        return cls(
            name=d["name"],
            registers=dict(d["registers"]),
            ops=list(d["ops"]),
            output_register=d.get("output_register", "out"),
            output_mode=d.get("output_mode", "register"),
        )
