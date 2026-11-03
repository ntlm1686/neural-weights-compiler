"""
Stage 2 Deterministic Weight Backend for Transformer Circuits.
Maps `TransformerProgram` (Neural IR) into explicit Transformer weights:
    W_Q, b_Q, W_K, b_K, W_V, b_V, W_O, b_O, W_fc, b_fc, W_proj, b_proj
across arbitrary target dimensions (d_model, n_layers, n_heads, d_mlp), enabling
zero-retraining Architecture Transfer and compositional algorithm execution.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from .ir import TransformerProgram


class ExecutableTransformer:
    """
    Standard Pre-Norm / Residual Transformer with compiled weight matrices.
    Every layer l = 0 .. L-1 executes:
      1) Multi-Head Self-Attention:
           Q_h = X @ W_Q[l, h] + b_Q[l, h]
           K_h = X @ W_K[l, h] + b_K[l, h]
           V_h = X @ W_V[l, h] + b_V[l, h]
           A_h = softmax(Q_h @ K_h^T / sqrt(d_head) + mask)
           X   = X + sum_h (A_h @ V_h) @ W_O[l, h] + b_O[l]
      2) Position-wise ReLU MLP:
           H   = ReLU(X @ W_fc[l] + b_fc[l])
           X   = X + H @ W_proj[l] + b_proj[l]
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_mlp: int,
        weights: Dict[str, np.ndarray],
        causal_masks: Dict[Tuple[int, int], bool],
        register_slices: Dict[str, slice],
        output_register: str,
        output_mode: str = "register",
    ):
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_mlp = d_mlp
        self.weights = weights
        self.causal_masks = causal_masks
        self.register_slices = register_slices
        self.output_register = output_register
        self.output_mode = output_mode

    def encode_inputs(self, reg_inputs: Dict[str, np.ndarray], seq_len: int) -> np.ndarray:
        """Packs named input arrays into the initial residual stream X of shape (seq_len, d_model)."""
        X = np.zeros((seq_len, self.d_model), dtype=np.float64)
        if "const_one" in self.register_slices:
            X[:, self.register_slices["const_one"]] = 1.0
        for reg_name, val in reg_inputs.items():
            if reg_name not in self.register_slices:
                raise KeyError(f"Unknown input register '{reg_name}'")
            sl = self.register_slices[reg_name]
            arr = np.asarray(val, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[:, None]
            X[:, sl] = arr
        return X

    def forward_residual(self, X: np.ndarray) -> np.ndarray:
        """Executes the exact compiled Transformer forward pass on residual stream X (seq_len, d_model)."""
        seq_len = X.shape[0]
        scale = np.sqrt(float(self.d_head))

        for l in range(self.n_layers):
            attn_delta = np.zeros_like(X)
            for h in range(self.n_heads):
                W_Q = self.weights[f"L{l}.H{h}.W_Q"]
                b_Q = self.weights[f"L{l}.H{h}.b_Q"]
                W_K = self.weights[f"L{l}.H{h}.W_K"]
                b_K = self.weights[f"L{l}.H{h}.b_K"]
                W_V = self.weights[f"L{l}.H{h}.W_V"]
                b_V = self.weights[f"L{l}.H{h}.b_V"]
                W_O = self.weights[f"L{l}.H{h}.W_O"]

                # Skip inactive heads (all zero W_O and W_V)
                if not (np.any(W_O) or np.any(W_V)):
                    continue

                Q = X @ W_Q + b_Q[None, :]
                K = X @ W_K + b_K[None, :]
                V = X @ W_V + b_V[None, :]

                scores = (Q @ K.T) / scale
                if self.causal_masks.get((l, h), False):
                    mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
                    scores = scores + mask

                scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
                weights_attn = np.exp(scores_shifted)
                weights_attn = weights_attn / np.sum(weights_attn, axis=-1, keepdims=True)

                head_out = weights_attn @ V
                attn_delta += head_out @ W_O

            X = X + attn_delta + self.weights[f"L{l}.b_O"][None, :]

            W_fc = self.weights[f"L{l}.W_fc"]
            b_fc = self.weights[f"L{l}.b_fc"]
            W_proj = self.weights[f"L{l}.W_proj"]
            b_proj = self.weights[f"L{l}.b_proj"]

            if np.any(W_proj) or np.any(b_proj):
                hidden = np.maximum(0.0, X @ W_fc + b_fc[None, :])
                X = X + hidden @ W_proj + b_proj[None, :]

        return X

    def run(self, reg_inputs: Dict[str, np.ndarray], seq_len: int) -> np.ndarray:
        X = self.encode_inputs(reg_inputs, seq_len)
        X_out = self.forward_residual(X)
        out_sl = self.register_slices[self.output_register]
        res = X_out[:, out_sl]
        if res.shape[1] == 1:
            res = res[:, 0]
        if self.output_mode == "last_pos_register":
            return res[-1]
        elif self.output_mode == "first_pos_register":
            return res[0]
        return res

    @property
    def nonzero_params(self) -> int:
        return int(sum(np.count_nonzero(v) for v in self.weights.values()))

    @property
    def total_params(self) -> int:
        return int(sum(v.size for v in self.weights.values()))


class TransformerCircuitBackend:
    """
    Compiles a `TransformerProgram` (Neural IR) into a target Transformer architecture
    specified by `(d_model, n_layers, n_heads, d_mlp)`.
    """

    @staticmethod
    def allocate_registers(prog: TransformerProgram, d_model: int) -> Dict[str, slice]:
        slices: Dict[str, slice] = {}
        cursor = 0
        # Always reserve slot 0 for constant 1.0 bias carrier if not explicitly declared
        if "const_one" not in prog.registers:
            slices["const_one"] = slice(0, 1)
            cursor = 1

        for name, width in prog.registers.items():
            if cursor + width > d_model:
                raise ValueError(
                    f"Register allocation overflow: program '{prog.name}' needs > {d_model} residual dims "
                    f"(failed allocating '{name}' of width {width} at offset {cursor})."
                )
            slices[name] = slice(cursor, cursor + width)
            cursor += width
        return slices

    @classmethod
    def compile(
        cls,
        prog: TransformerProgram,
        d_model: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,
        d_mlp: Optional[int] = None,
    ) -> ExecutableTransformer:
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        d_head = d_model // n_heads
        if d_mlp is None:
            d_mlp = 4 * d_model

        reg_slices = cls.allocate_registers(prog, d_model)
        weights: Dict[str, np.ndarray] = {}
        causal_masks: Dict[Tuple[int, int], bool] = {}

        for l in range(n_layers):
            for h in range(n_heads):
                weights[f"L{l}.H{h}.W_Q"] = np.zeros((d_model, d_head), dtype=np.float64)
                weights[f"L{l}.H{h}.b_Q"] = np.zeros((d_head,), dtype=np.float64)
                weights[f"L{l}.H{h}.W_K"] = np.zeros((d_model, d_head), dtype=np.float64)
                weights[f"L{l}.H{h}.b_K"] = np.zeros((d_head,), dtype=np.float64)
                weights[f"L{l}.H{h}.W_V"] = np.zeros((d_model, d_head), dtype=np.float64)
                weights[f"L{l}.H{h}.b_V"] = np.zeros((d_head,), dtype=np.float64)
                weights[f"L{l}.H{h}.W_O"] = np.zeros((d_head, d_model), dtype=np.float64)
            weights[f"L{l}.b_O"] = np.zeros((d_model,), dtype=np.float64)
            weights[f"L{l}.W_fc"] = np.zeros((d_model, d_mlp), dtype=np.float64)
            weights[f"L{l}.b_fc"] = np.zeros((d_mlp,), dtype=np.float64)
            weights[f"L{l}.W_proj"] = np.zeros((d_mlp, d_model), dtype=np.float64)
            weights[f"L{l}.b_proj"] = np.zeros((d_model,), dtype=np.float64)

        mlp_cursors = {l: 0 for l in range(n_layers)}

        for op in prog.ops:
            layer = int(op["layer"])
            if layer >= n_layers:
                raise ValueError(f"Program '{prog.name}' uses layer {layer}, but target n_layers={n_layers}")

            if op["block"] == "attn":
                head = int(op["head"])
                if head >= n_heads:
                    raise ValueError(f"Program '{prog.name}' uses head {head}, but target n_heads={n_heads}")
                cls._compile_attn_op(
                    op=op,
                    weights=weights,
                    causal_masks=causal_masks,
                    reg_slices=reg_slices,
                    d_head=d_head,
                )
            elif op["block"] == "mlp":
                mlp_cursors[layer] = cls._compile_mlp_op(
                    op=op,
                    weights=weights,
                    reg_slices=reg_slices,
                    cursor=mlp_cursors[layer],
                    d_mlp=d_mlp,
                )
            else:
                raise ValueError(f"Unknown block type: {op['block']}")

        return ExecutableTransformer(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_mlp=d_mlp,
            weights=weights,
            causal_masks=causal_masks,
            register_slices=reg_slices,
            output_register=prog.output_register,
            output_mode=prog.output_mode,
        )

    @staticmethod
    def _compile_attn_op(
        op: Dict,
        weights: Dict[str, np.ndarray],
        causal_masks: Dict[Tuple[int, int], bool],
        reg_slices: Dict[str, slice],
        d_head: int,
    ):
        l, h = int(op["layer"]), int(op["head"])
        op_type = op["op_type"]
        scale = float(op.get("scale", 50.0))
        sqrt_dh = np.sqrt(float(d_head))
        causal_masks[(l, h)] = bool(op.get("causal", False))

        W_Q = weights[f"L{l}.H{h}.W_Q"]
        b_Q = weights[f"L{l}.H{h}.b_Q"]
        W_K = weights[f"L{l}.H{h}.W_K"]
        b_K = weights[f"L{l}.H{h}.b_K"]
        W_V = weights[f"L{l}.H{h}.W_V"]
        W_O = weights[f"L{l}.H{h}.W_O"]

        v_sl = reg_slices[op["val_reg"]]
        o_sl = reg_slices[op["out_reg"]]
        v_width = v_sl.stop - v_sl.start
        o_width = o_sl.stop - o_sl.start
        if v_width != o_width:
            raise ValueError(f"val_reg width ({v_width}) != out_reg width ({o_width}) in {op_type}")
        if v_width > d_head:
            raise ValueError(f"Register width ({v_width}) exceeds d_head ({d_head}) in layer {l} head {h}")

        out_scale = float(op.get("params", {}).get("out_scale", 1.0))
        overwrite = bool(op.get("params", {}).get("overwrite", False))

        # Route V -> O through head dimensions 0 .. v_width-1
        for i in range(v_width):
            W_V[v_sl.start + i, i] = 1.0
            W_O[i, o_sl.start + i] = out_scale

        if op_type == "SCORE_MAX_BROADCAST":
            # Every position sets Q_0 = sqrt(d_head) * scale, K_0 = key_reg (scalar score).
            # Softmax concentrates attention weight ~ 1.0 on the maximum key_reg element.
            k_sl = reg_slices[op["key_reg"]]
            b_Q[0] = sqrt_dh * scale
            W_K[k_sl.start, 0] = 1.0

        elif op_type == "UNIFORM_MEAN":
            # Q = 0, K = 0 -> uniform attention across all visible positions (1/N or 1/(i+1) if causal)
            pass

        elif op_type == "DOT_MATCH":
            # Q = query_reg, K = key_reg (vector dot product scaled by temperature `scale`)
            q_sl = reg_slices[op["query_reg"]]
            k_sl = reg_slices[op["key_reg"]]
            q_width = q_sl.stop - q_sl.start
            if q_width > d_head:
                raise ValueError(f"DOT_MATCH width {q_width} exceeds d_head {d_head}")
            for i in range(q_width):
                W_Q[q_sl.start + i, i] = np.sqrt(sqrt_dh * scale)
                W_K[k_sl.start + i, i] = np.sqrt(sqrt_dh * scale)

        elif op_type == "SCALAR_DISTANCE_MATCH":
            # Attention score_ij = -scale * (q_i - k_j)^2 = -scale*q_i^2 + 2*scale*q_i*k_j - scale*k_j^2
            # Notice -scale*q_i^2 is row-constant in softmax!
            # Using 2 head dims: dim 0 has Q = 2*scale*q_i, K = k_j; dim 1 has Q = -scale, K = k_j^2 (if provided)
            # Even simpler for one-hot or unit-norm encodings, or when k_sq_reg is provided:
            q_sl = reg_slices[op["query_reg"]]
            k_sl = reg_slices[op["key_reg"]]
            k_sq_reg = op["params"].get("k_sq_reg")
            if k_sq_reg is None:
                raise ValueError("SCALAR_DISTANCE_MATCH requires params['k_sq_reg'] holding k^2")
            ksq_sl = reg_slices[k_sq_reg]
            W_Q[q_sl.start, 0] = 2.0 * scale * sqrt_dh
            W_K[k_sl.start, 0] = 1.0
            b_Q[1] = -scale * sqrt_dh
            W_K[ksq_sl.start, 1] = 1.0

        elif op_type == "TROPICAL_MIN_PLUS_ATTN":
            # Tropical (min, +) Attention Circuit:
            # Query node i has incoming edge weights `edge_w_reg` of width N: edge_w[i, k] = weight(k -> i)
            # Key node k has `node_oh_reg` of width N (one-hot at k) and scalar `dist_reg` (current distance of k)
            # Sets Q_i = [-edge_w[i, 0..N-1], 1.0] * sqrt(sqrt_dh * scale)
            # Sets K_k = [node_oh[k, 0..N-1], -dist[k]] * sqrt(sqrt_dh * scale)
            # Then Q_i . K_k / sqrt_dh = -scale * (dist[k] + edge_w[i, k])!
            # Softmax selects the winning predecessor k* = argmin_k (dist[k] + edge_w[i, k]).
            # Routes V_k = [node_oh[k] (width N), dist[k] (scalar)] into [win_oh_reg, win_dist_reg]!
            edge_sl = reg_slices[op["query_reg"]]
            oh_sl = reg_slices[op["key_reg"]]
            dist_reg = op["params"]["dist_reg"]
            win_dist_reg = op["params"]["win_dist_reg"]
            dist_sl = reg_slices[dist_reg]
            win_dist_sl = reg_slices[win_dist_reg]
            N = oh_sl.stop - oh_sl.start
            if N + 1 > d_head:
                raise ValueError(f"TROPICAL_MIN_PLUS_ATTN needs N+1={N+1} <= d_head={d_head}")
            factor = np.sqrt(sqrt_dh * scale)
            for k in range(N):
                W_Q[edge_sl.start + k, k] = -factor
                W_K[oh_sl.start + k, k] = factor
            b_Q[N] = factor
            W_K[dist_sl.start, N] = -factor

            # Also route dist[k] through head dimension N into win_dist_reg
            W_V[dist_sl.start, N] = 1.0
            W_O[N, win_dist_sl.start] = 1.0

        elif op_type == "BROADCAST_K_DISTANCE":
            # Broadcasts node k_idx's scalar distance `dist_reg` to all nodes!
            # Key node j has `node_oh[j, k_idx]` (1 at j == k_idx, 0 otherwise).
            # Query has constant `b_Q[0] = sqrt_dh * scale`.
            # Softmax puts weight ~ 1.0 on node `j = k_idx` and copies `V = dist_reg` into `out_reg`!
            oh_sl = reg_slices[op["key_reg"]]
            k_idx = int(op["params"]["k_idx"])
            b_Q[0] = sqrt_dh * scale
            W_K[oh_sl.start + k_idx, 0] = 1.0

        else:
            raise ValueError(f"Unsupported attention op_type: {op_type}")

    @staticmethod
    def _compile_mlp_op(
        op: Dict,
        weights: Dict[str, np.ndarray],
        reg_slices: Dict[str, slice],
        cursor: int,
        d_mlp: int,
    ) -> int:
        l = int(op["layer"])
        op_type = op["op_type"]
        W_fc = weights[f"L{l}.W_fc"]
        b_fc = weights[f"L{l}.b_fc"]
        W_proj = weights[f"L{l}.W_proj"]
        b_proj = weights[f"L{l}.b_proj"]

        out_sl = reg_slices[op["out_reg"]]
        params = op.get("params", {})

        if op_type == "GREATER_THAN":
            # Step indicator: out = ReLU(S*(a - b - threshold) + 1) - ReLU(S*(a - b - threshold))
            # Or for integer/separated inputs: ReLU(S*(a - b - margin)) - ReLU(S*(a - b - margin) - 1)
            # With slope S: equals 0 when a - b <= margin, and 1 when a - b >= margin + 1/S.
            if cursor + 2 > d_mlp:
                raise ValueError(f"MLP capacity exceeded at layer {l}")
            a_sl = reg_slices[op["in_regs"][0]]
            b_sl = reg_slices[op["in_regs"][1]]
            slope = float(params.get("slope", 100.0))
            margin = float(params.get("margin", 0.0))

            # Unit cursor: ReLU(slope * (a - b - margin))
            W_fc[a_sl.start, cursor] = slope
            W_fc[b_sl.start, cursor] = -slope
            b_fc[cursor] = -slope * margin
            W_proj[cursor, out_sl.start] = 1.0

            # Unit cursor+1: -ReLU(slope * (a - b - margin) - 1.0)
            W_fc[a_sl.start, cursor + 1] = slope
            W_fc[b_sl.start, cursor + 1] = -slope
            b_fc[cursor + 1] = -slope * margin - 1.0
            W_proj[cursor + 1, out_sl.start] = -1.0
            return cursor + 2

        elif op_type == "THRESHOLD_STEP":
            # Unary step indicator on a single register: out = clamp_01(slope * (x - threshold))
            if cursor + 2 > d_mlp:
                raise ValueError(f"MLP capacity exceeded at layer {l}")
            x_sl = reg_slices[op["in_regs"][0]]
            width = x_sl.stop - x_sl.start
            if cursor + 2 * width > d_mlp:
                raise ValueError(f"MLP capacity exceeded at layer {l}")
            slope = float(params.get("slope", 20.0))
            threshold = float(params.get("threshold", 0.05))
            clear_input = bool(params.get("overwrite_self", False))

            for i in range(width):
                u0 = cursor + 2 * i
                u1 = cursor + 2 * i + 1
                W_fc[x_sl.start + i, u0] = slope
                b_fc[u0] = -slope * threshold
                W_proj[u0, out_sl.start + i] = 1.0

                W_fc[x_sl.start + i, u1] = slope
                b_fc[u1] = -slope * threshold - 1.0
                W_proj[u1, out_sl.start + i] = -1.0
            return cursor + 2 * width

        elif op_type == "EQUAL_TO_CONST":
            # Indicator I(|x - c| < eps) via hat function:
            # ReLU((x - c + eps)/eps) - 2*ReLU((x - c)/eps) + ReLU((x - c - eps)/eps)
            if cursor + 3 > d_mlp:
                raise ValueError(f"MLP capacity exceeded at layer {l}")
            x_sl = reg_slices[op["in_regs"][0]]
            target = float(params["const"])
            eps = float(params.get("eps", 0.5))
            inv_e = 1.0 / eps
            scale_out = float(params.get("scale_out", 1.0))

            W_fc[x_sl.start, cursor] = inv_e
            b_fc[cursor] = (-target + eps) * inv_e
            W_proj[cursor, out_sl.start] = scale_out

            W_fc[x_sl.start, cursor + 1] = inv_e
            b_fc[cursor + 1] = -target * inv_e
            W_proj[cursor + 1, out_sl.start] = -2.0 * scale_out

            W_fc[x_sl.start, cursor + 2] = inv_e
            b_fc[cursor + 2] = (-target - eps) * inv_e
            W_proj[cursor + 2, out_sl.start] = scale_out
            return cursor + 3

        elif op_type == "EQUAL_PAIR":
            # Indicator I(|a - b| < eps) via hat function on (a - b)
            a_sl = reg_slices[op["in_regs"][0]]
            b_sl = reg_slices[op["in_regs"][1]]
            width = a_sl.stop - a_sl.start
            if cursor + 3 * width > d_mlp:
                raise ValueError(f"MLP capacity exceeded at layer {l}")
            eps = float(params.get("eps", 0.5))
            inv_e = 1.0 / eps
            scale_out = float(params.get("scale_out", 1.0))

            for i in range(width):
                u0, u1, u2 = cursor + 3 * i, cursor + 3 * i + 1, cursor + 3 * i + 2
                for u, shift, coeff in [(u0, eps, 1.0), (u1, 0.0, -2.0), (u2, -eps, 1.0)]:
                    W_fc[a_sl.start + i, u] = inv_e
                    W_fc[b_sl.start + i, u] = -inv_e
                    b_fc[u] = shift * inv_e
                    W_proj[u, out_sl.start] += coeff * scale_out
            return cursor + 3 * width

        elif op_type == "LINEAR_COMBINATION":
            # Exact linear map through ReLU pairs: x = ReLU(x) - ReLU(-x)
            in_regs = op["in_regs"]
            coeffs = params["coeffs"]
            width = out_sl.stop - out_sl.start
            if cursor + 2 * width > d_mlp:
                raise ValueError(f"MLP capacity exceeded at layer {l}")
            bias_val = float(params.get("bias", 0.0))

            for i in range(width):
                u_pos = cursor + 2 * i
                u_neg = cursor + 2 * i + 1
                for reg_name, c in zip(in_regs, coeffs):
                    r_sl = reg_slices[reg_name]
                    idx = r_sl.start + i if (r_sl.stop - r_sl.start) > 1 else r_sl.start
                    W_fc[idx, u_pos] += float(c)
                    W_fc[idx, u_neg] -= float(c)
                b_fc[u_pos] = bias_val
                b_fc[u_neg] = -bias_val
                W_proj[u_pos, out_sl.start + i] = 1.0
                W_proj[u_neg, out_sl.start + i] = -1.0
            return cursor + 2 * width

        elif op_type == "MULTIPLY_BY_ONEHOT":
            # Given scalar register `val_reg` and K-dim binary/one-hot register `oh_reg`,
            # writes `val_reg` into component k of `out_reg` where `oh_reg[k] == 1`
            # using exact gated ReLU: ReLU(val + M*(oh_k - 1)) - ReLU(-val + M*(oh_k - 1))
            val_sl = reg_slices[op["in_regs"][0]]
            oh_sl = reg_slices[op["in_regs"][1]]
            K = oh_sl.stop - oh_sl.start
            if cursor + 2 * K > d_mlp:
                raise ValueError(f"MLP capacity exceeded at layer {l}")
            M = float(params.get("M", 1000.0))
            out_width = out_sl.stop - out_sl.start

            for k in range(K):
                u_pos = cursor + 2 * k
                u_neg = cursor + 2 * k + 1
                W_fc[val_sl.start, u_pos] = 1.0
                W_fc[oh_sl.start + k, u_pos] = M
                b_fc[u_pos] = -M

                W_fc[val_sl.start, u_neg] = -1.0
                W_fc[oh_sl.start + k, u_neg] = M
                b_fc[u_neg] = -M

                target_col = out_sl.start + (k if out_width == K else 0)
                scale_k = float(params["slot_scales"][k]) if "slot_scales" in params else 1.0
                W_proj[u_pos, target_col] += scale_k
                W_proj[u_neg, target_col] -= scale_k
            return cursor + 2 * K

        elif op_type == "COUNT_GREATER_THAN_VECTOR":
            # Compares scalar register `in_regs[0]` against every component k=0..K-1 of vector register `in_regs[1]`
            # and sums the step indicators I(scalar > vec[k] + margin) into scalar `out_reg`!
            x_sl = reg_slices[op["in_regs"][0]]
            vec_sl = reg_slices[op["in_regs"][1]]
            K = vec_sl.stop - vec_sl.start
            if cursor + 2 * K > d_mlp:
                raise ValueError(f"MLP capacity exceeded at layer {l}")
            slope = float(params.get("slope", 200.0))
            margin = float(params.get("margin", 0.0))

            for k in range(K):
                u0 = cursor + 2 * k
                u1 = cursor + 2 * k + 1
                W_fc[x_sl.start, u0] = slope
                W_fc[vec_sl.start + k, u0] = -slope
                b_fc[u0] = -slope * margin
                W_proj[u0, out_sl.start] += 1.0

                W_fc[x_sl.start, u1] = slope
                W_fc[vec_sl.start + k, u1] = -slope
                b_fc[u1] = -slope * margin - 1.0
                W_proj[u1, out_sl.start] -= 1.0
            return cursor + 2 * K

        elif op_type == "SCALAR_TO_ONEHOT":
            # Converts an integer scalar register `in_regs[0]` in {0, ..., K-1} into a K-dim one-hot vector `out_reg`
            # using K hat functions of width eps: I(|x - k| < eps)
            x_sl = reg_slices[op["in_regs"][0]]
            K = out_sl.stop - out_sl.start
            if cursor + 3 * K > d_mlp:
                raise ValueError(f"MLP capacity exceeded at layer {l}")
            eps = float(params.get("eps", 0.45))
            inv_e = 1.0 / eps

            for k in range(K):
                u0, u1, u2 = cursor + 3 * k, cursor + 3 * k + 1, cursor + 3 * k + 2
                for u, shift, coeff in [(u0, eps, 1.0), (u1, 0.0, -2.0), (u2, -eps, 1.0)]:
                    W_fc[x_sl.start, u] = inv_e
                    b_fc[u] = (-float(k) + shift) * inv_e
                    W_proj[u, out_sl.start + k] += coeff
            return cursor + 3 * K

        elif op_type == "TROPICAL_MIN_PLUS_STEP":
            # Given current scalar `dist` (`in_regs[0]`), vector or scalar `all_dist` (`in_regs[1]`),
            # and edge weight row `edge_w` (`in_regs[2]`, length K),
            # relaxes dist <- min(dist, all_dist[k] + edge_w[k]) using 1 ReLU unit:
            #   delta = -ReLU(dist - (all_dist[k] + edge_w[k]))
            d_sl = reg_slices[op["in_regs"][0]]
            all_d_sl = reg_slices[op["in_regs"][1]]
            edge_sl = reg_slices[op["in_regs"][2]]
            k = int(params["neighbor_idx"])
            d_offset = k if (all_d_sl.stop - all_d_sl.start) > 1 else 0
            if cursor + 1 > d_mlp:
                raise ValueError(f"MLP capacity exceeded at layer {l}")
            W_fc[d_sl.start, cursor] = 1.0
            W_fc[all_d_sl.start + d_offset, cursor] = -1.0
            W_fc[edge_sl.start + k, cursor] = -1.0
            W_proj[cursor, out_sl.start] -= 1.0
            return cursor + 1

        elif op_type == "MIN_WITH_CANDIDATE":
            # Updates `dist_reg` with min(dist_reg, cand_reg):
            # Since residual adds to dist_reg, delta = min(d, c) - d = -ReLU(d - c)
            d_sl = reg_slices[op["in_regs"][0]]
            c_sl = reg_slices[op["in_regs"][1]]
            width = d_sl.stop - d_sl.start
            if cursor + width > d_mlp:
                raise ValueError(f"MLP capacity exceeded at layer {l}")
            for i in range(width):
                u = cursor + i
                W_fc[d_sl.start + i, u] = 1.0
                W_fc[c_sl.start + i, u] = -1.0
                W_proj[u, out_sl.start + i] = -1.0
            return cursor + width

        elif op_type == "TROPICAL_READOUT":
            # Given scalar `win_dist` (`in_regs[0]`), vector `edge_w` (`in_regs[1]`, width N),
            # and one-hot `win_oh` (`in_regs[2]`, width N), computes:
            #   out_reg = win_dist + sum_k (ReLU(edge_w[k] + M * (win_oh[k] - 1.0)) - ReLU(M * (win_oh[k] - 1.0)))
            # Look at that identity when win_oh[k*] == 1:
            #   ReLU(edge_w[k*] + 0) - ReLU(0) = edge_w[k*] (with ZERO shift bias!)
            # And when win_oh[k] < 1 - edge_w[k]/M (i.e. win_oh[k] < 0.98 for M=5000, edge_w<=100):
            #   both ReLUs are 0!
            wd_sl = reg_slices[op["in_regs"][0]]
            ew_sl = reg_slices[op["in_regs"][1]]
            oh_sl = reg_slices[op["in_regs"][2]]
            N = ew_sl.stop - ew_sl.start
            if cursor + 2 * N + 1 > d_mlp:
                raise ValueError(f"MLP capacity exceeded at layer {l}")
            M = float(params.get("M", 5000.0))

            W_fc[wd_sl.start, cursor] = 1.0
            W_proj[cursor, out_sl.start] += 1.0

            for k in range(N):
                u0 = cursor + 1 + 2 * k
                u1 = cursor + 1 + 2 * k + 1
                W_fc[ew_sl.start + k, u0] = 1.0
                W_fc[oh_sl.start + k, u0] = M
                b_fc[u0] = -M
                W_proj[u0, out_sl.start] += 1.0

                W_fc[oh_sl.start + k, u1] = M
                b_fc[u1] = -M
                W_proj[u1, out_sl.start] -= 1.0
            return cursor + 2 * N + 1

        else:
            raise ValueError(f"Unsupported MLP op_type: {op_type}")
