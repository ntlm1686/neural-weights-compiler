"""
LLM Semantic Compiler: Unseen Compositional Tasks for Stage 2 Transformer Circuits.

Compiles two unseen algorithms directly into `ExecutableTransformer(d_model=128, n_layers=6, n_heads=4)`
weights using ONLY `TransformerProgram` (Neural IR) and `TransformerCircuitBackend.compile(...)`
with zero gradient training:
  1. Task A (`Range_Count_In_Interval`): Count elements x_i in (low, high) for N=8.
  2. Task B (`Thresholded_Max_Distance_From_Source`): Max `node_val` among nodes reachable
     within 3 hops from the source in an N=6 directed graph.
"""

from typing import Dict, Tuple
import numpy as np

from neural_compiler.ir import TransformerProgram
from neural_compiler.transformer_backend import ExecutableTransformer, TransformerCircuitBackend


def build_range_count_program(seq_len: int = 8) -> TransformerProgram:
    """
    Unseen Task A: `Range_Count_In_Interval` (N = 8)
    Given token values `x` and threshold registers `low`, `high` (with low < high),
    outputs at position 0 the exact count of elements satisfying low < x_i < high.

    Circuit design:
      - Layer 0 MLP:
          `gt_low[i]  = GREATER_THAN(x[i], low[i])`   -> I(x_i > low)
          `lt_high[i] = GREATER_THAN(high[i], x[i])`  -> I(x_i < high)
      - Layer 1 MLP:
          `in_interval[i] = LINEAR_COMBINATION(gt_low[i], lt_high[i], coeffs=[1, 1], bias=-1)`
          Since low < high, I(low < x_i < high) = I(x_i > low) + I(x_i < high) - 1 in {0, 1}.
      - Layer 2 Attention (Head 0):
          `out = UNIFORM_MEAN(val_reg="in_interval", out_scale=N)`
          Computes N * (1/N) * sum_i in_interval[i] = exact count at all positions (read out at pos 0).
    """
    prog = TransformerProgram(
        name="Range_Count_In_Interval",
        output_register="out",
        output_mode="first_pos_register",
    )
    prog.declare_register("x", dim=1)
    prog.declare_register("low", dim=1)
    prog.declare_register("high", dim=1)
    prog.declare_register("gt_low", dim=1)
    prog.declare_register("lt_high", dim=1)
    prog.declare_register("in_interval", dim=1)
    prog.declare_register("out", dim=1)

    # Layer 0 MLP: parallel threshold comparisons I(x > low) and I(high > x)
    prog.add_mlp_op(
        layer=0,
        op_type="GREATER_THAN",
        in_regs=["x", "low"],
        out_reg="gt_low",
        params={"slope": 100.0, "margin": 0.0},
    )
    prog.add_mlp_op(
        layer=0,
        op_type="GREATER_THAN",
        in_regs=["high", "x"],
        out_reg="lt_high",
        params={"slope": 100.0, "margin": 0.0},
    )

    # Layer 1 MLP: exact intersection indicator in_interval = gt_low + lt_high - 1
    prog.add_mlp_op(
        layer=1,
        op_type="LINEAR_COMBINATION",
        in_regs=["gt_low", "lt_high"],
        out_reg="in_interval",
        params={"coeffs": [1.0, 1.0], "bias": -1.0},
    )

    # Layer 2 Attention: global uniform sum via UNIFORM_MEAN scaled by N
    prog.add_attention_op(
        layer=2,
        head=0,
        op_type="UNIFORM_MEAN",
        query_reg="in_interval",
        key_reg="in_interval",
        val_reg="in_interval",
        out_reg="out",
        causal=False,
        params={"out_scale": float(seq_len)},
    )

    return prog


def build_thresholded_max_distance_program(num_nodes: int = 6) -> TransformerProgram:
    """
    Unseen Task B: `Thresholded_Max_Distance_From_Source` (N = 6)
    Given an N=6 directed graph (`node_oh`, `in_neighbors`, `reach_0`) and node scores
    `node_val` (v_i in [1, 50] distinct), outputs the maximum `node_val` among all nodes
    reachable from the source within 3 hops.

    Circuit design:
      - Layers 0, 1, 2 (3-hop BFS / Graph Reachability):
          Attn Head 0: `DOT_MATCH(Q=in_neighbors, K=node_oh, V=reach_{t})` -> `att_{t+1}`
          MLP:         `THRESHOLD_STEP(att_{t+1}, threshold=0.05, slope=20.0)` -> `reach_{t+1}`
      - Layer 3 MLP (Mask unreachable nodes):
          `MULTIPLY_BY_ONEHOT(val_reg=node_val, oh_reg=reach_3)` -> `masked_val`
          Sets `masked_val[i] = node_val[i]` if `reach_3[i] == 1` else `0.0`.
      - Layer 4 Attention (Broadcast maximum reachable score):
          `SCORE_MAX_BROADCAST(key_reg=masked_val, val_reg=masked_val)` -> `out`
    """
    prog = TransformerProgram(
        name="Thresholded_Max_Distance_From_Source",
        output_register="out",
        output_mode="first_pos_register",
    )
    prog.declare_register("node_oh", dim=num_nodes)
    prog.declare_register("in_neighbors", dim=num_nodes)
    prog.declare_register("reach_0", dim=1)
    prog.declare_register("node_val", dim=1)

    for hop in range(1, 4):
        prog.declare_register(f"att_{hop}", dim=1)
        prog.declare_register(f"reach_{hop}", dim=1)

    prog.declare_register("masked_val", dim=1)
    prog.declare_register("out", dim=1)

    # Layers 0..2: 3 hops of directed reachability propagation
    for hop in range(3):
        prev_reach = f"reach_{hop}"
        att_reg = f"att_{hop + 1}"
        next_reach = f"reach_{hop + 1}"

        prog.add_attention_op(
            layer=hop,
            head=0,
            op_type="DOT_MATCH",
            query_reg="in_neighbors",
            key_reg="node_oh",
            val_reg=prev_reach,
            out_reg=att_reg,
            scale=50.0,
            causal=False,
        )
        prog.add_mlp_op(
            layer=hop,
            op_type="THRESHOLD_STEP",
            in_regs=[att_reg],
            out_reg=next_reach,
            params={"slope": 20.0, "threshold": 0.05},
        )

    # Layer 3 MLP: gate `node_val` by 3-hop reachability indicator `reach_3`
    prog.add_mlp_op(
        layer=3,
        op_type="MULTIPLY_BY_ONEHOT",
        in_regs=["node_val", "reach_3"],
        out_reg="masked_val",
        params={"M": 1000.0},
    )

    # Layer 4 Attention: broadcast maximum masked_val across all nodes
    prog.add_attention_op(
        layer=4,
        head=0,
        op_type="SCORE_MAX_BROADCAST",
        query_reg="masked_val",
        key_reg="masked_val",
        val_reg="masked_val",
        out_reg="out",
        scale=100.0,
        causal=False,
    )

    return prog


def evaluate_task_a(
    model: ExecutableTransformer,
    n_trials: int = 500,
    seq_len: int = 8,
    seed: int = 42,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    correct = 0
    max_abs_err = 0.0

    for _ in range(n_trials):
        x = rng.integers(-20, 51, size=(seq_len,)).astype(np.float64)
        bounds = rng.choice(np.arange(-20, 51), size=2, replace=False)
        low_val = float(np.min(bounds))
        high_val = float(np.max(bounds))

        low_reg = np.full((seq_len,), low_val, dtype=np.float64)
        high_reg = np.full((seq_len,), high_val, dtype=np.float64)

        target = int(np.sum((x > low_val) & (x < high_val)))
        pred = float(model.run({"x": x, "low": low_reg, "high": high_reg}, seq_len=seq_len))

        err = abs(pred - float(target))
        if err > max_abs_err:
            max_abs_err = err
        if int(np.round(pred)) == target and err < 1e-2:
            correct += 1

    return (100.0 * correct / n_trials), max_abs_err


def evaluate_task_b(
    model: ExecutableTransformer,
    n_trials: int = 500,
    num_nodes: int = 6,
    seed: int = 123,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    correct = 0
    max_abs_err = 0.0
    node_oh = np.eye(num_nodes, dtype=np.float64)

    for _ in range(n_trials):
        # Random directed graph with self-loops on diagonal for reflexive reachability
        adj = (rng.random((num_nodes, num_nodes)) < 0.30).astype(np.float64)
        np.fill_diagonal(adj, 1.0)

        src = int(rng.integers(0, num_nodes))
        reach_0 = np.zeros((num_nodes,), dtype=np.float64)
        reach_0[src] = 1.0

        node_val = rng.choice(np.arange(1, 51), size=(num_nodes,), replace=False).astype(np.float64)

        # Ground-truth 3-hop reachability: adj[i, j] == 1 means edge j -> i
        reach = reach_0.copy()
        for _ in range(3):
            reach = ((adj @ reach) > 0).astype(np.float64)
        target = float(np.max(node_val[reach > 0.5]))

        pred = float(
            model.run(
                {
                    "node_oh": node_oh,
                    "in_neighbors": adj,
                    "reach_0": reach_0,
                    "node_val": node_val,
                },
                seq_len=num_nodes,
            )
        )

        err = abs(pred - target)
        if err > max_abs_err:
            max_abs_err = err
        if int(np.round(pred)) == int(np.round(target)) and err < 1e-2:
            correct += 1

    return (100.0 * correct / n_trials), max_abs_err


def main():
    target_cfg = dict(d_model=128, n_layers=6, n_heads=4)

    prog_a = build_range_count_program(seq_len=8)
    model_a = TransformerCircuitBackend.compile(prog_a, **target_cfg)
    acc_a, err_a = evaluate_task_a(model_a, n_trials=500, seq_len=8)

    prog_b = build_thresholded_max_distance_program(num_nodes=6)
    model_b = TransformerCircuitBackend.compile(prog_b, **target_cfg)
    acc_b, err_b = evaluate_task_b(model_b, n_trials=500, num_nodes=6)

    print("=" * 80)
    print("LLM Semantic Compiler Evaluation on Unseen Compositional Tasks")
    print("Target Architecture: Transformer(d_model=128, n_layers=6, n_heads=4)")
    print("=" * 80)
    print(
        f"Task A ({prog_a.name}):\n"
        f"  - Functional Accuracy (500 random instances): {acc_a:.2f}%\n"
        f"  - Max Absolute Error:                         {err_a:.6e}\n"
        f"  - Nonzero Parameters / Total Parameters:      {model_a.nonzero_params} / {model_a.total_params}"
    )
    print("-" * 80)
    print(
        f"Task B ({prog_b.name}):\n"
        f"  - Functional Accuracy (500 random instances): {acc_b:.2f}%\n"
        f"  - Max Absolute Error:                         {err_b:.6e}\n"
        f"  - Nonzero Parameters / Total Parameters:      {model_b.nonzero_params} / {model_b.total_params}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
