"""
Stage 2 Benchmark: Compiling Symbolic Algorithms into Executable Transformer Weights
=====================================================================================
Benchmarks all 7 tasks from the NeuralCompiler roadmap plus 1 compositional task:
  1. Comparison         : x > y                                      [Easy]
  2. Maximum & Argmax   : max(x_1..x_n) and argmax(x_1..x_n)         [Easy]
  3. Counting           : Count tokens matching target query         [Medium]
  4. Prefix Sum         : Cumulative sum s_i = sum_{j<=i} x_j        [Medium]
  5. Sorting            : Sort sequence of N numbers                 [Medium]
  6. Graph Reachability : Multi-hop directed graph reachability      [Hard]
  7. Shortest Path      : Bounded weighted graph shortest path       [Hard]
  8. Compositional (Program Generalization):
       "Count how many elements are strictly greater than the prefix mean"
       (Composes Prefix Mean + Comparison + Counting with zero retraining)

Also tests Architecture Transfer across:
  - Config A: Transformer(d_model=64,  n_layers=6, n_heads=4)
  - Config B: Transformer(d_model=128, n_layers=8, n_heads=8)
  - Config C: Transformer(d_model=256, n_layers=8, n_heads=8)
The shortest-path circuit uses 20 layers at each width, not the depths above.
"""

import argparse
import json
import time
from typing import Dict, List, Tuple
import numpy as np
from neural_compiler import TransformerProgram, TransformerCircuitBackend


# ==============================================================================
# 1. Comparison: I(x > y)
# ==============================================================================
def build_comparison_ir() -> TransformerProgram:
    prog = TransformerProgram(name="Comparison_x_gt_y", output_register="out", output_mode="register")
    prog.declare_register("x", 1)
    prog.declare_register("y", 1)
    prog.declare_register("out", 1)
    prog.add_mlp_op(
        layer=0,
        op_type="GREATER_THAN",
        in_regs=["x", "y"],
        out_reg="out",
        params={"slope": 200.0, "margin": 0.0},
    )
    return prog


# ==============================================================================
# 2. Maximum & Argmax: max_i(x_i) and argmax_i(x_i)
# ==============================================================================
def build_maximum_ir() -> TransformerProgram:
    prog = TransformerProgram(name="Sequence_Max_and_Argmax", output_register="out", output_mode="first_pos_register")
    prog.declare_register("x", 1)
    prog.declare_register("val_and_pos", 2)
    prog.declare_register("out", 2)
    prog.add_attention_op(
        layer=0,
        head=0,
        op_type="SCORE_MAX_BROADCAST",
        query_reg="x",
        key_reg="x",
        val_reg="val_and_pos",
        out_reg="out",
        scale=120.0,
        causal=False,
    )
    return prog


# ==============================================================================
# 3. Counting: Count tokens matching target query across sequence of length N
# ==============================================================================
def build_counting_ir(seq_len: int = 8) -> TransformerProgram:
    prog = TransformerProgram(name="Count_Matching_Tokens", output_register="count", output_mode="first_pos_register")
    prog.declare_register("tok", 1)
    prog.declare_register("query", 1)
    prog.declare_register("is_match", 1)
    prog.declare_register("count", 1)

    prog.add_mlp_op(
        layer=0,
        op_type="EQUAL_PAIR",
        in_regs=["tok", "query"],
        out_reg="is_match",
        params={"eps": 0.5, "scale_out": 1.0},
    )
    prog.add_attention_op(
        layer=1,
        head=0,
        op_type="UNIFORM_MEAN",
        query_reg="is_match",
        key_reg="is_match",
        val_reg="is_match",
        out_reg="count",
        causal=False,
        params={"out_scale": float(seq_len)},
    )
    return prog


# ==============================================================================
# 4. Prefix Sum: Cumulative sum s_i = sum_{j <= i} x_j
# ==============================================================================
def build_prefix_sum_ir(max_n: int = 8) -> TransformerProgram:
    prog = TransformerProgram(name="Prefix_Sum", output_register="prefix_sum", output_mode="register")
    prog.declare_register("x", 1)
    prog.declare_register("pos_oh", max_n)
    prog.declare_register("prefix_mean", 1)
    prog.declare_register("prefix_sum", 1)

    prog.add_attention_op(
        layer=0,
        head=0,
        op_type="UNIFORM_MEAN",
        query_reg="x",
        key_reg="x",
        val_reg="x",
        out_reg="prefix_mean",
        causal=True,
    )
    prog.add_mlp_op(
        layer=0,
        op_type="MULTIPLY_BY_ONEHOT",
        in_regs=["prefix_mean", "pos_oh"],
        out_reg="prefix_sum",
        params={"M": 500.0, "slot_scales": [float(k + 1) for k in range(max_n)]},
    )
    return prog


# ==============================================================================
# 5. Sorting: Parallel Rank-and-Route Sort in 4 Transformer Layers
# ==============================================================================
def build_sorting_ir(n: int = 8) -> TransformerProgram:
    prog = TransformerProgram(name=f"Parallel_Sort_N{n}", output_register="sorted_x", output_mode="register")
    prog.declare_register("x", 1)
    prog.declare_register("pos_oh", n)
    prog.declare_register("x_slot", n)
    prog.declare_register("all_x", n)
    prog.declare_register("rank", 1)
    prog.declare_register("rank_oh", n)
    prog.declare_register("sorted_x", 1)

    prog.add_mlp_op(
        layer=0,
        op_type="MULTIPLY_BY_ONEHOT",
        in_regs=["x", "pos_oh"],
        out_reg="x_slot",
        params={"M": 500.0},
    )
    prog.add_attention_op(
        layer=1,
        head=0,
        op_type="UNIFORM_MEAN",
        query_reg="x_slot",
        key_reg="x_slot",
        val_reg="x_slot",
        out_reg="all_x",
        causal=False,
        params={"out_scale": float(n)},
    )
    prog.add_mlp_op(
        layer=1,
        op_type="COUNT_GREATER_THAN_VECTOR",
        in_regs=["x", "all_x"],
        out_reg="rank",
        params={"slope": 200.0, "margin": 0.0,
                "tie_break_reg": "pos_oh", "tie_break_scale": 1.0 / (2 * n)},
    )
    prog.add_mlp_op(
        layer=2,
        op_type="SCALAR_TO_ONEHOT",
        in_regs=["rank"],
        out_reg="rank_oh",
        params={"eps": 0.45},
    )
    prog.add_attention_op(
        layer=3,
        head=0,
        op_type="DOT_MATCH",
        query_reg="pos_oh",
        key_reg="rank_oh",
        val_reg="x",
        out_reg="sorted_x",
        scale=120.0,
        causal=False,
    )
    return prog


# ==============================================================================
# 6. Graph Reachability: Multi-hop BFS propagation on N-node directed graph
# ==============================================================================
def build_graph_reachability_ir(n_nodes: int = 6, max_hops: int = 4) -> TransformerProgram:
    prog = TransformerProgram(
        name=f"Graph_Reachability_N{n_nodes}_Hops{max_hops}",
        output_register=f"reach_{max_hops}",
        output_mode="register",
    )
    prog.declare_register("node_oh", n_nodes)
    prog.declare_register("in_neighbors", n_nodes)
    for hop in range(max_hops + 1):
        prog.declare_register(f"reach_{hop}", 1)
    for hop in range(max_hops):
        prog.declare_register(f"msg_{hop}", 1)

    for hop in range(max_hops):
        prog.add_attention_op(
            layer=hop,
            head=0,
            op_type="DOT_MATCH",
            query_reg="in_neighbors",
            key_reg="node_oh",
            val_reg=f"reach_{hop}",
            out_reg=f"msg_{hop}",
            scale=80.0,
            causal=False,
        )
        prog.add_mlp_op(
            layer=hop,
            op_type="THRESHOLD_STEP",
            in_regs=[f"msg_{hop}"],
            out_reg=f"reach_{hop + 1}",
            params={"slope": 200.0, "threshold": 0.5 / float(n_nodes)},
        )
    return prog


# ==============================================================================
# 7. Shortest Path: Tropical (min, +) Bellman-Ford in Transformer Layers
# ==============================================================================
def build_shortest_path_ir(n_nodes: int = 5, max_hops: int = None) -> TransformerProgram:
    """
    Sequential pivot relaxations for nonnegative, bounded edge weights.
    In each layer l = 0 .. n_nodes-1 (with pivot node k = l % n_nodes):
      1. Attention Head (l, H0) (`BROADCAST_K_DISTANCE`):
         Queries with constant 1 against `node_oh[:, k]`, broadcasting pivot node k's
         current shortest distance `dist[k]` into `pivot_dist_{l}` at every node i!
      2. MLP at layer l (`TROPICAL_MIN_PLUS_STEP`):
         Updates `dist[i]` in-place using the exact ReLU identity:
           `dist[i] <- dist[i] - ReLU(dist[i] - (pivot_dist_{l} + edge_w[i, k]))`
                     = `min(dist[i], dist[k] + edge_w[i, k])`!
    Repeating all pivots n_nodes-1 times covers all simple shortest paths.
    Softmax and bounded sentinel values impose numerical/domain limitations.
    """
    if max_hops is None:
        max_hops = n_nodes * (n_nodes - 1)
    prog = TransformerProgram(
        name=f"Shortest_Path_N{n_nodes}_Steps{max_hops}",
        output_register="dist_0",
        output_mode="register",
    )
    prog.declare_register("node_oh", n_nodes)
    prog.declare_register("edge_w", n_nodes)  # edge_w[i, k] = weight of edge k -> i (0 for self-loop i -> i)
    prog.declare_register("dist_0", 1)
    for step in range(max_hops):
        prog.declare_register(f"pivot_d_{step}", 1)

    for step in range(max_hops):
        k = step % n_nodes
        prog.add_attention_op(
            layer=step,
            head=0,
            op_type="BROADCAST_K_DISTANCE",
            query_reg="node_oh",
            key_reg="node_oh",
            val_reg="dist_0",
            out_reg=f"pivot_d_{step}",
            scale=200.0,
            causal=False,
            params={"k_idx": k},
        )
        prog.add_mlp_op(
            layer=step,
            op_type="TROPICAL_MIN_PLUS_STEP",
            in_regs=["dist_0", f"pivot_d_{step}", "edge_w"],
            out_reg="dist_0",
            params={"neighbor_idx": k},
        )
    return prog


# ==============================================================================
# 8. Compositional Task (Program Generalization):
#    "Count how many elements x_i in the sequence are strictly greater than
#     the causal prefix mean (1/(i+1)) * sum_{j<=i} x_j"
# ==============================================================================
def build_compositional_above_prefix_mean_ir(seq_len: int = 8) -> TransformerProgram:
    prog = TransformerProgram(
        name="Compositional_Count_Above_Prefix_Mean",
        output_register="count_above",
        output_mode="first_pos_register",
    )
    prog.declare_register("x", 1)
    prog.declare_register("prefix_mean", 1)
    prog.declare_register("is_above", 1)
    prog.declare_register("count_above", 1)

    # Layer 0 Causal Attn: compute prefix_mean[i] = (1/(i+1)) * sum_{j<=i} x_j
    prog.add_attention_op(
        layer=0,
        head=0,
        op_type="UNIFORM_MEAN",
        query_reg="x",
        key_reg="x",
        val_reg="x",
        out_reg="prefix_mean",
        causal=True,
    )
    # Layer 0 MLP: compare x[i] > prefix_mean[i]
    prog.add_mlp_op(
        layer=0,
        op_type="GREATER_THAN",
        in_regs=["x", "prefix_mean"],
        out_reg="is_above",
        params={"slope": 200.0, "margin": 0.0},
    )
    # Layer 1 Full Attn: count total positions where is_above[i] == 1
    prog.add_attention_op(
        layer=1,
        head=0,
        op_type="UNIFORM_MEAN",
        query_reg="is_above",
        key_reg="is_above",
        val_reg="is_above",
        out_reg="count_above",
        causal=False,
        params={"out_scale": float(seq_len)},
    )
    return prog


# ==============================================================================
# Evaluation Suite across Tasks and Target Transformer Architectures
# ==============================================================================
def evaluate_all_tasks(d_model: int = 64, n_layers: int = 6, n_heads: int = 4, n_trials: int = 500) -> Dict[str, Dict]:
    rng = np.random.RandomState(42)
    results: Dict[str, Dict] = {}

    # 1. Comparison
    prog1 = build_comparison_ir()
    model1 = TransformerCircuitBackend.compile(prog1, d_model=d_model, n_layers=n_layers, n_heads=n_heads)
    correct1 = 0
    for _ in range(n_trials):
        x = rng.choice(np.arange(-20, 21), size=8, replace=False).astype(np.float64)
        y = rng.choice(np.arange(-20, 21), size=8, replace=False).astype(np.float64)
        # Ensure x != y so strict comparison is well-defined
        y = np.where(x == y, y + 1.0, y)
        pred = np.round(model1.run({"x": x, "y": y}, seq_len=8))
        gt = (x > y).astype(np.float64)
        if np.array_equal(pred, gt):
            correct1 += 1
    results["1. Comparison (x > y)"] = {
        "accuracy": correct1 / n_trials,
        "nonzero_params": model1.nonzero_params,
        "total_params": model1.total_params,
    }

    # 2. Maximum & Argmax
    prog2 = build_maximum_ir()
    model2 = TransformerCircuitBackend.compile(prog2, d_model=d_model, n_layers=n_layers, n_heads=n_heads)
    correct2 = 0
    for _ in range(n_trials):
        x = rng.choice(np.arange(-50, 51), size=8, replace=False).astype(np.float64)
        pos = np.arange(8, dtype=np.float64)
        val_and_pos = np.stack([x, pos], axis=-1)
        out = model2.run({"x": x, "val_and_pos": val_and_pos}, seq_len=8)
        pred_max, pred_argmax = np.round(out[0]), int(np.round(out[1]))
        if pred_max == np.max(x) and pred_argmax == int(np.argmax(x)):
            correct2 += 1
    results["2. Maximum & Argmax"] = {
        "accuracy": correct2 / n_trials,
        "nonzero_params": model2.nonzero_params,
        "total_params": model2.total_params,
    }

    # 3. Counting
    prog3 = build_counting_ir(seq_len=8)
    model3 = TransformerCircuitBackend.compile(prog3, d_model=d_model, n_layers=n_layers, n_heads=n_heads)
    correct3 = 0
    for _ in range(n_trials):
        tok = rng.randint(0, 5, size=8).astype(np.float64)
        q_val = float(rng.randint(0, 5))
        query = np.full(8, q_val, dtype=np.float64)
        pred_cnt = int(np.round(model3.run({"tok": tok, "query": query}, seq_len=8)))
        gt_cnt = int(np.sum(tok == q_val))
        if pred_cnt == gt_cnt:
            correct3 += 1
    results["3. Counting Matches"] = {
        "accuracy": correct3 / n_trials,
        "nonzero_params": model3.nonzero_params,
        "total_params": model3.total_params,
    }

    # 4. Prefix Sum
    prog4 = build_prefix_sum_ir(max_n=8)
    model4 = TransformerCircuitBackend.compile(prog4, d_model=d_model, n_layers=n_layers, n_heads=n_heads)
    correct4 = 0
    pos_oh8 = np.eye(8, dtype=np.float64)
    for _ in range(n_trials):
        x = rng.randint(-10, 11, size=8).astype(np.float64)
        pred_ps = np.round(model4.run({"x": x, "pos_oh": pos_oh8}, seq_len=8))
        gt_ps = np.cumsum(x)
        if np.array_equal(pred_ps, gt_ps):
            correct4 += 1
    results["4. Prefix Sum"] = {
        "accuracy": correct4 / n_trials,
        "nonzero_params": model4.nonzero_params,
        "total_params": model4.total_params,
    }

    # 5. Sorting
    prog5 = build_sorting_ir(n=8)
    model5 = TransformerCircuitBackend.compile(prog5, d_model=d_model, n_layers=n_layers, n_heads=n_heads)
    correct5 = 0
    for _ in range(n_trials):
        x = rng.randint(-30, 31, size=8).astype(np.float64)
        pred_sorted = np.round(model5.run({"x": x, "pos_oh": pos_oh8}, seq_len=8))
        gt_sorted = np.sort(x)
        if np.array_equal(pred_sorted, gt_sorted):
            correct5 += 1
    results["5. Sorting (N=8)"] = {
        "accuracy": correct5 / n_trials,
        "nonzero_params": model5.nonzero_params,
        "total_params": model5.total_params,
    }

    # 6. Graph Reachability (N=6 nodes, up to 4 hops)
    n_nodes = 6
    prog6 = build_graph_reachability_ir(n_nodes=n_nodes, max_hops=4)
    model6 = TransformerCircuitBackend.compile(prog6, d_model=d_model, n_layers=n_layers, n_heads=n_heads)
    node_oh6 = np.eye(n_nodes, dtype=np.float64)
    correct6 = 0
    for _ in range(n_trials):
        # Random directed graph with edge probability 0.3 + self-loops
        adj = (rng.rand(n_nodes, n_nodes) < 0.3).astype(np.float64)
        np.fill_diagonal(adj, 1.0)
        # in_neighbors[i, j] = 1 iff edge j -> i
        in_neighbors = adj.T.copy()
        src = rng.randint(0, n_nodes)
        reach_0 = np.zeros(n_nodes, dtype=np.float64)
        reach_0[src] = 1.0

        pred_reach = np.round(
            model6.run({"node_oh": node_oh6, "in_neighbors": in_neighbors, "reach_0": reach_0}, seq_len=n_nodes)
        )
        # Ground truth 4-hop reachability
        gt_reach = reach_0.copy()
        for _ in range(4):
            gt_reach = np.minimum(1.0, in_neighbors @ gt_reach)
        if np.array_equal(pred_reach, gt_reach):
            correct6 += 1
    results["6. Graph Reachability"] = {
        "accuracy": correct6 / n_trials,
        "nonzero_params": model6.nonzero_params,
        "total_params": model6.total_params,
    }

    # 7. Shortest Path: 4 full pivot sweeps, checked against synchronous Bellman-Ford.
    n_sp = 5
    sp_layers = n_sp * (n_sp - 1)
    prog7 = build_shortest_path_ir(n_nodes=n_sp)
    model7 = TransformerCircuitBackend.compile(prog7, d_model=d_model, n_layers=max(n_layers, sp_layers), n_heads=n_heads)
    node_oh5 = np.eye(n_sp, dtype=np.float64)
    correct7 = 0
    INF = 100.0
    for _ in range(n_trials):
        edge_w = np.full((n_sp, n_sp), INF, dtype=np.float64)
        for u in range(n_sp):
            for v in range(n_sp):
                if u == v:
                    edge_w[v, u] = 0.0
                elif rng.rand() < 0.45:
                    edge_w[v, u] = float(rng.randint(1, 10))
        # Source is node 0; use a separate synchronous Bellman-Ford reference.
        dist_0 = np.full(n_sp, INF, dtype=np.float64)
        dist_0[0] = 0.0

        pred_dist = np.round(
            model7.run({"node_oh": node_oh5, "edge_w": edge_w, "dist_0": dist_0}, seq_len=n_sp)
        )
        gt_dist = dist_0.copy()
        for _ in range(n_sp - 1):
            gt_dist = np.minimum(gt_dist, np.min(edge_w + gt_dist[None, :], axis=1))
        if np.array_equal(pred_dist, np.round(gt_dist)):
            correct7 += 1
    results["7. Shortest Path (20 layers)"] = {
        "accuracy": correct7 / n_trials,
        "nonzero_params": model7.nonzero_params,
        "total_params": model7.total_params,
    }

    # 8. Compositional Task (Count elements strictly greater than causal prefix mean)
    prog8 = build_compositional_above_prefix_mean_ir(seq_len=8)
    model8 = TransformerCircuitBackend.compile(prog8, d_model=d_model, n_layers=n_layers, n_heads=n_heads)
    correct8 = 0
    for _ in range(n_trials):
        x = rng.choice(np.arange(-40, 41), size=8, replace=False).astype(np.float64)
        pmean = np.cumsum(x) / np.arange(1, 9, dtype=np.float64)
        # Ensure no exact equality x[i] == pmean[i] for i >= 1
        if np.any(np.abs(x[1:] - pmean[1:]) < 0.05):
            x[1:] += 0.2
            pmean = np.cumsum(x) / np.arange(1, 9, dtype=np.float64)
        pred_c = int(np.round(model8.run({"x": x}, seq_len=8)))
        gt_c = int(np.sum(x > pmean + 1e-9))
        if pred_c == gt_c:
            correct8 += 1
    results["8. Compositional (Above Prefix Mean)"] = {
        "accuracy": correct8 / n_trials,
        "nonzero_params": model8.nonzero_params,
        "total_params": model8.total_params,
    }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the bounded-domain transformer circuit benchmark")
    parser.add_argument("--output-json", help="write benchmark data for plotting")
    args = parser.parse_args()
    configs = [
        ("Small  (d=64,  L=6, H=4)", 64, 6, 4),
        ("Medium (d=128, L=8, H=8)", 128, 8, 8),
        ("Large  (d=256, L=8, H=8)", 256, 8, 8),
    ]
    all_arch_results = {}
    for label, d, L, H in configs:
        t0 = time.time()
        res = evaluate_all_tasks(d_model=d, n_layers=L, n_heads=H, n_trials=500)
        dt = time.time() - t0
        all_arch_results[label] = res
        print(f"\n=== Architecture: {label} (Eval time: {dt:.2f}s) ===")
        print("Shortest path uses 20 layers; all other tasks use the depth above.")
        print(f"{'Task':<38} | {'Accuracy':<10} | {'Nonzero / Total Weights'}")
        print("-" * 78)
        for task_name, info in res.items():
            acc_str = f"{info['accuracy']*100:6.2f}%"
            wt_str = f"{info['nonzero_params']:>4} / {info['total_params']:<9} ({info['nonzero_params']/info['total_params']*100:.4f}%)"
            print(f"{task_name:<38} | {acc_str:<10} | {wt_str}")
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"trials_per_task": 500, "seed": 42, "architectures": all_arch_results}, f, indent=2)
            f.write("\n")
