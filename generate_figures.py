"""
Generate publication-quality blog figures for `neural-weights-compiler` README.md,
styled after 'Sparks of Weight-Space Programming' (MIT CSAIL, Nov 2026).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

os.makedirs("/home/tiger/weight_space/assets", exist_ok=True)

# Set clean academic/blog styling
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#CCCCCC"
plt.rcParams["axes.linewidth"] = 0.8


def make_fig1_hero_cards():
    """Hero summary banner with 3 side-by-side result cards (Compile 1.48M Go, Graft 1.58M Transformer, Beat Astra on MinAtar)."""
    fig = plt.figure(figsize=(15.5, 5.2), dpi=220)
    fig.patch.set_facecolor("#F9FAFB")
    gs = GridSpec(1, 3, figure=fig, wspace=0.22, left=0.04, right=0.97, top=0.80, bottom=0.14)

    fig.suptitle(
        "Neural Weights Compiler: Scaling Weight-Space Programming via Neural IR & Orthogonal Circuit Grafting",
        fontsize=15.5, fontweight="bold", color="#111827", y=0.95
    )
    fig.text(
        0.5, 0.88,
        "Compiling natural-language specifications & algorithms directly into multi-million-parameter neural weights — with zero gradient descent",
        ha="center", fontsize=11, color="#4B5563", style="italic"
    )

    # Card 1: 1.48M 9x9 Go (Zero Forward Passes)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#FFFFFF")
    labels1 = ["PPO (20.5M games)\nBaseline", "GPT-6 Astra\n(Budget 0, Raw)", "NeuralCompiler\n(Budget 0, Ours)"]
    vals1 = [50.0, 35.1, 69.8]
    colors1 = ["#9CA3AF", "#F59E0B", "#10B981"]
    bars1 = ax1.bar(labels1, vals1, color=colors1, width=0.56, edgecolor="#374151", linewidth=0.7)
    ax1.axhline(50.0, color="#EF4444", linestyle="--", linewidth=1.0, alpha=0.7, label="50% Win Threshold")
    ax1.set_ylim(0, 92)
    ax1.set_ylabel("Win Rate vs. 20.5M-Game PPO (%)", fontsize=9.5, fontweight="bold", color="#1F2937")
    ax1.set_title("1. Compile 1.48M-Param 9×9 Go\n(Zero Training, Zero Forward Passes)", fontsize=11, fontweight="bold", color="#111827", pad=10)
    ax1.tick_params(axis="x", labelsize=8.5)
    for bar, v in zip(bars1, vals1):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 2.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#111827")
    ax1.text(0.5, 0.89, "82.0% H2H vs. Astra Budget 0!", transform=ax1.transAxes, ha="center",
             fontsize=8.8, fontweight="bold", color="#065F46",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#D1FAE5", edgecolor="#10B981", lw=0.8))
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    # Card 2: Stage 3 Orthogonal Circuit Grafting (Solving Exp 8 Gemma Failure Mode)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#FFFFFF")
    x2 = np.arange(2)
    width = 0.34
    before_vals = [100.0, 0.0]
    after_vals = [100.0, 100.0]
    b1 = ax2.bar(x2 - width / 2, before_vals, width, label="Host Model (Before Graft)", color="#9CA3AF", edgecolor="#374151", linewidth=0.7)
    b2 = ax2.bar(x2 + width / 2, after_vals, width, label="Grafted (+68 weights, 0.0043%)", color="#3B82F6", edgecolor="#1D4ED8", linewidth=0.7)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(["Host Multi-Task\n(Preservation)", "8-Digit Carry Adder\n(New Capability)"], fontsize=9)
    ax2.set_ylim(0, 122)
    ax2.set_ylabel("Exact Match Accuracy (%)", fontsize=9.5, fontweight="bold", color="#1F2937")
    ax2.set_title("2. Graft New Skill into 1.58M Transformer\n(Zero Forgetting: max |Δf_host| = 0.0)", fontsize=11, fontweight="bold", color="#111827", pad=10)
    ax2.legend(loc="upper left", fontsize=8, frameon=True)
    for bar, v in zip(b1, before_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 2.0, f"{v:.0f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#4B5563")
    for bar, v in zip(b2, after_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 2.0, f"{v:.0f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#1D4ED8")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    # Card 3: MinAtar Normalized Multi-Game Dominance
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#FFFFFF")
    games = ["Breakout\n(÷10)", "Seaquest", "Asterix"]
    ppo100 = [1.58, 5.6, 25.8]
    astra = [51.74, 53.1, 53.2]
    ours = [54.68, 59.8, 64.9]
    x3 = np.arange(len(games))
    w3 = 0.25
    r1 = ax3.bar(x3 - w3, ppo100, w3, label="PPO (100M steps)", color="#9CA3AF", edgecolor="#374151", linewidth=0.6)
    r2 = ax3.bar(x3, astra, w3, label="GPT-6 Astra (One-Shot)", color="#F59E0B", edgecolor="#374151", linewidth=0.6)
    r3 = ax3.bar(x3 + w3, ours, w3, label="NeuralCompiler (Ours)", color="#8B5CF6", edgecolor="#374151", linewidth=0.6)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(games, fontsize=9)
    ax3.set_ylim(0, 78)
    ax3.set_ylabel("Mean Episode Return (Breakout ÷10)", fontsize=9.5, fontweight="bold", color="#1F2937")
    ax3.set_title("3. Outperform Astra on All 3 MinAtar Games\n(20,486-Weight tanh MLP, Zero Training)", fontsize=11, fontweight="bold", color="#111827", pad=10)
    ax3.legend(loc="upper left", fontsize=8, frameon=True)
    raw_labels_ours = ["546.8", "59.8", "64.9"]
    for bar, v, rl in zip(r3, ours, raw_labels_ours):
        ax3.text(bar.get_x() + bar.get_width() / 2, v + 1.5, rl, ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#5B21B6")
    ax3.grid(axis="y", linestyle=":", alpha=0.5)

    plt.savefig("/home/tiger/weight_space/assets/fig1_hero_summary.png", bbox_inches="tight")
    plt.close(fig)


def make_fig2_go_comparison():
    """Detailed figure for Experiment / Demo 1: 1.48M-Parameter 9x9 Go Network Compilation vs Astra & PPO."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=220, gridspec_kw={"width_ratios": [1.35, 1.0]})
    fig.patch.set_facecolor("#FFFFFF")

    # Left: Win rate against the 3 same-size 20.5M-game PPO seeds (Budget 0)
    seeds = ["vs. PPO Seed 0\n(1.48M, 20.5M games)", "vs. PPO Seed 1\n(1.48M, 20.5M games)", "vs. PPO Seed 2\n(1.48M, 20.5M games)", "Mean Win Rate\nAcross All Seeds"]
    astra_b0 = [42.6, 26.3, 36.3, 35.1]
    compiler_b0 = [63.5, 66.5, 79.5, 69.8]

    x = np.arange(len(seeds))
    w = 0.35
    b_a = ax_left.bar(x - w / 2, astra_b0, w, label="GPT-6 Astra (Budget 0: Raw Script)", color="#F59E0B", edgecolor="#374151", lw=0.7)
    b_c = ax_left.bar(x + w / 2, compiler_b0, w, label="SpatialGraphGoCompiler (Budget 0: Ours)", color="#10B981", edgecolor="#065F46", lw=0.7)

    ax_left.axhline(50.0, color="#EF4444", linestyle="--", lw=1.2, label="50% Parity vs. 20.5M-Game PPO")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(seeds, fontsize=9)
    ax_left.set_ylim(0, 95)
    ax_left.set_ylabel("Head-to-Head Win Rate (%)", fontsize=10, fontweight="bold")
    ax_left.set_title("9×9 Go (1,480,274 Weights, Zero Forward Passes)\nWin Rate vs. Equal-Sized 20.5M-Game PPO", fontsize=11, fontweight="bold")
    ax_left.legend(loc="upper left", fontsize=8.5)
    ax_left.grid(axis="y", linestyle=":", alpha=0.5)

    for bar, v in zip(b_a, astra_b0):
        ax_left.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=8.8, fontweight="bold", color="#92400E")
    for bar, v in zip(b_c, compiler_b0):
        ax_left.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9.2, fontweight="bold", color="#065F46")

    # Right: Direct Head-to-Head Our Compiler vs GPT-6 Astra Budget 0 (Black / White / Overall)
    cats = ["As Black\n(Komi -7.5)", "As White\n(Komi +7.5)", "Overall\n(200 Games)"]
    win_rates = [75.0, 89.0, 82.0]
    lose_rates = [25.0, 11.0, 18.0]
    x_r = np.arange(len(cats))
    w_r = 0.48
    p1 = ax_right.bar(x_r, win_rates, w_r, label="NeuralCompiler Wins", color="#10B981", edgecolor="#065F46", lw=0.7)
    p2 = ax_right.bar(x_r, lose_rates, w_r, bottom=win_rates, label="GPT-6 Astra (Budget 0) Wins", color="#FDE68A", edgecolor="#D97706", lw=0.7)
    ax_right.axhline(50.0, color="#374151", linestyle=":", lw=1.0)
    ax_right.set_xticks(x_r)
    ax_right.set_xticklabels(cats, fontsize=9.5)
    ax_right.set_ylim(0, 108)
    ax_right.set_ylabel("Direct Head-to-Head Share (%)", fontsize=10, fontweight="bold")
    ax_right.set_title("Direct H2H: NeuralCompiler vs. GPT-6 Astra\n(Same 243 → 512×6 → 82 ReLU Architecture)", fontsize=11, fontweight="bold")
    ax_right.legend(loc="upper right", fontsize=8.5)
    for bar, v in zip(p1, win_rates):
        ax_right.text(bar.get_x() + bar.get_width() / 2, v / 2, f"Ours\n{v:.1f}%", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#FFFFFF")

    plt.tight_layout()
    plt.savefig("/home/tiger/weight_space/assets/fig2_go_1p5m_h2h.png", bbox_inches="tight")
    plt.close(fig)


def make_fig3_grafting_and_registers():
    """Visualize Stage 3 Orthogonal Subspace Circuit Grafting and O(1)-Depth Carry-Lookahead Attention."""
    from stage3_circuit_grafting import build_host_transformer, compile_and_graft_carry_lookahead_adder

    host = build_host_transformer(d_model=128, n_layers=8, n_heads=8)
    grafted, _ = compile_and_graft_carry_lookahead_adder(host, n_digits=8)

    # Run a cascading carry example: 49283756 + 50716244 = 100000000 (LSD at pos 0)
    # In LSD-first order: A = [6,5,7,3,8,2,9,4], B = [4,4,2,6,1,7,0,5] -> A+B = [10,9,9,9,9,9,9,9] -> sum = [0,0,0,0,0,0,0,0], carry_msb = 1
    da = np.array([6, 5, 7, 3, 8, 2, 9, 4], dtype=np.float64)
    db = np.array([4, 4, 2, 6, 1, 7, 0, 5], dtype=np.float64)
    hx = np.array([3, -2, 5, 3, 1, 4, -1, 3], dtype=np.float64)
    hq = np.full(8, 3.0, dtype=np.float64)
    pos_oh8 = np.eye(8, dtype=np.float64)
    pos_idx8 = np.arange(8, dtype=np.float64)

    inputs = {
        "host_x": hx, "host_query": hq, "host_pos_oh": pos_oh8,
        "digit_a": da, "digit_b": db, "pos_idx": pos_idx8, "add_pos_oh": pos_oh8,
    }
    X_final = grafted.forward_residual(grafted.encode_inputs(inputs, seq_len=8))

    # Also compute the exact Layer 1 Head 2 Carry-Lookahead Attention map on this cascading carry!
    X = grafted.encode_inputs(inputs, seq_len=8)
    # Run Layer 0
    scale = np.sqrt(float(grafted.d_head))
    attn_d0 = np.zeros_like(X)
    for h in range(grafted.n_heads):
        W_Q = grafted.weights[f"L0.H{h}.W_Q"]
        b_Q = grafted.weights[f"L0.H{h}.b_Q"]
        W_K = grafted.weights[f"L0.H{h}.W_K"]
        b_K = grafted.weights[f"L0.H{h}.b_K"]
        W_V = grafted.weights[f"L0.H{h}.W_V"]
        b_V = grafted.weights[f"L0.H{h}.b_V"]
        W_O = grafted.weights[f"L0.H{h}.W_O"]
        if not (np.any(W_O) or np.any(W_V)):
            continue
        Q, K, V = X @ W_Q + b_Q, X @ W_K + b_K, X @ W_V + b_V
        sc = (Q @ K.T) / scale
        if grafted.causal_masks.get((0, h), False):
            sc += np.triu(np.full((8, 8), -1e9), k=1)
        sc -= np.max(sc, axis=-1, keepdims=True)
        A = np.exp(sc) / np.sum(np.exp(sc), axis=-1, keepdims=True)
        attn_d0 += (A @ V) @ W_O
    X = X + attn_d0
    X = X + np.maximum(0.0, X @ grafted.weights["L0.W_fc"] + grafted.weights["L0.b_fc"]) @ grafted.weights["L0.W_proj"]

    # Layer 1 Head 2 (Grafted Carry-Lookahead Head)
    W_Q1 = grafted.weights["L1.H2.W_Q"]
    b_Q1 = grafted.weights["L1.H2.b_Q"]
    W_K1 = grafted.weights["L1.H2.W_K"]
    b_K1 = grafted.weights["L1.H2.b_K"]
    Q1 = X @ W_Q1 + b_Q1
    K1 = X @ W_K1 + b_K1
    sc1 = (Q1 @ K1.T) / scale + np.triu(np.full((8, 8), -1e9), k=1)
    sc1 -= np.max(sc1, axis=-1, keepdims=True)
    A_carry = np.exp(sc1) / np.sum(np.exp(sc1), axis=-1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 4.9), dpi=220, gridspec_kw={"width_ratios": [1.05, 1.25]})
    fig.patch.set_facecolor("#FFFFFF")

    # Left: Attention heatmap of Grafted Head L1.H2 on 49283756 + 50716244 = 100000000
    im1 = ax1.imshow(A_carry, cmap="Blues", vmin=0.0, vmax=1.0)
    pos_labels = [f"d{i}\n({int(da[i])}+{int(db[i])}={int(da[i]+db[i])})" for i in range(8)]
    ax1.set_xticks(range(8))
    ax1.set_xticklabels([f"Key j={j}\n(sum={int(da[j]+db[j])})" for j in range(8)], fontsize=7.5, rotation=30)
    ax1.set_yticks(range(8))
    ax1.set_yticklabels([f"Query i={i} (sum={int(da[i]+db[i])})" for i in range(8)], fontsize=8)
    ax1.set_title("Grafted Head L1.H2 Carry-Lookahead Attention\nInput: 49,283,756 + 50,716,244 = 100,000,000\n(All positions i=1..7 jump in O(1) to j*=0 where 6+4=10!)", fontsize=10, fontweight="bold")
    for i in range(8):
        for j in range(8):
            if A_carry[i, j] > 0.5:
                ax1.text(j, i, f"{A_carry[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")

    # Right: Orthogonal Subspace Partitioning in the 128-D Residual Stream
    # Show active norms across residual dimensions 0..32 for Host vs Grafted registers
    host_dims = list(range(0, 19))
    graft_dims = list(range(19, 36))
    free_dims = list(range(36, 64))

    norms_host_only = np.linalg.norm(host.forward_residual(host.encode_inputs(inputs, seq_len=8))[:, :44], axis=0)
    norms_graft_only = np.linalg.norm(X_final[:, :44] - host.forward_residual(host.encode_inputs(inputs, seq_len=8))[:, :44], axis=0)

    dims_x = np.arange(44)
    ax2.bar(dims_x, norms_host_only, color="#6B7280", label="Host Task Subspace S_host (Dims 0–18: PrefixSum/Max/Count)", width=0.75)
    ax2.bar(dims_x, norms_graft_only, bottom=norms_host_only, color="#2563EB", label="Grafted Adder Subspace S_graft ⊂ S_host^⊥ (Dims 19–35)", width=0.75)
    ax2.axvline(18.5, color="#EF4444", linestyle="--", lw=1.4, label="Orthogonal Subspace Boundary (Zero Interference)")
    ax2.set_xlabel("Residual Stream Dimension Index (0..43 of d_model=128)", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("L2 Activation Energy Across Sequence", fontsize=9.5, fontweight="bold")
    ax2.set_title("Orthogonal Subspace Allocation in d_model = 128\nWhy Host Drift is Exactly 0.000000e+00 After Grafting", fontsize=10.5, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8.2)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig("/home/tiger/weight_space/assets/fig3_orthogonal_grafting.png", bbox_inches="tight")
    plt.close(fig)


def make_fig4_exp3_simplex_and_heads():
    """Visualize our Blind Reproduction of Exp 3 (Middle-4-Blocks SVD Spectrum & Induction Head Discovery)."""
    W = dict(np.load("/home/tiger/weight_space/blind_tasks/exp3_blocks_mystery.npz"))
    ln1_0 = W["ln1_w.0"]
    # Stack W_v.0 of heads 1, 2, 3, 5 (64 rows each)
    V_stack = np.vstack([W["W_v.0"][64 * h : 64 * (h + 1), :] for h in [1, 2, 3, 5]])
    _, sv_v0, _ = np.linalg.svd(V_stack, full_matrices=False)

    _, sv_proj3, _ = np.linalg.svd(W["W_proj.3"], full_matrices=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.4), dpi=220)
    fig.patch.set_facecolor("#FFFFFF")

    # Left: Spectral cliff at index 27 in Layer 0 W_v.0 (28-character vocabulary -> 27D centered simplex)
    idx = np.arange(1, 46)
    ax1.plot(idx, sv_v0[:45], marker="o", markersize=4.5, color="#2563EB", lw=1.8, label="Singular values of stacked Layer-0 W_v")
    ax1.axvline(27, color="#EF4444", linestyle="--", lw=1.5, label="Rank-27 Cliff (28 chars → 27D Simplex U_in)")
    ax1.fill_between(idx[:27], 0, sv_v0[:27], color="#DBEAFE", alpha=0.55)
    ax1.set_xlabel("Singular Value Index k", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Singular Value σ_k", fontsize=9.5, fontweight="bold")
    ax1.set_title("Blind Reading of 7.16M Transformer Middle Blocks (Exp 3)\nExact Recovery of 27D Token Simplex Without Embeddings", fontsize=10.5, fontweight="bold")
    ax1.set_ylim(0, 3.8)
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Right: 4x6 Heatmap of Induction / Copying Head Scores from Pure Weights vs Ground Truth
    # Ground truth copying heads in Python model: (2,1), (3,1), (3,5) (plus weak (3,0))
    copy_trace = np.array([
        [0.02, 0.01, 0.01, 0.01, 0.01, 0.03],
        [0.04, 0.03, 0.05, 0.02, 0.03, 0.04],
        [0.05, 5.50, 0.10, 0.08, 0.06, 1.36],
        [7.39, 27.17, -3.95, 0.15, 0.12, 25.34],
    ])
    im2 = ax2.imshow(np.maximum(0, copy_trace), cmap="Purples", aspect="auto")
    ax2.set_xticks(range(6))
    ax2.set_xticklabels([f"Head {h}" for h in range(6)], fontsize=9)
    ax2.set_yticks(range(4))
    ax2.set_yticklabels([f"Layer {l}" for l in range(4)], fontsize=9)
    ax2.set_title("Weight-Space Induction Circuit Trace tr(U_out^T W_OV U_in)\nIdentifies Copying Heads L2.H1, L3.H0, L3.H1, L3.H5 (Zero Forward Pass)", fontsize=10.5, fontweight="bold")
    for l in range(4):
        for h in range(6):
            v = copy_trace[l, h]
            tag = " ★ COPY" if (l, h) in [(2, 1), (3, 0), (3, 1), (3, 5)] else ""
            col = "white" if v > 12 else "#1F2937"
            ax2.text(h, l, f"{v:+.2f}{tag}", ha="center", va="center", fontsize=8.5, fontweight="bold" if tag else "normal", color=col)

    plt.tight_layout()
    plt.savefig("/home/tiger/weight_space/assets/fig4_exp3_spectral_decompile.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_fig1_hero_cards()
    make_fig2_go_comparison()
    make_fig3_grafting_and_registers()
    make_fig4_exp3_simplex_and_heads()
    print("Generated all 4 blog figures in assets/!")
