# Weight-Space Programming Blind Reproduction (`weight_space`)

Blind reproduction of experiments from [Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, Nov 2026) using zero-context subagents.

## Summary of Blind Reproduction Results

| Experiment | Task | Ground Truth / Baseline | Zero-Context Subagent Result |
| --- | --- | --- | --- |
| **Exp 1 (`exp1_network_A`)** | Read 17,253 MLP weights directly from text (no tools, no code) and identify which game the PPO policy was trained on | **Breakout** (Astra: Breakout, 99% confidence) | **Breakout** (98% confidence, exact identification of row 8 paddle, rows 1–3 bricks, and `no-op`/`up`/`down` output symmetry) |
| **Exp 2 (`exp2_diffusion_mystery`)** | Read 23,824 DDPM MLP weights directly from text (no tools, no forward pass) and identify generated 12×12 images | **Filled Upright Triangle** (Astra: hollow triangle outline, 88% confidence) | **Filled White Upright Triangle** (95% confidence — accurately distinguished filled interior vs. outline via `b3` and `W3` interior blocks) |
| **Exp 5 (`exp5_breakout`)** | Write 20,486 weights (`600 -> 32 -> 32 -> 6` tanh MLP) from scratch in one shot (no tools, no training, no rollouts) for MinAtar Breakout | PPO 100M steps: **15.8**<br>PPO 500M steps: **25.3**<br>Astra one-shot: **517.4** | **77.92 ± 2.61** over 1,000 episodes (3.1× PPO 500M, 4.9× PPO 100M; 649 nonzero weights using 5 `h1` units and 4 `h2` units) |

## Repository Structure

```
weight_space/
├── README.md
├── blind_tasks/                  # Blind prompts and weights given to zero-context subagents
│   ├── exp1_network_A.txt        # Exp 1: Mystery PPO game network A (Breakout)
│   ├── exp1_network_B.txt        # Exp 1: Mystery PPO game network B (Pac-Man)
│   ├── exp2_diffusion_mystery.txt# Exp 2: Mystery 12x12 DDPM weights (Triangle)
│   ├── exp3_prompt.txt           # Exp 3: Middle-4-block character Transformer prompt
│   ├── exp3_blocks_mystery.npz   # Exp 3: Middle-4-block weights (Python corpus)
│   └── exp5_breakout_prompt.txt  # Exp 5: MinAtar Breakout zero-shot weight writing prompt
├── results/                      # Full verbatim outputs from the zero-context subagents
│   ├── exp1_subagent_response.md
│   ├── exp2_subagent_response.md
│   └── exp5_subagent_response.md
└── eval_exp5_breakout.py         # Standalone 1,000-episode evaluation script for Exp 5 handwritten weights
```

## Quick Start

To evaluate the zero-shot handwritten MinAtar Breakout policy network from Experiment 5:

```bash
python3 eval_exp5_breakout.py
```
