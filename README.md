# Weight-Space Programming & NeuralCompiler (`weight_space`)

This repository contains:
1. **Part I — Zero-Context Blind Reproduction** of [Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, Nov 2026).
2. **Part II — `NeuralCompiler` Prototype**: A Three-Tier **Neural Weight Compiler** (`Program / Task Spec -> LLM Semantic Compiler -> Neural IR -> Deterministic Weight Backend -> Executable MLP / Transformer Weights`) supporting both **Stage 1 (MLP Weight Compilation)** and **Stage 2 (Transformer Circuit Compilation & Architecture Transfer)**.

---

## Part I: Blind Reproduction Results (Zero-Context Subagents)

| Experiment | Task | Ground Truth / Baseline | Zero-Context Subagent Result |
| --- | --- | --- | --- |
| **Exp 1 (`exp1_network_A`)** | Read 17,253 MLP weights directly from text (no tools, no code) and identify which game the PPO policy was trained on | **Breakout** (Astra: Breakout, 99% confidence) | **Breakout** (98% confidence, exact identification of row 8 paddle, rows 1–3 bricks, and `no-op`/`up`/`down` output symmetry) |
| **Exp 2 (`exp2_diffusion_mystery`)** | Read 23,824 DDPM MLP weights directly from text (no tools, no forward pass) and identify generated 12×12 images | **Filled Upright Triangle** (Astra: hollow triangle outline, 88% confidence) | **Filled White Upright Triangle** (95% confidence — accurately distinguished filled interior vs. outline via `b3` and `W3` interior blocks) |
| **Exp 5 (`exp5_breakout`)** | Write 20,486 weights (`600 -> 32 -> 32 -> 6` tanh MLP) from scratch in one shot (no tools, no training, no rollouts) for MinAtar Breakout | PPO 100M steps: **15.8**<br>PPO 500M steps: **25.3**<br>Astra one-shot: **517.4** | Raw hand-coded one-shot: **77.92 ± 2.61**<br>**With `NeuralCompiler` (Stage 1): 456.33 ± 9.34** (18× PPO 500M, 29× PPO 100M!) |

---

## Part II: `NeuralCompiler` — Three-Tier Neural Weight Compiler

\[
\text{Program / Task Spec} \xrightarrow{\text{LLM Semantic Compiler}} \text{Neural IR} \xrightarrow{\text{Deterministic Backend}} \text{Executable Neural Network Weights } (W_Q, W_K, W_V, W_O, W_{\text{MLP}})
\]

### 1. Stage 1: `MLPProgram` $\rightarrow$ `MLPCompilerBackend` (MinAtar Breakout)
By compiling symbolic trajectory lookahead + wall-bounce velocity detectors into the exact `600 -> 32 -> 32 -> 6` `tanh` MLP architecture (`stage1_mlp_benchmark.py`):
- **H1 units used**: `10 / 32`
- **H2 units used**: `8 / 32`
- **Nonzero parameters**: `1,240 / 20,486` (6.05% sparsity)
- **1,000-Episode Mean Score**: **456.33 ± 9.34** (Mean episode length: `5,544.6` steps) — up from `77.92` with raw manual indexing and **18× higher than 500M-step PPO (`25.3`)**.

### 2. Stage 2: `TransformerProgram` $\rightarrow$ `TransformerCircuitBackend`
Compiles symbolic algorithms and unseen program compositions directly into standard multi-head self-attention + ReLU MLP Transformer weights (`stage2_transformer_benchmark.py`), with **automatic residual-stream register allocation** and **zero-shot Architecture Transfer** across varying widths ($d \in \{64, 128, 256\}$), depths ($L \in \{6, 8\}$), and heads ($H \in \{4, 8\}$):

| Task | Difficulty | Compiled Circuit Mechanism | Small (`d=64, L=6, H=4`) Acc | Medium (`d=128, L=8, H=8`) Acc | Large (`d=256, L=8, H=8`) Acc | Nonzero Weights |
| --- | --- | --- | :---: | :---: | :---: | :---: |
| **1. Comparison ($x > y$)** | Easy | `GREATER_THAN` step comparator in L0 MLP | **100.0%** | **100.0%** | **100.0%** | 7 |
| **2. Maximum & Argmax** | Easy | `SCORE_MAX_BROADCAST` in L0 Attention | **100.0%** | **100.0%** | **100.0%** | 6 |
| **3. Counting Matches** | Medium | `EQUAL_PAIR` hat detector + `UNIFORM_MEAN` | **100.0%** | **100.0%** | **100.0%** | 13 |
| **4. Prefix Sum** | Medium | Causal `UNIFORM_MEAN` + `MULTIPLY_BY_ONEHOT` position rescaling | **100.0%** | **100.0%** | **100.0%** | 66 |
| **5. Sorting ($N=8$)** | Medium | 4-Layer Parallel Rank-and-Route (`MULTIPLY_BY_ONEHOT` $\rightarrow$ `UNIFORM_MEAN` $\rightarrow$ `COUNT_GREATER_THAN_VECTOR` $\rightarrow$ `SCALAR_TO_ONEHOT` $\rightarrow$ `DOT_MATCH`) | **100.0%** | **100.0%** | **100.0%** | 225 |
| **6. Graph Reachability** | Hard | 4-Hop BFS (`DOT_MATCH` adjacency propagation + `THRESHOLD_STEP`) | **100.0%** | **100.0%** | **100.0%** | 80 |
| **7. Shortest Path** | Hard | 6-Layer Tropical $(\min, +)$ Pivot Relaxation (`BROADCAST_K_DISTANCE` + `TROPICAL_MIN_PLUS_STEP`) | **100.0%** | **100.0%** | **100.0%** | 48 |
| **8. Compositional (Above Prefix Mean)** | Compositional | Causal `UNIFORM_MEAN` + `GREATER_THAN` + global `UNIFORM_MEAN` | **100.0%** | **100.0%** | **100.0%** | 11 |

### 3. Zero-Context LLM Semantic Compiler Verification (`llm_compiled_unseen_tasks.py`)
A fresh subagent with zero access to the benchmark solutions was given only `neural_compiler/ir.py` and `neural_compiler/transformer_backend.py` and asked to compile **two unseen compositional tasks** into a 1.18M-parameter `Transformer(d_model=128, n_layers=6, n_heads=4)`:
- **Unseen Task A (`Range_Count_In_Interval`, count $x_i \in (\text{low}, \text{high})$)**: **100.00% accuracy** (500/500 trials, `0.0` max error, **24** nonzero weights).
- **Unseen Task B (`Thresholded_Max_Distance_From_Source`, max node value among nodes reachable within 3 hops)**: **100.00% accuracy** (500/500 trials, `0.0` max error, **72** nonzero weights).

---

## Repository Structure

```
weight_space/
├── README.md
├── neural_compiler/                  # Core Three-Tier Neural Weight Compiler package
│   ├── __init__.py
│   ├── ir.py                         # Neural IR: MLPProgram & TransformerProgram DSL
│   ├── mlp_backend.py                # Stage 1 deterministic MLP weight synthesis backend
│   └── transformer_backend.py        # Stage 2 deterministic Transformer circuit synthesis backend
├── stage1_mlp_benchmark.py           # Stage 1 MinAtar Breakout compiler benchmark (456.33 mean score)
├── stage2_transformer_benchmark.py   # Stage 2 Transformer Circuit benchmark (8 tasks x 3 architectures)
├── llm_compiled_unseen_tasks.py      # Unseen compositional tasks compiled by zero-context LLM subagent
├── eval_exp5_breakout.py             # Evaluation script for Part I raw handwritten Breakout weights
├── blind_tasks/                      # Blind prompts and weights given to zero-context subagents in Part I
└── results/                          # Verbatim outputs from zero-context subagents in Part I
```

## Quick Start

```bash
# 1. Run Stage 2 Transformer Circuit Benchmark (8 algorithms across 3 Transformer sizes)
python3 stage2_transformer_benchmark.py

# 2. Run Zero-Context LLM-Compiled Unseen Compositional Tasks
python3 llm_compiled_unseen_tasks.py

# 3. Run Stage 1 Compiled MinAtar Breakout Policy (1,000 episodes)
python3 stage1_mlp_benchmark.py
```
