# Tic-tac-toe game value: compiled Transformer (no training)

## Result
- `--dev` runs: 1. Result: `{"correct": 500, "total": 500, "nonzero": 54216, "total_params": 37518048, "d_model": 512, "n_layers": 18, "n_heads": 4}`
- I also tested on all 4,520 reachable non-terminal positions against `ttt_value` (reference used only for testing; see the bottom of this file).
- Architecture: d_model 512, 18 layers, 4 heads (d_head 128), d_mlp 1008 (only the widest layer uses all of it), seq_len 383, no causal mask. Nonzero parameters: 54,216.
- Runtime is about 0.35 s per example in numpy, so a 500-case check takes about 3 minutes.

## Algorithm: full minimax search over all continuations of the input board
The network runs a complete retrograde minimax evaluation of the game tree rooted at the input board. It does not use any evaluation heuristic.

A search node is a pair (A, B). A is the set of cells the root's side to move ("mover") has added, and B is the set the opponent has added. The node's board is root + A + B.
- **Tokens**: one token for each subset A of the 9 cells with |A| <= 5 (382 tokens), plus one sink token. Token index -> A is a fixed bijection that enumerates subsets of {0..8} by size and then in lexicographic order. This is a position code. It does not enumerate positions or values.
- **Residual dims inside a token**: register `V` (256 dims) is indexed by subsets B with |B| <= 4, grouped into "classes" by |B|. At a token with |A| = a:
  - class a holds MO(A,B): the node with the mover to move (|B| = |A|). It uses the g-encoding g(-1)=1, g(0)=eps, g(+1)=0.
  - class a-1 holds OP(A,B): the node with the opponent to move (|B| = |A|-1). It uses the f-encoding f(+1)=1, f(0)=eps, f(-1)=0.
  - Both encodings are 0 for invalid nodes (A or B overlap each other or the root's stones). eps = 0.02.
  - All values are from the root mover's point of view.
- **Recurrences** (the rules of the game only):
  - OP(A,B) = +1 if root-mover stones + A contain a line. Otherwise 0 if the board is full. Otherwise min over legal c of MO(A, B+c). This min is computed inside the token with a fixed linear map over subset dims.
  - MO(A,B) = -1 if root-opponent stones + B contain a line. Otherwise 0 if the board is full. Otherwise max over legal c of OP(A+c, B). This max is computed across tokens: token A attends uniformly to its children tokens A+c.
  - Answer: MO(∅,∅) at token 0.
- **Exact max/min by averaging**: values are in {-1,0,1}. The mean over children of f(OP) is >= 1/9 if any child is +1, lies in [eps/9, eps] if the best child is 0, and is 0 if all children are -1 (illegal children contribute 0). Two clamped ReLU steps at thresholds 0.04 and 0.0005 recover the max exactly. In the same way, the sum of g(MO) over the B+c children gives the min (thresholds 0.5 and 0.005).
- **Order**: levels are processed from deepest to shallowest. OP at |A|=5, then for k = 4..0: MO at level k, then OP at level k. That is 10 half-plies, which covers the maximum game length of 9 plies from the empty board.

## Layer by layer
| Layer | Attention | MLP |
|---|---|---|
| L0 | H0 broadcast: every token attends uniformly to the 9 cell tokens (key `iscell`, score 40). The value is each cell's X/O one-hot in its own slot. ×9 gives `X[9]`, `O[9]` at every token. | Side to move from diff = ΣX−ΣO ∈ {0,1}. Mover stones `M_c = ReLU(X_c−diff) + ReLU(O_c+diff−1)`, opponent stones `R_c` symmetric. `nroh` = one-hot of the stone count (hat functions). |
| L1 | – | `aocc = Σ_c ReLU(A_c+occ_c−1)` (whether token A hits an occupied cell). `nml = Σ_lines ReLU(Σ_{c∈L}(A_c+M_c)−2)` (number of mover lines). |
| L2 | – | OP level 5 (class 4). Only relevant when the root is the empty board. |
| per k=4..0: layer a | H0 "children": score = 40·(|A∩A'| + [lvl(A')=lvl(A)+1] − lvl(A) − 1). This is 0 exactly for the children A' = A+c and ≤ −40 otherwise. The sink token catches level-5 queries. It copies V class k (C(9,k) ≤ 126 dims) into the same dims, so the mean over children lands in V class k of level-k tokens. | `NL[B]` = number of opponent lines in R+B, for every k-subset B (scratch register). |
| layer b | – | MO step: writes g(MO) into V class k. It is gated by level k (penalty 1e5·(1−lvl_k)), by validity (Σ_{c∈B}(occ_c+A_c)+aocc), by an opponent line (NL), and by full board (`nroh[9−2k]`). It also clears the raw mean and the NL scratch. |
| layer c (k≥1) | – | OP step for level k: the sum over B+c of g(MO) gives f(OP) in V class k−1. Gated by level, validity, mover line (`nml`) and full board (`nroh[10−2k]`). |
| L17 | – | OUT = 1 − step(g ≥ 0.005) − step(g ≥ 0.3) applied to g = V[class 0] (token 0), giving the value in {−1,0,+1}. Output mode `first_pos_register`. |

Because every MLP is gated by level, dims a token does not use stay at 0. This means the attention copy of class k only produces nonzero values at the level-k parents. Parents at level k−1 also receive class-k entries from their children, but those entries are still 0 at that point and are never used.

## Exact vs approximate
- Everything is piecewise-linear and exact on the integer and {0, eps, 1} values that occur. All step functions have margins of at least 2× (e.g. 1/9 vs eps=0.02).
- The only approximation is softmax leakage: about 380·e^-40 ≈ 1e-15 relative on non-selected tokens. The final output is an exact integer to within about 1e-12.

## encode / decode
- `encode`: at token c (c = 0..8), register `xin` slot c = [cell c is X] and `oin` slot c = [cell c is O]. It also writes position codes that are fixed per index: `A` (9-dim membership of the token's subset), `lvl` (one-hot of |A|, 7 dims with the sink at 6), and `iscell` (tokens 0..8). It does no computation over the board, so it does not compute whose turn it is either.
- `decode`: `round(raw)`.

## Primitives added
I did not use `TransformerCircuitBackend`. The weight dict is built directly from formulas in `solution.py`:
- `MLPLayer.unit` / `MLPLayer.step`: a generic ReLU unit, and a clamped step `clamp01(slope·(x−θ))` built from 2 ReLUs with additive gating penalties.
- A "children" attention head: subset-inclusion plus level-match score.
- A uniform broadcast head.

## Knowledge from enumeration?
- None is stored in the weights. The weights encode only:
  - the 8 winning lines,
  - the subset lattice of the 9 cells (which subsets contain which),
  - the move-alternation rules.
- No position values, no position tables and no fitted parameters. All constants (eps, thresholds, slopes, penalties) were chosen by reasoning, without search.
- The reference solver was used only for testing.

## Limitations
- The search is exhaustive (5,478-ish node slots for the empty board). It is specific to a 3×3 board with this token layout, and it would not scale to larger games.
- It assumes the input is reachable and non-terminal, as the task specifies. A terminal root is not detected.
