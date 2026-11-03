# Game of Life (8x8 torus, one step): compiled Transformer

## Result
`python3 experiments/compiler_frontier/check.py life --dev` gives
`{"correct": 500, "total": 500, "nonzero": 94, "total_params": 2028, "d_model": 20, "n_layers": 1, "n_heads": 1}`.
I made **one** --dev run. I also ran a separate stress test (2000 boards with density from 0 to 1, including the all-0 and all-1 boards; seed 123, not the dev seed): 0 wrong. The largest |raw output - target| was **0.0** in float64.

## Architecture
d_model = 20, n_layers = 1, n_heads = 1 (d_head = 20), d_mlp = 8. 94 nonzero parameters.
The sequence has 64 tokens, one per cell; token i is cell (i // 8, i % 8).
Residual registers: `const_one`(1), `alive`(1), `row_oh`(8), `col_oh`(8), `box`(1), `next`(1).

`encode` writes only the cell value and the row/column one-hots of the position index. `decode` thresholds at 0.5 and reshapes to 8x8.

## Algorithm
Let S be the 3x3 box sum, with the cell itself included. A cell is alive next step iff S == 3, or (alive and S == 4).
This is the standard B3/S23 rule, restated using S.

**L0 attention (1 head): torus box mean.**
- The query maps the row one-hot of r through an 8x8 circulant band matrix. The result is the indicator of {r-1, r, r+1} mod 8. The column one-hot is handled the same way.
- The key is the row and column one-hots of the key cell.
- Score(i, j) = 40 * ([row_j in band(r_i)] + [col_j in band(c_i)]).
- The 9 cells of the torus 3x3 box score 80. Every other cell scores at most 40.
- V = `alive`. The head writes into `box`, so box = S / 9.
- The pattern is approximate only in the softmax: the off-box mass is at most 55 * e^-40 / 9, about 2.6e-17, which is below float64 resolution here.
- No per-offset heads and no shifted copies are needed. A single head counts all 8 neighbours plus self, because the box is a product of two 1-D bands and the score adds a row term and a column term.

**L0 MLP (7 active ReLU units): integer lookup table.**
- z = 9*box + 10*alive = S + 10*alive is an integer. When the cell is dead, z is in 0..8; when it is alive, z is in 11..19.
- f(z) = 1 for z in {3, 13, 14}, otherwise 0.
- f is written as the exact piecewise-linear interpolant f(z) = sum_k Δslope_k * ReLU(z - z_k), with knots at 2, 3, 4, 12, 13, 14 and 15. It is exact on integers.
- f is the rule table on (S, own state), which is allowed. It is not fitted: the knots and slopes come directly from the rule.
- The output goes to `next`.

## Exact vs approximate
- **MLP:** exact on integer z.
- **Attention:** the only approximation is softmax leakage, about 1e-17. The reachable z values are spaced 1 apart and each hat function has unit width, so the decision margin is 0.5 against an error of about 1e-15.

## Primitives added (`life_primitives.py`)
- `torus_box_mean_attn`: writes Q/K/V/O for the band-matrix box-mean head described above. Works for any n_rows x n_cols torus.
- `integer_lut_mlp`: writes W_fc/b_fc/W_proj for the ReLU-knot expansion of an arbitrary function on consecutive integers of a linear combination of registers.

The registers and zero weights are allocated by `TransformerCircuitBackend.compile` on a `TransformerProgram` with an empty op list. The two primitives then fill in the weights.

## Limitations and scaling
- **Larger N x N tori:** the same 1 layer and 1 head work. Requirements:
  - d_model = 2N + 5 and d_head >= 2N + 1. For example, N = 32 needs d_model = 69. With d_model <= 512 the limit is about N <= 253 under the one-head layout.
  - Sequence length N^2.
  - The margin must exceed ln(N^2) + some slack. A margin of 40 is fine up to very large N.
- **Non-torus boards:** with dead cells beyond the edge, the band matrices would drop the wrap entries. The average is then still over 9 slots only if out-of-board cells count as 0, which a mean over the fewer visible cells would not give. Fix this by using sum = mean * (box size), with a box size known per position, or by giving V the value 9*alive and adding a constant-0 "pad" key with the same score.
- **Larger neighbourhoods or other totalistic rules:** change the band width and the LUT points.
