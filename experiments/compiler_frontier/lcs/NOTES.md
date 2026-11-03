# LCS length (two length-6 strings over {0,1,2,3}): compiled Transformer

## Result
`check.py lcs --dev`: **500/500**. Dev runs used: **1**.
Before that, a self-test on 2000 random cases (seed 12345, not a dev seed) plus hand-picked edge cases
(identical strings, disjoint alphabets, reversed pairs) gave 0 errors. The largest |raw output - true LCS|
was 0.0 in float64.

Architecture: d_model = 192, n_layers = 12, n_heads = 3 (d_head = 64), d_mlp = 8, no LayerNorm.
Nonzero parameters: 2667 (of 1,817,952 allocated).

## Encoding (input only, no computation)
- One sequence position per DP cell (i, j), p = 6i + j (36 positions), plus a "null" position p = 36.
- `tok_a[p]` = one-hot of the raw symbol a[i]. `tok_b[p]` = one-hot of the raw symbol b[j]. Both are zero at the null position.
- `cell[p]` = one-hot of the position index p (37 dims), which is a fixed function of the position alone.
- `decode` reads register `L` at position 35 (cell (5,5)) and rounds it.

## Algorithm
The recurrence is L(i,j) = eq(i,j) ? L(i-1,j-1)+1 : max(L(i-1,j), L(i,j-1)), with out-of-range cells = 0.
Every layer relaxes this recurrence once at all cells in parallel (Jacobi iteration). Cell (i,j) only depends on
cells on earlier anti-diagonals, so it becomes exact after i+j+1 relaxations and stays exact after that.
11 relaxations are therefore enough for (5,5).

- **Layer 0 MLP (4 hidden units):** eq = sum_c ReLU(a_c + b_c - 1). This is exact for one-hots. Layer 0 attention is unused.
- **Layers 1..11, attention (3 heads, one per neighbour: up, left, diag):** Q = 400 * onehot(neighbour(p)) and
  K = onehot(p), both linear in the `cell` register. Q is a fixed permutation/shift matrix. If a neighbour is out of range,
  the query points to the null position, whose L stays 0 forever. V = the L register. W_O writes the fetched value into a
  fresh per-layer scratch register (`up_l`, `left_l`, `diag_l`; 33 dims total). Fresh registers avoid having to clear accumulations.
- **Layers 1..11, MLP (4 hidden units):** this computes the whole recurrence in one ReLU layer, with BIG = 20 > 7:
  `new = ReLU(up - BIG*eq) + ReLU(left - up - BIG*eq) + ReLU(diag + 1 - BIG + BIG*eq)`.
  If eq = 0, the sum is up + ReLU(left-up) = max(up, left). If eq = 1, it is diag + 1.
  A fourth unit, ReLU(L_old), is projected with weight -1, so the residual update is new - L_old. This overwrites L.
  At the null position eq = 0 and all its neighbours are the null position itself, so its L stays 0.

## Exact vs approximate
- The MLPs are exact piecewise-linear identities on integer inputs. Every ReLU input is either an integer value or below -13.
- Attention is the only approximate part. The score gap between the target key and every other key is 400/sqrt(64) = 50.
  So the weight leaking to wrong positions is at most 36 * e^-50 ≈ 7e-21 per fetch, times values ≤ 6.
  This is far below float64 resolution at these magnitudes, and the observed error was exactly 0.
  The rounding margin in `decode` is 0.5.

## Primitives added
None from the compiler library. `solution.py` writes the ExecutableTransformer weight dict directly from the formulas above:
one-hot shift-attention ("fetch neighbour cell") and the one-layer "LCS cell update" ReLU MLP.
No training, fitting or search was used.

## Limitations / scaling
- For strings of length n and m: n*m + 1 positions, n+m-1 relaxation layers (+1 for eq), 3 heads,
  and d_head ≥ n*m + 1 for the one-hot cell ids.
- The one-hot query could be replaced by factored (row, col) one-hots with an additive score. That would make d_head
  O(n+m), keeping a gap of one unit of scale between exact and partial matches.
- Fresh scratch registers cost 3 dims per layer. For long inputs, the MLP could clear them instead
  (2 extra hidden units per register), giving O(1) residual width per step.
- n = m = 6 uses 12 of 48 layers. Within these limits, the current design reaches roughly n = m = 23 if fetches are
  factored and the depth is the binding constraint. An alphabet of size V only changes the eq MLP (V hidden units).
