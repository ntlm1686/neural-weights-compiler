# multiply: exact 6x6-digit multiplication by a compiled Transformer

**Result: `--dev` 500/500** (2 dev runs, both 500/500). The model has d_model=128, 7 layers and 16 heads (12 used,
all in layer 0). The MLP width is 576 and there is no LayerNorm. 5,055 parameters are nonzero, out of 1.50 M total.
The weights are written directly from closed-form formulas in `solution.py`. There is no training and no search.
The only table is the implicit single-digit product identity.

## I/O
- `encode`: seq_len=24. Positions 0..5 hold the digits of x (least significant first), positions 6..11 hold the
  digits of y, and positions 12..23 are blank output slots. It writes two registers: `digit` (the raw digit value,
  0 at output slots) and `pos` (a 24-wide one-hot position code). Python does no arithmetic.
- `decode`: rounds `out` at positions 12..23 and joins the 12 digits by place value.

## Algorithm per layer
Residual registers: const_one, digit, pos(24), G(12 gathered digits), C(12 column values), g(12), p(12), out(1).

| Layer | Block | What it computes | Units |
|---|---|---|---|
| 0 | attn | Head i (i<12): Q = const, K = 60·pos_onehot[i], V = digit, so every position copies digit i into G[i]. | 12 heads |
| 0 | MLP | Column sums C[k] = Σ_{i+j=k} x_i y_j. Uses the exact identity a·b = Σ_{v=1..9} ReLU(b+10a−10v) − 10·Σ_{v=1..8} ReLU(a−v), which holds for integers a,b ∈ 0..9. The correction term for x_i is shared across its 6 products. | 372 |
| 1–3 | MLP | Carry-save normalisation: q_k = ⌊C_k/10⌋ as a sum of step functions, then C_k −= 10q_k and C_{k+1} += q_k, all columns in parallel. Per-column bounds go 486 → 57 → 14 → 10. | 576, 64, 20 |
| 4 | MLP | g_k = [C_k ≥ 10], p_k = [C_k = 9]. Both are binary. | 48 |
| 5 | MLP | Carry lookahead: carry_k = Σ_{m<k} ReLU(g_m + Σ_{m<j<k} p_j − (k−m−1)), an AND of binary flags. The terms are mutually exclusive, so the sum is an OR. Then C_k += carry_k and C_{k−1} −= 10·carry_k, which makes C_k the k-th digit. | 66 |
| 6 | MLP | Output slot 12+k computes out = Σ_k ReLU(C_k + 10·pos_onehot[12+k] − 10), so it picks out digit k. | 12 |

The final digit identity is digit_k = C_k + carry_k − 10·carry_{k+1}, which holds whenever C_k ≤ 10.

## Exactness and margins
- Every intermediate value is a small integer: at most 486 at layer 0, then at most 57, 14, 10, and finally digits.
  The values never contain powers of ten, so float64 has no precision problem.
- Step functions use ReLU(2(x−T)+1.5) − ReLU(2(x−T)+0.5). This is exactly 0 for x ≤ T−0.75 and exactly 1 for
  x ≥ T−0.25, so it tolerates ±0.25 error on integer inputs.
- Attention leakage per gathered digit is at most 23·e^{−60}·9 ≈ 2e−25.
- The observed maximum of |out − round(out)| is **0.0** on every test.

## Tests (local, not --dev)
- `test_local.py` ran 3,031 cases with 0 failures. It covered 3,000 random pairs plus 999999², 10⁵·10⁵, 999999·100001
  and every 6×6 factor pair of 10¹¹, 10¹¹±1 and 10¹²−1. Those factor pairs give long 0/9 runs and therefore long
  carry chains.
- `test_carry.py` ran layers 4–6 on 14,987 adversarial column vectors with C_k ∈ {0,5,9,10} and many 9-runs ending
  in a 10. It found 0 failures, so carry chains up to 11 columns are covered.
- `test_scale.py` uses the same builder with `build_model(nd)`:
  - nd=10: d_model 128, 8 layers, MLP width 1,602, 14k nonzero parameters. 0/302 failures.
  - **nd=20: d_model 256, 10 layers, MLP width 6,444, 61k nonzero parameters. 0/302 failures**, including (10²⁰−1)².

## What limits scale
The parameter costs are:
- Gathering needs 2n attention heads, which is ⌈2n/16⌉ layers.
- The product layer needs about 9n² MLP units.
- The first normalisation layer needs about 16n² units.
- Normalisation needs about log₁₀(81n) layers.
- The carry-lookahead layer needs (2n)²/2 units, and each unit has up to 2n inputs.
- The residual width grows as about 8n (dominated by the one-hot position code and the G, C, g, p registers).

So depth grows like O(log n) and width like O(n²). Precision is not the bottleneck, because every value stays a small
integer. At 20 digits everything fits easily within d_model ≤ 1024 and 96 layers; the task as specified does not limit
MLP width. With the conventional d_mlp = 4·d_model = 1024 cap, the 9n² product units and 16n² normalisation units
would have to be split across more layers (about 4 product + 7 first-normalisation layers at n=20, i.e. ~19 layers total), which still fits the layer budget.

The real ceiling is the d_model ≤ 1024 limit, because the scheme keeps every digit at every position. It is reached
at roughly n ≈ 100 digits.

A version that scales further would be column-local:
- Each output position k gathers only its own diagonal pairs (x_i, y_{k−i}) through attention with position-difference
  keys.
- Neighbour shifts are done by attention instead of register re-indexing.
- The carry is resolved with a single "most recent non-propagate position" attention head (as in the stage-3 adder)
  instead of the O(n²) AND layer.

That version needs O(n) heads over several layers for the diagonal gather and O(1) residual width per position. It was
not needed for 6 digits.

A design that fails at scale is the tempting shortcut of one scalar prefix value s_k = Σ c_m 10^m. It is exact only
while s < 2^53 ≈ 9·10¹⁵, so it breaks after about 15 result digits. This solution avoids that by never forming such a
scalar.

## Files
`solution.py` contains the builder (parameterised by `nd`), `encode` and `decode`. `test_local.py`, `test_carry.py` and
`test_scale.py` are the local worst-case and scaling tests. No files outside this directory were modified, and no
new modules were added.
