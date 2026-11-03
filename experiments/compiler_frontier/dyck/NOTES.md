# Dyck-1 (length 16) via a hand-constructed Transformer

## Algorithm
Balanced iff (a) no prefix depth d_i is negative and (b) the final depth d_15 = 0.
Uniform causal attention computes prefix means, i.e. d_i/(i+1). A second uniform head over a
"first position" indicator gives 1/(i+1) inside the model. These two values are enough to
build exact sign tests on d_i, because d_i is an integer.

## Encoding / decoding
- `x_i` = +1 for '(' and -1 for ')' (fixed per-symbol code).
- `first_i` = 1 if i == 0 else 0 (a position code that depends only on the index).
- `const_one` is set by the backend. No other inputs.
- decode: `raw > 0` on the last-position output register (`output_mode="last_pos_register"`).

## Architecture
d_model=16, n_layers=2, n_heads=2 (d_head=8), d_mlp=4, all heads causal, no LayerNorm.
Residual registers (1 dim each): const_one, x, first, m, p, bad, out.
Nonzero parameters: **23** (2472 total). Weights are written directly into the
`ExecutableTransformer` weight dict from formulas in `solution.py`.

| Layer | Component | Function |
|---|---|---|
| L0 H0 | Q=K=0, causal, V=x, O->m | m_i = mean_{j<=i} x_j = d_i/(i+1) |
| L0 H1 | Q=K=0, causal, V=first, O->p | p_i = 1/(i+1) |
| L0 MLP (4 ReLUs) | a=32(-m - p/2), b=32(m - p/2); step(a)=ReLU(a)-ReLU(a-1), step(b) likewise | bad_i = [d_i<0]; out_i = 0.5 - [d_i != 0] |
| L1 H0 | Q=K=0, causal, V=bad, O->out with gain -16 | out_15 -= 16 * mean bad = number of positions with negative depth |

Output at position 15 is `0.5 - [d_15 != 0] - #{i : d_i < 0}`, so it is +0.5 when the string is balanced and <= -0.5 otherwise.

## Exactness / margins
- Uniform attention is exact: Q=K=0 gives equal scores, and the causal mask makes the weights 1/(i+1). There is no sharpness approximation, and the only error is float rounding (~1e-16).
- Step functions: a = 32(-d_i - 1/2)/(i+1). If d_i <= -1 then a >= 32*(1/2)/16 = 1, so the step saturates at 1. If d_i >= 0 then a <= -16/(i+1) < 0, so the step is 0. The same holds for b. The step outputs are therefore exactly 0 or 1.
- Exhaustive check over all 2^16 = 65,536 strings: 0 errors. Balanced strings always give 0.5, and unbalanced strings give at most -0.5.

## Primitives added
`uniform_prefix_mean(w, layer, head, src, dst, scale)` in `solution.py` builds a causal head with zero Q/K that writes scale * (prefix mean of src) into dst. The weight-dict builder is also local. No existing files were modified.

## Dev runs
1 `--dev` run: `{"correct": 500, "total": 500, "nonzero": 23, d_model 16, n_layers 2, n_heads 2}`.
(I also ran the default-seed check once (500/500) and did the exhaustive verification above.)

## Limitations / longer lengths
The step gain K must satisfy K >= 2n for length n: the margin is (1/2)/(i+1) >= 1/(2n). The final-count gain must be n.
For n = 16 this means K = 32. For longer n, set K = 2n and the count gain to n. Float64 keeps this exact for n up to about 1e6 or more.
The construction is length-specific only through these two constants (it reads the answer at the last position).
It does not handle Dyck-k with several bracket types, which would need a stack-matching mechanism.
