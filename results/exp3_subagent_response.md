# Experiment 3 Blind Subagent Response (`exp3_blocks_mystery.npz` — Middle 4 Transformer Blocks Only)

Observations:
1. Subspace separation of token embeddings (27D) vs. positional embeddings in Layer 0 (`ln1_w.0`, `W_v.0`, `W_q.0`, `W_k.0`):
 - `ln1_w.0` (mean `0.268`, std `0.215`, min `0.017`, max `1.013`) is strongly bimodal: 217 of 384 coordinates are scaled down (`< 0.20`), while 76 coordinates remain large (`> 0.40`).
 - Stacking the raw value matrices `W_v.0` of `L0H1`, `L0H2`, `L0H3`, and `L0H5` ($256 \times 384$) reveals a sharp spectral cliff between the 27th and 28th singular values: `sv[0..26] = [3.43, 3.34, 3.24, ..., 1.98, 1.88, 1.84]` followed by `sv[27..30] = [1.29, 1.20, 1.11, 1.09]`. Since `VOCAB` has 28 characters, centered token embeddings span an exact 27-dimensional subspace $U_{\text{in}} \subset \mathbb{R}^{384}$ on which `ln1_w.0` has norm `0.18..0.24`.
 - In Layer 0, `L0H1` and `L0H5` put `87.7%` and `84.8%` of their raw `W_v.0` Frobenius energy into $U_{\text{in}}$, while putting `96.5%` and `98.0%` of their `ln1_w.0`-scaled `W_q.0` energy and `98.2%` and `98.9%` of their scaled `W_k.0` energy into the orthogonal positional complement $U_{\text{in}}^\perp$. Thus `L0H5` and `L0H1` attend purely by position and move token identities into orthogonal 27D subspaces of the residual stream.

2. Relative positional lags of `L0H5` ('previous') vs. `L0H1` ('local'):
 - Restricted to the 30D positional subspace $U_{\text{pos30}} \subset U_{\text{in}}^\perp$, `QK = (W_q.0 * ln1_w.0)^T (W_k.0 * ln1_w.0) / 8` decomposes into 2D complex-conjugate Fourier rotation blocks. Across all shared positional Fourier modes, `L0H5` rotates by `1x` the base frequency angle, whereas `L0H1` rotates the exact same planes by `2.3x` to `2.6x` the angle. This identifies `L0H5` as lag-1 (`'previous'`) and `L0H1` as lag-2..3 (`'local'`).

3. Output logit subspace (27D) in `W_proj.3` and the induction ('copy') circuit in Layers 2 and 3:
 - `W_proj.3` has a sharp singular value drop after index 27 (`sv[0..27] = [28.18, 20.29, ..., 8.57, 7.93]`), concentrating `92.5%` of its Frobenius norm into the 27D row space $U_{\text{out}}$ of `W_out`.
 - Projecting every head's `OV` matrix into $C^{(l,h)} = U_{\text{out}}^T OV^{(l,h)} U_{\text{in}}$ ($27 \times 27$) reveals that `L3H1`, `L3H5`, `L2H1`, and `L3H0` implement the token-copying map, with direct token-copying traces `+27.17` (`L3H1`), `+25.34` (`L3H5`), `+7.39` (`L3H0`), and `+5.50` (`L2H1`).
 - Testing QK induction composition confirms that `L2H1`, `L3H1`, `L3H5`, and `L3H0` match the current token at query $i$ ($U_{\text{in}}$) with the `L0H5`-copied previous token at key $j$ ($U_{05}$): $\text{tr}(U_{\text{in}}^T QK^{(l,h)} U_{05}) = +0.852$ (`L2H1`), `+0.605` (`L3H1`), `+0.864` (`L3H5`), and `+0.226` (`L3H0`).

4. Exact recovery of the 28 character simplex vertices in `W_fc.0` and the $28 \times 28$ bigram transition table:
 - Projecting the 1,536 rows of `W_fc.0 * ln2_w.0` onto $U_{\text{in}}$ and cosine clustering yields all 28 pure character clusters $c \in \{0, \dots, 27\}$ forming a 27-simplex plus a 4-neuron whitespace disjunction cluster (`'\n'` or `' '`).
 - Because trailing spaces were removed in normalization, `' ' -> '\n'` never occurs, uniquely identifying `Tok 10` as `' '` and `Tok 9` as `'\n'` via the single most negative entry `T_comb[10, 9] = -0.79`.
 - `T_comb[9, 10]` (`'\n' -> ' '`) is strongly positive (`+0.14`), and the bigram `(' ', ' ')` (consecutive leading spaces from code indentation) has `12` dedicated bigram detector neurons in `W_fc.0`—the `#1` highest bigram count in the entire network—which uniquely identifies `python` (where leading indentation spaces are preserved).

Heads:
L0H0: 'broad'
L0H1: 'local'
L0H2: 'broad'
L0H3: 'broad'
L0H4: 'broad'
L0H5: 'previous'
L1H0: 'local'
L1H1: 'local'
L1H2: 'local'
L1H3: 'local'
L1H4: 'local'
L1H5: 'local'
L2H0: 'local'
L2H1: 'copy'
L2H2: 'previous'
L2H3: 'local'
L2H4: 'local'
L2H5: 'broad'
L3H0: 'copy'
L3H1: 'copy'
L3H2: 'broad'
L3H3: 'broad'
L3H4: 'broad'
L3H5: 'copy'

corpus: python
order: as is
probabilities: python=0.96 romaji=0.01 english=0.03; as is=0.99 shuffled=0.01
