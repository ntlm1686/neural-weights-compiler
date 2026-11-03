# Poker showdown: compiled Transformer

**Result:** `--dev` gives 500/500. The model has d_model=512, 7 layers and 2 heads (d_head=256), with d_mlp=452 and no LayerNorm. It has 9,630 nonzero parameters out of 10.6M allocated.

I made 1 `--dev` run. Before it, I checked the model against the reference `_hand7_rank` in three ways:
- **Random deals:** 2,000 deals from `gen_poker` with a separate seed. All were correct, and the raw output was exactly an integer (max |raw - round| = 0).
- **Biased deals for rare categories:** 3,671 deals biased toward few ranks, few suits or runs of consecutive ranks, including A-5. Every one of the 9 categories occurs at least 95 times. All were correct.
  - I also sorted the 7,342 per-player scores `S` by the reference rank tuple. `S` is strictly increasing between different tuples and equal for equal tuples, with 0 violations. So the score is an exact numeric version of the reference ordering on all these hands.
- **Hand-written cases:** I checked these one at a time:
  - wheel vs 6-high straight
  - wheel straight flush vs straight
  - counterfeited two pair (three pairs on 7 cards)
  - board plays (tie)
  - kicker tie
  - flush vs straight
  - quads on the board with an ace-kicker tie

## Encoding (encode/decode)

`encode` uses 9 positions, one per card. Each position holds a rank one-hot (13), a suit one-hot (4) and an owner one-hot (A hole / B hole / board). Python does nothing else.

`decode` returns `round(out[pos 0])`.

## Hand strength as a number

For each player P, the network computes the score `S_P = c * 2^26 + T_c`, where c is the best category (0 = high card … 8 = straight flush) and `T_c < 2^26` is that category's tie-break.

The key trick is how tie-breaks over sets of ranks are written. A set of k ranks is encoded as `sum 2^r`. For sets of the same size, comparing these sums gives the same result as comparing the sorted ranks lexicographically, so kicker comparison becomes a single integer.

A primary rank (pair, trips or quad rank, or the set of the top two pairs) is multiplied by `KB = 2^13`. This puts it above all kickers, because any kicker sum is less than 2^13.

| c | category | T_c |
|---|---|---|
| 0 | high | top-5 of present ranks, as `sum 2^r` |
| 1 | pair | `KB*r_pair` + top-3 of singles |
| 2 | two pair | `KB*sum 2^r` (top-2 pair ranks) + top-1 of the remaining ranks (this handles counterfeiting and a third pair used as kicker) |
| 3 | trips | `KB*r_trips` + top-2 of the other ranks |
| 4 | straight | `2^h` of the highest straight (h = 0 is the wheel) |
| 5 | flush | top-5 of the flush-suit ranks |
| 6 | full house | `KB*2^r` (top trips rank) + top-1 of the pair ranks, excluding that trips rank (a second trips counts as the pair) |
| 7 | quads | `KB*r_quad` + top-1 of the other ranks |
| 8 | straight flush | `2^h` of the highest straight flush |

`T_c` only needs to be correct when c is the best category. Each "contains category c" indicator `I_c` only has to be sound: it may fire for a hand that also contains a higher category, because the gate below picks the highest one.

## Layers (all exact integer arithmetic; every ReLU step is applied to integer-valued inputs)

Notation: P is the player, r is a rank index (0..12, where 12 is the ace), s is a suit and h is a straight's high-card index.

- **L0 MLP:** `x_P[r,s] = ReLU(rank_r + suit_s + owner_P + owner_board - 2)`. This is the per-card visible indicator (104 units).
- **L1 attention:** two heads (one per player) with Q = K = 0, so attention is uniform over the 9 positions. V = x, and `W_O = 9`. This gives the 52-bit card set `X_P` for each player's 7 cards.
- **L1 MLP:** integer steps `[c >= k] = ReLU(c-k+1) - ReLU(c-k)`.
  - `p_k[r] = [count_r >= k]` for k = 1..4.
  - Flush suit: `fl[s] = [suitcount_s >= 5]`.
- **L2 MLP:**
  - Flush-suit ranks: `y[r] = sum_s ReLU(X[r,s] + fl[s] - 1)`.
  - Straights: `st[h] = ReLU(sum of p1 over the 5 ranks - 4)`, with the wheel using {A,2,3,4,5}.
  - Indicators: pair, two pair, trips and quads (steps on sums of p2, p2, p3 and p4).
  - All top-k selections that depend only on p (see the primitive below). These write `2^r`-weighted sums straight into the `T` registers. They also write the sets top-2 pairs (`t2p`) and top-1 trips (`t1p3`) for the next layer.
- **L3 MLP:**
  - Straight flushes: `sf[h]` from y.
  - Top-5 of y (flush tie-break).
  - Top-1 of st (straight tie-break).
  - Two-pair kicker: top-1 of `p1 - t2p`.
  - Full-house pair: top-1 of `p2 - t1p3`.
  - Indicators: `I_straight`, and `I_fh = ReLU(I_trips + I_twopair - 1)`.
- **L4 MLP:** top-1 of sf, and `I_sf`.
- **L5 MLP:** one gated unit per category, `ReLU(c*2^26 + T_c + lin_c - 2^30*(1 - I_c) - 2^30 * sum_{c'>c} I_c')`, summed into `S_P`. Here `lin_c` is the linear primary-rank term for pair, trips and quads.
- **L6 MLP:** `out = ReLU(d) - ReLU(d-1) - ReLU(-d) + ReLU(-d-1)` with `d = S_A - S_B`, which gives -1, 0 or +1.

## Primitive added (`topk_units` in solution.py)

This primitive selects the top-k members of a binary set in one ReLU layer with 2 units per element:

`t_r = ReLU(b_r - a_r) - ReLU(-a_r)`, where `a_r = #{r' > r : b_r'} - (k-1)`

This equals `b_r` when `a_r <= 0` and 0 otherwise. The inputs `b_r` can be any linear forms that are binary. For example, `p1 - p2` gives the singletons, and `p1 - t2p` gives the ranks outside the top-2 pairs.

The other primitives are:
- the integer step `[s >= k]` (2 units);
- AND as `ReLU(sum - (n-1))`;
- a "uniform attention × 9" head for set aggregation.

Weights are written from these formulas into an `ExecutableTransformer` weight dict by a small `Builder`, also in solution.py.

## Exact vs approximate; margins

Every quantity is an integer in exact arithmetic. The only rounding comes from softmax weights of 1/9 in floating point, which are then multiplied by 9. It is about 1e-16 relative, and even multiplied by the largest constant (2^30) it stays below 1e-6. On the 2,000 random deals checked, the observed output error was 0. In the decision, `d` is either 0 or has |d| >= 1, so the margin is 0.5 against a rounding error below 1e-6.

## Cost driver / blocker

There was no real blocker. Once the 52-bit card set is in every position (one attention layer), hand evaluation is a fixed-depth Boolean/integer circuit. The parts that look hard are in fact cheap:
- Kicker comparison is solved by the `sum 2^r` encoding.
- "Best 5 of 7" is solved by single-layer top-k selection.
- The category hierarchy is solved by gating each category with a large constant.

Width is driven by L2 (452 hidden units). The two players are evaluated in parallel, and each needs 6 top-k selections over 13 ranks, at 26 units each. Depth (7 layers) is driven by the dependency chain counts → flush suit → flush ranks → straight flush → top-1 straight flush → gate → sign.
