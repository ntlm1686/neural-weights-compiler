# Mahjong: is a 14-tile hand 4 melds + 1 pair?

## Result
`--dev`: **500/500** (1 dev run total). `{"correct": 500, "nonzero": 1907, "d_model": 256, "n_layers": 11, "n_heads": 1}`.
Extra self-check on 3000 hands from my own generator (random melds, 0-2 random tile replacements, 1055 complete,
checked against `_mj_complete`): 0 errors. The raw output was exactly 0.0 or 1.0 in every case.

## Algorithm (fixed rules I worked out myself; no tables)
1. Histogram c[0..33] of the hand.
2. For every candidate pair p (34 choices, run in parallel), take 2 copies of p out, then scan each suit
   left to right with a greedy rule. At tile i: r = c_i - 2[p==i] - k_{i-1} - k_{i-2}, where k_t is the number of chows
   starting at t. Rule: **k_i = r mod 3**. Why it holds: at the lowest remaining tile the number of chows k satisfies
   k ≡ r (mod 3). Three identical chows are the same tiles as three pungs, so k ∈ {0,1,2} without loss of generality,
   and the remaining r - k tiles form pungs. Fail when r < 0, or when a chow would start at number 8 or 9 (k_i > 0 at offset 7 or 8).
   Honours have no chows, so they fail when r mod 3 ≠ 0.
3. The hand is complete iff some candidate p gives no failure. The pair-removal step checks c_p ≥ 2, because r < 0
   otherwise. With 14 tiles and no failure there are exactly 4 melds.

## Implementation (weights written directly from formulas, `solution.py`)
- Sequence: 48 positions. Positions 0-13 hold the tile one-hot (34). Positions 14-47 are blank candidate positions.
  Every position has a 48-dim position one-hot. Candidate p sits at position 14+p, and its pair indicator is read
  linearly from the position one-hot. encode does no counting.
- L0 attention (1 head, Q=K=0): uniform attention over the 48 positions, with V = tile one-hot and W_O = 48·I. This puts
  the exact histogram c at every position.
- L0..L8 MLPs: step j of the scan, for the 3 suits at once (tile s*9+j), plus all 7 honours in L0. For each tile:
  1 unit ReLU(-r) adds to the penalty register. Three hat functions at r = 1, 2, 4 write k_i = 1, 2, 1 (r mod 3 on
  r ∈ {0..4}). At offsets 7 and 8 the hats also add k_i to the penalty, and for honours they add only to the penalty.
  All penalties are ≥ 0 and integer-valued, so once a branch has failed it stays failed even when later r values
  are garbage.
- L9 MLP: ok = ReLU(is_candidate − penalty), where is_candidate is the sum of the candidate position one-hots.
- L10 attention: uniform mean of ok × 48 = number of valid pair choices. The L10 MLP clamps it to [0,1] with
  ReLU(n) − ReLU(n−1). The output is read at the last position, and decode returns out > 0.5.
- Everything is exact: integer-valued piecewise-linear functions, and the only floating-point error is ×48/48.
  Nothing is approximate.
- Size: d_model 256 (about 188 used), 11 layers, 1 head, d_mlp 128 (at most 100 units used in a layer),
  1907 nonzero parameters.
- Primitives: a small unit allocator (`_MLP.unit`, `_MLP.hat`) in solution.py. Nothing outside this directory was
  changed, and `neural_compiler`'s IR was not used.

## Cost driver / blocker
Nothing blocked the task. The cost driver is the sequential carry of the greedy scan: depth is 9 MLP layers, one per
number within a suit. Width is cheap because the 34 pair choices go to separate positions, and the suits and honours
run in parallel inside each layer. The step that turns a counting problem into a linear chain of local rules is the
greedy "k = r mod 3" argument. Without it, the network would have to search over decompositions.

## Harder variant estimate (13-tile hand → list of winning tiles)
Add one more axis of parallelism: a position for each (added tile w, pair p), which is 34×34 = 1156 positions.
Alternatively, keep 34 positions per w and loop over p inside the MLP width (34 × ~100 units = 3400 > d_mlp), or
reuse the per-suit structure: a suit is OK if it decomposes with or without a pair. That needs 34 w-positions ×
(pair-in-suit-or-not branches) and about the same 9-layer depth. Then an attention per w aggregates over p, and the
output is a 34-way multi-label read at 34 positions. Estimate: about 12 layers, d_model ≤ 512, and either about 1.2k
positions or about 4 times the MLP width. The logic is unchanged and the cost grows only by the 34× branching.
