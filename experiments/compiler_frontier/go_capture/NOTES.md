# Go move with captures (9x9): compiled Transformer

## Result
- `solution.py` (submitted): `{"correct": 500, "total": 500, "nonzero": 14423, "d_model": 80, "n_layers": 81, "n_heads": 4}` on `--dev`.
- `solution_logdepth.py` (alternative, same task, also within limits): `{"correct": 500, "total": 500, "nonzero": 13670, "d_model": 1024, "n_layers": 10, "n_heads": 8}` on `--dev`.
- --dev runs used: **2** (one per file). Both are exact for **every** legal 9x9 input, not only small groups (argument below). No bound on group size or shape.

## Encoding / decoding (no board computation in Python)
Sequence = 81 board points (row-major) + 1 blank "null" token (index 81).
Registers placed by `encode`: state one-hot (empty/black/white), move flag, null flag, row one-hot (9), col one-hot (9),
and in the log-depth variant also the 81-dim position one-hot. `decode` = `rint(raw[:81]).reshape(9,9)`.

## Algorithm A: `solution.py` — BFS of "has a liberty", one layer per step
Let empty' = empty − move (exact: the move point is empty), white unchanged by the move.

    alive_0 = 0
    alive_{k+1}[p] = white[p] AND ( Σ_{q ∈ N4(p)} (empty'[q] + alive_k[q]) ≥ 1 )
    out = black + move − alive_K

`alive_k` = white stones within in-group distance k−1 of a stone touching a liberty (monotone). A white group
has ≤ 80 stones, so its in-group distance is ≤ 79 and K = 80 iterations suffice for **any** group; we use 81 layers.
Captured white stones are exactly the white stones with alive = 0 → they become 0, survivors stay −1.

Per layer (identical layers, `neighbour_read_head` + `alive_update_mlp` primitives):
- **Attention, 4 heads (up/down/left/right)**: query at (r,c) has a shifted row one-hot (r+dr) and shifted col
  one-hot (c+dc) and a constant; keys are row/col one-hots and the null flag.
  score = A·[row match] + A·[col match] + 1.5A·[key is null], A = 60.
  The true neighbour scores 2A, the null token 1.5A (wins when the neighbour is off-board because the shifted
  one-hot is zero), every other position ≤ A. Margin 0.5A = 30 → leaked weight ≤ 81·e^−30 ≈ 8e−12.
  Value = empty − move + alive (linear), all 4 heads add into scratch register `nsum` ∈ {0..4}.
- **MLP, 6 ReLU units**: `alive += [white]·clamp01(2(nsum − 0.25)) − alive_old` using the gated pair
  ReLU(2nsum − 0.5 − 10(1−white)) − ReLU(2nsum − 1.5 − 10(1−white)) and −ReLU(alive_old);
  `nsum` cleared exactly with ReLU(x) − ReLU(−x). The last layer also writes `out = black + move − alive`.
- Dead zone (nsum ≤ 0.25 → 0, ≥ 0.75 → 1, true values are integers) re-quantises every layer, so the 8e−12
  attention leak never accumulates: observed max |raw − round(raw)| = 0.0 on all tests.

Architecture: d_model 80 (37 used), 81 layers, 4 heads × d_head 20 (19 used), d_mlp 8 (6 used). 14,423 nonzero params
(≈178/layer). Exact parts: everything except the softmax (approximate to ~1e−11, absorbed by dead zones).

## Algorithm B: `solution_logdepth.py` — reachability doubling (10 layers)
Each position carries an 81-wide row R[p] (which points of its own white group it has reached).
- L0 MLP: wp[p] = white[p]·onehot(p) (81 gated units).
- L1: 4 neighbour heads (same primitive, 82-wide value) add neighbours' wp into `nbr` and count empty' neighbours;
  MLP: R0[p] = white[p]·(wp[p] + nbr[p]) (distance ≤ 1), lib[p] = white[p]·[≥1 empty' neighbour].
- L2..L9 (8 squaring layers, one head each): Q = A·R[p], K = onehot(q) → uniform attention over support(R[p]);
  V = [R[q], lib[q]]; average entry is ≥ 1/80 if reachable through some q, ~1e−11 otherwise.
  MLP: dead-zone threshold (0.002 → 0, 0.010 → 1) gated by white gives R_{k+1} (distance ≤ 2^{k+1}) and
  alive = OR over the support of lib. Last layer writes out = black + move − alive.
  R_7 covers distance 128 ≥ 79, so any 9x9 group is handled.
Architecture: d_model 1024 (435 used), 10 layers, 8 heads × 128 (uses ≤ 5 heads, ≤ 82 dims), d_mlp 512 (≤ 413 used).
13,670 nonzero params.

## Adversarial tests (`adversarial.py`, not check.py)
152 legal hand/random cases: all-white board with one liberty (80-stone capture) and with a shared second liberty;
random induced (snake-like) self-avoiding white walks of length 10–40 enclosed by black with exactly one white
liberty at the tail, move either filling it (capture) or elsewhere (no capture; needs up to 35 propagation steps);
100 dense boards (60% white). Plus an out-of-distribution stress case: a 47-stone serpentine white snake with its
only liberty at the tail (needs 47 steps; some black groups lack liberties, which the white-side computation
ignores and `go_play` also leaves alone).

| model | legal cases | serpentine-47 |
|---|---|---|
| A, 81 layers | 152/152 | OK |
| A truncated to 46 layers | 152/152 | FAIL (needs 47) |
| A truncated to 25 layers | 142/152 (all fails are snakes needing ≥ 28 steps) | FAIL |
| B, 8 squarings | 152/152 | OK |
| B, 6 squarings (covers 32) | 147/152 (fails: snakes needing 35) | FAIL |

The truncation experiments confirm the failure mode is exactly depth < in-group distance, as predicted.

## What limits correctness / scale
- **Depth vs group diameter (A)**: layers ≥ 1 + max in-group distance from a stone to a liberty-touching stone.
  Worst case on N×N is O(N²/2) (serpentine ≈ N²/2; ≤ #stones − 1 always). 9x9 → ≤ 80 fits in 96 layers;
  13x13 serpentines (7·13+6 = 97 stones) already need ~97 > 96 layers; 19x19 needs ~190–360 layers → does not fit.
  Width is trivial (d_model 80), so A is purely depth-bound.
- **Width vs depth (B)**: depth 2 + ⌈log2(max distance)⌉ + 1, but each position needs ~4 P-wide registers
  (wp, nbr, R, avg; nbr/avg could be shared → ~3P) and one head with d_head ≥ P + 1. With d_model ≤ 1024:
  P ≤ ~255 with 4 heads (d_head 256) and ~3P+30 ≤ 1024 → P ≤ ~330, so B reaches about **15x15** (P = 225);
  19x19 (P = 361) needs d_model ≈ 1.1–1.5k and d_head ≥ 362 (over the 1024 limit).
- **Precision**: not a limit here. Every MLP re-quantises to exact 0/1 with dead zones; attention leak is ≤ P·e^−30.
  For B the true signal is ≥ 1/P, so the threshold must sit below 1/P (fine in float32 even for P = 361;
  in bf16, 1/361 ≈ 0.0028 is still representable but the dead zone must shrink, LO ≈ 0.3/P).
- **To go further (19x19 within these limits)**: a hybrid — run ~D steps of BFS (A, 1 layer each) to label
  locally, or compress R to group IDs via a min-label propagation with pointer-jumping (each position stores a
  scalar label and a pointer; pointer jumping needs only O(1) width per position and O(log P) layers, but
  "min over neighbours + jump" requires roughly 2 layers per round and careful tie handling) — estimated
  ~20–30 layers at d_model ≈ 128 for 19x19. Not attempted within the time budget.

## Primitives added (in the task directory, no repo files modified)
- `neighbour_read_head(dr, dc)`: exact-neighbour hard attention with an off-board null sink.
- `alive_update_mlp` / `MLP.gated_clamp`: AND(gate, dead-zone threshold) with exact output, `clear`, `remove_binary`.
- `square_head`: reachability-row squaring (set-OR via uniform attention over a support).
No training, no search over weights, no answer tables; all weights are written from the formulas above.
