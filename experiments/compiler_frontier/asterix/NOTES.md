# Asterix policy: compiled, not trained

`solution.py` builds a 600 -> 32 -> 32 -> 6 tanh MLP from explicit formulas (`primitives.py`).
`verify.py` checks the H1 bits exhaustively and the H2 units on 20,000 random bit patterns (0 mismatches,
all |activations| >= 0.9999).

## The main idea: pin the player to column 5
Left, right and fire get logit -60, so the player never moves sideways and stays in its start column 5.
Every entity crosses the whole row, so every gold still passes through column 5, and the game reduces to a
1-D problem: pick a row at each step. Relative-position logic then needs **one H1 unit per row** and no
units per (player cell, entity cell) pair. Asterix keeps at most one entity per row (slot = row - 1), and the
trail cell gives its direction (trail at x-1 means it moves right). So "this row's entity is approaching column
5 within distance k" is a single linear threshold:
* Trail weights b <= 0 everywhere, so a row holding only the *other* entity type (trail only) stays silent.
* b strictly decreases towards column 5 (b[x-1] - b[x+1] >= 1), so approaching and receding entities separate.
* Entity weight a[x] = 1 - b[behind]. The unit fires iff a[x] + b[trail] >= 1 and is silent iff <= 0. With
  gain 10 and threshold 0.5, every case has |z| >= 5.

## Units
**H1 (31/32 used, all exact on any reachable observation):**
| units | meaning |
|---|---|
| B0..B2 (3) | player row r in binary (bit i of r-1). Saves 5 units compared with a one-hot row code |
| A[2..7] (6) | an enemy approaches col 5 at any distance, or is on it |
| E[1..8] (8) | an enemy approaches within 2 cells, or is on col 5 (deadly soon) |
| D[1..8] (8) | an enemy approaches within 1 cell, or is on col 5 (entering kills) |
| G[2..7] (6) | a gold approaches col 5 at any distance, or is on it (safe row that pays) |

A and G are not computed for edge rows 1 and 8, because the policy never enters an edge row except to escape.

**H2 (26/32 used, exact linear thresholds over H1 bits).** "Player in row r" is a gate:
sum_i (+-M) * B_i reaches M*popcount(r-1) only when the code matches.
* `MOVE[r,d]` (12): move from r to an inner target t = r+d. Hard gates: P[r], !E[t], !A[t], !G[r] (stay on
  your gold), and !E[t+d], because a 10% sticky repeat would carry the player one more row. The unit then
  fires if E[r] (escape), A[r] (pre-empt), r is an edge row (leave it), G[t], or there are more golds on
  that side than on the other.
* `ESC[r,d]` (14): an acceptable escape. It fires iff E[r]+D[r]-E[t]-D[t] >= 1, which means (E[r] & !E[t]) or
  (D[r] & E[t] & !D[t]), i.e. the target's enemy is exactly 2 away, so entering survives this tick.

**Output** (strict priority, gaps >= 20 logits): noop = -24*[any ESC]; up/down = -20 + 40*[MOVE] + 24*[ESC];
l/r/f = -60. So a good move beats an acceptable escape, an acceptable escape beats staying, and staying is
the default. A tie between up and down (both good) is broken at random.

Approximate parts: none in the logic, since every bit is exact. The policy is still a 1-step-lookahead
heuristic with distance measured in cells, not ticks.

## Primitives added (`primitives.py`)
`h1_row_detector` uses the weight tables ENEMY_K1/ENEMY_K2/APPROACH_ANY(=GOLD_ANY) from the derivation above.
`h1_any` is an OR over cells. `h2_threshold` is an integer-coefficient threshold over H1 bits with a half-integer
margin. `make_mlp` wraps the weights in an ExecutableMLP.

## Dev history (`check.py asterix --dev`, 5 runs)
1. 39.9: one-hot P (8) + E(dist<=3) + G with "escape / seek gold" units.
2. 40.0: added imminent-danger emergency units (no effect).
3. 70.2: binary player code, A/E/D/G bits, larger logit gaps, no gold trips into edge rows, sticky check, and
   the split into good MOVE and acceptable ESC.
4. 62.8: the sticky check (!E[t+d]) became a hard gate for every MOVE.
5. 62.8: final (same policy as 4, re-run after cleanup).

Between dev runs I also ran a private diagnostic simulator (my own loop: softmax sampling, 10% sticky, my own
seeds, 100-500 episodes). It logged the death cause (sticky? what the row and its neighbours held) and printed
the last frames of deaths. Every change was motivated by those death traces, not chosen by a score search.
That simulator gave about 8 comparisons. Two of them decided changes: version 4 vs version 3 (73.2 vs 67.7
over 500 episodes) and a two-step-escape ESC variant (72.5, no gain, reverted). Measured on 500 episodes, the
final policy scores about **73**. The dev score of 62.8 comes from only 50 episodes (se 5).

## Known weaknesses
* Most deaths now happen at maximum difficulty (entities move every frame, a spawn every frame). There the
  player gets trapped when its own and both neighbouring enemies arrive in sync. Seeing 2+ steps ahead would
  need distance/time per row, and there is no H1 capacity left (1 spare unit).
* Distances are in cells, not ticks: when move_speed > 1 the policy is over-cautious.
* Edge rows have no A/G bits, and gold in rows 1 and 8 is never pursued.
* A sticky repeated noop (10%) can still waste the single escape step.
