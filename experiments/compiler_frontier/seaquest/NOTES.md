# Seaquest policy by construction: notes

Final `--dev` result: `{"score": 71.4, "se": 5.84, "length": 3299.9, "nonzero": 3682, "episodes": 50, "seed": 100}`

Architecture: 600 -> 32 -> 32 -> 6, tanh. **H1: 32/32 units used, H2: 22/32 used.** No training, no fitting and no
search over weights. Every weight comes from a formula in `primitives.py`.

## Key idea: a relative-row test in one unit (lexicographic row weights)
Seaquest needs logic relative to the submarine's row y, for example "is there an enemy in my row on the left?". The
built-in `adjacent` primitive pays one unit per anchor cell, so it does not fit in 32 units. `Policy.rel_rows` answers
"is there an X in rows >= y+k" (or "<= y-k") in **one** unit:

    S = sum_{cells in zone} K^{pos(r)} x[r,c,ch]  -  sum_y K^{pos(y+k)} * (ch0 cells in row y)/2,   unit = [S > -0.5]

with pos(r)=r for "down" and 9-r for "up". The sub always shows exactly 2 ch0 cells in its row, so the second sum
picks out the threshold for the current row. Row weights in base K=4 make the most significant row decide. The unit is
**exact** whenever each row has at most 3 X cells in the zone (checked: no mismatches).

Two such units u_k ⊇ u_k2 form a **pair**. The pair (u_k - u_k2)/2 ∈ {0,1} is a 0/1 literal meaning "X in rows y+k..y+k2-1
**and none further out**". This is the one approximation in the design. A pair never gives a false positive, but it
misses when another X lies further out. The impossibility is real: no single threshold unit can compute "X in exactly
row y". For the important features I take the union of an "up"-family pair and a "down"-family pair. That union misses
only when there is interference on both sides.

A signed variant (`neg_ch`) uses digits x[ch2]-x[ch1] (diver +1, enemy 0, enemy bullet -1) in base 5. It is compared
against K^p/2. It reports "the most significant non-zero row beyond y holds more divers than bullets". This is an
approximation that I use to seek divers.

Layer 2 consists of linear threshold gates over 0/1 literals (single unit (u+1)/2, pair (a-b)/2), with gain 20 per
half-unit margin. The output uses strict priority. The weight of each level is 20 plus the largest logit that lower
levels can give any other action, so the gap is at least 20 logits. The default `fire` has bias 10. Max logit is about 3.7e4.

## Positional convention
The sub only ever sits at x=4 facing left or x=5 facing right. In both cases it occupies cols 4-5, so zones are fixed
columns. Facing left ⇔ the sub's back (ch4) is at (y,5). Turning (l/r) switches between the two states. `col3` /
`col6` units recentre after sticky-action drift. The start position, x=5 facing left (cols 5-6), is fixed by `center_l`.

## Units
H1 (32):
| units | meaning | exactness |
|---|---|---|
| wl, wr (2 pairs) | ch1 (enemy or bullet) in my row, cols 0-4 / 5-9 (aim) | pair (down family) |
| ild/ilu, ird/iru (4 pairs) | ch1 in my row, cols 3-4 / 5-6 (immediate threat), both families OR'ed | union of pairs |
| face_d (pair) | ch4 at (y,5) means facing left | pair |
| du1, du2 / dd1, dd2 (4 pairs) | ch1 near my column (cols 3-6) in the next 2 rows above / below, both families. du1_a / dd1_a alone = "any danger near my column in any row above / below" (exact, conservative). Row 8 is forced "danger below" | union of pairs; _a units exact |
| t_below, t_above | ch2 (enemy or diver) in rows > y / rows 2..y-1 | exact |
| d_below, d_above | diver-dominant row below / above (signed digits) | approximate |
| col27_ge0 | ch1 at cols 2/7 in rows >= y (turn guard) | exact, conservative |
| oxy_low | (9,2) of ch5 off, i.e. oxygen < 60 | exact |
| have_diver | (9,8) of ch5 on, i.e. at least 1 diver (when oxygen < 180) | exact in that range |
| row1, col3, col6 | sub in row 1 / col 3 / col 6 | exact |

H2 (22 gates) and output priority, highest first:
1. `surface`: oxy_low & have_diver & no danger anywhere above (conservative). Then `surface_window`: the same with the 2-row window check. Action u.
2. Dodge an immediate threat (left or right): up if nothing above (conservative), else down (conservative), else up (window), else down (window). Up is never taken from row 1, because surfacing with 0 divers kills.
3. `flee_l/r`: if both vertical escapes are blocked, move horizontally away from a one-sided threat.
4. `center_l/r`: recentre after drift.
5. Aim: `fire_l/r` when ch1 is on the side I face. `turn_l/r` when it is only on the other side and nothing is at distance 2 (`col27_ge0`).
6. Need a diver (none visible): `div_dn` / `div_up` toward the diver-dominant row, if the 2-row window is clear.
7. `seek_dn` / `seek_up` toward any enemy or diver row, if the whole direction is clear near my column. `seek_dn` is suppressed while I want to surface.
8. Default: fire (harmless when the shot timer is not ready).

Verification (scratch script, not part of the deliverable): on about 8k states from env rollouts, all 22 H2 gates agree
with a Python reference of the intended (pair-semantics) logic, with |h2| >= 0.99999994.

## Primitives added (`primitives.py`)
- `rel_rows(ch, cols, k, direction, rows, force_true, neg_ch, base)`: one-unit relative-row existence test (signed variant optional).
- `row_pair(..., k, k2)`: the monotone pair literal described above.
- `gate(terms, thr)`: an H2 linear threshold over 0/1 literals (single or pair).
- `any_cells` / `none_cells`: plain H1 tests.
- `compile()`: strict-priority output weights.

## Development history
Design iterations were manual. I used 6 `--dev` runs, the last two only to confirm a refactor that did not change behaviour:
| # | change | dev score |
|---|---|---|
| 1 | first design (single-row pairs everywhere) | 9.28 |
| 2 | conservative vertical checks, oxygen thresholds 60/40, logit gap 6 -> 10 | 59.98 |
| 3 | immediate-zone threats with both families, diver seeking, 2-row windows (union of families), turn guard | 64.92 |
| 4 | flee fallback, no seek-down while surfacing | 71.4 |
| 5 | smaller output weights (same decisions) | 71.4 |
| 6 | cosmetic cleanup | 71.4 |

Between dev runs I also used a local diagnostic simulator in the scratchpad: my own sampling loop with sticky actions,
on seeds different from the dev seeds, logging death causes and board snapshots. I ran it about 12 times (40-120
episodes each) to find failure modes. Two candidate changes were rejected on its numbers:
- Widening the threat zones to cols 2-4 / 5-7 regardless of facing: about 51 vs 56, within noise, and I kept the facing-aware design.
- "Forced surfacing at oxygen < 20" in place of `seek_up`: 39, much worse.

No constants were tuned numerically. Every threshold, zone and priority is a design choice argued from the game rules.

## Known weaknesses
- Pair semantics miss threats or dangers when the same zone is occupied in rows beyond on both sides. This is common in
  the crowded late game (move_speed 2, spawn every few steps), and most remaining deaths are collisions there.
- Sticky actions (10%) repeat a vertical move or a turn. Checks look 2 rows ahead and the turn guard keeps distance 2,
  but a repeat into a fresh danger still kills.
- When surfacing is blocked for a long time the sub can run out of oxygen (about 15% of deaths). After a 6-diver surface
  MinAtar's `diver_count` becomes -1, and the gauge cannot tell that state apart from 0.
- The diver-seeking units are approximate: an enemy bullet in a more significant row masks divers.
- H1 is fully used. Exact "content of row y±1" would need about 8 units per feature (one per row), which does not fit.
