"""Hand-designed weight primitives for the Asterix policy (no fitting: every weight is an explicit formula).

Layer conventions (tanh MLP 600 -> H1 -> H2 -> 6):
  * H1 units are linear threshold units over the binary grid:  h = tanh(G1 * (sum w_i x_i - theta)).
    All designs use integer weights and half-integer thresholds, so |z| >= 0.5 * G1 = 5 -> |h| >= 0.9999.
  * H2 units are linear threshold units over H1 *bits* b = (h + 1) / 2 in {0, 1}:
        v = tanh(G2 * (sum c_j b_j - theta)),  again integer c_j and half-integer theta.
  * Output logits are affine in H2 bits.
"""
import numpy as np

from neural_compiler import ExecutableMLP

G1 = 10.0
G2 = 10.0
ROWS, COLS, CHS = 10, 10, 6


def idx(r, c, ch):
    return (r * COLS + c) * CHS + ch


# Per-row "approaching entity" detectors for a player pinned at column 5 (see NOTES.md for the derivation).
# Exactness argument (Asterix holds at most one entity per row): an entity at x moving right has its trail at
# x-1, moving left at x+1, and no trail when it sits on the edge it spawned from.  The unit fires (threshold 0.5)
# iff a[x] + b[back] >= 1 and is silent iff a[x] + b[back] <= 0.  Choosing b <= 0 everywhere makes a row that
# holds only the *other* entity type (its trail) silent; choosing b strictly decreasing towards column 5
# (b[x-1] - b[x+1] >= 1 left of 5, mirrored right of 5) separates approaching from receding, and
# a[x] = 1 - b[x-1] (left side) / 1 - b[x+1] (right side).
# ENEMY_K3: enemy within distance 3 of column 5 approaching it, or on column 5.
ENEMY_K3 = ({2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 1, 8: 1},
            {3: -1, 4: -1, 5: -2, 6: -1, 7: -1})
# ENEMY_K1: enemy adjacent to column 5 approaching it, or on column 5 (imminent).
ENEMY_K1 = ({4: 1, 5: 2, 6: 1},
            {4: -1, 5: -1, 6: -1})
# ENEMY_K2: enemy within distance 2 of column 5 approaching it, or on column 5.
ENEMY_K2 = ({3: 1, 4: 1, 5: 2, 6: 1, 7: 1},
            {4: -1, 5: -1, 6: -1})
# GOLD_ANY / APPROACH_ANY: gold anywhere in the row approaching column 5, or on column 5.
GOLD_ANY = ({0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 4, 6: 2, 7: 2, 8: 1, 9: 1},
            {1: -1, 2: -1, 3: -2, 4: -2, 5: -3, 6: -2, 7: -1, 8: -1})
APPROACH_ANY = GOLD_ANY   # same weights, applied to the enemy channel: enemy approaching col 5 at any distance


def h1_row_detector(W1, b1, j, row, ent_ch, trail_ch, weights):
    a, b = weights
    for c, w in a.items():
        W1[idx(row, c, ent_ch), j] += G1 * w
    for c, w in b.items():
        W1[idx(row, c, trail_ch), j] += G1 * w
    b1[j] = -G1 * 0.5


def h1_any(W1, b1, j, cells_ch):
    for (r, c, ch) in cells_ch:
        W1[idx(r, c, ch), j] += G1
    b1[j] = -G1 * 0.5


def h2_threshold(W2, b2, j, terms, theta):
    """H2 unit j fires iff sum_k c_k * bit_k >= theta + 0.5 (bits of H1 units), theta integer.
    terms: dict h1_index -> integer coefficient c_k.  With bit = (h+1)/2:  z = G2*(sum c (h+1)/2 - (theta+0.5))."""
    const = 0.0
    for k, c in terms.items():
        W2[k, j] += G2 * c / 2.0
        const += c / 2.0
    b2[j] = G2 * (const - (theta + 0.5))


def make_mlp(W1, b1, W2, b2, W3, b3, unit_map=None):
    return ExecutableMLP(W1.astype(np.float32), b1.astype(np.float32), W2.astype(np.float32),
                         b2.astype(np.float32), W3.astype(np.float32), b3.astype(np.float32), "tanh", unit_map)
