# chess_check: is the side to move in check?

**Result:** `--dev`: `{"correct": 500, "total": 500, "nonzero": 1992, "d_model": 1024, "n_layers": 4, "n_heads": 8}`.
I made 1 `--dev` run (500/500, first attempt). An extra debug script (`test_margins.py`) compares the model with
python-chess on 1500 more generator positions from its own RNG: 0 wrong. The lowest output on a positive position
was 0.99999999794 and the highest on a negative position was 4e-18. All 65 positions produce the same output.

## Encoding (raw symbols only)
There are 65 positions: 64 squares (index 8*rank+file) plus one blank "sink" position (64).
Each position holds these registers:
- `pc`: piece one-hot (12 = 6 white + 6 black). An empty square or the sink gets all zeros.
- `tw`: 1 if White is to move.
- Fixed position codes: rank, file, diagonal (r-f+7) and anti-diagonal (r+f) one-hots, a 64-wide square one-hot,
  thermometers `[r>i]`, `[r<i]`, `[f>i]`, `[f<i]`, and a `sink` flag.
`decode` reads the `att` register at the sink position and applies the threshold 0.5.
Python does no chess logic.

## Layers (weights written by hand from formulas in `solution.py`; no compiler IR ops)
| layer | block | what it computes | nonzero |
|---|---|---|---|
| 0 | MLP (12 ReLU units) | Turns pieces into side-relative features using ReLU gates on binary inputs: `x*(1-t)=ReLU(x-t)`, `x*t=ReLU(x+t-1)`. Features: ownK, enemy R/Q, enemy B/Q, enemy N, enemy K, "black pawn & White to move", "white pawn & Black to move". | 46 |
| 1 | attn, 1 head | Broadcasts the king location. The logit is `A*ownK_j`, so every position copies the king's rank, file, diagonal, anti-diagonal and square one-hots (110 dims) into `kR,kF,kD,kAD,kSQ`. | 222 |
| 2 | attn, 8 heads | **Sliding attacks with blockers.** There is one head per ray direction (N, S, E, W, NE, SW, NW, SE). Each logit is bilinear in (king codes at the query, square codes at the key): `A*[same line] + A*[correct side of king] + A*occupied_j ± S*coord_j`. Only occupied squares on that ray reach about 3A, and the coordinate term ranks them by distance, so the nearest one wins by at least S=40 logits. The sink scores `3A-7.5S` or `3A-0.5S`, which puts it below every real ray candidate and above every non-candidate (at most 2A+7S, with A=2000). The value is "enemy R/Q" for orthogonal heads and "enemy B/Q" for diagonal heads, so a blocker of any other kind gives 0 and an empty ray gives the sink's value, also 0. | 496 |
| 3 | attn, 4 heads | **Contact attacks.** The heads are knight, king, and two pawn heads (one per side to move, since pawn direction depends on colour). The query is `M·kSQ`, where M is the fixed 64x64 attack-offset adjacency for that piece; the key is the square one-hot. The logit is `A*[attack offset] + A*feature_j`, against 1.5A for the sink. The output is about 1 if any enemy piece of that type sits on an attacking square, otherwise about 0. | 1228 |

All 12 heads add into `att` (normally 0 or 1; a double check gives 2), and decode tests `att > 0.5`.

**Exact vs approximate:** Every condition is exact: the logits are integer combinations of one-hots. The only
approximation is softmax leakage, at most about 64·e^-40 (around 1e-16) per head. On the tests the output is 0 or
1 to within 1e-8, and in practice the model computes `is_check` exactly.

**Primitives:** I added no compiler primitives. The weights are built directly into an `ExecutableTransformer`
weight dict. The patterns used are: the ReLU gated-AND, "attend-to-flag broadcast", "nearest occupied square on a
ray = bilinear line/side test + linear distance tie-break + sink", and "adjacency-matrix query against a square
one-hot key".

**Architecture / cost:** d_model=1024 (275 dims used), 4 layers, 8 heads (d_head=128, needed for the 64+1-wide
contact heads), d_mlp=16. There are 1992 nonzero parameters out of 16.9M total.

## Main idea / why it was easy
The blocker problem never needs an explicit scan along the ray. "First occupied square on a ray" is an argmax, and
one softmax head computes it: the ray-membership logits fix the candidate set and the linear distance term breaks
ties. The king location is the only thing that depends on the input, so one broadcast head turns every geometric
relation into a bilinear form in (king code, square code). The number of attention layers is constant (3).

## Extending to full legal-move generation (estimate)
The main cost driver is the number of (piece, square, ray) queries, not depth.
- **Pseudo-legal moves.** Each square would query its own 8 rays, using the same nearest-blocker head with the
  square's own codes in place of the king's. That gives target squares up to the blocker, which is 8 heads in one
  layer. The output is then a 64x64 from-to move matrix, which needs a much larger readout: about 4096 outputs
  spread over positions.
- **Legality: pins and checks.**
  - One way is to apply the "is the king attacked" test above to the position *after* each move. That multiplies
    the work by the number of moves (about 30-40), which means either one sequence copy per move, or recomputing
    rays with the moved piece's from-square treated as empty and its to-square as occupied. In attention this is
    one extra key feature per candidate move.
  - The cheaper route is direct pin detection: for each king ray, find the first and second occupied squares.
    The second can be found with a second head that excludes the first (an extra layer).
  - Checks can be handled with "block the checking ray or capture the checker", using the attacker location found
    above plus between-square tests, which are bilinear once both endpoints are broadcast.
  - King moves need an attacked-square map for all 64 squares. That is the same 12 heads with per-square queries,
    with the king itself removed as a blocker.
  - Castling and en passant add a few special heads (en passant has the rare horizontal discovered-pin case).
- **My estimate:** about 8-12 attention layers, 16 heads, and d_model around 1024. Most of the cost is in output
  width and bookkeeping, not in any operation that is hard to express.

## Files
- `solution.py`: model, encode and decode.
- `test_margins.py`: debug comparison against python-chess with output margins. It uses the generator from
  `check.py` with its own RNG and is not used for scoring.
