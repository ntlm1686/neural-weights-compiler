# Register-machine interpreter compiled into Transformer weights

**Result:** `check.py program --dev` -> `{"correct": 500, "total": 500, "nonzero": 173141, "d_model": 192, "n_layers": 25, "n_heads": 1}`.
--dev runs used: **1** (500/500). There was also a private sanity run with my own RandomState(123): 2000/2000 correct, with output exactly equal to the one-hot answer.

## Architecture
- d_model 192 (166 used), 1 head (d_head 192), d_mlp 2404, 25 layers = 1 init layer + 2 layers per step x 12 steps.
- No LayerNorm, no training. All weights are written from explicit formulas in `solution.py`. I did not use the IR/backend ops: I fill the `ExecutableTransformer` weight dict directly. A small `MLPBuilder` holds ReLU units of the form `ReLU(sum w x + bias) -> outputs`.
- Sequence of 9 positions. Positions 0..7 each hold instruction p: a one-hot `op`(6), `a`(4) and `b`(16), plus a position one-hot. Position 8 is blank. Every position runs the same machine and decode reads position 8 (`last_pos_register`).
- Residual registers: `pc` is a one-hot of 8 and `R` is 4 x 16 one-hot registers. The per-step scratch registers are `fop`(6), `fa`(4), `fb`(16), `va`(16) and `vb`(16).
- encode places only the per-instruction one-hots and position one-hots. decode is an argmax over each 16-slot block of R.

## Per-layer implementation
- **Layer 0 (init):** `b_O` adds pc[0]=1 and R_i[0]=1. This is the constant initial state.
- **Step s, layer 1+2s:**
  - *Fetch (attention):* Q = c * pc one-hot and K = c * pos one-hot, with c^2/sqrt(d_head) = 60. The current instruction scores 60 and every other position scores 0. V/O copy op/a/b into fop/fa/fb. The leakage is <= 8e^-60 (about 1e-25).
  - *Operand decode (MLP, 128 units):* `va[v] = sum_i ReLU(fa_i + R_i[v] - 1)` and `vb[v] = sum_i ReLU(fb_i + R_i[v] - 1)`, which give the one-hots of r[a] and r[b]. Each is an exact AND of binaries.
- **Step s, layer 2+2s (execute + pc update + cleanup, MLP, 2404 units):**
  - ADD: `ReLU(fa_i + va[x] + vb[y] + fop1 - 3) -> R_i[(x+y)%16]`, with 4*256 units. SUB is the same pattern with `(x-y)%16`.
  - DEC: `ReLU(fa_i + va[x] + fop4 - 2) -> R_i[(x-1)%16]`. LOADI: `ReLU(fa_i + fb[v] + fop0 - 2) -> R_i[v]`.
  - Erase the old value of the written register: `ReLU(fa_i + R_i[v] + fop0+fop1+fop2+fop4 - 2) -> -R_i[v]`.
  - pc:
    - jump: `ReLU(fop3 + fb[p] - va[0] - 1) -> pc[p]`
    - advance: `ReLU(pc[p] - fop3 - fop5)` and `ReLU(pc[p] + fop3 + va[0] - 2)`, both `-> pc[p+1 mod 8]`
    - HALT stays: `ReLU(pc[p] + fop5 - 1) -> pc[p]`
    - remove old pc: `-(ReLU(pc)-ReLU(-pc))`
  - HALT freezes the machine: no write fires, and pc stays put for the remaining steps.
  - Cleanup: fop/fa/fb/va/vb are zeroed with `-(ReLU(x)-ReLU(-x))`, so the next fetch starts clean.

## Exact vs approximate
Every MLP unit is an AND of {0,1} values, so it is exact. The only approximation is softmax leakage in the fetch, with a gap of 60. Across all 25 layers on 200 programs, the largest residual deviation from an integer that I measured is 6e-26. The decision margin is 0.5 at every threshold, and none of it is consumed in practice.

## Parameters
- 173,141 nonzero out of 26.8M allocated.
- Most of the nonzeros are the 2048 ADD/SUB units per update layer (5 nonzeros each) times 12 layers, about 123k.

## Primitives added (local to `solution.py`)
- `MLPBuilder.unit` (a generic AND/threshold unit)
- `linear_copy` (an exact signed copy/clear)
- a one-hot pointer-fetch attention head

## What limits scale
- **Steps:** each step needs 2 layers, because of the chain fetch -> operand select -> (update + pc). With the 96-layer cap that allows (96-1)/2 = **47 steps**. Getting to 1 layer per step would need operand selection inside attention, for example registers stored in separate positions and read by a second head keyed on `a` and `b`. But the result must then be written back into the register position and the fetch still depends on the new pc. I think 2 layers per step is close to the floor for this design.
- **Register width V (one-hot):**
  - Residual dims are about N_reg*V (registers) + 3V (fb, va, vb) + P-related terms.
  - The ADD/SUB tables need 2*N_reg*V^2 hidden units per step: about 2k at V=16, about 131k at V=128 (d_mlp has no cap, but parameters grow as V^2). V=256 with 4 registers already needs about 1.8k residual dims, which is over the 1024 cap.
  - Going wider would need a binary encoding with a ripple-carry adder: about log2(V) extra MLP layers per step, or 1-2 layers with a carry-lookahead in a wider MLP. Alternatively, a scalar encoding with a mod-wrap threshold (about 3 extra layers per step), but that needs exact scalar margins.
- **Program length P:** costs 2P dims (pos one-hot + pc) and about 4P units for the pc. The fetch margin does not depend on P (one-hot dot product; leakage is P*e^-60). P around 400 fits in d_model 1024 with V=16.
- **Number of registers N:** it is linear in dims (16N) and units (2N*V^2 + 2N*V).
- The binding constraint for this task is none of these: 12 steps x 2 layers sits well under every limit. The first wall would be **steps x 2 > 95 layers** (more than 47 steps), followed by **V^2 hidden units / V*N residual dims** once registers are wider than about 128 values.
