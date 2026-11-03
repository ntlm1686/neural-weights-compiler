"""Conway's Game of Life (8x8 torus), one step, as a 1-layer / 1-head compiled Transformer.

Token i = cell (r, c) = (i // 8, i % 8).  Registers: alive (cell value), row_oh / col_oh (position one-hots).
L0 attention : box  = mean of alive over the 3x3 torus box around the cell  (= S / 9, S = 3x3 sum incl. self)
L0 MLP       : next = f(z), z = 9*box + 10*alive  (integer), f = 1 iff z == 3 (dead, S=3) or z in {13, 14}
               (alive, S in {3, 4} i.e. 2 or 3 neighbours)
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
from neural_compiler import TransformerProgram, TransformerCircuitBackend  # noqa: E402
from life_primitives import torus_box_mean_attn, integer_lut_mlp  # noqa: E402

N = 8


def build_model():
    prog = TransformerProgram(name="life8_torus", output_register="next")
    for name, w in [("alive", 1), ("row_oh", N), ("col_oh", N), ("box", 1), ("next", 1)]:
        prog.declare_register(name, w)
    # empty op list: the backend allocates registers and zero weights; our primitives fill them in
    model = TransformerCircuitBackend.compile(prog, d_model=20, n_layers=1, n_heads=1, d_mlp=8)

    torus_box_mean_attn(model, layer=0, head=0, row_reg="row_oh", col_reg="col_oh",
                        val_reg="alive", out_reg="box", n_rows=N, n_cols=N, margin=40.0)

    def rule(z):  # z = S + 10*alive, S = 3x3 sum including self
        alive, S = divmod(z, 10)
        return int(S == 3 or (alive == 1 and S == 4))
    points = [(z, rule(z)) for z in range(0, 20)]
    integer_lut_mlp(model, layer=0, in_coeffs={"box": 9.0, "alive": 10.0}, points=points, out_reg="next")
    return model


def encode(example):
    g = np.asarray(example).reshape(N * N).astype(float)
    pos = np.arange(N * N)
    return {"alive": g, "row_oh": np.eye(N)[pos // N], "col_oh": np.eye(N)[pos % N]}, N * N


def decode(raw_output):
    return (np.asarray(raw_output) > 0.5).astype(int).reshape(N, N)
