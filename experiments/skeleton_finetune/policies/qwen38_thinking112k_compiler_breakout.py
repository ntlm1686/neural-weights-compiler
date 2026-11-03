# Extracted verbatim from experiments/minatar_model_comparison/qwen38_thinking112k_compiler_breakout.txt (final answer after </think>).
# Only the two import lines below were added; the policy logic is unchanged.
from neural_compiler import MLPProgram, MLPCompilerBackend, ExecutableMLP
import numpy as np
def build_policy():
    prog = MLPProgram(
        input_dim=600,
        h1_dim=32,
        h2_dim=32,
        output_dim=6,
        activation="tanh",
    )

    G = 20.0
    G2 = 20.0
    S = 10.0

    def idx(r, c, ch):
        return (r * 10 + c) * 6 + ch

    def make_weights(bx=0.0, by=0.0, tx=0.0, ty=0.0, px=0.0):
        w = {}
        for r in range(10):
            for c in range(10):
                v = 0.0
                if bx:
                    v += bx * c
                if by:
                    v += by * r
                if v:
                    i = idx(r, c, 3)
                    w[i] = w.get(i, 0.0) + v

                v = 0.0
                if tx:
                    v += tx * c
                if ty:
                    v += ty * r
                if v:
                    i = idx(r, c, 4)
                    w[i] = w.get(i, 0.0) + v

        if px:
            for c in range(10):
                v = px * c
                if v:
                    i = idx(9, c, 0)
                    w[i] = w.get(i, 0.0) + v
        return w

    def add_h1(name, bx, by, tx, ty, px, threshold):
        prog.add_h1_threshold(
            name,
            make_weights(bx, by, tx, ty, px),
            threshold,
            G,
        )

    # Direction indicators.
    # dx_right: ball_x - trail_x == 1
    # dx_left:  trail_x - ball_x == 1
    # dy_up:    trail_y - ball_y == 1
    # dy_nonneg: ball_y - trail_y >= 0  (down or vertical-bounce ambiguity)
    add_h1("dx_right", 1, 0, -1, 0, 0, 0.5)
    add_h1("dx_left", -1, 0, 1, 0, 0, 0.5)
    add_h1("dy_up", 0, -1, 0, 1, 0, 0.5)
    add_h1("dy_nonneg", 0, 1, 0, -1, 0, -0.5)

    # Down-right:
    # q = x + 9 - y, range 0..18.
    # target = q if q <= 9, else 19 - q.
    add_h1("DR_A", -1, 1, 0, 0, 0, -0.5)   # q <= 9
    add_h1("DR_B", 1, -1, 0, 0, 0, 0.5)    # q >= 10
    add_h1("DR_C", 1, -1, 0, 0, -1, -8.5)  # q - p > 0
    add_h1("DR_D", -1, 1, 0, 0, 1, 9.5)    # p - q > 0
    add_h1("DR_E", -1, 1, 0, 0, -1, -9.5)  # 19 - q - p > 0
    add_h1("DR_F", 1, -1, 0, 0, 1, 10.5)   # q + p - 19 > 0

    # Down-left:
    # q = 28 - x - y, range 10..28.
    # If q <= 19: target = x + y - 9.
    # If q >= 20: target = 8 - x - y.
    add_h1("DL_G", 1, 1, 0, 0, 0, 8.5)     # x + y >= 9  (q <= 19)
    add_h1("DL_H", -1, -1, 0, 0, 0, -8.5)  # x + y <= 8  (q >= 20)
    add_h1("DL_I", 1, 1, 0, 0, -1, 9.5)    # (x + y - 9) - p > 0
    add_h1("DL_J", -1, -1, 0, 0, 1, -8.5)  # p - (x + y - 9) > 0
    add_h1("DL_K", -1, -1, 0, 0, -1, -7.5) # (8 - x - y) - p > 0
    add_h1("DL_L", 1, 1, 0, 0, 1, 8.5)     # x + y + p - 8 > 0

    def add_and(name, terms):
        k = len(terms)
        prog.add_h2_gate(
            name,
            {t: G2 for t in terms},
            bias=-(k - 0.5) * G2,
        )

    # Down-moving right/left gates.
    add_and("DR_r1", ["dx_right", "dy_nonneg", "DR_A", "DR_C"])
    add_and("DR_r2", ["dx_right", "dy_nonneg", "DR_B", "DR_E"])
    add_and("DR_l1", ["dx_right", "dy_nonneg", "DR_A", "DR_D"])
    add_and("DR_l2", ["dx_right", "dy_nonneg", "DR_B", "DR_F"])

    add_and("DL_r1", ["dx_left", "dy_nonneg", "DL_G", "DL_I"])
    add_and("DL_r2", ["dx_left", "dy_nonneg", "DL_H", "DL_K"])
    add_and("DL_l1", ["dx_left", "dy_nonneg", "DL_G", "DL_J"])
    add_and("DL_l2", ["dx_left", "dy_nonneg", "DL_H", "DL_L"])

    # Up-moving: follow horizontal direction.
    add_and("UR_pass", ["dx_right", "dy_up"])
    add_and("UL_pass", ["dx_left", "dy_up"])

    # Output scoring.
    prog.set_output_action(0, {}, 0.0)
    prog.set_output_action(
        1,
        {
            "DR_l1": S,
            "DR_l2": S,
            "DL_l1": S,
            "DL_l2": S,
            "UL_pass": S,
        },
        40.0,
    )
    prog.set_output_action(2, {}, -20.0)
    prog.set_output_action(
        3,
        {
            "DR_r1": S,
            "DR_r2": S,
            "DL_r1": S,
            "DL_r2": S,
            "UR_pass": S,
        },
        40.0,
    )
    prog.set_output_action(4, {}, -20.0)
    prog.set_output_action(5, {}, -20.0)

    return MLPCompilerBackend.compile(prog)
