# Extracted verbatim from experiments/minatar_model_comparison/qwen38_thinking112k_compiler_seaquest.txt (final answer after </think>).
# Only the two import lines below were added; the policy logic is unchanged.
from neural_compiler import MLPProgram, MLPCompilerBackend, ExecutableMLP
import numpy as np
def build_policy():
    def idx(r, c, ch):
        return (r * 10 + c) * 6 + ch

    prog = MLPProgram(
        input_dim=600,
        h1_dim=32,
        h2_dim=32,
        output_dim=6,
        activation="tanh",
    )

    # ------------------------------------------------------------------
    # Layer 1: 32 hand-designed feature detectors.
    # ------------------------------------------------------------------

    # Surface: sub is in row 0.
    w = {idx(0, c, 0): 1.0 for c in range(10)}
    prog.add_h1_threshold("surface", w, 0.5, 10.0)

    # Six divers: use the right-aligned diver gauge pattern.
    # When oxygen is low enough, count=6 gives col3=1, col4=1, col2=0.
    w = {
        idx(9, 3, 5): 1.0,
        idx(9, 4, 5): 1.0,
        idx(9, 2, 5): -1.0,
    }
    prog.add_h1_threshold("six", w, 1.5, 10.0)

    # Enemy immediately adjacent to the sub.
    def add_enemy_adj(name, dr, dc):
        w = {}
        for r in range(10):
            for c in range(10):
                nr, nc = r + dr, c + dc
                if 0 <= nr < 10 and 0 <= nc < 10:
                    w[idx(r, c, 0)] = w.get(idx(r, c, 0), 0.0) + 1.0
                    w[idx(nr, nc, 1)] = w.get(idx(nr, nc, 1), 0.0) + 1.0
        prog.add_h1_threshold(name, w, 1.5, 10.0)

    add_enemy_adj("enemy_adj_left", 0, -1)
    add_enemy_adj("enemy_adj_right", 0, 1)
    add_enemy_adj("enemy_adj_up", -1, 0)
    add_enemy_adj("enemy_adj_down", 1, 0)

    # Diver immediately adjacent to the sub.
    # Diver indicator is channel 2 without channel 1.
    def add_diver_adj(name, dr, dc):
        w = {}
        for r in range(10):
            for c in range(10):
                nr, nc = r + dr, c + dc
                if 0 <= nr < 10 and 0 <= nc < 10:
                    w[idx(r, c, 0)] = w.get(idx(r, c, 0), 0.0) + 1.0
                    w[idx(nr, nc, 2)] = w.get(idx(nr, nc, 2), 0.0) + 3.0
                    w[idx(nr, nc, 1)] = w.get(idx(nr, nc, 1), 0.0) - 1.0
        prog.add_h1_threshold(name, w, 3.5, 10.0)

    add_diver_adj("diver_adj_left", 0, -1)
    add_diver_adj("diver_adj_right", 0, 1)
    add_diver_adj("diver_adj_up", -1, 0)
    add_diver_adj("diver_adj_down", 1, 0)

    # Coarse enemy half-space detectors.
    def add_channel_sum(name, ch, rows, cols):
        w = {}
        for r in rows:
            for c in cols:
                w[idx(r, c, ch)] = w.get(idx(r, c, ch), 0.0) + 1.0
        prog.add_h1_threshold(name, w, 0.5, 10.0)

    add_channel_sum("enemy_left", 1, range(9), range(5))
    add_channel_sum("enemy_right", 1, range(9), range(5, 10))
    add_channel_sum("enemy_top", 1, range(5), range(10))
    add_channel_sum("enemy_bottom", 1, range(5, 9), range(10))

    # Coarse diver half-space detectors.
    def add_diver_region(name, rows, cols):
        w = {}
        for r in rows:
            for c in cols:
                w[idx(r, c, 2)] = w.get(idx(r, c, 2), 0.0) + 3.0
                w[idx(r, c, 1)] = w.get(idx(r, c, 1), 0.0) - 1.0
        prog.add_h1_threshold(name, w, 0.5, 10.0)

    add_diver_region("diver_left", range(9), range(5))
    add_diver_region("diver_right", range(9), range(5, 10))

    # Sub half-space detectors.
    add_channel_sum("sub_left", 0, range(9), range(5))
    add_channel_sum("sub_right", 0, range(9), range(5, 10))

    # Sub row comparison features: sub_y < k.
    for k in range(2, 9):
        w = {}
        for r in range(k):
            for c in range(10):
                w[idx(r, c, 0)] = w.get(idx(r, c, 0), 0.0) + 1.0
        prog.add_h1_threshold(f"sub_lt_{k}", w, 0.5, 10.0)

    # Diver row comparison features: some diver is in rows < k.
    for k in range(2, 9):
        w = {}
        for r in range(k):
            for c in range(10):
                w[idx(r, c, 2)] = w.get(idx(r, c, 2), 0.0) + 3.0
                w[idx(r, c, 1)] = w.get(idx(r, c, 1), 0.0) - 1.0
        prog.add_h1_threshold(f"diver_lt_{k}", w, 0.5, 10.0)

    # ------------------------------------------------------------------
    # Layer 2: 32 symbolic gates / passes.
    # ------------------------------------------------------------------

    def pass_h1(name, h1_name):
        prog.add_h2_gate(name, {h1_name: 10.0}, 0.0)

    def and_h1(name, a, b):
        prog.add_h2_gate(name, {a: 10.0, b: 10.0}, -15.0)

    def and_not_h1(name, a, b):
        prog.add_h2_gate(name, {a: 10.0, b: -10.0}, -15.0)

    # Surface / diver-count state.
    prog.add_h2_gate("h2_surface_six", {"surface": 10.0, "six": 10.0}, -15.0)
    prog.add_h2_gate("h2_surface_not_six", {"surface": 10.0, "six": -10.0}, -15.0)
    prog.add_h2_gate("h2_six_not_surface", {"six": 10.0, "surface": -10.0}, -15.0)

    # Bottom row: sub_y == 8 is equivalent to NOT sub_y < 8.
    prog.add_h2_gate("h2_bottom", {"sub_lt_8": -10.0}, 5.0)

    # Immediate danger passes.
    pass_h1("h2_danger_adj_left", "enemy_adj_left")
    pass_h1("h2_danger_adj_right", "enemy_adj_right")
    pass_h1("h2_danger_adj_up", "enemy_adj_up")
    pass_h1("h2_danger_adj_down", "enemy_adj_down")

    # Coarse danger passes.
    pass_h1("h2_danger_left", "enemy_left")
    pass_h1("h2_danger_right", "enemy_right")
    pass_h1("h2_danger_top", "enemy_top")
    pass_h1("h2_danger_bottom", "enemy_bottom")

    # Immediate diver passes.
    pass_h1("h2_diver_adj_left", "diver_adj_left")
    pass_h1("h2_diver_adj_right", "diver_adj_right")
    pass_h1("h2_diver_adj_up", "diver_adj_up")
    pass_h1("h2_diver_adj_down", "diver_adj_down")

    # Coarse diver direction passes.
    pass_h1("h2_diver_left", "diver_left")
    pass_h1("h2_diver_right", "diver_right")

    # Vertical diver direction using row comparisons.
    for k in range(2, 9):
        and_not_h1(f"h2_diver_up_{k}", f"diver_lt_{k}", f"sub_lt_{k}")
        and_not_h1(f"h2_diver_down_{k}", f"sub_lt_{k}", f"diver_lt_{k}")

    # ------------------------------------------------------------------
    # Output logits.
    #
    # Biases are chosen so that when all listed H2 gates are false (-1),
    # the logit is a small base value. A true gate (+1) adds 2 * weight.
    # ------------------------------------------------------------------

    # 0 = no-op: surface only when carrying six divers.
    prog.set_output_action(
        0,
        {"h2_surface_six": 50.0},
        bias=50.0,
    )

    # 1 = left.
    left_terms = {
        "h2_danger_adj_right": 40.0,
        "h2_danger_right": 30.0,
        "h2_diver_adj_left": 25.0,
        "h2_diver_left": 15.0,
    }
    prog.set_output_action(1, left_terms, bias=sum(left_terms.values()))

    # 2 = up.
    up_terms = {
        "h2_six_not_surface": 50.0,
        "h2_danger_adj_down": 40.0,
        "h2_danger_bottom": 30.0,
        "h2_diver_adj_up": 25.0,
        "h2_bottom": 40.0,
    }
    for k in range(2, 9):
        up_terms[f"h2_diver_up_{k}"] = 3.0
    prog.set_output_action(2, up_terms, bias=sum(up_terms.values()))

    # 3 = right.
    right_terms = {
        "h2_danger_adj_left": 40.0,
        "h2_danger_left": 30.0,
        "h2_diver_adj_right": 25.0,
        "h2_diver_right": 15.0,
    }
    prog.set_output_action(3, right_terms, bias=sum(right_terms.values()))

    # 4 = down.
    down_terms = {
        "h2_surface_not_six": 50.0,
        "h2_danger_adj_up": 40.0,
        "h2_danger_top": 30.0,
        "h2_diver_adj_down": 25.0,
    }
    for k in range(2, 9):
        down_terms[f"h2_diver_down_{k}"] = 3.0
    prog.set_output_action(4, down_terms, bias=sum(down_terms.values()) + 10.0)

    # 5 = fire: intentionally low priority / unused in this policy.
    prog.set_output_action(5, {}, bias=0.0)

    return MLPCompilerBackend.compile(prog)
