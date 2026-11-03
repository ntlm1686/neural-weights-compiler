def build_policy():
    from neural_compiler import MLPProgram, MLPCompilerBackend
    p = MLPProgram(600, 32, 32, 6)
    def idx(y, x, c):
        return (y * 10 + x) * 6 + c
    for y in range(9):
        p.add_h1_threshold(f"player_{y}", {idx(y, x, 0): 1.0 for x in range(10)}, threshold=0.5, gain=8.0)
        weights = {}
        for x in range(10):
            weights[idx(y, x, 2)] = 1.0
            weights[idx(y, x, 1)] = -1.0
        p.add_h1_threshold(f"diver_{y}", weights, threshold=0.5, gain=8.0)
    p.add_h1_threshold("oxygen_ok", {idx(9, 1, 5): 1.0}, threshold=0.5, gain=8.0)
    p.add_h1_threshold("has_diver", {idx(9, 8, 5): 1.0}, threshold=0.5, gain=8.0)
    horizontal = {}
    for y in range(1, 9):
        for x in range(10):
            offset = x - 4.5
            horizontal[idx(y, x, 2)] = offset
            horizontal[idx(y, x, 1)] = -offset
            horizontal[idx(y, x, 0)] = -0.5 * offset
    p.add_h1_threshold("target_right", horizontal, threshold=0.5, gain=0.3)
    p.add_h1_threshold("target_left", {i: -w for i, w in horizontal.items()}, threshold=0.5, gain=0.3)
    up_terms = {}
    down_terms = {}
    for y in range(1, 9):
        if y >= 2:
            rows = list(range(1, y))
            terms = {f"player_{y}": 3.0, "oxygen_ok": 3.0}
            terms.update({f"diver_{r}": 3.0 for r in rows})
            name = f"seek_up_{y}"
            p.add_h2_gate(name, terms, bias=3.0 * (len(rows) - 3))
            up_terms[name] = 3.0
        if y < 8:
            rows = list(range(y + 1, 9))
            terms = {f"player_{y}": 3.0, "oxygen_ok": 3.0}
            terms.update({f"diver_{r}": 3.0 for r in rows})
            name = f"seek_down_{y}"
            p.add_h2_gate(name, terms, bias=3.0 * (len(rows) - 3))
            down_terms[name] = 3.0
    p.add_h2_gate("surface", {"oxygen_ok": -4.0, "has_diver": 4.0}, bias=-4.0)
    p.add_h2_gate("leave_surface", {"player_0": 4.0})
    p.add_h2_gate("go_left", {"target_left": 3.0, "oxygen_ok": 3.0}, bias=-3.0)
    p.add_h2_gate("go_right", {"target_right": 3.0, "oxygen_ok": 3.0}, bias=-3.0)
    up_terms["surface"] = 10.0
    down_terms["leave_surface"] = 8.0
    p.set_output_action(0, {}, bias=-5.0)
    p.set_output_action(1, {"go_left": 2.0}, bias=0.0)
    p.set_output_action(2, up_terms, bias=0.0)
    p.set_output_action(3, {"go_right": 2.0}, bias=0.0)
    p.set_output_action(4, down_terms, bias=0.0)
    p.set_output_action(5, {}, bias=2.5)
    return MLPCompilerBackend.compile(p)
