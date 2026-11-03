def build_policy():
    import math
    from neural_compiler.ir import MLPProgram
    from neural_compiler.mlp_backend import MLPCompilerBackend
    p = MLPProgram(600, 32, 32, 6, activation="tanh")
    ix = lambda y, x, c: (10 * y + x) * 6 + c
    p.add_h1_threshold("y", {ix(y, x, 0): 0.05 * (y - 4.5) for y in range(9) for x in range(10)}, gain=1.0)
    for r in range(1, 8):
        w = {}
        for x in range(10):
            value = 1.0 if 2 <= x <= 7 else 0.8
            w[ix(r, x, 2)] = value
            w[ix(r, x, 1)] = -value
        p.add_h1_threshold(f"diver{r}", w, 0.5, 8.0)
    for r in range(1, 9):
        w = {ix(y, x, 0): -0.75 * abs(y - r) for y in range(9) for x in range(10)}
        for x, value in [(3, 1.0), (4, 4.0), (5, 5.0), (6, 4.0), (7, 1.0)]:
            w[ix(r, x, 1)] = value
        p.add_h1_threshold(f"danger{r}", w, 1.5, 5.0)
    p.add_h1_threshold("low_air", {ix(9, 1, 5): -1.0}, -0.5, 12)
    p.add_h1_threshold("carrying", {ix(9, 8, 5): 1.0}, 0.5, 12)
    p.add_h1_threshold("surface", {ix(0, x, 0): 0.5 for x in range(10)}, 0.5, 12)
    p.add_h1_threshold("row_one", {ix(1, x, 0): 0.5 for x in range(10)}, 0.5, 12)
    up = {}
    down = {}
    up_bias = -1.0
    down_bias = -1.0
    def gated_comparison(name, gate, row, direction):
        boundary = math.tanh(0.1 * (row - 4.5))
        p.add_h2_gate(name, {"y": 30.0 * direction, gate: 25.0}, bias=-25.0 - 30.0 * direction * boundary)
    for r in range(1, 8):
        u, d = f"seek_up{r}", f"seek_down{r}"
        gated_comparison(u, f"diver{r}", r + 0.5, 1)
        gated_comparison(d, f"diver{r}", r - 0.5, -1)
        up[u] = 2.5
        down[d] = 2.5
        up_bias += 2.5
        down_bias += 2.5
    for r in range(1, 9):
        if r > 1:
            u = f"escape_up{r}"
            gated_comparison(u, f"danger{r}", r + 0.5, -1)
            up[u] = 5.0
            up_bias += 5.0
        d = f"escape_down{r}"
        gated_comparison(d, f"danger{r}", r - 0.5, 1)
        down[d] = 5.0
        down_bias += 5.0
    p.add_h2_gate("refill", {"low_air": 10, "carrying": 10}, -10)
    p.add_h2_gate("dive", {"surface": 10})
    p.add_h2_gate("protect_surface", {"row_one": 10, "carrying": -10}, -10)
    up.update({"refill": 12.0, "dive": -12.0, "protect_surface": -18.0})
    down.update({"refill": -12.0, "dive": 8.0})
    up_bias += 12.0 - 12.0 - 18.0
    down_bias += -12.0 + 8.0
    p.set_output_action(0, {}, -6.0)
    p.set_output_action(1, {}, -25.0)
    p.set_output_action(2, up, up_bias)
    p.set_output_action(3, {}, -25.0)
    p.set_output_action(4, down, down_bias)
    p.set_output_action(5, {}, 3.5)
    return MLPCompilerBackend.compile(p)
