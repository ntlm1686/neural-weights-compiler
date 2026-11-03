def build_policy():
    from neural_compiler.ir import MLPProgram
    from neural_compiler.mlp_backend import MLPCompilerBackend
    p = MLPProgram(600, 32, 32, 6, activation="tanh")
    ix = lambda y, x, c: (10 * y + x) * 6 + c
    y_weights = {ix(y, x, 0): float(y) for y in range(1, 9) for x in range(10)}
    for r in range(1, 9):
        p.add_h1_threshold(f"below{r}", y_weights, r + 0.5, 10.0)
        p.add_h1_threshold(f"above{r}", {i: -v for i, v in y_weights.items()}, -r + 0.5, 10.0)
        gold = {ix(y, x, 0): -0.18 * abs(y - r) for y in range(1, 9) for x in range(10)}
        for x in range(10):
            gold[ix(r, x, 2)] = 3.0 - 0.25 * abs(x - 5)
        p.add_h1_threshold(f"gold{r}", gold, 0.5, 6.0)
        danger = {ix(y, x, 0): -3.0 * abs(y - r) for y in range(1, 9) for x in range(10)}
        for x, value in [(3, 2.0), (4, 5.0), (5, 6.0), (6, 5.0), (7, 2.0)]:
            danger[ix(r, x, 1)] = value
        p.add_h1_threshold(f"danger{r}", danger, 1.5, 6.0)
    up = {}
    down = {}
    up_bias = -1.0
    down_bias = -1.0
    for r in range(1, 9):
        gu, gd = f"gold_up{r}", f"gold_down{r}"
        eu, ed = f"escape_up{r}", f"escape_down{r}"
        p.add_h2_gate(gu, {f"gold{r}": 10, f"below{r}": 10}, -10)
        p.add_h2_gate(gd, {f"gold{r}": 10, f"above{r}": 10}, -10)
        p.add_h2_gate(eu, {f"danger{r}": 10, f"below{r}": -10}, -10)
        p.add_h2_gate(ed, {f"danger{r}": 10, f"above{r}": -10}, -10)
        up[gu] = 2.5
        down[gd] = 2.5
        up_bias += 2.5
        down_bias += 2.5
        if r > 1:
            up[eu] = 9.0
            up_bias += 9.0
        if r < 8:
            down[ed] = 9.0
            down_bias += 9.0
    p.set_output_action(0, {}, 2.5)
    p.set_output_action(1, {}, -30.0)
    p.set_output_action(2, up, up_bias)
    p.set_output_action(3, {}, -30.0)
    p.set_output_action(4, down, down_bias)
    p.set_output_action(5, {}, -30.0)
    return MLPCompilerBackend.compile(p)
