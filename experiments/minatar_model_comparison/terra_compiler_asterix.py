def build_policy():
    from neural_compiler import MLPProgram, MLPCompilerBackend
    p = MLPProgram(600, 32, 32, 6)
    def idx(y, x, c):
        return (y * 10 + x) * 6 + c
    for y in range(1, 9):
        for name, channel in (("player", 0), ("enemy", 1), ("gold", 2)):
            p.add_h1_threshold(f"{name}_{y}", {idx(y, x, channel): 1.0 for x in range(10)}, threshold=0.5, gain=8.0)
    up_terms = {}
    down_terms = {}
    for y in range(1, 9):
        for direction, target, rows, output in (
            ("up", y - 1, list(range(1, y)), up_terms),
            ("down", y + 1, list(range(y + 1, 9)), down_terms),
        ):
            if not rows:
                continue
            # Pursue gold in that direction only when the adjacent row has no visible enemy.
            seek = f"seek_{direction}_{y}"
            seek_inputs = {f"player_{y}": 3.0, f"enemy_{target}": -3.0}
            seek_inputs.update({f"gold_{r}": 3.0 for r in rows})
            p.add_h2_gate(seek, seek_inputs, bias=3.0 * (len(rows) - 3))
            output[seek] = 3.0
            # Leave an occupied enemy row for a clear adjacent row.
            escape = f"escape_{direction}_{y}"
            p.add_h2_gate(escape, {f"player_{y}": 3.0, f"enemy_{y}": 3.0, f"enemy_{target}": -3.0}, bias=-6.0)
            output[escape] = 4.0
    p.set_output_action(0, {}, bias=1.5)
    p.set_output_action(1, {}, bias=-5.0)
    p.set_output_action(2, up_terms, bias=0.0)
    p.set_output_action(3, {}, bias=-5.0)
    p.set_output_action(4, down_terms, bias=0.0)
    p.set_output_action(5, {}, bias=-5.0)
    return MLPCompilerBackend.compile(p)
