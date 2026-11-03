from neural_compiler import MLPProgram, MLPCompilerBackend


def build_policy():
    p = MLPProgram(input_dim=600, h1_dim=32, h2_dim=32, output_dim=6)

    def weights(ball_fn, paddle_fn=lambda c: 0.0, trail_fn=lambda r, c: 0.0):
        w = {}
        for r in range(10):
            for c in range(10):
                i = (r * 10 + c) * 6
                w[i + 3] = float(ball_fn(r, c))
                w[i + 4] = float(trail_fn(r, c))
                w[i] = float(paddle_fn(c))
        return w

    # The walls retain the ball for one step at x=0 or x=9.
    p.add_h1_threshold('rightward', weights(
        lambda r, c: c + (0.5 if c == 0 else 0.0) - (0.5 if c == 9 else 0.0),
        trail_fn=lambda r, c: -c), gain=10.0)
    # q is the unreflected column when the ball reaches row 8. Reflection
    # occurs around -0.5 and 9.5 because a wall consumes one time step.
    p.add_h1_threshold('right_wall', weights(lambda r, c: c + 8 - r), threshold=9.5, gain=10.0)
    p.add_h1_threshold('left_wall', weights(lambda r, c: -(c - 8 + r)), threshold=0.5, gain=10.0)

    targets = {
        'right_direct': lambda r, c: c + 8 - r,
        'right_reflect': lambda r, c: 19 - (c + 8 - r),
        'left_direct': lambda r, c: c - 8 + r,
        'left_reflect': lambda r, c: -1 - (c - 8 + r),
    }
    for name, target in targets.items():
        p.add_h1_threshold(name + '_right', weights(target, lambda c: -c), threshold=0.5, gain=10.0)
        p.add_h1_threshold(name + '_left', weights(lambda r, c: -target(r, c), lambda c: c), threshold=0.5, gain=10.0)

    cases = [
        ('right_direct', 1.0, 'right_wall', -1.0),
        ('right_reflect', 1.0, 'right_wall', 1.0),
        ('left_direct', -1.0, 'left_wall', -1.0),
        ('left_reflect', -1.0, 'left_wall', 1.0),
    ]
    for case, direction, wall, wall_sign in cases:
        for move in ('left', 'right'):
            p.add_h2_gate(case + '_' + move,
                {'rightward': 8.0 * direction, wall: 8.0 * wall_sign,
                 case + '_' + move: 8.0}, bias=-16.0)

    p.output_bias = [0.0, 15.0, -20.0, 15.0, -20.0, -20.0]
    p.set_output_action(1, {case + '_left': 5.0 for case, _, _, _ in cases}, bias=15.0)
    p.set_output_action(3, {case + '_right': 5.0 for case, _, _, _ in cases}, bias=15.0)
    return MLPCompilerBackend.compile(p)
