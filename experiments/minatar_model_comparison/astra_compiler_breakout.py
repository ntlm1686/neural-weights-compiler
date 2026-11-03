from neural_compiler.ir import MLPProgram
from neural_compiler.mlp_backend import ExecutableMLP, MLPCompilerBackend


def build_policy() -> ExecutableMLP:
    program = MLPProgram(input_dim=600, h1_dim=32, h2_dim=32, output_dim=6, activation="tanh")
    def index(row, column, channel):
        return (row * 10 + column) * 6 + channel
    def reflected_column(unfolded):
        # The wall repeats its edge cell for one step.
        phase = unfolded % 20
        return phase if phase <= 9 else 19 - phase
    def landing_target(row, column, direction, desired_direction):
        steps_to_eight = max(0, 8 - row)
        before = reflected_column(column + direction * steps_to_eight)
        after = reflected_column(column + direction * (steps_to_eight + 1))
        if before == after:
            return before
        incoming_direction = 1 if after > before else -1
        return before if incoming_direction == desired_direction else after
    horizontal_weights = {}
    descending_weights = {}
    brick_weights = {}
    for row in range(10):
        for column in range(10):
            ball = index(row, column, 3)
            trail = index(row, column, 4)
            edge_correction = 0.25 if column == 0 else -0.25 if column == 9 else 0.0
            horizontal_weights[ball] = column + edge_correction
            horizontal_weights[trail] = -column
            descending_weights[ball] = row
            descending_weights[trail] = -row
            brick_weights[index(row, column, 2)] = column - 4.5
    program.add_h1_threshold("moving_right", horizontal_weights, threshold=0.0, gain=24.0)
    program.add_h1_threshold("descending", descending_weights, threshold=0.5, gain=16.0)
    program.add_h1_threshold("prefer_right", brick_weights, threshold=-0.25, gain=24.0)
    left_terms = {}
    right_terms = {}
    for direction in (-1, 1):
        for desired_direction in (-1, 1):
            suffix = (("east" if direction > 0 else "west") + "_" + ("east" if desired_direction > 0 else "west"))
            difference = {}
            for column in range(10):
                difference[index(9, column, 0)] = float(column)
            for row in range(10):
                for column in range(10):
                    target = landing_target(row, column, direction, desired_direction)
                    difference[index(row, column, 3)] = -float(target)
            for action_name, sign, output_terms in (("left", 1.0, left_terms), ("right", -1.0, right_terms)):
                comparator = "need_" + action_name + "_" + suffix
                program.add_h1_threshold(comparator, {input_index: sign * weight for input_index, weight in difference.items()}, threshold=0.5, gain=16.0)
                gate = "catch_" + action_name + "_" + suffix
                program.add_h2_gate(gate, {comparator: 8.0, "moving_right": 8.0 * direction, "prefer_right": 8.0 * desired_direction, "descending": 8.0}, bias=-24.0)
                output_terms[gate] = 9.0
    paddle_weights = {index(9, column, 0): float(column) for column in range(10)}
    program.add_h1_threshold("right_of_center", paddle_weights, threshold=5.5, gain=16.0)
    program.add_h1_threshold("left_of_center", {input_index: -weight for input_index, weight in paddle_weights.items()}, threshold=-3.5, gain=16.0)
    program.add_h2_gate("center_left", {"right_of_center": 8.0, "descending": -8.0}, bias=-8.0)
    program.add_h2_gate("center_right", {"left_of_center": 8.0, "descending": -8.0}, bias=-8.0)
    left_terms["center_left"] = 9.0
    right_terms["center_right"] = 9.0
    program.set_output_action(0, {}, bias=0.0)
    program.set_output_action(1, left_terms, bias=36.0)
    program.set_output_action(2, {}, bias=-30.0)
    program.set_output_action(3, right_terms, bias=36.0)
    program.set_output_action(4, {}, bias=-30.0)
    program.set_output_action(5, {}, bias=-30.0)
    return MLPCompilerBackend.compile(program)
