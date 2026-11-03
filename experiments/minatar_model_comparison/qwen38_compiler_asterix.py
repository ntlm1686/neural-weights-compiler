import numpy as np
from ir import MLPProgram
from backend import MLPCompilerBackend


def build_policy():
    prog = MLPProgram(
        input_dim=600,
        h1_dim=32,
        h2_dim=32,
        output_dim=6,
        activation="tanh"
    )

    # Grid shape: 10x10x6
    rows, cols, chs = 10, 10, 6

    # Channel indices in the 6-channel observation:
    # 0: player
    # 1: enemy
    # 2: gold
    # 3: (unused in obs wrapper, but present in grid)
    # 4: trail
    # 5: (unused)

    # We need to detect:
    # 1. Player position (to know where we are)
    # 2. Enemy positions (to avoid)
    # 3. Gold positions (to collect)
    # 4. Trail positions (to avoid or use for context)

    # Strategy:
    # - Detect if there's an enemy in each of the 4 cardinal directions
    # - Detect if there's gold in each of the 4 cardinal directions
    # - Prefer moving toward gold if safe, otherwise move away from enemies
    # - If no clear direction, fire or no-op

    # H1 units: Detect presence of entities in specific relative positions
    # We'll use threshold units to detect if any enemy/gold/trail is in a specific row/col offset

    # For simplicity, let's detect:
    # - Enemy directly left, right, up, down
    # - Gold directly left, right, up, down
    # - Trail directly left, right, up, down

    # But we don't know player position directly. However, the player is always at exactly one position.
    # We can use the fact that player channel has exactly one 1.

    # Alternative approach: Use spatial coordinate extractors to compute weighted sums
    # that indicate direction of nearest enemy/gold.

    # Let's create H1 units that compute:
    # - Sum of enemy positions weighted by x-coordinate (to know if enemies are left/right)
    # - Sum of enemy positions weighted by y-coordinate (to know if enemies are up/down)
    # - Same for gold

    # But this is complex. Let's use a simpler heuristic:
    # Detect if there's an enemy in the same row to the left/right
    # Detect if there's an enemy in the same column to the up/down

    # Since we can't easily do relative position detection without knowing player pos,
    # let's use a different approach:
    # - Count total enemies, gold, trails
    # - Use these counts to bias actions

    # H1 unit 0: Total enemy count
    enemy_indices = []
    for r in range(rows):
        for c in range(cols):
            idx = (r * cols + c) * chs + 1  # enemy channel
            enemy_indices.append(idx)

    prog.add_h1_threshold(
        name="enemy_count",
        input_weights={idx: 1.0 for idx in enemy_indices},
        threshold=0.5,
        gain=5.0
    )

    # H1 unit 1: Total gold count
    gold_indices = []
    for r in range(rows):
        for c in range(cols):
            idx = (r * cols + c) * chs + 2  # gold channel
            gold_indices.append(idx)

    prog.add_h1_threshold(
        name="gold_count",
        input_weights={idx: 1.0 for idx in gold_indices},
        threshold=0.5,
        gain=5.0
    )

    # H1 unit 2: Total trail count
    trail_indices = []
    for r in range(rows):
        for c in range(cols):
            idx = (r * cols + c) * chs + 4  # trail channel
            trail_indices.append(idx)

    prog.add_h1_threshold(
        name="trail_count",
        input_weights={idx: 1.0 for idx in trail_indices},
        threshold=0.5,
        gain=5.0
    )

    # H1 unit 3: Player x-coordinate (weighted sum of x positions where player is)
    player_x_weights = {}
    for r in range(rows):
        for c in range(cols):
            idx = (r * cols + c) * chs + 0  # player channel
            player_x_weights[idx] = float(c)

    prog.add_h1_threshold(
        name="player_x",
        input_weights=player_x_weights,
        threshold=0.0,
        gain=1.0
    )

    # H1 unit 4: Player y-coordinate
    player_y_weights = {}
    for r in range(rows):
        for c in range(cols):
            idx = (r * cols + c) * chs + 0  # player channel
            player_y_weights[idx] = float(r)

    prog.add_h1_threshold(
        name="player_y",
        input_weights=player_y_weights,
        threshold=0.0,
        gain=1.0
    )

    # H1 unit 5: Enemy x-weighted sum (to know average enemy x position)
    enemy_x_weights = {}
    for r in range(rows):
        for c in range(cols):
            idx = (r * cols + c) * chs + 1  # enemy channel
            enemy_x_weights[idx] = float(c)

    prog.add_h1_threshold(
        name="enemy_x_avg",
        input_weights=enemy_x_weights,
        threshold=0.0,
        gain=1.0
    )

    # H1 unit 6: Enemy y-weighted sum
    enemy_y_weights = {}
    for r in range(rows):
        for c in range(cols):
            idx = (r * cols + c) * chs + 1  # enemy channel
            enemy_y_weights[idx] = float(r)

    prog.add_h1_threshold(
        name="enemy_y_avg",
        input_weights=enemy_y_weights,
        threshold=0.0,
        gain=1.0
    )

    # H1 unit 7: Gold x-weighted sum
    gold_x_weights = {}
    for r in range(rows):
        for c in range(cols):
            idx = (r * cols + c) * chs + 2  # gold channel
            gold_x_weights[idx] = float(c)

    prog.add_h1_threshold(
        name="gold_x_avg",
        input_weights=gold_x_weights,
        threshold=0.0,
        gain=1.0
    )

    # H1 unit 8: Gold y-weighted sum
    gold_y_weights = {}
    for r in range(rows):
        for c in range(cols):
            idx = (r * cols + c) * chs + 2  # gold channel
            gold_y_weights[idx] = float(r)

    prog.add_h1_threshold(
        name="gold_y_avg",
        input_weights=gold_y_weights,
        threshold=0.0,
        gain=1.0
    )

    # H2 units: Combine H1 signals to produce action scores
    # H2 unit 0: Move left score
    prog.add_h2_gate(
        name="move_left_score",
        h1_terms={
            "player_x": 1.0,
            "enemy_x_avg": -1.0,
            "gold_x_avg": 1.0,
            "enemy_count": -0.5,
            "gold_count": 0.5
        },
        bias=0.0
    )

    # H2 unit 1: Move right score
    prog.add_h2_gate(
        name="move_right_score",
        h1_terms={
            "player_x": -1.0,
            "enemy_x_avg": 1.0,
            "gold_x_avg": -1.0,
            "enemy_count": -0.5,
            "gold_count": 0.5
        },
        bias=0.0
    )

    # H2 unit 2: Move up score
    prog.add_h2_gate(
        name="move_up_score",
        h1_terms={
            "player_y": 1.0,
            "enemy_y_avg": -1.0,
            "gold_y_avg": 1.0,
            "enemy_count": -0.5,
            "gold_count": 0.5
        },
        bias=0.0
    )

    # H2 unit 3: Move down score
    prog.add_h2_gate(
        name="move_down_score",
        h1_terms={
            "player_y": -1.0,
            "enemy_y_avg": 1.0,
            "gold_y_avg": -1.0,
            "enemy_count": -0.5,
            "gold_count": 0.5
        },
        bias=0.0
    )

    # H2 unit 4: Fire score
    prog.add_h2_gate(
        name="fire_score",
        h1_terms={
            "enemy_count": 1.0,
            "trail_count": -0.5
        },
        bias=0.0
    )

    # H2 unit 5: No-op score
    prog.add_h2_gate(
        name="noop_score",
        h1_terms={
            "enemy_count": -0.5,
            "gold_count": -0.5
        },
        bias=0.5
    )

    # Output rules: Map H2 scores to action logits
    # Action 0: no-op
    prog.set_output_action(
        action_idx=0,
        h2_terms={"noop_score": 1.0},
        bias=0.0
    )

    # Action 1: left
    prog.set_output_action(
        action_idx=1,
        h2_terms={"move_left_score": 1.0},
        bias=0.0
    )

    # Action 2: up
    prog.set_output_action(
        action_idx=2,
        h2_terms={"move_up_score": 1.0},
        bias=0.0
    )

    # Action 3: right
    prog.set_output_action(
        action_idx=3,
        h2_terms={"move_right_score": 1.0},
        bias=0.0
    )

    # Action 4: down
    prog.set_output_action(
        action_idx=4,
        h2_terms={"move_down_score": 1.0},
        bias=0.0
    )

    # Action 5: fire
    prog.set_output_action(
        action_idx=5,
        h2_terms={"fire_score": 1.0},
        bias=0.0
    )

    return MLPCompilerBackend.compile(prog)
