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

    # Helper to get input index for (row, col, channel)
    def idx(r, c, ch):
        return (r * cols + c) * chs + ch

    # We need to design a policy that:
    # 1. Detects enemies (channel 1: enemy_bullet, channel 2: enemy_fish/sub/diver)
    # 2. Detects oxygen level (channel 5)
    # 3. Detects sub position (channel 0)
    # 4. Makes decisions: move up to surface if oxygen low, fire if enemy ahead, etc.

    # Let's create some H1 units to detect key features:
    
    # H1 unit 0: Detect if there's an enemy in the same row as sub, ahead of sub
    # First, we need to know where the sub is. Channel 0 has sub_front, channel 1 has sub_back.
    # Let's create a unit that detects the sub's row.
    
    # Actually, let's think about this differently. The observation is 10x10x6 flattened.
    # Channel 0: sub_front | sub_back
    # Channel 1: enemy_bullet | enemy_fish | enemy_sub
    # Channel 2: enemy_fish | enemy_sub | diver
    # Channel 3: friendly_bullet
    # Channel 4: trail | sub_back
    # Channel 5: oxygen_gauge | diver_gauge

    # Let's create H1 units to detect:
    # 1. Sub position (row, col) - we can use the fact that channel 0 has exactly one 1
    # 2. Enemy presence in various directions
    # 3. Oxygen level

    # For simplicity, let's create units that detect:
    # - Sub row (10 units, one per row)
    # - Sub col (10 units, one per col)
    # - Enemy ahead (same row, col > sub_col if facing right, or col < sub_col if facing left)
    # - Oxygen low (channel 5 has few 1s)

    # But we only have 32 H1 units. Let's be strategic.

    # H1 unit 0-9: Detect sub row r (sum of channel 0 in row r)
    for r in range(10):
        weights = {}
        for c in range(cols):
            weights[idx(r, c, 0)] = 1.0
        prog.add_h1_threshold(f"sub_row_{r}", weights, threshold=0.5, gain=10.0)

    # H1 unit 10-19: Detect sub col c (sum of channel 0 in col c)
    for c in range(10):
        weights = {}
        for r in range(rows):
            weights[idx(r, c, 0)] = 1.0
        prog.add_h1_threshold(f"sub_col_{c}", weights, threshold=0.5, gain=10.0)

    # H1 unit 20: Detect if there's an enemy in the same row as sub, to the right
    # This is complex because we need to know sub row and col. Let's approximate.
    # Actually, let's detect enemy presence in general.
    
    # H1 unit 20: Total enemy count (channel 1 + channel 2)
    weights = {}
    for r in range(rows):
        for c in range(cols):
            weights[idx(r, c, 1)] = 1.0
            weights[idx(r, c, 2)] = 1.0
    prog.add_h1_threshold("enemy_count", weights, threshold=0.5, gain=5.0)

    # H1 unit 21: Oxygen level (channel 5, row 9)
    # Oxygen gauge is in row 9, channel 5. The number of 1s indicates oxygen level.
    weights = {}
    for c in range(cols):
        weights[idx(9, c, 5)] = 1.0
    prog.add_h1_threshold("oxygen_level", weights, threshold=0.5, gain=5.0)

    # H1 unit 22: Detect if sub is at surface (row 0)
    weights = {}
    for c in range(cols):
        weights[idx(0, c, 0)] = 1.0
    prog.add_h1_threshold("at_surface", weights, threshold=0.5, gain=10.0)

    # H1 unit 23: Detect if there's an enemy in the same row as sub
    # This requires knowing sub row. Let's create a unit for each row that detects enemy in that row.
    # But we're running out of units. Let's use a different approach.

    # Let's use H1 units 23-31 to detect enemy in specific rows
    for r in range(9):  # rows 1-9
        weights = {}
        for c in range(cols):
            weights[idx(r, c, 1)] = 1.0
            weights[idx(r, c, 2)] = 1.0
        prog.add_h1_threshold(f"enemy_row_{r+1}", weights, threshold=0.5, gain=5.0)

    # Now H2 units:
    # H2 unit 0: Fire if enemy ahead
    # H2 unit 1: Move up if oxygen low
    # H2 unit 2: Move down if not at surface and oxygen high
    # H2 unit 3: Move left/right to avoid enemies
    # H2 unit 4: No-op if safe

    # Let's create H2 units that combine H1 signals:
    
    # H2 unit 0: Fire signal - fire if enemy_count > 0 and sub is not at surface
    prog.add_h2_gate("fire_signal", {
        "enemy_count": 1.0,
        "at_surface": -1.0
    }, bias=-0.5)

    # H2 unit 1: Move up signal - move up if oxygen_level is low
    # Oxygen level is in [0, 10] roughly. Low means < 5.
    prog.add_h2_gate("move_up_signal", {
        "oxygen_level": -1.0
    }, bias=2.0)

    # H2 unit 2: Move down signal - move down if not at surface and oxygen is high
    prog.add_h2_gate("move_down_signal", {
        "at_surface": -1.0,
        "oxygen_level": 1.0
    }, bias=-1.0)

    # H2 unit 3: Move left signal - move left if enemy is to the right
    # This is complex. Let's just use enemy_row signals.
    prog.add_h2_gate("move_left_signal", {
        "enemy_row_1": 0.5,
        "enemy_row_2": 0.5,
        "enemy_row_3": 0.5,
        "enemy_row_4": 0.5,
        "enemy_row_5": 0.5,
        "enemy_row_6": 0.5,
        "enemy_row_7": 0.5,
        "enemy_row_8": 0.5,
        "enemy_row_9": 0.5
    }, bias=-2.0)

    # H2 unit 4: Move right signal - move right if enemy is to the left
    prog.add_h2_gate("move_right_signal", {
        "enemy_row_1": 0.5,
        "enemy_row_2": 0.5,
        "enemy_row_3": 0.5,
        "enemy_row_4": 0.5,
        "enemy_row_5": 0.5,
        "enemy_row_6": 0.5,
        "enemy_row_7": 0.5,
        "enemy_row_8": 0.5,
        "enemy_row_9": 0.5
    }, bias=-2.0)

    # H2 unit 5: No-op signal - no-op if no enemies and oxygen is fine
    prog.add_h2_gate("noop_signal", {
        "enemy_count": -1.0,
        "oxygen_level": 0.5
    }, bias=1.0)

    # Output rules:
    # Action 0: no-op
    prog.set_output_action(0, {"noop_signal": 1.0}, bias=0.0)
    
    # Action 1: left
    prog.set_output_action(1, {"move_left_signal": 1.0}, bias=0.0)
    
    # Action 2: up
    prog.set_output_action(2, {"move_up_signal": 1.0}, bias=0.0)
    
    # Action 3: right
    prog.set_output_action(3, {"move_right_signal": 1.0}, bias=0.0)
    
    # Action 4: down
    prog.set_output_action(4, {"move_down_signal": 1.0}, bias=0.0)
    
    # Action 5: fire
    prog.set_output_action(5, {"fire_signal": 1.0}, bias=0.0)

    return MLPCompilerBackend.compile(prog)
