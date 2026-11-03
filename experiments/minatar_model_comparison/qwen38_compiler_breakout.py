def build_policy():
    from ir import MLPProgram
    from backend import MLPCompilerBackend

    prog = MLPProgram(
        input_dim=600,
        h1_dim=32,
        h2_dim=32,
        output_dim=6,
        activation="tanh",
    )

    # --- Layer 1: extract ball column, trail column, paddle column, ball row ---
    # Units 0-9: ball_col_c
    for c in range(10):
        iw = {}
        for r in range(10):
            idx = (r * 10 + c) * 6 + 3  # channel 3 = projectile (ball)
            iw[idx] = 1.0
        prog.add_h1_threshold(f"ball_col_{c}", iw, threshold=0.5, gain=10.0)

    # Units 10-19: trail_col_c
    for c in range(10):
        iw = {}
        for r in range(10):
            idx = (r * 10 + c) * 6 + 4  # channel 4 = trail
            iw[idx] = 1.0
        prog.add_h1_threshold(f"trail_col_{c}", iw, threshold=0.5, gain=10.0)

    # Units 20-29: paddle_col_c (row 9 only, channel 0)
    for c in range(10):
        idx = (9 * 10 + c) * 6 + 0  # channel 0 = player/paddle
        prog.add_h1_threshold(f"paddle_col_{c}", {idx: 1.0}, threshold=0.5, gain=10.0)

    # Unit 30: ball_row (weighted sum of row indices)
    iw = {}
    for r in range(10):
        for c in range(10):
            idx = (r * 10 + c) * 6 + 3
            iw[idx] = float(r)
    prog.add_h1_threshold("ball_row", iw, threshold=0.0, gain=1.0)

    # Unit 31: ball_row_high (ball in lower half, row >= 5)
    iw = {}
    for r in range(5, 10):
        for c in range(10):
            idx = (r * 10 + c) * 6 + 3
            iw[idx] = 1.0
    prog.add_h1_threshold("ball_row_high", iw, threshold=0.5, gain=10.0)

    # --- Layer 2: compute target column for each c (0..9) ---
    # For each target column c, we need to detect: ball is at column b, trail at column t,
    # and the predicted landing column is c.
    # Direction: left if t = b+1, right if t = b-1.
    # If left: target = b - (9 - ball_row)
    # If right: target = b + (9 - ball_row)
    # We approximate by using ball_row as a continuous value. For each (b, t) pair,
    # the target depends on ball_row. We use a weighted sum over possible ball_rows.
    # Since ball_row is in [0, 9], and target = b ± (9 - ball_row), we can write:
    # target = b ± 9 ∓ ball_row
    # So for a given (b, t), the target column is a linear function of ball_row.
    # We'll create H2 units that fire when the predicted target is c, by combining
    # ball_col_b, trail_col_t, and ball_row with appropriate weights.

    # For each target column c (0..9):
    for c in range(10):
        h1_terms = {}
        # Case 1: moving left (trail at b+1, ball at b)
        # target = b - (9 - ball_row) = b - 9 + ball_row
        # We want this to equal c, so ball_row = c - b + 9
        # For each b in 0..8 (so b+1 <= 9):
        for b in range(0, 9):
            t = b + 1
            # ball_row should be c - b + 9
            # ball_row is in [0, 9], so c - b + 9 must be in [0, 9]
            # i.e., b - 9 <= c <= b
            # We add a term that fires when ball_col_b AND trail_col_t AND ball_row ≈ c - b + 9
            # Approximate: use ball_row with weight 1, and add bias to match
            # h2 = tanh(w_b * ball_col_b + w_t * trail_col_t + w_r * ball_row + bias)
            # We want this to be +1 when ball_col_b=1, trail_col_t=1, ball_row = c - b + 9
            # and -1 otherwise.
            # Use AND-gate style: positive weights for ball_col_b and trail_col_t,
            # and a term for ball_row that peaks at the right value.
            # Since ball_row is a continuous value in [0,9], we use:
            # w_r * ball_row + bias_r, where bias_r = -(c - b + 9) * w_r
            # This gives 0 when ball_row = c - b + 9, and we add a small positive
            # offset to make it fire.
            # Actually, let's use a simpler approach: for each (b, t) pair, create
            # a term that contributes positively when ball_col_b and trail_col_t are
            # both active, weighted by how close ball_row is to the required value.
            # We'll use: weight = 1.0 for ball_col_b and trail_col_t, and
            # ball_row weight = 0.5, bias = -(c - b + 9) * 0.5 + 0.5
            # This gives a peak at ball_row = c - b + 9.
            w_bt = 2.0
            w_r = 0.5
            bias_r = -(c - b + 9) * w_r + 0.5
            h1_terms[f"ball_col_{b}"] = h1_terms.get(f"ball_col_{b}", 0.0) + w_bt
            h1_terms[f"trail_col_{t}"] = h1_terms.get(f"trail_col_{t}", 0.0) + w_bt
            h1_terms["ball_row"] = h1_terms.get("ball_row", 0.0) + w_r
            # We'll accumulate bias separately
            if "bias" not in h1_terms:
                h1_terms["bias"] = 0.0
            h1_terms["bias"] += bias_r

        # Case 2: moving right (trail at b-1, ball at b)
        # target = b + (9 - ball_row) = b + 9 - ball_row
        # We want this to equal c, so ball_row = b + 9 - c
        for b in range(1, 10):
            t = b - 1
            w_bt = 2.0
            w_r = 0.5
            bias_r = -(b + 9 - c) * w_r + 0.5
            h1_terms[f"ball_col_{b}"] = h1_terms.get(f"ball_col_{b}", 0.0) + w_bt
            h1_terms[f"trail_col_{t}"] = h1_terms.get(f"trail_col_{t}", 0.0) + w_bt
            h1_terms["ball_row"] = h1_terms.get("ball_row", 0.0) + w_r
            if "bias" not in h1_terms:
                h1_terms["bias"] = 0.0
            h1_terms["bias"] += bias_r

        # Extract bias from h1_terms
        bias_val = h1_terms.pop("bias", 0.0)
        prog.add_h2_gate(f"target_col_{c}", h1_terms, bias=bias_val)

    # --- Output: map target column and paddle column to actions ---
    # Action 1 (left): paddle is to the right of target
    # For each target c, if paddle is at c+1..9, move left
    left_terms = {}
    for c in range(10):
        for p in range(c + 1, 10):
            left_terms[f"target_col_{c}"] = left_terms.get(f"target_col_{c}", 0.0) + 1.0
            left_terms[f"paddle_col_{p}"] = left_terms.get(f"paddle_col_{p}", 0.0) + 1.0
    prog.set_output_action(1, left_terms, bias=-2.0)

    # Action 3 (right): paddle is to the left of target
    right_terms = {}
    for c in range(10):
        for p in range(0, c):
            right_terms[f"target_col_{c}"] = right_terms.get(f"target_col_{c}", 0.0) + 1.0
            right_terms[f"paddle_col_{p}"] = right_terms.get(f"paddle_col_{p}", 0.0) + 1.0
    prog.set_output_action(3, right_terms, bias=-2.0)

    # Action 0 (no-op): paddle is at target
    noop_terms = {}
    for c in range(10):
        noop_terms[f"target_col_{c}"] = noop_terms.get(f"target_col_{c}", 0.0) + 1.0
        noop_terms[f"paddle_col_{c}"] = noop_terms.get(f"paddle_col_{c}", 0.0) + 1.0
    prog.set_output_action(0, noop_terms, bias=-1.0)

    # Actions 2, 4, 5: always low
    prog.set_output_action(2, {}, bias=-10.0)
    prog.set_output_action(4, {}, bias=-10.0)
    prog.set_output_action(5, {}, bias=-10.0)

    return MLPCompilerBackend.compile(prog)
