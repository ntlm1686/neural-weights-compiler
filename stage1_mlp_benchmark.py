"""
Stage 1 Neural Weight Compiler Benchmark: MinAtar Breakout
==========================================================
Demonstrates compiling a multi-mode trajectory-lookahead + sticky-action-resilient
paddle controller into the exact `600 -> 32 -> 32 -> 6` tanh MLP architecture via
`MLPProgram` and `MLPCompilerBackend`, with zero gradient training.
"""

import numpy as np
from neural_compiler import MLPProgram, MLPCompilerBackend
from eval_exp5_breakout import Env, state_to_obs


def simulate_landing_col(row: int, col: int, vx: int, vy: int, bounce_row: int = 3) -> float:
    """
    Exact symbolic trajectory lookahead for a diagonal ball at (row, col) with velocity (vx, vy):
    traces wall reflections (cols 0 and 9, row 0) and anticipated brick-layer reflection
    at `bounce_row` when ascending, returning the column `0..9` where the ball reaches row 8.
    """
    x, y = col, row
    dx, dy = vx, vy
    for _ in range(32):
        if y == 8 and dy > 0:
            return float(x)
        nx, ny = x + dx, y + dy
        if nx < 0 or nx > 9:
            nx = 0 if nx < 0 else 9
            dx = -dx
        if ny < 0:
            ny = 0
            dy = -dy
        elif ny <= bounce_row and dy < 0:
            dy = -dy
            ny = y
        x, y = nx, ny
    return float(x)


def compile_breakout_policy() -> MLPProgram:
    """
    Compiles a trajectory-lookahead Breakout controller into `MLPProgram` Neural IR:
      1. Direction detectors in H1:
         - `vx_right`: +1 when ball moves right, -1 when moving left
         - `vy_down`:  +1 when ball moves down, -1 when moving up
      2. Four trajectory-lookahead target comparators (for rows 0..6) + exact close-range
         next-step / bounce-step comparators (for rows 7..8):
         - For rows 0..6 descending (`vy=+1, vx=+1/-1`), targets the exact wall-reflected
           landing column at row 8 so the paddle arrives early and waits with `no-op`
           (making 10% sticky actions repeat `no-op` instead of overshooting!).
         - For rows 7..8, smoothly hands over to exact next-step tracking (`new_x`) at row 7
           and current-column (`ball_x`) at row 8.
    """
    prog = MLPProgram(
        input_dim=600,
        h1_dim=32,
        h2_dim=32,
        output_dim=6,
        activation="tanh",
    )
    CH_PADDLE = 0
    CH_BALL = 3
    CH_TRAIL = 4

    # 1. Horizontal velocity detector: vx_right in {-1, +1}
    # Compare ball_x - last_x, with side-wall bounce adjustment at c=0 and c=9:
    #   - If ball_x == 0 and last_x == 1: moving left (-0.5)
    #   - If ball_x == 0 and last_x == 0: bounced off left wall, now moving right (+0.5)
    #   - If ball_x == 9 and last_x == 8: moving right (+0.5)
    #   - If ball_x == 9 and last_x == 9: bounced off right wall, now moving left (-0.5)
    w_vx = {}
    for r in range(10):
        for c in range(10):
            b_idx = (r * 10 + c) * 6 + CH_BALL
            t_idx = (r * 10 + c) * 6 + CH_TRAIL
            if c == 0:
                w_vx[b_idx] = 0.5
            elif c == 9:
                w_vx[b_idx] = 8.5
            else:
                w_vx[b_idx] = float(c)
            w_vx[t_idx] = -float(c)
    prog.add_h1_threshold("vx_right", w_vx, threshold=0.0, gain=15.0)

    # 2. Vertical velocity detector: vy_down in {-1, +1}
    # At row 8 after paddle bounce, ball_y == 8 and last_y == 8 -> ball is now ascending (-0.5)
    # At rows 0..3 after brick/top bounce, ball_y == last_y -> ball is now descending (+0.5)
    w_vy = {}
    for r in range(10):
        for c in range(10):
            if r >= 8:
                w_vy[(r * 10 + c) * 6 + CH_BALL] = float(r) - 0.5
            else:
                w_vy[(r * 10 + c) * 6 + CH_BALL] = float(r) + 0.5
            w_vy[(r * 10 + c) * 6 + CH_TRAIL] = -float(r)
    prog.add_h1_threshold("vy_down", w_vy, threshold=0.0, gain=15.0)

    # 3. Four Quadrant Lookahead Error Signals (Rows 0..7 -> landing col, Row 8 -> ball_x):
    #   Q0: descending right (vy=+1, vx=+1)
    #   Q1: descending left  (vy=+1, vx=-1)
    #   Q2: ascending right  (vy=-1, vx=+1)
    #   Q3: ascending left   (vy=-1, vx=-1)
    quadrants = [
        ("desc_R", +1, +1),
        ("desc_L", -1, +1),
        ("asc_R",  +1, -1),
        ("asc_L",  -1, -1),
    ]
    for q_name, vx, vy in quadrants:
        w_q = {}
        for r in range(10):
            for c in range(10):
                if r >= 8 and vy > 0:
                    target_c = float(c)
                elif r == 7 and vy > 0:
                    # At row 7 descending, next step at row 8 is new_x; midpoint betweenr7 and r8
                    # accepts both ball_x (at r8) and new_x (at r9) or steps directly to new_x
                    target_c = simulate_landing_col(r, c, vx=vx, vy=+1)
                else:
                    target_c = simulate_landing_col(r, c, vx=vx, vy=vy, bounce_row=3)
                w_q[(r * 10 + c) * 6 + CH_BALL] = target_c
        for c in range(10):
            w_q[(9 * 10 + c) * 6 + CH_PADDLE] = -float(c)

        prog.add_h1_threshold(f"{q_name}_R", w_q, threshold=0.5, gain=10.0)
        prog.add_h1_threshold(f"{q_name}_L", {k: -v for k, v in w_q.items()}, threshold=0.5, gain=10.0)

    # Layer 2: 3-input AND gates selecting the active quadrant's Right/Left signal!
    # An H2 unit with weights (+10 signal, +10 vy_sign, +10 vx_sign) and bias -25
    # outputs tanh(+5) ~ +1.0 ONLY when all three inputs are +1, and tanh(<= -15) ~ -1.0 otherwise!
    for q_name, vx_sign, vy_sign in [
        ("desc_R", +1.0, +1.0),
        ("desc_L", -1.0, +1.0),
        ("asc_R",  +1.0, -1.0),
        ("asc_L",  -1.0, -1.0),
    ]:
        prog.add_h2_gate(
            f"act_{q_name}_R",
            {f"{q_name}_R": 10.0, "vx_right": 10.0 * vx_sign, "vy_down": 10.0 * vy_sign},
            bias=-25.0,
        )
        prog.add_h2_gate(
            f"act_{q_name}_L",
            {f"{q_name}_L": 10.0, "vx_right": 10.0 * vx_sign, "vy_down": 10.0 * vy_sign},
            bias=-25.0,
        )

    # Output Layer:
    # Exactly one quadrant is active; if neither R nor L is triggered in that quadrant,
    # all 8 H2 units are -1, so Action 0 (no-op) wins!
    # Let's check the algebra when all 8 H2 units are -1:
    #   If R_active = +1 (and other 7 units are -1):
    #     sum(R_units) = -2, sum(L_units) = -4
    #   If no move needed (all 8 units are -1):
    #     sum(R_units) = -4, sum(L_units) = -4
    r_units = [f"act_{q}_R" for q, _, _ in quadrants]
    l_units = [f"act_{q}_L" for q, _, _ in quadrants]

    # Action 0 (no-op): -20 * sum(R_units + L_units) - 140
    #   When all 8 are -1: -20 * (-8) - 140 = +20
    #   When one is +1 (sum = -6): -20 * (-6) - 140 = -20
    prog.set_output_action(
        0,
        {u: -20.0 for u in (r_units + l_units)},
        bias=-140.0,
    )
    # Action 1 (LEFT): +20 * sum(L_units) - 20 * sum(R_units)
    #   When one L_unit is +1: 20*(-2) - 20*(-4) = +40 (beats no-op -20 and RIGHT -40!)
    #   When all 8 are -1: 20*(-4) - 20*(-4) = 0 (loses to no-op +20!)
    prog.set_output_action(
        1,
        {**{u: 20.0 for u in l_units}, **{u: -20.0 for u in r_units}},
        bias=0.0,
    )
    # Action 3 (RIGHT): +20 * sum(R_units) - 20 * sum(L_units)
    prog.set_output_action(
        3,
        {**{u: 20.0 for u in r_units}, **{u: -20.0 for u in l_units}},
        bias=0.0,
    )
    for unused_a in [2, 4, 5]:
        prog.set_output_action(unused_a, {}, bias=-100.0)

    return prog


def evaluate_mlp_policy(mlp, n_episodes: int = 1000, seed: int = 0):
    rng = np.random.RandomState(seed)
    env = Env()
    env.random = np.random.RandomState(seed)
    scores, lengths = [], []
    for _ in range(n_episodes):
        env.reset()
        last_action = 0
        ep_return = 0
        steps = 0
        while not env.terminal and steps < 10000:
            x = state_to_obs(env.state())
            logits = mlp.forward(x)
            logits = logits - np.max(logits)
            probs = np.exp(logits) / np.sum(np.exp(logits))
            action = rng.choice(6, p=probs)
            if env.random.rand() < 0.1:
                action = last_action
            last_action = action
            r, _ = env.act(action)
            ep_return += r
            steps += 1
        scores.append(ep_return)
        lengths.append(steps)
    return float(np.mean(scores)), float(np.std(scores) / np.sqrt(n_episodes)), float(np.mean(lengths))


if __name__ == "__main__":
    prog = compile_breakout_policy()
    compiled_mlp = MLPCompilerBackend.compile(prog)
    mean_score, se_score, mean_len = evaluate_mlp_policy(compiled_mlp, n_episodes=1000)
    print(f"[Stage 1 NeuralCompiler - Breakout]")
    print(f"  H1 units used: {len(prog.h1_units)} / {prog.h1_dim}")
    print(f"  H2 units used: {len(prog.h2_units)} / {prog.h2_dim}")
    print(f"  Nonzero weights: {compiled_mlp.nonzero_params} / {compiled_mlp.total_params}")
    print(f"  1000-episode Mean Score: {mean_score:.2f} +/- {se_score:.2f} (Mean length: {mean_len:.1f})")
