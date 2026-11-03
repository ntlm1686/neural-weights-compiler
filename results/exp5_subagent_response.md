# Experiment 5 Blind Subagent Response (`exp5_breakout_prompt.txt`)

### Design & Strategy

#### 1. Game Physics & Optimal Paddle Strategy
In MinAtar Breakout:
- The ball moves diagonally by 1 cell per step. Critically, bricks (`channel 2`) and the top wall (`row 0`) only reflect the ball's **vertical** velocity (`ball_dir = [3,2,1,0][ball_dir]`); they never alter its horizontal coordinate `new_x`.
- Side walls (`col 0` and `col 9`) clamp `new_x` to `0` or `9` for one step and flip horizontal velocity.
- Because the ball's horizontal coordinate `ball_x` (`channel 3`) and previous coordinate `last_x` (`channel 4`, trail) are one-hot across columns `0..9`, the exact next horizontal coordinate `new_x` across all 20 states of the horizontal cycle (including side-wall bounces and `reset()` states `(0, 0)` and `(9, 9)`) is given **exact** by a single linear combination of column indicators:
  $$\text{new\_x} = \sum_{c=0}^{9} w^{(b)}_c\,\mathbb{I}(\text{ball\_x}=c) + \sum_{c=0}^{9} w^{(t)}_c\,\mathbb{I}(\text{last\_x}=c)$$
  with $w^{(b)} = [1, 2, 4, 6, 8, 10, 12, 14, 16, 17]$ and $w^{(t)} = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]$.
- When the ball is at `ball_y == 8` moving down (`last_y == 7`, so `new_y == 9` on the current step), `act()` checks:
  - `if self.ball_x == self.pos`: normal vertical reflection (`[3,2,1,0]`, preserving horizontal direction),
  - `elif new_x == self.pos`: retro-reflection (`[2,3,0,1]`, reversing both horizontal and vertical directions).
- **Two-phase tracking rule**:
  1. **In flight (`ball_y < 8` or moving up)**: target `new_x`. This keeps `pos == ball_x` at the start of every step while continuously outputting the action in the direction of ball motion (`left` or `right`), making sticky actions (10% repeat of `last_action`) harmless because `last_action` already matches the ball's horizontal velocity.
  2. **At paddle bounce (`ball_y == 8` and `last_y == 7`)**: target `ball_x`. Since `pos == ball_x` already at the start of `ball_y == 8`, the policy outputs `no-op` (`self.pos = ball_x`, giving normal reflection so the ball sweeps all 10 columns without getting trapped in retro-reflection loops), while if a 10% sticky action repeats the previous movement action `vx`, the paddle moves to `self.pos = ball_x + vx = new_x` and still catches the ball. And if an earlier sticky slip left `pos = ball_x - vx`, targeting `ball_x` moves `pos` to `ball_x` and still catches the ball.

#### 2. Hidden Units
- **Layer 1 (`h1`)**:
  - `h1[0]`: $R_{\text{new}} = \tanh(10(\text{new\_x} - \text{pos} - 0.5)) \in \{-1, +1\}$, $+1$ iff $\text{new\_x} - \text{pos} \ge 1$.
  - `h1[1]`: $L_{\text{new}} = \tanh(10(\text{pos} - \text{new\_x} - 0.5)) \in \{-1, +1\}$, $+1$ iff $\text{new\_x} - \text{pos} \le -1$.
  - `h1[2]`: $R_{\text{curr}} = \tanh(10(\text{ball\_x} - \text{pos} - 0.5)) \in \{-1, +1\}$, $+1$ iff $\text{ball\_x} - \text{pos} \ge 1$.
  - `h1[3]`: $L_{\text{curr}} = \tanh(10(\text{pos} - \text{ball\_x} - 0.5)) \in \{-1, +1\}$, $+1$ iff $\text{ball\_x} - \text{pos} \le -1$.
  - `h1[4]`: $\text{gate} = \tanh(20\,\mathbb{I}(\text{ball\_y}=8) + 20\,\mathbb{I}(\text{last\_y}=7) - 30) \in \{-1, +1\}$, $+1$ iff the ball is at row 8 moving down from row 7.
- **Layer 2 (`h2`)**:
  - `h2[0]`, `h2[1]` pass $R_{\text{new}}, L_{\text{new}}$ when $\text{gate} = -1$ and saturate to $-1$ when $\text{gate} = +1$.
  - `h2[2]`, `h2[3]` pass $R_{\text{curr}}, L_{\text{curr}}$ when $\text{gate} = +1$ and saturate to $-1$ when $\text{gate} = -1$.
- **Output (`logits`)**:
  - Combines $R = \text{h2}[0] + \text{h2}[2] + 1$ and $L = \text{h2}[1] + \text{h2}[3] + 1$ to select action `3` (`right`) when $R = +1$, action `1` (`left`) when $L = +1$, and action `0` (`no-op`) when $R = L = -1$.

```edits
# h1[0]: R_new = tanh(10 * (new_x - pos) - 5)
# h1[1]: L_new = tanh(-10 * (new_x - pos) - 5)
# Channel 3 (ball) column weights for 10 * new_x: [10, 20, 40, 60, 80, 100, 120, 140, 160, 170]
W1[3:600:60, 0] = 10
W1[9:600:60, 0] = 20
W1[15:600:60, 0] = 40
W1[21:600:60, 0] = 60
W1[27:600:60, 0] = 80
W1[33:600:60, 0] = 100
W1[39:600:60, 0] = 120
W1[45:600:60, 0] = 140
W1[51:600:60, 0] = 160
W1[57:600:60, 0] = 170

W1[3:600:60, 1] = -10
W1[9:600:60, 1] = -20
W1[15:600:60, 1] = -40
W1[21:600:60, 1] = -60
W1[27:600:60, 1] = -80
W1[33:600:60, 1] = -100
W1[39:600:60, 1] = -120
W1[45:600:60, 1] = -140
W1[51:600:60, 1] = -160
W1[57:600:60, 1] = -170

# Channel 4 (trail) column weights for 10 * new_x: [0, -10, -20, -30, -40, -50, -60, -70, -80, -90]
W1[10:600:60, 0] = -10
W1[16:600:60, 0] = -20
W1[22:600:60, 0] = -30
W1[28:600:60, 0] = -40
W1[34:600:60, 0] = -50
W1[40:600:60, 0] = -60
W1[46:600:60, 0] = -70
W1[52:600:60, 0] = -80
W1[58:600:60, 0] = -90

W1[10:600:60, 1] = 10
W1[16:600:60, 1] = 20
W1[22:600:60, 1] = 30
W1[28:600:60, 1] = 40
W1[34:600:60, 1] = 50
W1[40:600:60, 1] = 60
W1[46:600:60, 1] = 70
W1[52:600:60, 1] = 80
W1[58:600:60, 1] = 90

# h1[2]: R_curr = tanh(10 * (ball_x - pos) - 5)
# h1[3]: L_curr = tanh(-10 * (ball_x - pos) - 5)
# Channel 3 (ball) column weights for 10 * ball_x: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
W1[9:600:60, 2] = 10
W1[15:600:60, 2] = 20
W1[21:600:60, 2] = 30
W1[27:600:60, 2] = 40
W1[33:600:60, 2] = 50
W1[39:600:60, 2] = 60
W1[45:600:60, 2] = 70
W1[51:600:60, 2] = 80
W1[57:600:60, 2] = 90

W1[9:600:60, 3] = -10
W1[15:600:60, 3] = -20
W1[21:600:60, 3] = -30
W1[27:600:60, 3] = -40
W1[33:600:60, 3] = -50
W1[39:600:60, 3] = -60
W1[45:600:60, 3] = -70
W1[51:600:60, 3] = -80
W1[57:600:60, 3] = -90

# Channel 0 (paddle, row 9) column weights for -10 * pos and +10 * pos
W1[540:600:6, 0] = [0, -10, -20, -30, -40, -50, -60, -70, -80, -90]
W1[540:600:6, 1] = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
W1[540:600:6, 2] = [0, -10, -20, -30, -40, -50, -60, -70, -80, -90]
W1[540:600:6, 3] = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

# h1[4]: gate = tanh(20 * I(ball_y == 8) + 20 * I(last_y == 7) - 30)
W1[483:540:6, 4] = 20
W1[424:480:6, 4] = 20

b1[0:4] = -5
b1[4] = -30

# Layer 2: gate between (R_new, L_new) when h1[4] == -1 and (R_curr, L_curr) when h1[4] == +1
W2[0, 0] = 10
W2[4, 0] = -20
W2[1, 1] = 10
W2[4, 1] = -20
W2[2, 2] = 10
W2[4, 2] = 20
W2[3, 3] = 10
W2[4, 3] = 20
b2[0:4] = -20

# Output layer:
# Action 0 (no-op): -20 * R - 20 * L - 20
# Action 1 (left):  -20 * R + 20 * L
# Action 3 (right): +20 * R - 20 * L
W3[0:4, 0] = [-20, -20, -20, -20]
W3[0:4, 1] = [-20, 20, -20, 20]
W3[0:4, 3] = [20, -20, 20, -20]
b3[0:6] = [-60, 0, -100, 0, -100, -100]
```
