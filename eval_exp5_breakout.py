import numpy as np


class Env:
    def __init__(self, ramping=None):
        self.channels = {'paddle': 0, 'ball': 1, 'trail': 2, 'brick': 3}
        self.action_map = ['n', 'l', 'u', 'r', 'd', 'f']
        self.random = np.random.RandomState()
        self.reset()

    def act(self, a):
        r = 0
        if self.terminal:
            return r, self.terminal
        a = self.action_map[a]
        if a == 'l':
            self.pos = max(0, self.pos - 1)
        elif a == 'r':
            self.pos = min(9, self.pos + 1)
        self.last_x = self.ball_x
        self.last_y = self.ball_y
        if self.ball_dir == 0:
            new_x, new_y = self.ball_x - 1, self.ball_y - 1
        elif self.ball_dir == 1:
            new_x, new_y = self.ball_x + 1, self.ball_y - 1
        elif self.ball_dir == 2:
            new_x, new_y = self.ball_x + 1, self.ball_y + 1
        elif self.ball_dir == 3:
            new_x, new_y = self.ball_x - 1, self.ball_y + 1

        strike_toggle = False
        if new_x < 0 or new_x > 9:
            new_x = 0 if new_x < 0 else 9
            self.ball_dir = [1, 0, 3, 2][self.ball_dir]
        if new_y < 0:
            new_y = 0
            self.ball_dir = [3, 2, 1, 0][self.ball_dir]
        elif self.brick_map[new_y, new_x] == 1:
            strike_toggle = True
            if not self.strike:
                r += 1
                self.strike = True
                self.brick_map[new_y, new_x] = 0
                new_y = self.last_y
                self.ball_dir = [3, 2, 1, 0][self.ball_dir]
        elif new_y == 9:
            if np.count_nonzero(self.brick_map) == 0:
                self.brick_map[1:4, :] = 1
            if self.ball_x == self.pos:
                self.ball_dir = [3, 2, 1, 0][self.ball_dir]
                new_y = self.last_y
            elif new_x == self.pos:
                self.ball_dir = [2, 3, 0, 1][self.ball_dir]
                new_y = self.last_y
            else:
                self.terminal = True
        if not strike_toggle:
            self.strike = False
        self.ball_x = new_x
        self.ball_y = new_y
        return r, self.terminal

    def state(self):
        state = np.zeros((10, 10, len(self.channels)), dtype=bool)
        state[self.ball_y, self.ball_x, self.channels['ball']] = 1
        state[9, self.pos, self.channels['paddle']] = 1
        state[self.last_y, self.last_x, self.channels['trail']] = 1
        state[:, :, self.channels['brick']] = self.brick_map
        return state

    def reset(self):
        self.ball_y = 3
        ball_start = self.random.randint(2)
        self.ball_x, self.ball_dir = [(0, 2), (9, 3)][ball_start]
        self.pos = 4
        self.brick_map = np.zeros((10, 10))
        self.brick_map[1:4, :] = 1
        self.strike = False
        self.last_x = self.ball_x
        self.last_y = self.ball_y
        self.terminal = False


def build_subagent_weights():
    W1 = np.zeros((600, 32), dtype=np.float32)
    b1 = np.zeros((32,), dtype=np.float32)
    W2 = np.zeros((32, 32), dtype=np.float32)
    b2 = np.zeros((32,), dtype=np.float32)
    W3 = np.zeros((32, 6), dtype=np.float32)
    b3 = np.zeros((6,), dtype=np.float32)

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

    W1[540:600:6, 0] = [0, -10, -20, -30, -40, -50, -60, -70, -80, -90]
    W1[540:600:6, 1] = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    W1[540:600:6, 2] = [0, -10, -20, -30, -40, -50, -60, -70, -80, -90]
    W1[540:600:6, 3] = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

    W1[483:540:6, 4] = 20
    W1[424:480:6, 4] = 20

    b1[0:4] = -5
    b1[4] = -30

    W2[0, 0] = 10
    W2[4, 0] = -20
    W2[1, 1] = 10
    W2[4, 1] = -20
    W2[2, 2] = 10
    W2[4, 2] = 20
    W2[3, 3] = 10
    W2[4, 3] = 20
    b2[0:4] = -20

    W3[0:4, 0] = [-20, -20, -20, -20]
    W3[0:4, 1] = [-20, 20, -20, 20]
    W3[0:4, 3] = [20, -20, 20, -20]
    b3[0:6] = [-60, 0, -100, 0, -100, -100]
    return W1, b1, W2, b2, W3, b3


def state_to_obs(s):
    obs = np.zeros((10, 10, 6), dtype=np.float32)
    obs[:, :, 0] = s[:, :, 0]
    obs[:, :, 2] = s[:, :, 3]
    obs[:, :, 3] = s[:, :, 1]
    obs[:, :, 4] = s[:, :, 2]
    return obs.reshape(-1)


if __name__ == '__main__':
    W1, b1, W2, b2, W3, b3 = build_subagent_weights()
    nonzero = sum(np.count_nonzero(t) for t in [W1, b1, W2, b2, W3, b3])

    rng = np.random.RandomState(0)
    env = Env()
    env.random = np.random.RandomState(0)
    scores, lengths = [], []
    for _ in range(1000):
        env.reset()
        last_action = 0
        ep_return = 0
        steps = 0
        while not env.terminal and steps < 10000:
            x = state_to_obs(env.state())
            h1 = np.tanh(x @ W1 + b1)
            h2 = np.tanh(h1 @ W2 + b2)
            logits = h2 @ W3 + b3
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

    print(f"Nonzero weights: {nonzero} / 20486")
    print(f"1000-episode Mean Score: {np.mean(scores):.2f} +/- {np.std(scores)/np.sqrt(len(scores)):.2f}")
    print(f"Mean Episode Length: {np.mean(lengths):.1f}")
