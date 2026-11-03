"""
Exact MinAtar 1.0.15 Environments (Breakout, Seaquest, Asterix) and 6-Channel Observation Wrappers
as specified in Experiment 5 of 'Sparks of Weight-Space Programming'.
"""

import numpy as np


class BreakoutEnv:
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


class SeaquestEnv:
    def __init__(self, ramping=True):
        self.channels = {
            'sub_front': 0, 'sub_back': 1, 'friendly_bullet': 2, 'trail': 3,
            'enemy_bullet': 4, 'enemy_fish': 5, 'enemy_sub': 6,
            'oxygen_guage': 7, 'diver_guage': 8, 'diver': 9,
        }
        self.action_map = ['n', 'l', 'u', 'r', 'd', 'f']
        self.ramping = ramping
        self.random = np.random.RandomState()
        self.reset()

    def act(self, a):
        r = 0
        if self.terminal:
            return r, self.terminal
        a = self.action_map[a]
        if self.e_spawn_timer == 0:
            self._spawn_enemy()
            self.e_spawn_timer = self.e_spawn_speed
        if self.d_spawn_timer == 0:
            self._spawn_diver()
            self.d_spawn_timer = 30
        if a == 'f' and self.shot_timer == 0:
            self.f_bullets += [[self.sub_x, self.sub_y, self.sub_or]]
            self.shot_timer = 5
        elif a == 'l':
            self.sub_x = max(0, self.sub_x - 1)
            self.sub_or = False
        elif a == 'r':
            self.sub_x = min(9, self.sub_x + 1)
            self.sub_or = True
        elif a == 'u':
            self.sub_y = max(0, self.sub_y - 1)
        elif a == 'd':
            self.sub_y = min(8, self.sub_y + 1)

        for bullet in reversed(self.f_bullets):
            bullet[0] += 1 if bullet[2] else -1
            if bullet[0] < 0 or bullet[0] > 9:
                self.f_bullets.remove(bullet)
            else:
                removed = False
                for x in self.e_fish:
                    if bullet[0:2] == x[0:2]:
                        self.e_fish.remove(x)
                        self.f_bullets.remove(bullet)
                        r += 1
                        removed = True
                        break
                if not removed:
                    for x in self.e_subs:
                        if bullet[0:2] == x[0:2]:
                            self.e_subs.remove(x)
                            self.f_bullets.remove(bullet)
                            r += 1
                            break

        for diver in reversed(self.divers):
            if diver[0:2] == [self.sub_x, self.sub_y] and self.diver_count < 6:
                self.divers.remove(diver)
                self.diver_count += 1
            else:
                if diver[3] == 0:
                    diver[3] = 5
                    diver[0] += 1 if diver[2] else -1
                    if diver[0] < 0 or diver[0] > 9:
                        self.divers.remove(diver)
                    elif diver[0:2] == [self.sub_x, self.sub_y] and self.diver_count < 6:
                        self.divers.remove(diver)
                        self.diver_count += 1
                else:
                    diver[3] -= 1

        for sub in reversed(self.e_subs):
            if sub[0:2] == [self.sub_x, self.sub_y]:
                self.terminal = True
            if sub[3] == 0:
                sub[3] = self.move_speed
                sub[0] += 1 if sub[2] else -1
                if sub[0] < 0 or sub[0] > 9:
                    self.e_subs.remove(sub)
                elif sub[0:2] == [self.sub_x, self.sub_y]:
                    self.terminal = True
                else:
                    for x in self.f_bullets:
                        if sub[0:2] == x[0:2]:
                            self.e_subs.remove(sub)
                            self.f_bullets.remove(x)
                            r += 1
                            break
            else:
                sub[3] -= 1
            if sub[4] == 0:
                sub[4] = 10
                self.e_bullets += [[sub[0], sub[1], sub[2]]]
            else:
                sub[4] -= 1

        for bullet in reversed(self.e_bullets):
            if bullet[0:2] == [self.sub_x, self.sub_y]:
                self.terminal = True
            bullet[0] += 1 if bullet[2] else -1
            if bullet[0] < 0 or bullet[0] > 9:
                self.e_bullets.remove(bullet)
            else:
                if bullet[0:2] == [self.sub_x, self.sub_y]:
                    self.terminal = True

        for fish in reversed(self.e_fish):
            if fish[0:2] == [self.sub_x, self.sub_y]:
                self.terminal = True
            if fish[3] == 0:
                fish[3] = self.move_speed
                fish[0] += 1 if fish[2] else -1
                if fish[0] < 0 or fish[0] > 9:
                    self.e_fish.remove(fish)
                elif fish[0:2] == [self.sub_x, self.sub_y]:
                    self.terminal = True
                else:
                    for x in self.f_bullets:
                        if fish[0:2] == x[0:2]:
                            self.e_fish.remove(fish)
                            self.f_bullets.remove(x)
                            r += 1
                            break
            else:
                fish[3] -= 1

        self.e_spawn_timer -= (self.e_spawn_timer > 0)
        self.d_spawn_timer -= (self.d_spawn_timer > 0)
        self.shot_timer -= (self.shot_timer > 0)
        if self.oxygen <= 0:
            self.terminal = True
        if self.sub_y > 0:
            self.oxygen -= 1
            self.surface = False
        else:
            if not self.surface:
                if self.diver_count == 0:
                    self.terminal = True
                else:
                    r += self._surface()
        return r, self.terminal

    def _surface(self):
        self.surface = True
        if self.diver_count == 6:
            self.diver_count = 0
            r = self.oxygen * 10 // 200
        else:
            r = 0
        self.oxygen = 200
        self.diver_count -= 1
        if self.ramping and (self.e_spawn_speed > 1 or self.move_speed > 2):
            if self.move_speed > 2 and self.ramp_index % 2:
                self.move_speed -= 1
            if self.e_spawn_speed > 1:
                self.e_spawn_speed -= 1
            self.ramp_index += 1
        return r

    def _spawn_enemy(self):
        lr = self.random.rand() < 0.5
        is_sub = self.random.rand() < (1.0 / 3.0)
        x = 0 if lr else 9
        y = self.random.randint(low=1, high=9)
        if any([z[1] == y and z[2] != lr for z in self.e_subs + self.e_fish]):
            return
        if is_sub:
            self.e_subs += [[x, y, lr, self.move_speed, 10]]
        else:
            self.e_fish += [[x, y, lr, self.move_speed]]

    def _spawn_diver(self):
        lr = self.random.rand() < 0.5
        x = 0 if lr else 9
        y = self.random.randint(low=1, high=9)
        self.divers += [[x, y, lr, 5]]

    def state(self):
        state = np.zeros((10, 10, 10), dtype=bool)
        state[self.sub_y, self.sub_x, 0] = 1
        back_x = self.sub_x - 1 if self.sub_or else self.sub_x + 1
        state[self.sub_y, back_x, 1] = 1
        state[9, 0:max(0, self.oxygen) * 10 // 200, 7] = 1
        state[9, 9 - self.diver_count:9, 8] = 1
        for bullet in self.f_bullets:
            state[bullet[1], bullet[0], 2] = 1
        for bullet in self.e_bullets:
            state[bullet[1], bullet[0], 4] = 1
        for fish in self.e_fish:
            state[fish[1], fish[0], 5] = 1
            back_x = fish[0] - 1 if fish[2] else fish[0] + 1
            if 0 <= back_x <= 9:
                state[fish[1], back_x, 3] = 1
        for sub in self.e_subs:
            state[sub[1], sub[0], 6] = 1
            back_x = sub[0] - 1 if sub[2] else sub[0] + 1
            if 0 <= back_x <= 9:
                state[sub[1], back_x, 3] = 1
        for diver in self.divers:
            state[diver[1], diver[0], 9] = 1
            back_x = diver[0] - 1 if diver[2] else diver[0] + 1
            if 0 <= back_x <= 9:
                state[diver[1], back_x, 3] = 1
        return state

    def reset(self):
        self.oxygen = 200
        self.diver_count = 0
        self.sub_x = 5
        self.sub_y = 0
        self.sub_or = False
        self.f_bullets = []
        self.e_bullets = []
        self.e_fish = []
        self.e_subs = []
        self.divers = []
        self.e_spawn_speed = 20
        self.e_spawn_timer = self.e_spawn_speed
        self.d_spawn_timer = 30
        self.move_speed = 5
        self.ramp_index = 0
        self.shot_timer = 0
        self.surface = True
        self.terminal = False


class AsterixEnv:
    def __init__(self, ramping=True):
        self.channels = {'player': 0, 'enemy': 1, 'trail': 2, 'gold': 3}
        self.action_map = ['n', 'l', 'u', 'r', 'd', 'f']
        self.ramping = ramping
        self.random = np.random.RandomState()
        self.reset()

    def act(self, a):
        r = 0
        if self.terminal:
            return r, self.terminal
        a = self.action_map[a]
        if self.spawn_timer == 0:
            self._spawn_entity()
            self.spawn_timer = self.spawn_speed
        if a == 'l':
            self.player_x = max(0, self.player_x - 1)
        elif a == 'r':
            self.player_x = min(9, self.player_x + 1)
        elif a == 'u':
            self.player_y = max(1, self.player_y - 1)
        elif a == 'd':
            self.player_y = min(8, self.player_y + 1)

        for i in range(len(self.entities)):
            x = self.entities[i]
            if x is not None and x[0:2] == [self.player_x, self.player_y]:
                if x[3]:
                    self.entities[i] = None
                    r += 1
                else:
                    self.terminal = True
        if self.move_timer == 0:
            self.move_timer = self.move_speed
            for i in range(len(self.entities)):
                x = self.entities[i]
                if x is not None:
                    x[0] += 1 if x[2] else -1
                    if x[0] < 0 or x[0] > 9:
                        self.entities[i] = None
                    if x[0:2] == [self.player_x, self.player_y]:
                        if x[3]:
                            self.entities[i] = None
                            r += 1
                        else:
                            self.terminal = True
        self.spawn_timer -= 1
        self.move_timer -= 1
        if self.ramping and (self.spawn_speed > 1 or self.move_speed > 1):
            if self.ramp_timer >= 0:
                self.ramp_timer -= 1
            else:
                if self.move_speed > 1 and self.ramp_index % 2:
                    self.move_speed -= 1
                if self.spawn_speed > 1:
                    self.spawn_speed -= 1
                self.ramp_index += 1
                self.ramp_timer = 100
        return r, self.terminal

    def _spawn_entity(self):
        lr = self.random.rand() < 0.5
        is_gold = self.random.rand() < (1.0 / 3.0)
        x = 0 if lr else 9
        slot_options = [i for i in range(len(self.entities)) if self.entities[i] is None]
        if not slot_options:
            return
        slot = slot_options[self.random.randint(len(slot_options))]
        self.entities[slot] = [x, slot + 1, lr, is_gold]

    def state(self):
        state = np.zeros((10, 10, 4), dtype=bool)
        state[self.player_y, self.player_x, 0] = 1
        for x in self.entities:
            if x is not None:
                c = 3 if x[3] else 1
                state[x[1], x[0], c] = 1
                back_x = x[0] - 1 if x[2] else x[0] + 1
                if 0 <= back_x <= 9:
                    state[x[1], back_x, 2] = 1
        return state

    def reset(self):
        self.player_x = 5
        self.player_y = 5
        self.entities = [None] * 8
        self.shot_timer = 0
        self.spawn_speed = 10
        self.spawn_timer = self.spawn_speed
        self.move_speed = 5
        self.move_timer = self.move_speed
        self.ramp_timer = 100
        self.ramp_index = 0
        self.terminal = False


def breakout_obs(s):
    obs = np.zeros((10, 10, 6), dtype=np.float32)
    obs[:, :, 0] = s[:, :, 0]
    obs[:, :, 2] = s[:, :, 3]
    obs[:, :, 3] = s[:, :, 1]
    obs[:, :, 4] = s[:, :, 2]
    return obs.reshape(-1)


def seaquest_obs(s):
    obs = np.zeros((10, 10, 6), dtype=np.float32)
    obs[:, :, 0] = s[:, :, 0] | s[:, :, 1]
    obs[:, :, 1] = s[:, :, 5] | s[:, :, 6] | s[:, :, 4]
    obs[:, :, 2] = s[:, :, 5] | s[:, :, 6] | s[:, :, 9]
    obs[:, :, 3] = s[:, :, 2]
    obs[:, :, 4] = s[:, :, 3] | s[:, :, 1]
    obs[:, :, 5] = s[:, :, 7] | s[:, :, 8]
    return obs.reshape(-1)


def asterix_obs(s):
    obs = np.zeros((10, 10, 6), dtype=np.float32)
    obs[:, :, 0] = s[:, :, 0]
    obs[:, :, 1] = s[:, :, 1]
    obs[:, :, 2] = s[:, :, 3]
    obs[:, :, 4] = s[:, :, 2]
    return obs.reshape(-1)
