"""
1.48M-Parameter Spatial-Graph Neural Compiler for 9x9 Go (243 -> 512x6 -> 82 ReLU)
==================================================================================
Demonstrates compiling a 6-layer spatial-graph tactical & territorial program into
the exact 1,480,274-parameter architecture (`243 -> 512 x 6 -> 82` ReLU MLP) from
Experiment 6 of 'Sparks of Weight-Space Programming', and evaluates it head-to-head
against:
  1. GPT-6 Astra's zero-pass (`budget_0`) 1.48M-parameter network
  2. The blog's 20.5-million-game same-size PPO network (`ppo_same_size_lr3e-5_s0`)
"""

import os
from typing import Dict, List, Tuple
import numpy as np


# ==============================================================================
# 1. Exact 9x9 Go Environment (Tromp-Taylor / PGX Area Scoring, Komi 7.5)
# ==============================================================================
N_PTS = 81
PASS_MOVE = 81

ADJ: List[List[int]] = []
DIAG: List[List[int]] = []
RINGS: List[List[Tuple[int, int, int]]] = []
for p in range(N_PTS):
    r, c = divmod(p, 9)
    ADJ.append([
        rr * 9 + cc
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
        if 0 <= rr < 9 and 0 <= cc < 9
    ])
    DIAG.append([
        rr * 9 + cc
        for rr, cc in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1))
        if 0 <= rr < 9 and 0 <= cc < 9
    ])
    RINGS.append([(q, abs(r - q // 9), abs(c - q % 9)) for q in range(N_PTS) if q != p])


class Go9x9Env:
    """Fast 9x9 Go environment matching PGX `go_9x9` rules (area scoring, komi 7.5, simple ko, superko loss)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros(N_PTS, dtype=np.int8)  # 0: empty, 1: Black, -1: White
        self.turn = 1  # 1: Black, -1: White
        self.ko_point = -1
        self.passes = 0
        self.moves = 0
        self.terminal = False
        self.winner = 0  # 1: Black, -1: White
        self.history = {self.board.tobytes()}

    def _find_group_and_liberties(self, board: np.ndarray, start: int) -> Tuple[List[int], int]:
        color = board[start]
        group = [start]
        visited = {start}
        liberties = set()
        idx = 0
        while idx < len(group):
            curr = group[idx]
            idx += 1
            for q in ADJ[curr]:
                v = board[q]
                if v == 0:
                    liberties.add(q)
                elif v == color and q not in visited:
                    visited.add(q)
                    group.append(q)
        return group, len(liberties)

    def legal_mask(self) -> np.ndarray:
        mask = np.zeros(82, dtype=bool)
        mask[PASS_MOVE] = True
        c = self.turn
        opp = -c
        for p in range(N_PTS):
            if self.board[p] != 0 or p == self.ko_point:
                continue
            # Fast check: if any adjacent point is empty, move has at least 1 liberty -> legal
            has_empty_adj = False
            for q in ADJ[p]:
                if self.board[q] == 0:
                    has_empty_adj = True
                    break
            if has_empty_adj:
                mask[p] = True
                continue
            # Otherwise check if it captures an opponent group or connects to a friendly group with >= 2 liberties
            legal = False
            for q in ADJ[p]:
                if self.board[q] == opp:
                    _, libs = self._find_group_and_liberties(self.board, q)
                    if libs == 1:
                        legal = True
                        break
                elif self.board[q] == c:
                    _, libs = self._find_group_and_liberties(self.board, q)
                    if libs >= 2:
                        legal = True
                        break
            mask[p] = legal
        return mask

    def step(self, action: int):
        if self.terminal:
            return
        self.moves += 1
        if action == PASS_MOVE:
            self.passes += 1
            self.ko_point = -1
            if self.passes >= 2 or self.moves >= 162:
                self.terminal = True
                self.winner = self._score_winner()
            else:
                self.turn = -self.turn
            return

        self.passes = 0
        c = self.turn
        opp = -c
        self.board[action] = c

        captured = []
        visited_opp = set()
        for q in ADJ[action]:
            if self.board[q] == opp and q not in visited_opp:
                grp, libs = self._find_group_and_liberties(self.board, q)
                visited_opp.update(grp)
                if libs == 0:
                    captured.extend(grp)

        for q in captured:
            self.board[q] = 0

        # Simple ko check
        if len(captured) == 1:
            own_grp, own_libs = self._find_group_and_liberties(self.board, action)
            if len(own_grp) == 1 and own_libs == 1:
                self.ko_point = captured[0]
            else:
                self.ko_point = -1
        else:
            self.ko_point = -1

        b_bytes = self.board.tobytes()
        if b_bytes in self.history:
            # Superko rule: player who repeats whole-board position loses immediately
            self.terminal = True
            self.winner = -c
            return
        self.history.add(b_bytes)

        if self.moves >= 162:
            self.terminal = True
            self.winner = self._score_winner()
        else:
            self.turn = -self.turn

    def _score_winner(self) -> int:
        # Tromp-Taylor area scoring: stones + empty points surrounded only by that color; White gets 7.5 komi
        black_score = float(np.sum(self.board == 1))
        white_score = float(np.sum(self.board == -1)) + 7.5
        visited = set()
        for p in range(N_PTS):
            if self.board[p] == 0 and p not in visited:
                region = [p]
                visited.add(p)
                borders = set()
                idx = 0
                while idx < len(region):
                    curr = region[idx]
                    idx += 1
                    for q in ADJ[curr]:
                        v = self.board[q]
                        if v == 0 and q not in visited:
                            visited.add(q)
                            region.append(q)
                        elif v != 0:
                            borders.add(v)
                if borders == {1}:
                    black_score += len(region)
                elif borders == {-1}:
                    white_score += len(region)
        return 1 if black_score > white_score else -1

    def obs(self) -> np.ndarray:
        x = np.zeros((N_PTS, 3), dtype=np.float32)
        c = self.turn
        x[:, 0] = (self.board == c)
        x[:, 1] = (self.board == -c)
        if c == -1:
            x[:, 2] = 1.0
        return x.reshape(-1)


# ==============================================================================
# 2. Spatial-Graph Neural Compiler for 9x9 Go (243 -> 512x6 -> 82 ReLU)
# ==============================================================================
class SpatialGraphGoCompiler:
    """
    Declarative Spatial-Graph Compiler that maps board-wide relational operators
    (liberty counting, certified liberty propagation along chains, 1-stone ko/snapback
    repeat-risk suppression, eye preservation, and territory boundary valuation)
    into the 1,480,274 weights of a `243 -> 512 x 6 -> 82` ReLU MLP.
    """

    @staticmethod
    def compile(overrides: Dict = None) -> Dict[str, np.ndarray]:
        cfg = dict(
            capture=9.5,
            rescue=6.2,
            attack=1.25,
            self_atari=7.8,
            eye=11.5,
            connection=0.95,
            contact=1.45,
            own_territory=1.65,
            enemy_territory=1.05,
            territory_threshold=3.0,
            own_territory_threshold=0.5,
            pass_base=-9.5,
            pass_per_stone=0.10,
            pass_white=0.35,
            own_adj=0.26,
            own_diag=0.68,
            own_two=0.42,
            own_knight=0.14,
            enemy_adj=0.78,
            enemy_diag=0.26,
            enemy_two=0.12,
            temperature_scale=2.2,
            surrounded_penalty=10.5,
            repeat_risk=18.5,
            rescue_size=0.42,
            framework_bonus=2.1,
        )
        if overrides:
            cfg.update(overrides)

        W = [np.zeros((243 if k == 0 else 512, 512), np.float32) for k in range(6)]
        B = [np.zeros(512, np.float32) for _ in range(6)]
        OW = np.zeros((512, 82), np.float32)
        OB = np.zeros(82, np.float32)

        def u(ch: int, p: int) -> int:
            return 81 * ch + p

        # Layer 1: Board state, exact local liberties, and 0-liberty indicators
        for p in range(N_PTS):
            W[0][3 * p, u(0, p)] = 1.0
            W[0][3 * p + 1, u(1, p)] = 1.0
            for colour in range(2):
                j = u(2 + colour, p)
                B[0][j] = len(ADJ[p]) - 4.0
                W[0][3 * p + colour, j] += 4.0
                for q in ADJ[p]:
                    W[0][3 * q, j] -= 1.0
                    W[0][3 * q + 1, j] -= 1.0

                # Zero-liberty indicator (distinguishes shared single liberty on bent 2-stone chain)
                j0 = u(4 + colour, p)
                B[0][j0] = -float(len(ADJ[p]))
                W[0][3 * p + colour, j0] = 1.0
                for q in ADJ[p]:
                    W[0][3 * q, j0] += 1.0
                    W[0][3 * q + 1, j0] += 1.0

        # Global stone counters & White-to-move flag in registers 486, 487, 488
        for p in range(N_PTS):
            W[0][3 * p, 486] = 1.0
            W[0][3 * p + 1, 487] = 1.0
        W[0][2, 488] = 1.0

        for k in range(1, 6):
            for g in (486, 487, 488):
                W[k][g, g] = 1.0

        def carry_board(k: int):
            for ch in (0, 1):
                for p in range(N_PTS):
                    W[k][u(ch, p), u(ch, p)] = 1.0

        # Layer 2: Monochromatic liberty aggregation + diagonal shared-liberty de-duplication
        carry_board(1)
        for p in range(N_PTS):
            for colour in range(2):
                j = u(2 + colour, p)
                B[1][j] = -21.0
                W[1][u(colour, p), j] = 20.0
                for q in [p] + ADJ[p]:
                    W[1][u(2 + colour, q), j] += 1.0
                W[1][u(4 + colour, p), j] -= 1.0

                j_diag = u(4 + colour, p)
                B[1][j_diag] = -21.0 - len(DIAG[p])
                W[1][u(colour, p), j_diag] = 20.0
                for q in [p] + ADJ[p]:
                    W[1][u(2 + colour, q), j_diag] += 1.0
                for q in DIAG[p]:
                    W[1][u(0, q), j_diag] += 1.0
                    W[1][u(1, q), j_diag] += 1.0

        # Layer 3: Multi-hop chain safety propagation + post-extension liberty estimator + 1-stone ko detector
        for p in range(N_PTS):
            # Pack colour into single register u(0, p) in {1.0 (own), 0.5 (empty), 0.0 (enemy)}
            j_col = u(0, p)
            B[2][j_col] = 0.5
            W[2][u(0, p), j_col] = 0.5
            W[2][u(1, p), j_col] = -0.5

            # Carry raw enemy safety certificate in u(1, p) for short-range weakness detection
            W[2][u(3, p), u(1, p)] = 1.0

            for colour in range(2):
                j = u(2 + colour, p)
                B[2][j] = -100.0
                W[2][u(colour, p), j] = 100.0
                for q in [p] + ADJ[p]:
                    W[2][u(2 + colour, q), j] += 1.0
                    W[2][u(4 + colour, q), j] += 1.0

            # Post-extension available liberty estimate for friendly move at p
            j_ext = u(4, p)
            B[2][j_ext] = float(len(ADJ[p]))
            for q in ADJ[p]:
                W[2][u(0, q), j_ext] -= 1.0
                W[2][u(1, q), j_ext] -= 1.0
                W[2][u(2, q), j_ext] += 1.0
                W[2][u(4, q), j_ext] += 1.0

            # Single-stone capture detector (for superko / snapback repeat-risk guard)
            j_ko = u(5, p)
            B[2][j_ko] = -float(len(ADJ[p]))
            W[2][u(1, p), j_ko] = 2.0
            for q in ADJ[p]:
                W[2][u(0, q), j_ko] += 1.0
                W[2][u(1, q), j_ko] -= 1.0

        # Layer 4: Whole-group atari, size-weighted rescue, self-atari, and ko-repeat risk
        for p in range(N_PTS):
            W[3][u(0, p), u(0, p)] = 1.0

            # Weak enemy group detector
            j_weak = u(1, p)
            B[3][j_weak] = 3.0
            W[3][u(0, p), j_weak] = -20.0
            W[3][u(1, p), j_weak] = -1.0
            for q in ADJ[p]:
                W[3][u(1, q), j_weak] -= 0.5

            # Size-weighted endangered friendly group (rescue target)
            j_resc = u(2, p)
            sz = cfg["rescue_size"]
            B[3][j_resc] = 1.0 - 0.5 * sz - 20.0
            W[3][u(0, p), j_resc] = 20.0
            for q in ADJ[p]:
                W[3][u(0, q), j_resc] += sz
            for q in [p] + ADJ[p]:
                W[3][u(2, q), j_resc] -= 10.0

            # Endangered enemy group (capture target)
            j_cap = u(3, p)
            B[3][j_cap] = 1.0
            W[3][u(0, p), j_cap] = -20.0
            for q in [p] + ADJ[p]:
                W[3][u(3, q), j_cap] -= 1.0

            # Self-atari pre-detector
            B[3][u(4, p)] = 2.0
            W[3][u(4, p), u(4, p)] = -1.0

            # Ko / repetition risk detector
            j_rep = u(5, p)
            for q in ADJ[p]:
                W[3][u(5, q), j_rep] += 2.0
                W[3][u(0, q), j_rep] -= 2.0

        # Layer 5: Decode board planes, tactical capture/attack/repeat-risk, and capture-aware self-atari
        for p in range(N_PTS):
            for ch, sign, bias in ((0, 2.0, -1.0), (1, -2.0, 1.0)):
                j = u(ch, p)
                B[4][j] = bias
                W[4][u(0, p), j] = sign

            j_tac = u(2, p)
            B[4][j_tac] = 100.0
            for q in ADJ[p]:
                W[4][u(3, q), j_tac] += cfg["capture"]
                W[4][u(1, q), j_tac] += cfg["attack"]
            W[4][u(5, p), j_tac] -= cfg["repeat_risk"]

            # Capture-exempted self-atari detector
            j_sa = u(3, p)
            W[4][u(4, p), j_sa] = 1.0
            for q in ADJ[p]:
                W[4][u(3, q), j_sa] -= 2.0

            # Safe rescue detector (only rescue if move actually gains liberties)
            j_sr = u(4, p)
            W[4][u(4, p), j_sr] = -12.0
            for q in ADJ[p]:
                W[4][u(2, q), j_sr] += 1.0

            # Surrounded friendly empty point
            j_sur = u(5, p)
            B[4][j_sur] = 1.0 - 2.0 * len(ADJ[p])
            for q in ADJ[p]:
                W[4][u(0, q), j_sur] = 2.0

        # Layer 6: Combine tactics, ring shape kernels, connections, territory boundaries, and eyes
        line_prior = [-1.5, 0.25, 1.2, 0.9, 1.0]
        for p in range(N_PTS):
            r, c = divmod(p, 9)
            j0 = u(0, p)
            B[5][j0] = line_prior[min(r, 8 - r)] + line_prior[min(c, 8 - c)]
            W[5][u(2, p), j0] = 1.0
            W[5][u(3, p), j0] = -cfg["self_atari"]
            W[5][u(4, p), j0] = cfg["rescue"]
            W[5][u(5, p), j0] = -cfg["surrounded_penalty"]

            for q, dr, dc in RINGS[p]:
                if dr + dc == 1:
                    a, b = cfg["own_adj"], cfg["enemy_adj"]
                elif dr == 1 and dc == 1:
                    a, b = cfg["own_diag"], cfg["enemy_diag"]
                elif dr + dc == 2:
                    a, b = cfg["own_two"], cfg["enemy_two"]
                elif sorted((dr, dc)) == [1, 2]:
                    a, b = cfg["own_knight"], 0.0
                else:
                    continue
                W[5][u(0, q), j0] += a
                W[5][u(1, q), j0] += b

            # Connection without filled-triangle clumping
            j1 = u(1, p)
            B[5][j1] = -1.0
            for q in ADJ[p]:
                W[5][u(0, q), j1] = 1.0
            for q in DIAG[p]:
                W[5][u(0, q), j1] = -0.65

            # Unsupported enemy contact penalty
            j2 = u(2, p)
            B[5][j2] = -1.0
            for q in ADJ[p]:
                W[5][u(1, q), j2] = 1.0
                W[5][u(0, q), j2] = -0.75
            for q in DIAG[p]:
                W[5][u(0, q), j2] -= 0.20

            # Own & enemy territory interior detectors
            for colour, ch, thresh in (
                (0, 3, cfg["own_territory_threshold"]),
                (1, 4, cfg["territory_threshold"]),
            ):
                jt = u(ch, p)
                B[5][jt] = -thresh
                for q, dr, dc in RINGS[p]:
                    if dr + dc == 1:
                        wt = 1.0
                    elif dr == 1 and dc == 1:
                        wt = 0.65
                    elif dr + dc == 2:
                        wt = 0.45
                    elif sorted((dr, dc)) == [1, 2]:
                        wt = 0.20
                    else:
                        continue
                    W[5][u(colour, q), jt] += wt
                    W[5][u(1 - colour, q), jt] -= 1.2 * wt

            # True eye detector
            j5 = u(5, p)
            B[5][j5] = 1.0 - len(ADJ[p])
            for q in ADJ[p]:
                W[5][u(0, q), j5] += 1.0
            diag_pen = 1.0 if len(ADJ[p]) < 4 else 0.5
            for q in DIAG[p]:
                W[5][u(1, q), j5] -= diag_pen

            for ch, coeff in (
                (0, 1.0),
                (1, cfg["connection"]),
                (2, -cfg["contact"]),
                (3, -cfg["own_territory"]),
                (4, -cfg["enemy_territory"]),
                (5, -cfg["eye"]),
            ):
                OW[u(ch, p), p] = coeff
            OB[p] = -100.0
            if max(abs(r - 4), abs(c - 4)) == 2:
                OB[p] += cfg["framework_bonus"]

        OB[PASS_MOVE] = cfg["pass_base"]
        OW[486, PASS_MOVE] = cfg["pass_per_stone"]
        OW[487, PASS_MOVE] = cfg["pass_per_stone"]
        OW[488, PASS_MOVE] = cfg["pass_white"]

        OW *= cfg["temperature_scale"]
        OB *= cfg["temperature_scale"]

        tensors = {f"L{k+1}.W": W[k] for k in range(6)}
        tensors.update({f"L{k+1}.b": B[k] for k in range(6)})
        tensors["out.W"] = OW
        tensors["out.b"] = OB
        return tensors


# ==============================================================================
# 3. Forward Pass & Head-to-Head Match Evaluator
# ==============================================================================
def forward_go_net(x: np.ndarray, net: Dict[str, np.ndarray]) -> np.ndarray:
    h = x
    for k in range(1, 7):
        h = np.maximum(0.0, h @ net[f"L{k}.W"] + net[f"L{k}.b"])
    return h @ net["out.W"] + net["out.b"]


def load_go_checkpoint(folder: str) -> Dict[str, np.ndarray]:
    net = {}
    for k in range(1, 7):
        net[f"L{k}.W"] = np.load(os.path.join(folder, f"L{k}.W.npy"))
        net[f"L{k}.b"] = np.load(os.path.join(folder, f"L{k}.b.npy"))
    net["out.W"] = np.load(os.path.join(folder, "out.W.npy"))
    net["out.b"] = np.load(os.path.join(folder, "out.b.npy"))
    return net


def play_h2h(net_a: Dict[str, np.ndarray], net_b: Dict[str, np.ndarray], n_games: int = 200, seed: int = 31337) -> Dict:
    rng = np.random.RandomState(seed)
    env = Go9x9Env()
    wins_as_black = 0
    wins_as_white = 0
    half = n_games // 2

    for g in range(n_games):
        a_is_black = (g < half)
        env.reset()
        while not env.terminal:
            x = env.obs()
            active_net = net_a if ((env.turn == 1) == a_is_black) else net_b
            logits = forward_go_net(x, active_net).astype(np.float64)
            mask = env.legal_mask()
            logits[~mask] = -1e9
            logits -= np.max(logits)
            probs = np.exp(logits)
            probs /= np.sum(probs)
            move = int(rng.choice(82, p=probs))
            env.step(move)

        a_won = (env.winner == 1 and a_is_black) or (env.winner == -1 and not a_is_black)
        if a_won:
            if a_is_black:
                wins_as_black += 1
            else:
                wins_as_white += 1

    total_wins = wins_as_black + wins_as_white
    return {
        "win_rate": total_wins / n_games,
        "win_as_black": wins_as_black / half,
        "win_as_white": wins_as_white / (n_games - half),
        "games": n_games,
    }


if __name__ == "__main__":
    compiled_net = SpatialGraphGoCompiler.compile()
    total_params = sum(v.size for v in compiled_net.values())
    nonzero_params = sum(np.count_nonzero(v) for v in compiled_net.values())
    print("=== Demo 2: 1.48M-Parameter Spatial-Graph Neural Compiler for 9x9 Go ===")
    print(f"Architecture   : 243 -> 512 x 6 -> 82 (ReLU)")
    print(f"Total Weights  : {total_params:,}")
    print(f"Nonzero Weights: {nonzero_params:,} ({nonzero_params / total_params * 100:.2f}%)")

    if os.path.exists("/tmp/go_checkpoints/budget_0/L1.W.npy"):
        astra_b0 = load_go_checkpoint("/tmp/go_checkpoints/budget_0")
        ppo_s0 = load_go_checkpoint("/tmp/go_checkpoints/ppo_s0")

        res_vs_b0 = play_h2h(compiled_net, astra_b0, n_games=200, seed=31337)
        print(
            f"Head-to-Head vs GPT-6 Astra (budget_0, 1.48M) : "
            f"{res_vs_b0['win_rate']*100:5.1f}% Win Rate "
            f"(Black: {res_vs_b0['win_as_black']*100:5.1f}%, White: {res_vs_b0['win_as_white']*100:5.1f}%)"
        )

        res_vs_ppo = play_h2h(compiled_net, ppo_s0, n_games=200, seed=31337)
        print(
            f"Head-to-Head vs 20.5M-Game PPO (same-size s0) : "
            f"{res_vs_ppo['win_rate']*100:5.1f}% Win Rate "
            f"(Black: {res_vs_ppo['win_as_black']*100:5.1f}%, White: {res_vs_ppo['win_as_white']*100:5.1f}%) "
            f"[Astra budget_0 had 42.6%]"
        )
