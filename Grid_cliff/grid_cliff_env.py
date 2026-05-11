# grid_cliff_env.py
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ===== Designed characteristics =====
ROWS = 4
COLS = 12
MAP_SEED = 34       # fixed map per run (change to get a different map)
N_CLIFFS = 12     # number of cliff tiles (not counting S/G)
MAX_TRIES = 5000    # retries to ensure a valid path exists
MAX_STEPS = 100

ENV_ID = "RandomCliffWalking-v1"

# Actions: 0=Up, 1=Right, 2=Down, 3=Left
ACTIONS = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}


def _idx(r: int, c: int, cols: int = COLS) -> int:
    return r * cols + c


def _neighbors(r: int, c: int, rows: int = ROWS, cols: int = COLS) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ACTIONS.values():
        rr = int(np.clip(r + dr, 0, rows - 1))
        cc = int(np.clip(c + dc, 0, cols - 1))
        out.append((rr, cc))
    return out


def _path_exists(
    cliff: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    rows: int,
    cols: int,
) -> bool:
    # BFS on safe cells
    q = [start]
    seen = {start}
    while q:
        r, c = q.pop(0)
        if (r, c) == goal:
            return True
        for rr, cc in _neighbors(r, c, rows=rows, cols=cols):
            if (rr, cc) in seen:
                continue
            if cliff[rr, cc]:
                continue
            seen.add((rr, cc))
            q.append((rr, cc))
    return False


def generate_random_cliffs_with_path(
    rows: int,
    cols: int,
    n_cliffs: int,
    seed: int,
    max_tries: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    start = (rows - 1, 0)
    goal = (rows - 1, cols - 1)

    candidates = [(r, c) for r in range(rows) for c in range(cols)]
    candidates.remove(start)
    candidates.remove(goal)

    n_cliffs = min(n_cliffs, len(candidates))

    for _ in range(max_tries):
        cliff = np.zeros((rows, cols), dtype=bool)
        chosen = rng.choice(len(candidates), size=n_cliffs, replace=False)
        for i in chosen:
            r, c = candidates[int(i)]
            cliff[r, c] = True

        cliff[start] = False
        cliff[goal] = False

        if _path_exists(cliff, start, goal, rows=rows, cols=cols):
            return cliff

    raise RuntimeError(
        "Failed to generate a random cliff map with a valid path. "
        "Try reducing N_CLIFFS or increasing MAX_TRIES."
    )


class RandomCliffWalkingEnv(gym.Env):
    """
    Random cliffs placed once at env creation. Always at least one path S->G.
    Observation: Discrete(ROWS*COLS)
    Actions: Discrete(4)
    Reward: -1 per step, -100 on cliff (terminate), -1 on goal (terminate)
    """

    metadata = {"render_modes": ["ansi", "human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        render_mode=None,
        map_seed: int = MAP_SEED,
        n_cliffs: int = N_CLIFFS,
        max_steps: int = MAX_STEPS,
    ):
        super().__init__()
        self.rows = ROWS
        self.cols = COLS
        self.render_mode = render_mode
        self.max_steps = max_steps

        self.observation_space = spaces.Discrete(self.rows * self.cols)
        self.action_space = spaces.Discrete(4)

        self.start = (self.rows - 1, 0)
        self.goal = (self.rows - 1, self.cols - 1)

        # cliffs fixed at env creation (same map all episode resets)
        self.cliff = generate_random_cliffs_with_path(
            rows=self.rows,
            cols=self.cols,
            n_cliffs=n_cliffs,
            seed=map_seed,
            max_tries=MAX_TRIES,
        )

        self._agent_rc = self.start
        self._steps = 0

        # pygame lazy init
        self._pg = None
        self._screen = None
        self._canvas = None
        self._clock = None
        self._cell = 50

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_rc = self.start
        self._steps = 0
        if self.render_mode in {"human", "rgb_array"}:
            self.render()
        return _idx(*self._agent_rc, cols=self.cols), {}

    def step(self, action: int):
        dr, dc = ACTIONS[int(action)]
        r, c = self._agent_rc
        r2 = int(np.clip(r + dr, 0, self.rows - 1))
        c2 = int(np.clip(c + dc, 0, self.cols - 1))
        self._agent_rc = (r2, c2)
        self._steps += 1

        terminated = False
        truncated = False
        reward = -1.0

        if self.cliff[r2, c2]:
            reward = -100.0
            terminated = True
        elif (r2, c2) == self.goal:
            terminated = True
        elif self._steps >= self.max_steps:
            truncated = True

        if self.render_mode in {"human", "rgb_array"}:
            self.render()

        return _idx(r2, c2, cols=self.cols), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "ansi":
            grid = np.full((self.rows, self.cols), ".", dtype="<U1")
            grid[self.cliff] = "C"
            sr, sc = self.start
            gr, gc = self.goal
            grid[sr, sc] = "S"
            grid[gr, gc] = "G"
            ar, ac = self._agent_rc
            grid[ar, ac] = "A"
            return "\n".join(" ".join(row.tolist()) for row in grid)

        if self.render_mode in {"human", "rgb_array"}:
            try:
                import pygame
            except Exception as e:
                raise RuntimeError(
                    "pygame is required for render_mode='human' or 'rgb_array'"
                ) from e

            if self._pg is None:
                self._pg = pygame
                pygame.init()
                w = self.cols * self._cell
                h = self.rows * self._cell
                self._canvas = pygame.Surface((w, h))
                if self.render_mode == "human":
                    self._screen = pygame.display.set_mode((w, h))
                    pygame.display.set_caption("RandomCliffWalking")
                self._clock = pygame.time.Clock()

            if self.render_mode == "human":
                # handle quit so window doesn't freeze
                for event in self._pg.event.get():
                    if event.type == self._pg.QUIT:
                        self.close()
                        return None

            self._canvas.fill((245, 245, 245))

            # draw cells
            for r in range(self.rows):
                for c in range(self.cols):
                    x = c * self._cell
                    y = r * self._cell

                    if (r, c) == self.start:
                        color = (80, 200, 120)  # green
                    elif (r, c) == self.goal:
                        color = (80, 120, 240)  # blue
                    elif self.cliff[r, c]:
                        color = (220, 70, 70)   # red
                    else:
                        color = (230, 230, 230) # light gray

                    self._pg.draw.rect(self._canvas, color, (x, y, self._cell, self._cell))
                    self._pg.draw.rect(self._canvas, (200, 200, 200), (x, y, self._cell, self._cell), 1)

            # draw agent
            ar, ac = self._agent_rc
            ax = ac * self._cell + self._cell // 2
            ay = ar * self._cell + self._cell // 2
            self._pg.draw.circle(self._canvas, (20, 20, 20), (ax, ay), self._cell // 4)

            if self.render_mode == "rgb_array":
                frame = self._pg.surfarray.array3d(self._canvas)
                return np.transpose(frame, (1, 0, 2))

            self._screen.blit(self._canvas, (0, 0))
            self._pg.display.flip()
            if self._clock is not None:
                self._clock.tick(self.metadata.get("render_fps", 10))
            return None

        return None

    def close(self):
        if self._pg is not None:
            try:
                self._pg.quit()
            except Exception:
                pass
        self._pg = None
        self._screen = None
        self._canvas = None
        self._clock = None


def make_cliff_env(
    render_mode=None,
    map_seed: int = MAP_SEED,
    n_cliffs: int = N_CLIFFS,
    max_steps: int = MAX_STEPS,
):
    return RandomCliffWalkingEnv(
        render_mode=render_mode,
        map_seed=map_seed,
        n_cliffs=n_cliffs,
        max_steps=max_steps,
    )

def seed_everything(env, seed: int):
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    return obs, info
