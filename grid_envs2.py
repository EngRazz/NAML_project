import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
from typing import Optional
import pygame

class GridWorldMovingObstacle(gym.Env):
    """
    GridWorld con ostacolo mobile, stilizzato secondo GridFlagEnv.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    CELL_SIZE  = 64
    HUD_HEIGHT = 80
    FPS = 10

    def __init__(self, size: int = 5, max_steps: int = 50, render_mode: Optional[str] = None):
        super().__init__()
        self.size = size
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Rendering state
        self._window = None
        self._clock = None
        self._tick = 0

        # Observation space
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, size - 1, shape=(2,), dtype=np.int32),
            "target": spaces.Box(0, size - 1, shape=(2,), dtype=np.int32),
        })

        # Action space: 0=R, 1=U, 2=L, 3=D
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([0, 1]),   # right
            1: np.array([-1, 0]),  # up
            2: np.array([0, -1]),  # left
            3: np.array([1, 0]),   # down
        }

    def _get_obs(self):
        return {"agent": self.agent_location, "target": self.target_location}

    def _get_info(self):
        return {"distance": np.linalg.norm(self.agent_location - self.target_location, ord=1)}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Inizializzazione Pygame per il rendering
        if self.render_mode == "human" and self._window is None:
            pygame.init()
            W, H = self.size * self.CELL_SIZE, self.size * self.CELL_SIZE + self.HUD_HEIGHT
            pygame.display.set_caption("GridWorld Moving Obstacle")
            self._window = pygame.display.set_mode((W, H))
            self._clock = pygame.time.Clock()
            try:
                self._font_big = pygame.font.SysFont("monospace", 22, bold=True)
                self._font_small = pygame.font.SysFont("monospace", 14)
            except:
                self._font_big = pygame.font.Font(None, 26)
                self._font_small = pygame.font.Font(None, 18)

        self.agent_location = np.array([4, 0], dtype=np.int32)
        self.target_location = np.array([0, 4], dtype=np.int32)
        self.reward_function = np.zeros((self.size, self.size))
        self.reward_function[2, 4] = -10
        
        self.current_step = 0
        self._tick = 0

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1
        self._tick += 1

        # Movimento agente
        direction = self._action_to_direction[action]
        self.agent_location = np.clip(self.agent_location + direction, 0, self.size - 1)

        # Logica ostacolo mobile
        current_cell_reward = float(self.reward_function[self.agent_location[0], self.agent_location[1]])
        
        # Aggiornamento posizione ostacolo
        new_map = np.zeros_like(self.reward_function)
        for i in range(self.size):
            for j in range(self.size):
                if self.reward_function[i, j] == -10:
                    new_j = max(0, j - 1)
                    new_map[i, new_j] = -10
        self.reward_function = new_map

        # Terminazione
        terminated = bool(np.array_equal(self.agent_location, self.target_location))
        truncated = self.current_step >= self.max_steps
        reward = 1.0 if terminated else current_cell_reward

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode not in ("human", "rgb_array"):
            return

        CELL = self.CELL_SIZE
        W, H = self.size * CELL, self.size * CELL + self.HUD_HEIGHT
        
        # --- AGGIUNTA: Inizializzazione sicura dei font ---
        if not hasattr(self, "_font_small"):
            pygame.init()
            try:
                self._font_big   = pygame.font.SysFont("monospace", 22, bold=True)
                self._font_small = pygame.font.SysFont("monospace", 14)
            except Exception:
                self._font_big   = pygame.font.Font(None, 26)
                self._font_small = pygame.font.Font(None, 18)

        if self._window is None and self.render_mode == "human":
            pygame.display.set_caption("GridWorld Moving Obstacle")
            self._window = pygame.display.set_mode((W, H))
            self._clock = pygame.time.Clock()
        # --------------------------------------------------

        # Palette colori stile GridFlagEnv
        COL_BG = (10, 12, 20)
        COL_GRID = (25, 30, 50)
        COL_AGENT_CORE = (80, 200, 255)
        COL_AGENT_GLOW = (30, 100, 200)
        COL_TARGET = (60, 200, 60)
        COL_OBSTACLE = (220, 80, 80)
        COL_TEXT = (180, 200, 240)

        canvas = pygame.Surface((W, H))
        canvas.fill(COL_BG)

        # Disegno Griglia
        for x in range(self.size + 1):
            pygame.draw.line(canvas, COL_GRID, (x * CELL, 0), (x * CELL, self.size * CELL))
        for y in range(self.size + 1):
            pygame.draw.line(canvas, COL_GRID, (0, y * CELL), (W, y * CELL))

        # Ostacoli e Target (Usa i dati correnti se disponibili, altrimenti salta per il check_env)
        if hasattr(self, 'reward_function'):
            t = self._tick
            for r in range(self.size):
                for c in range(self.size):
                    if self.reward_function[r, c] == -10:
                        pygame.draw.rect(canvas, COL_OBSTACLE, (c * CELL + 8, r * CELL + 8, CELL - 16, CELL - 16), border_radius=4)
            
            tr, tc = self.target_location
            pygame.draw.rect(canvas, COL_TARGET, (tc * CELL + 12, tr * CELL + 12, CELL - 24, CELL - 24), border_radius=6)

            # Agente
            ar, ac = self.agent_location
            cx, cy = ac * CELL + CELL // 2, ar * CELL + CELL // 2
            bob = int(math.sin(t * 0.22) * 2)
            pygame.draw.circle(canvas, COL_AGENT_CORE, (cx, cy + bob), 12)

        # HUD
        hud_y = self.size * CELL
        pygame.draw.rect(canvas, (14, 17, 30), (0, hud_y, W, self.HUD_HEIGHT))
        pygame.draw.line(canvas, (40, 50, 90), (0, hud_y), (W, hud_y), 2)
        
        # Testo DISTANCE (ora i font esistono sicuramente)
        dist = self._get_info()["distance"] if hasattr(self, 'agent_location') else 0
        canvas.blit(self._font_small.render("DISTANCE", True, COL_TEXT), (20, hud_y + 10))
        canvas.blit(self._font_big.render(f"{dist:.1f}", True, (100, 160, 255)), (20, hud_y + 28))

        if self.render_mode == "human":
            self._window.blit(canvas, (0, 0))
            pygame.display.flip()
            self._clock.tick(self.FPS)
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()
            self._window = None