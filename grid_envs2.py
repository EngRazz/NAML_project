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

    def __init__(self, size: int = 5, max_steps: int = 50, render_mode = None):
        super().__init__()
        self.size = size
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Rendering state
        self._window = None
        self._clock = None
        self._tick = 0
        self.metadata['render_fps'] = self.FPS


        # Observation space
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, size - 1, shape=(2,), dtype=np.int32),
            "target": spaces.Box(0, size - 1, shape=(2,), dtype=np.int32),
            "obstacle": spaces.Box(0, size - 1, shape=(2,), dtype=np.int32),
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
        return {
            "agent": self.agent_location, 
            "target": self.target_location,
            "obstacle": np.array([self.gap_row, self.obstacle_col], dtype=np.int32),
            }

    def _get_info(self):
        return {"distance": np.linalg.norm(self.agent_location - self.target_location, ord=1)}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Inizializzazione Pygame per il rendering
        if self.render_mode == "human" and self._window is None:
            pygame.init()
            W = self.size * self.CELL_SIZE 
            H = self.size * self.CELL_SIZE + self.HUD_HEIGHT
            pygame.display.set_caption("GridWorld_Moving_Obstacle")
            self._window = pygame.display.set_mode((W, H))
            self._clock = pygame.time.Clock()
            try:
                self._font_big = pygame.font.SysFont("monospace", 22, bold=True)
                self._font_small = pygame.font.SysFont("monospace", 14)
            except:
                self._font_big = pygame.font.Font(None, 26)
                self._font_small = pygame.font.Font(None, 18)

        self.agent_location = np.array([self.size-1, 0], dtype=np.int32) #because I didn't parametrized the agent start location 
        self.target_location = np.array([0, self.size-1], dtype=np.int32) #because I didn't parametrized the target location 
        
        self.gap_row = int(self.np_random.integers(1, self.size - 1))
        self.obstacle_col = self.size - 1   # parte dal bordo destro
        
        # NUOVO: reward_function = parete con buco nel gap_row
        self.reward_function = np.zeros((self.size, self.size))
        for i in range(self.size):
            if i != self.gap_row:
                self.reward_function[i, self.obstacle_col] = -10
        
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
        # MODIFICATO: la parete si sposta a sinistra e fa loop (ricompare a destra)
        new_map = np.zeros_like(self.reward_function)
        new_col = self.obstacle_col - 1
        if new_col < 0:
            new_col = self.size - 1   # loop: ricompare dal bordo destro
        
        for i in range(self.size):
            if i != self.gap_row:
                new_map[i, new_col] = -10

        self.reward_function = new_map
        self.obstacle_col = new_col   # NUOVO: teniamo traccia della colonna

        # Terminazione
        terminated = bool(np.array_equal(self.agent_location, self.target_location))
        truncated = self.current_step >= self.max_steps
        reward = 1.0 if terminated else current_cell_reward

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def close(self):
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()
            self._window = None

    def render(self):
        if self.render_mode not in ("human", "rgb_array"):
            return

        CELL = self.CELL_SIZE
        W    = self.size * CELL
        H    = self.size * CELL + self.HUD_HEIGHT

        # ── Palette ──────────────────────────────────────────────────
        COL_BG         = (10,  12,  20)
        COL_GRID       = (25,  30,  50)
        COL_HUD_BG     = (14,  17,  30)
        COL_HUD_BORDER = (40,  50,  90)
        COL_AGENT_CORE = (80, 200, 255)
        COL_AGENT_GLOW = (30, 100, 200)
        COL_TARGET     = (60, 200,  60)
        COL_OBSTACLE   = (220, 80,  80)
        COL_GAP        = (40,  80,  40)
        COL_TEXT       = (180, 200, 240)
        COL_ACCENT     = (100, 160, 255)
        COL_BAR_BG     = (30,  35,  60)
        COL_BAR_FG     = (80, 200, 120)
        COL_BAR_LOW    = (220, 80,  80)

        # ── Init sicuro di pygame e font ──────────────────────────────
        if not pygame.get_init():
            pygame.init()

        if not hasattr(self, "_font_small") or self._font_small is None:
            try:
                self._font_big   = pygame.font.SysFont("monospace", 22, bold=True)
                self._font_small = pygame.font.SysFont("monospace", 14)
            except Exception:
                self._font_big   = pygame.font.Font(None, 26)
                self._font_small = pygame.font.Font(None, 18)

        if self._window is None and self.render_mode == "human":
            pygame.display.set_caption("GridWorld_Moving_Obstacle")
            self._window = pygame.display.set_mode((W, H))
            self._clock  = pygame.time.Clock()

        # ── Canvas ───────────────────────────────────────────────────
        canvas = pygame.Surface((W, H))
        canvas.fill(COL_BG)

        # ── Griglia ──────────────────────────────────────────────────
        for x in range(self.size + 1):
            pygame.draw.line(canvas, COL_GRID, (x * CELL, 0), (x * CELL, self.size * CELL))
        for y in range(self.size + 1):
            pygame.draw.line(canvas, COL_GRID, (0, y * CELL), (W, y * CELL))

        # ── Ostacolo/i (celle con reward == -10) ─────────────────────
        # MODIFICATO: disegna la parete + evidenzia il gap
        for i in range(self.size):
            rect = pygame.Rect(self.obstacle_col * CELL + 3, i * CELL + 3, CELL - 6, CELL - 6)
            if i == self.gap_row:
                # gap: cella libera evidenziata in verde scuro
                pygame.draw.rect(canvas, COL_GAP, rect, border_radius=8)
                lbl = self._font_small.render("GAP", True, (100, 255, 100))
                canvas.blit(lbl, lbl.get_rect(center=rect.center))
            else:
                pygame.draw.rect(canvas, COL_OBSTACLE, rect, border_radius=8)
                lbl = self._font_big.render("✕", True, (255, 255, 255))
                canvas.blit(lbl, lbl.get_rect(center=rect.center))

        # ── Target ───────────────────────────────────────────────────
        tr, tc = self.target_location
        t_rect = pygame.Rect(tc * CELL + 3, tr * CELL + 3, CELL - 6, CELL - 6)
        pygame.draw.rect(canvas, COL_TARGET, t_rect, border_radius=8)
        lbl = self._font_big.render("★", True, (255, 255, 255))
        canvas.blit(lbl, lbl.get_rect(center=t_rect.center))

        # ── Agente (con effetto glow) ─────────────────────────────────
        ar, ac = self.agent_location
        cx = ac * CELL + CELL // 2
        cy = ar * CELL + CELL // 2
        radius = CELL // 2 - 6

        glow = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
        pygame.draw.circle(glow, (*COL_AGENT_GLOW, 90), (CELL // 2, CELL // 2), radius + 10)
        canvas.blit(glow, (ac * CELL, ar * CELL))
        pygame.draw.circle(canvas, COL_AGENT_CORE, (cx, cy), radius)
        lbl = self._font_small.render("A", True, COL_BG)
        canvas.blit(lbl, lbl.get_rect(center=(cx, cy)))

        # ── HUD ──────────────────────────────────────────────────────
        hud_y = self.size * CELL
        pygame.draw.rect(canvas, COL_HUD_BG, (0, hud_y, W, self.HUD_HEIGHT))
        pygame.draw.line(canvas, COL_HUD_BORDER, (0, hud_y), (W, hud_y), 2)

        # Testo step
        step_surf = self._font_big.render(
            f"Step: {self.current_step} / {self.max_steps}", True, COL_TEXT
        )
        canvas.blit(step_surf, (10, hud_y + 8))

        # Testo distanza
        dist = int(np.linalg.norm(self.agent_location - self.target_location, ord=1))
        dist_surf = self._font_small.render(
            f"Manhattan dist: {dist}  |  Gap row: {self.gap_row}", True, COL_ACCENT
        )
        canvas.blit(dist_surf, (10, hud_y + 38))

        # Barra progresso step
        bar_x = W // 2
        bar_y = hud_y + 20
        bar_w = W // 2 - 20
        bar_h = 16
        progress   = self.current_step / max(self.max_steps, 1)
        bar_color  = COL_BAR_FG if progress < 0.75 else COL_BAR_LOW
        pygame.draw.rect(canvas, COL_BAR_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        pygame.draw.rect(canvas, bar_color,  (bar_x, bar_y, int(bar_w * progress), bar_h), border_radius=4)
        bar_lbl = self._font_small.render("progress", True, COL_TEXT)
        canvas.blit(bar_lbl, (bar_x, bar_y + bar_h + 4))

        # ── Output ───────────────────────────────────────────────────
        if self.render_mode == "human":
            # Gestione eventi pygame (evita freeze della finestra)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
            self._window.blit(canvas, (0, 0))
            pygame.display.flip()
            self._clock.tick(self.FPS)

        else:  # rgb_array — obbligatorio per RecordVideo
            # surfarray restituisce (W, H, 3), gymnasium vuole (H, W, 3)
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

