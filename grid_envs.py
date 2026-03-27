import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math


class GridFlagEnv(gym.Env):
    """
    A 10x10 grid world where the agent collects reward cells before time runs out.

    Grid cell encoding:
        0 = empty
        1 = agent
        2 = reward cell

    Actions:
        0 = stay
        1 = up
        2 = down
        3 = left
        4 = right
    """

    metadata = {"render_modes": ["human"]}

    CELL_SIZE  = 64
    HUD_HEIGHT = 80
    FPS = 10

    def __init__(self, grid_size, max_step, agent_start, flag_value, flag_cells, render_mode=None):
        super().__init__()

        # Attributes of the environment
        self.grid_w, self.grid_h = grid_size
        self.max_step = max_step
        self.flag_value = flag_value
        self.agent_pos = agent_start
        self.flag_cells = flag_cells
        
        # Attributes for rendering
        self.render_mode = render_mode
        self._window = None
        self._clock = None
        self._tick = 0
        self._collected_flash = {}
        self.metadata['render_fps'] = self.FPS

        # Observation: a dict with agent position and binary flags for remaining rewards
        self.observation_space = spaces.Dict({
            "agent" : spaces.Box(low=0, high=max(self.grid_w, self.grid_h), shape=(2,), dtype=np.int32),
            "flags": spaces.MultiBinary(len(self.flag_cells))
        })

        # Action: 0=stay, 1=up, 2=down, 3=left, 4=right
        self.action_space = spaces.Discrete(5)

        # Movement deltas (row, col) for each action
        self._action_to_delta = {
            0: (0, 0),
            1: (-1, 0),
            2: (1, 0),
            3: (0, -1),
            4: (0, 1)
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize pygame here so it's ready before the event loop
        if self.render_mode == "human" and self._window is None:
            import pygame
            pygame.init()
            CELL = self.CELL_SIZE
            W = self.grid_w * CELL
            H = self.grid_h * CELL + self.HUD_HEIGHT
            pygame.display.set_caption("GridWorld")
            self._window = pygame.display.set_mode((W, H))
            self._clock = pygame.time.Clock()
            try:
                self._font_big   = pygame.font.SysFont("monospace", 22, bold=True)
                self._font_small = pygame.font.SysFont("monospace", 14)
            except Exception:
                self._font_big   = pygame.font.Font(None, 26)
                self._font_small = pygame.font.Font(None, 18)

        self.agent_pos = list(self.agent_pos)
        self.remaining_flags = set(self.flag_cells)
        self.current_step = 0
        self.total_reward = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.current_step += 1
        self._tick += 1

        # Move agent
        dr, dc = self._action_to_delta[action]
        new_row = self.agent_pos[0] + dr
        new_col = self.agent_pos[1] + dc

        # Clamp to grid boundaries (walls block movement)
        new_row = max(0, min(self.grid_h - 1, new_row))
        new_col = max(0, min(self.grid_w - 1, new_col))
        self.agent_pos = [new_row, new_col]

        # Check for reward collection
        pos_tuple = tuple(self.agent_pos)
        if pos_tuple in self.remaining_flags:
            reward = self.flag_value
            self.remaining_flags.remove(pos_tuple)
            self._collected_flash[pos_tuple] = 12
        else:
            reward = 0

        self.total_reward += reward

        # Episode ends when time runs out
        terminated = False
        truncated = self.current_step >= self.max_step

        obs = self._get_obs()
        info = {
            "total_reward": self.total_reward,
            "remaining_flags": len(self.remaining_flags),
            "steps": self.current_step,
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        flags = np.array([1 if cell in self.remaining_flags else 0 for cell in self.flag_cells], dtype=np.int8)
        return {
            "agent": np.array(self.agent_pos, dtype=np.int32),
            "flags": flags
        }
    
    def close(self):
        if self._window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self._window = None

    # def render(self):
    #     if self.render_mode != "human":
    #         return

    #     symbols = {0: ".", 1: "A", 2: "R"}
    #     print(f"\nStep: {self.current_step}/{self.max_step}  |  Score: {self.total_reward}")
    #     print("-" * (self.grid_w * 2 + 1))
    #     obs = self._get_obs()
    #     for row in obs:
    #         print("|" + " ".join(symbols[v] for v in row) + "|")
    #     print("-" * (self.grid_w * 2 + 1))

    def render(self):
        if self.render_mode not in ("human", "rgb_array"):
            return

        try:
            import pygame
        except ImportError:
            raise ImportError("pygame is required for rendering. Run: pip install pygame")

        CELL = self.CELL_SIZE
        W = self.grid_w * CELL
        H = self.grid_h * CELL + self.HUD_HEIGHT

        # Palette
        COL_BG          = (10,  12,  20)
        COL_GRID_LINE   = (25,  30,  50)
        COL_HUD_BG      = (14,  17,  30)
        COL_HUD_BORDER  = (40,  50,  90)
        COL_AGENT_CORE  = (80, 200, 255)
        COL_AGENT_GLOW  = (30, 100, 200)
        COL_REWARD      = (255, 210,  50)
        COL_REWARD_GLW  = (200, 100,   0)
        COL_FLASH       = (255, 255, 180)
        COL_TEXT        = (180, 200, 240)
        COL_ACCENT      = (100, 160, 255)
        COL_BAR_BG      = (30,  35,  60)
        COL_BAR_FG      = (80, 200, 120)
        COL_BAR_LOW     = (220,  80,  80)

        # Init pygame
        if self._window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("GridWorld")
            self._window = pygame.display.set_mode((W, H))
            self._clock = pygame.time.Clock()
            try:
                self._font_big   = pygame.font.SysFont("monospace", 22, bold=True)
                self._font_small = pygame.font.SysFont("monospace", 14)
            except Exception:
                self._font_big   = pygame.font.Font(None, 26)
                self._font_small = pygame.font.Font(None, 18)

        canvas = pygame.Surface((W, H))
        canvas.fill(COL_BG)

        # --- Grid lines ---
        for x in range(self.grid_w + 1):
            pygame.draw.line(canvas, COL_GRID_LINE, (x * CELL, 0), (x * CELL, self.grid_h * CELL))
        for y in range(self.grid_h + 1):
            pygame.draw.line(canvas, COL_GRID_LINE, (0, y * CELL), (W, y * CELL))

        # --- Subtle checkerboard shading ---
        for r in range(self.grid_h):
            for c in range(self.grid_w):
                if (r + c) % 2 == 0:
                    s = pygame.Surface((CELL - 1, CELL - 1), pygame.SRCALPHA)
                    s.fill((255, 255, 255, 4))
                    canvas.blit(s, (c * CELL + 1, r * CELL + 1))

        t = self._tick

        # --- Reward cells (pulsing golden diamonds) ---
        for (r, c) in self.remaining_flags:
            cx = c * CELL + CELL // 2
            cy = r * CELL + CELL // 2
            pulse = 0.5 + 0.5 * math.sin(t * 0.18 + r + c)
            glow_r = int(18 + 14 * pulse)

            # Outer glow
            glow_surf = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
            glow_alpha = int(60 + 40 * pulse)
            pygame.draw.circle(glow_surf, (*COL_REWARD_GLW, glow_alpha), (CELL // 2, CELL // 2), glow_r + 4)
            canvas.blit(glow_surf, (c * CELL, r * CELL))

            # Diamond shape
            gem_size = int(14 + 3 * pulse)
            points = [
                (cx,            cy - gem_size),
                (cx + gem_size, cy),
                (cx,            cy + gem_size),
                (cx - gem_size, cy),
            ]
            pygame.draw.polygon(canvas, COL_REWARD, points)
            inner = [
                (cx,                cy - gem_size + 4),
                (cx + gem_size - 4, cy),
                (cx,                cy + gem_size - 4),
                (cx - gem_size + 4, cy),
            ]
            bright = tuple(min(255, v + 60) for v in COL_REWARD)
            pygame.draw.polygon(canvas, bright, inner)

        # --- Collection flash ---
        expired = []
        for pos, frames in self._collected_flash.items():
            r, c = pos
            alpha = int(200 * (frames / 12))
            flash_surf = pygame.Surface((CELL - 2, CELL - 2), pygame.SRCALPHA)
            flash_surf.fill((*COL_FLASH, alpha))
            canvas.blit(flash_surf, (c * CELL + 1, r * CELL + 1))
            self._collected_flash[pos] = frames - 1
            if frames - 1 <= 0:
                expired.append(pos)
        for p in expired:
            del self._collected_flash[p]

        # --- Agent (glowing circle with bob animation) ---
        ar, ac = self.agent_pos
        cx = ac * CELL + CELL // 2
        cy = ar * CELL + CELL // 2
        bob = int(math.sin(t * 0.22) * 2)

        # Glow layers
        for radius, alpha in [(26, 18), (20, 35), (14, 60)]:
            g = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
            pygame.draw.circle(g, (*COL_AGENT_GLOW, alpha), (CELL // 2, CELL // 2 + bob), radius)
            canvas.blit(g, (ac * CELL, ar * CELL))

        # Core
        pygame.draw.circle(canvas, COL_AGENT_CORE, (cx, cy + bob), 11)
        # Specular highlight
        pygame.draw.circle(canvas, (220, 240, 255), (cx - 3, cy + bob - 3), 4)

        # --- HUD background ---
        hud_y = self.grid_h * CELL
        pygame.draw.rect(canvas, COL_HUD_BG, (0, hud_y, W, self.HUD_HEIGHT))
        pygame.draw.line(canvas, COL_HUD_BORDER, (0, hud_y), (W, hud_y), 2)

        # Score
        canvas.blit(self._font_small.render("SCORE", True, COL_TEXT),  (20, hud_y + 10))
        canvas.blit(self._font_big.render(str(self.total_reward), True, COL_ACCENT), (20, hud_y + 28))

        # Remaining targets
        canvas.blit(self._font_small.render("TARGETS", True, COL_TEXT), (160, hud_y + 10))
        canvas.blit(self._font_big.render(
            f"{len(self.remaining_flags)} / {len(self.flag_cells)}", True, COL_REWARD),
            (160, hud_y + 28))

        # Time progress bar
        bar_x, bar_y = 310, hud_y + 14
        bar_w, bar_h = W - bar_x - 20, 16
        progress = self.current_step / self.max_step
        bar_color = COL_BAR_FG if progress < 0.7 else COL_BAR_LOW

        pygame.draw.rect(canvas, COL_BAR_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        pygame.draw.rect(canvas, bar_color,  (bar_x, bar_y, int(bar_w * progress), bar_h), border_radius=4)
        pygame.draw.rect(canvas, COL_HUD_BORDER, (bar_x, bar_y, bar_w, bar_h), 1, border_radius=4)

        canvas.blit(self._font_small.render("TIME", True, COL_TEXT), (bar_x, hud_y + 38))
        canvas.blit(self._font_small.render(
            f"{self.current_step} / {self.max_step}", True, COL_TEXT),
            (bar_x + bar_w - 68, hud_y + 38))

        # --- Output ---
        if self.render_mode == "human" and self._window is not None and self._clock is not None:
            self._window.blit(canvas, (0, 0))
            pygame.display.flip()
            self._clock.tick(self.FPS)

        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
