from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
import pygame


class DiffDriveEnv(gym.Env):
    """
    2D differential drive robot that navigates a room with static obstacles to reach a goal.

    Observation (18 values):
        - 16 lidar ray distances (rotating with the robot heading)
        - distance to goal
        - relative angle to goal (in robot frame)

    Action (2 values, continuous):
        - v_linear:  forward speed  [m/s],  in [-0.5, 1.0]
        - v_angular: turning rate   [rad/s], in [-π,   π  ]

    Reward:
        - Dense: improvement in distance to goal at each step
        - Collision: -10, episode ends
        - Goal reached: +100, episode ends
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    # Render config
    RENDER_SCALE = 80   # pixels per meter
    HUD_HEIGHT   = 60
    FPS          = 30

    def __init__(
        self,
        room_size       = (10.0, 10.0),  # (width, height) in metres
        obstacles       = None,           # list of (x, y, w, h) axis-aligned rectangles
        random_obst     = False,          # if True, ignore `obstacles` and sample random ones
        robot_start     = (1.0, 1.0),    # (x, y) initial position
        goal_pos        = (8.0, 8.0),    # (x, y) goal position
        max_step        = 500,
        n_lidar_rays    = 16,
        lidar_max_range = 5.0,
        robot_radius    = 0.3,
        dt              = 0.1,           # seconds per step
        render_mode     = None,
    ):
        super().__init__()

        self.room_w, self.room_h = room_size
        self.obstacles       = obstacles or []
        self.random_obst     = random_obst
        self.robot_start     = np.array(robot_start, dtype=np.float32)
        self.goal_pos        = np.array(goal_pos,    dtype=np.float32)
        self.max_step        = max_step
        self.n_lidar_rays    = n_lidar_rays
        self.lidar_max_range = lidar_max_range
        self.robot_radius    = robot_radius
        self.dt              = dt
        self.render_mode     = render_mode

        # Render state
        self._window     = None
        self._clock      = None
        self._last_lidar = None #initially the robot didn't make any move so the lidar didn't register any value
        self.metadata["render_fps"] = self.FPS

        # Observation space: [lidar x16, dist_to_goal, angle_to_goal]
        max_dist = math.sqrt(self.room_w ** 2 + self.room_h ** 2)
        obs_low  = np.array([0.0] * n_lidar_rays + [0.0, -math.pi], dtype=np.float32) #lowest observation is 0 for each lidar with 0 distance from goal and -pi as angle 
        obs_high = np.array([lidar_max_range] * n_lidar_rays + [max_dist, math.pi], dtype=np.float32) #highest observation is 5 from each lidar and max distance from goal and pi as angle
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action space: [v_linear, v_angular]
        self.action_space = spaces.Box(
            low  = np.array([-0.5, -math.pi], dtype=np.float32), #can go in retro at 0.5 and rotating of -pi
            high = np.array([ 1.0,  math.pi], dtype=np.float32),
            dtype= np.float32,
        )

    def _sample_obstacles(self):
        obstacles = []
        protected = [self.robot_start, self.goal_pos]
        for _ in range(3):
            for _ in range(20):   # max attempts
                x = np.random.uniform(0.0, 9.0)
                y = np.random.uniform(0.0, 9.0)
                w = np.random.uniform(1.0, self.room_w-x)
                h = np.random.uniform(1.0, self.room_h-y)
                # reject if too close to start or goal
                too_close = any(
                    x < px < x + w and y < py < y + h #check if initial or goal are inside the obstacle --> reject the obstacle
                    for (px, py) in protected
                )
                if not too_close:
                    obstacles.append((x, y, w, h))
                    break
        return obstacles

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if self.random_obst:
            self.obstacles = self._sample_obstacles()

        self.robot_pos    = self.robot_start.copy()
        self.robot_theta  = 0.0
        self.current_step = 0
        self.prev_dist    = float(np.linalg.norm(self.goal_pos - self.robot_pos))
        self._last_lidar  = None

        if self.render_mode == "human" and self._window is None:
            self._init_pygame()

        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1

        v_linear  = float(action[0])
        v_angular = float(action[1])

        # --- Differential drive kinematics ---
        self.robot_theta += v_angular * self.dt #rotation
        self.robot_theta  = (self.robot_theta + math.pi) % (2 * math.pi) - math.pi  # wrap to [-π, π]

        new_pos = self.robot_pos + np.array([
            v_linear * math.cos(self.robot_theta),
            v_linear * math.sin(self.robot_theta),
        ], dtype=np.float32) * self.dt

        # --- Collision ---
        collision = self._check_collision(new_pos)
        if not collision:
            self.robot_pos = new_pos

        # --- Reward ---
        dist         = float(np.linalg.norm(self.goal_pos - self.robot_pos))
        goal_reached = dist < self.robot_radius + 0.2 #reach goal if closer than 0.2 from the goal

        if collision:
            reward, terminated = -10.0, True

        elif goal_reached:
            reward, terminated = 100.0, True

        else:
            # 1. Progress: reward getting closer, penalise moving away
            progress = (self.prev_dist - dist) * 10
            
            min_lidar = np.min(self._last_lidar)
            safety_reward = -2 * np.exp(-5.0 * min_lidar) if min_lidar < 0.5 else 1e-8

            # 2. Orientation: reward facing the goal
            diff       = self.goal_pos - self.robot_pos
            angle_glob = math.atan2(float(diff[1]), float(diff[0]))
            angle_err  = abs((angle_glob - self.robot_theta + math.pi) % (2 * math.pi) - math.pi) #reward term for the robot not orientated toward the goal
            orientation = (1.0 - angle_err / math.pi)  # 1.0 = facing goal, 0.0 = facing away

            # 3. Time penalty: small cost per step to discourage spinning in place
            time_penalty = -0.05 

            reward     = progress + 0.3 * orientation + time_penalty
            terminated = False

        self.prev_dist = dist
        truncated = self.current_step >= self.max_step

        info = {
            "dist_to_goal": dist, #isn't already provided in the get_obs?
            "collision"   : collision,
            "goal_reached": goal_reached,
            "steps"       : self.current_step,
        }
        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self):
        lidar = self._get_lidar()
        self._last_lidar = lidar

        diff       = self.goal_pos - self.robot_pos
        dist       = float(np.linalg.norm(diff))
        angle_glob = math.atan2(float(diff[1]), float(diff[0]))
        angle_rel  = (angle_glob - self.robot_theta + math.pi) % (2 * math.pi) - math.pi

        return np.concatenate([lidar, [dist, angle_rel]]).astype(np.float32)

    #
    # Each Lidar returns the distance from the robot and the nearest (if dist < max_lidar_range) object (wall or osbtacle) pointed by the lidar, using n_lidar uniformely distribuited arounf the robot
    #
    def _get_lidar(self):
        angles = np.linspace(0, 2 * math.pi, self.n_lidar_rays, endpoint=False) + self.robot_theta 
        return np.array([self._cast_ray(a) for a in angles], dtype=np.float32)

    def _cast_ray(self, angle):
        ox, oy   = float(self.robot_pos[0]), float(self.robot_pos[1])
        dx, dy   = math.cos(angle), math.sin(angle)
        min_dist = self.lidar_max_range

        walls = [
            (0.0,         0.0,         0.0,         self.room_h),
            (self.room_w, 0.0,         self.room_w, self.room_h),
            (0.0,         0.0,         self.room_w, 0.0),
            (0.0,         self.room_h, self.room_w, self.room_h),
        ]
        for seg in walls:
            d = self._ray_segment(ox, oy, dx, dy, *seg)
            if d is not None:
                min_dist = min(min_dist, d)

        for (rx, ry, rw, rh) in self.obstacles:
            for seg in [
                (rx,      ry,      rx + rw, ry),
                (rx + rw, ry,      rx + rw, ry + rh),
                (rx + rw, ry + rh, rx,      ry + rh),
                (rx,      ry + rh, rx,      ry),
            ]:
                d = self._ray_segment(ox, oy, dx, dy, *seg)
                if d is not None:
                    min_dist = min(min_dist, d)

        return min_dist

    def _ray_segment(self, ox, oy, dx, dy, x1, y1, x2, y2):
        """Ray–segment intersection. Returns distance t ≥ 0 or None."""
        sx, sy = x2 - x1, y2 - y1
        denom  = dx * sy - dy * sx
        if abs(denom) < 1e-10:
            return None
        t = ((x1 - ox) * sy - (y1 - oy) * sx) / denom
        u = ((x1 - ox) * dy  - (y1 - oy) * dx) / denom
        return t if (t >= 0 and 0.0 <= u <= 1.0) else None

    # ------------------------------------------------------------------
    # Collision detection
    # ------------------------------------------------------------------

    def _check_collision(self, pos):
        x, y = float(pos[0]), float(pos[1])
        r    = self.robot_radius
        #check if robot went outside the perimeter of the room
        if x - r < 0 or x + r > self.room_w or y - r < 0 or y + r > self.room_h:
            return True
        
        #check if robot is inside an obstacle
        for (ox, oy, ow, oh) in self.obstacles:
            cx = max(ox, min(x, ox + ow))
            cy = max(oy, min(y, oy + oh))
            if math.sqrt((x - cx) ** 2 + (y - cy) ** 2) < r:
                return True

        return False

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _init_pygame(self):
        pygame.init() #start the graphic engine
        W = int(self.room_w * self.RENDER_SCALE) #rescale the room to give more pixel to single posizion
        H = int(self.room_h * self.RENDER_SCALE) + self.HUD_HEIGHT # as above and add a space where will be written some informative text
        pygame.display.set_caption("DiffDrive")
        self._window = pygame.display.set_mode((W, H)) #display the window of decided dimension
        self._clock  = pygame.time.Clock() #create an object to manage the FPS
        try:
            self._font = pygame.font.SysFont("monospace", 15, bold=True)
        except Exception:
            self._font = pygame.font.Font(None, 18)

    def render(self):
        if self.render_mode not in ("human", "rgb_array"):
            return

        S  = self.RENDER_SCALE
        W  = int(self.room_w * S)
        H  = int(self.room_h * S)
        TH = H + self.HUD_HEIGHT

        COL_BG       = (15,  18,  28) #color of the background 
        COL_WALL     = (40,  50,  80) #color of the wall
        COL_OBS      = (50,  60, 100) #color of the obstacles
        COL_OBS_EDGE = (80,  95, 140) #color of the obstacle edges
        COL_ROBOT    = (80, 200, 255) #color of the robot
        COL_GLOW     = (30, 100, 200) #color of the glow for the robot
        COL_GOAL     = (80, 220, 120) #color of the goal position
        COL_GOAL_GLW = (20, 120,  50) #color of the glow for the goal
        COL_LIDAR    = (255,  80,  80) #color of the lidar
        COL_HUD_BG   = (14,  17,  30) #color of the background space dedicated to other info
        COL_HUD_BDR  = (40,  50,  90) #color of the border for the hud space
        COL_TEXT     = (180, 200, 240) #color for the text
        COL_ACCENT   = (100, 160, 255)

        def to_px(x, y): #return the cordinates adapted to the Scale factor with origin adapted to be (0,0) 
            return int(x * S), int((self.room_h - y) * S)

        if not hasattr(self, "_font") or not pygame.get_init():
            self._init_pygame()
        if self._window is None and self.render_mode == "human":
            self._init_pygame()

        canvas = pygame.Surface((W, TH))
        canvas.fill(COL_BG)

        # Room boundary
        pygame.draw.rect(canvas, COL_WALL, (0, 0, W, H), 3) #draw the rectangle on the surface that represent the walls

        # Obstacles
        for (ox, oy, ow, oh) in self.obstacles:
            px, py = to_px(ox, oy + oh) #obtain corrected coordinated
            pw, ph = int(ow * S), int(oh * S) #obstain scaled width
            pygame.draw.rect(canvas, COL_OBS,      (px, py, pw, ph))
            pygame.draw.rect(canvas, COL_OBS_EDGE, (px, py, pw, ph), 2)

        # Lidar rays
        if self._last_lidar is not None:
            rx, ry = to_px(*self.robot_pos)
            angles = np.linspace(0, 2 * math.pi, self.n_lidar_rays, endpoint=False) + self.robot_theta
            ray_surf = pygame.Surface((W, H), pygame.SRCALPHA)
            for dist, angle in zip(self._last_lidar, angles):
                ex = float(self.robot_pos[0]) + dist * math.cos(angle)
                ey = float(self.robot_pos[1]) + dist * math.sin(angle)
                epx, epy = to_px(ex, ey)
                pygame.draw.line(ray_surf,   (*COL_LIDAR,  50), (rx, ry), (epx, epy), 1)
                pygame.draw.circle(ray_surf, (*COL_LIDAR, 200), (epx, epy), 3)
            canvas.blit(ray_surf, (0, 0))

        # Goal
        gx, gy    = to_px(*self.goal_pos)
        goal_surf = pygame.Surface((W, H), pygame.SRCALPHA)
        pygame.draw.circle(goal_surf, (*COL_GOAL_GLW,  60), (gx, gy), 22)
        pygame.draw.circle(goal_surf, (*COL_GOAL,     160), (gx, gy), 12)
        canvas.blit(goal_surf, (0, 0))
        pygame.draw.circle(canvas, COL_GOAL, (gx, gy), 12)

        # Robot glow + body
        rx, ry   = to_px(*self.robot_pos)
        r_px     = int(self.robot_radius * S)
        rob_surf = pygame.Surface((W, H), pygame.SRCALPHA)
        for radius, alpha in [(r_px + 12, 18), (r_px + 7, 35), (r_px + 3, 60)]:
            pygame.draw.circle(rob_surf, (*COL_GLOW, alpha), (rx, ry), radius)
        canvas.blit(rob_surf, (0, 0))
        pygame.draw.circle(canvas, COL_ROBOT, (rx, ry), r_px)
        pygame.draw.circle(canvas, (220, 240, 255), (rx - 3, ry - 3), max(2, r_px // 4))

        # Heading arrow
        hx = rx + int(r_px * 1.6 * math.cos(self.robot_theta))
        hy = ry - int(r_px * 1.6 * math.sin(self.robot_theta))
        pygame.draw.line(canvas, (220, 240, 255), (rx, ry), (hx, hy), 2)

        # HUD
        pygame.draw.rect(canvas, COL_HUD_BG,  (0, H, W, self.HUD_HEIGHT))
        pygame.draw.line(canvas, COL_HUD_BDR, (0, H), (W, H), 2)
        dist = float(np.linalg.norm(self.goal_pos - self.robot_pos))
        canvas.blit(self._font.render(f"STEP  {self.current_step} / {self.max_step}",          True, COL_TEXT),   (20,  H + 10))
        canvas.blit(self._font.render(f"DIST TO GOAL  {dist:.2f} m",                           True, COL_ACCENT), (20,  H + 32))
        canvas.blit(self._font.render(f"theta = {math.degrees(self.robot_theta):+.1f}°",           True, COL_TEXT),   (320, H + 10))
        canvas.blit(self._font.render(f"pos ({self.robot_pos[0]:.1f}, {self.robot_pos[1]:.1f})", True, COL_TEXT), (320, H + 32))

        if self.render_mode == "human" and self._window is not None:
            self._window.blit(canvas, (0, 0))
            pygame.display.flip()
            self._clock.tick(self.FPS)
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()
            self._window = None