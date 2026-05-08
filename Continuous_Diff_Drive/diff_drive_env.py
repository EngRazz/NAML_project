from collections import deque
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
        - LiDAR shaping: safer moves near obstacles are rewarded
        - Collision: -50, episode ends
        - Goal reached: +100, episode ends
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    # Render config
    RENDER_SCALE = 80   # pixels per meter
    HUD_HEIGHT   = 60
    FPS          = 30

    GOAL_REWARD = 100.0
    COLLISION_REWARD = -50.0
    SAFE_DISTANCE = 1.2
    FRONT_SAFE_DISTANCE = 1.0
    GOAL_BLOCK_DISTANCE = 2.0
    GOAL_PROGRESS_WEIGHT = 8.0
    BLOCKED_GOAL_PROGRESS_WEIGHT = 3.0
    DANGER_REDUCTION_WEIGHT = 3.0
    DANGER_PENALTY_WEIGHT = 2.0
    FRONT_DANGER_PENALTY_WEIGHT = 1.0
    ORIENTATION_WEIGHT = 0.2
    TIME_PENALTY = -0.02

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
        obstacle_mode   = "curriculum",  # curriculum, random_blocks, evaluate_like_detour, wall_with_gap
        obstacle_mix    = None,
    ):
        super().__init__()

        self.room_w, self.room_h = room_size
        self.obstacles       = list(obstacles) if obstacles is not None else []
        self.random_obst     = random_obst
        self.robot_start     = np.array(robot_start, dtype=np.float32)
        self.goal_pos        = np.array(goal_pos,    dtype=np.float32)
        self.max_step        = max_step
        self.n_lidar_rays    = n_lidar_rays
        self.lidar_max_range = lidar_max_range
        self.robot_radius    = robot_radius
        self.dt              = dt
        self.render_mode     = render_mode
        self.obstacle_mode   = obstacle_mode
        self.obstacle_mix    = obstacle_mix or {
            "evaluate_like_detour": 0.4,
            "wall_with_gap": 0.4,
            "random_blocks": 0.2,
        }

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

    @staticmethod
    def _point_rect_distance(point, rect):
        px, py = float(point[0]), float(point[1])
        rx, ry, rw, rh = rect
        cx = max(rx, min(px, rx + rw))
        cy = max(ry, min(py, ry + rh))
        return math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    
    def _lidar_clearance(self, lidar):
        """Robust obstacle clearance estimate in metres from LiDAR readings."""
        lidar = np.asarray(lidar, dtype=np.float32)
        return float(np.percentile(lidar, 20))

    def _front_clearance(self, lidar):
        """Minimum LiDAR distance in the forward +/-45 degree sector."""
        lidar = np.asarray(lidar, dtype=np.float32)
        angles = np.linspace(0, 2 * math.pi, self.n_lidar_rays, endpoint=False)
        rel_angles = (angles + math.pi) % (2 * math.pi) - math.pi
        front_mask = np.abs(rel_angles) <= math.pi / 4
        if not np.any(front_mask):
            return float(lidar[0])
        return float(np.min(lidar[front_mask]))

    def _goal_direction_clearance(self, lidar, angle_rel):
        """LiDAR distance along the ray closest to the goal direction."""
        lidar = np.asarray(lidar, dtype=np.float32)
        angles = np.linspace(0, 2 * math.pi, self.n_lidar_rays, endpoint=False)
        diffs = (angles - angle_rel + math.pi) % (2 * math.pi) - math.pi
        return float(lidar[int(np.argmin(np.abs(diffs)))])

    def _reward_components(self, dist, curr_lidar):
        """Compute shaped reward terms for goal progress and LiDAR safety."""
        goal_progress = self.prev_dist - dist

        diff = self.goal_pos - self.robot_pos
        angle_glob = math.atan2(float(diff[1]), float(diff[0]))
        angle_rel = (angle_glob - self.robot_theta + math.pi) % (2 * math.pi) - math.pi
        angle_err = abs(angle_rel)
        orientation = 1.0 - angle_err / math.pi

        clearance = self._lidar_clearance(curr_lidar)
        prev_clearance = self._lidar_clearance(self._prev_lidar)
        danger = max(0.0, (self.SAFE_DISTANCE - clearance) / self.SAFE_DISTANCE)
        prev_danger = max(0.0, (self.SAFE_DISTANCE - prev_clearance) / self.SAFE_DISTANCE)
        danger_reduction = prev_danger - danger

        front_clearance = self._front_clearance(curr_lidar)
        front_danger = max(
            0.0,
            (self.FRONT_SAFE_DISTANCE - front_clearance) / self.FRONT_SAFE_DISTANCE,
        )

        goal_direction_lidar = self._goal_direction_clearance(curr_lidar, angle_rel)
        goal_blocked = goal_direction_lidar < self.GOAL_BLOCK_DISTANCE
        goal_weight = (
            self.BLOCKED_GOAL_PROGRESS_WEIGHT
            if goal_blocked
            else self.GOAL_PROGRESS_WEIGHT
        )

        reward_goal_progress = goal_weight * goal_progress
        reward_danger_reduction = self.DANGER_REDUCTION_WEIGHT * danger_reduction
        reward_danger_penalty = -self.DANGER_PENALTY_WEIGHT * danger ** 2
        reward_front_penalty = -self.FRONT_DANGER_PENALTY_WEIGHT * front_danger ** 2
        reward_orientation = self.ORIENTATION_WEIGHT * orientation

        reward = (
            reward_goal_progress
            + reward_danger_reduction
            + reward_danger_penalty
            + reward_front_penalty
            + reward_orientation
            + self.TIME_PENALTY
        )

        return reward, {
            "reward_goal_progress": reward_goal_progress,
            "reward_danger_reduction": reward_danger_reduction,
            "reward_danger_penalty": reward_danger_penalty,
            "reward_front_penalty": reward_front_penalty,
            "reward_orientation": reward_orientation,
            "goal_blocked": goal_blocked,
            "lidar_clearance": clearance,
            "front_clearance": front_clearance,
            "goal_direction_clearance": goal_direction_lidar,
        }

    def _empty_reward_components(self, curr_lidar):
        return {
            "reward_goal_progress": 0.0,
            "reward_danger_reduction": 0.0,
            "reward_danger_penalty": 0.0,
            "reward_front_penalty": 0.0,
            "reward_orientation": 0.0,
            "goal_blocked": False,
            "lidar_clearance": self._lidar_clearance(curr_lidar),
            "front_clearance": self._front_clearance(curr_lidar),
            "goal_direction_clearance": self.lidar_max_range,
        }

    def _sample_obstacles(self):
        for _ in range(100):
            mode = self._choose_obstacle_mode()
            if mode == "evaluate_like_detour":
                obstacles = self._sample_evaluate_like_detour()
            elif mode == "wall_with_gap":
                obstacles = self._sample_wall_with_gap()
            else:
                obstacles = self._sample_random_blocks()

            if self._obstacle_set_is_valid(obstacles):
                return obstacles

        return []

    def _choose_obstacle_mode(self):
        if self.obstacle_mode != "curriculum":
            return self.obstacle_mode

        modes = list(self.obstacle_mix.keys())
        weights = np.array([self.obstacle_mix[m] for m in modes], dtype=np.float64)
        if len(modes) == 0 or weights.sum() <= 0:
            return "random_blocks"
        weights = weights / weights.sum()
        return str(self.np_random.choice(modes, p=weights))

    def _sample_random_blocks(self):
        obstacles = []
        protected = [self.robot_start, self.goal_pos]
        clearance = self.robot_radius + 0.5
        margin = self.robot_radius + 0.2

        for _ in range(3):
            for _ in range(100):   # max attempts
                max_w = max(0.5, min(2.0, self.room_w - 2 * margin))
                max_h = max(0.3, min(1.5, self.room_h - 2 * margin))
                w = float(self.np_random.uniform(0.5, max_w))
                h = float(self.np_random.uniform(0.3, max_h))
                x_hi = max(margin, self.room_w - margin - w)
                y_hi = max(margin, self.room_h - margin - h)
                x = float(self.np_random.uniform(margin, x_hi))
                y = float(self.np_random.uniform(margin, y_hi))
                rect = (x, y, w, h)
                # Reject obstacles that block the start or goal clearance zone.
                too_close = any(
                    self._point_rect_distance(point, rect) < clearance
                    for point in protected
                )
                if not too_close:
                    obstacles.append(rect)
                    break
        return obstacles

    def _sample_evaluate_like_detour(self):
        base = [
            (2.0, 4.0, 3.0, 0.3),
            (5.0, 2.0, 0.3, 3.0),
            (7.0, 6.0, 1.5, 0.3),
        ]
        obstacles = []
        for x, y, w, h in base:
            jx = float(self.np_random.uniform(-0.45, 0.45))
            jy = float(self.np_random.uniform(-0.45, 0.45))
            jw = float(self.np_random.uniform(-0.25, 0.35))
            jh = float(self.np_random.uniform(-0.15, 0.35))
            nw = max(0.25, w + jw)
            nh = max(0.25, h + jh)
            nx = float(np.clip(x + jx, 0.5, self.room_w - nw - 0.5))
            ny = float(np.clip(y + jy, 0.5, self.room_h - nh - 0.5))
            obstacles.append((nx, ny, nw, nh))
        return obstacles

    def _sample_wall_with_gap(self):
        gap_size = float(self.np_random.uniform(1.5, 2.3))
        thickness = float(self.np_random.uniform(0.25, 0.45))
        obstacles = []

        if bool(self.np_random.integers(0, 2)):
            x = float(self.np_random.uniform(3.5, 6.3))
            gap_center = float(self.np_random.uniform(3.0, 7.0))
            gap_low = max(0.8, gap_center - gap_size / 2)
            gap_high = min(self.room_h - 0.8, gap_center + gap_size / 2)
            if gap_low > 0.8:
                obstacles.append((x, 0.8, thickness, gap_low - 0.8))
            if gap_high < self.room_h - 0.8:
                obstacles.append((x, gap_high, thickness, self.room_h - 0.8 - gap_high))
        else:
            y = float(self.np_random.uniform(3.5, 6.3))
            gap_center = float(self.np_random.uniform(3.0, 7.0))
            gap_low = max(0.8, gap_center - gap_size / 2)
            gap_high = min(self.room_w - 0.8, gap_center + gap_size / 2)
            if gap_low > 0.8:
                obstacles.append((0.8, y, gap_low - 0.8, thickness))
            if gap_high < self.room_w - 0.8:
                obstacles.append((gap_high, y, self.room_w - 0.8 - gap_high, thickness))

        return obstacles

    def _obstacle_set_is_valid(self, obstacles):
        protected_clearance = self.robot_radius + 0.2
        for point in (self.robot_start, self.goal_pos):
            if any(
                self._point_rect_distance(point, rect) < protected_clearance
                for rect in obstacles
            ):
                return False
        return self._has_feasible_path(obstacles)

    def _has_feasible_path(self, obstacles, resolution=0.2):
        inflate = self.robot_radius + 0.05

        def to_idx(point):
            x, y = float(point[0]), float(point[1])
            return int(round(x / resolution)), int(round(y / resolution))

        max_ix = int(round(self.room_w / resolution))
        max_iy = int(round(self.room_h / resolution))
        start = to_idx(self.robot_start)
        goal = to_idx(self.goal_pos)

        def free(idx):
            ix, iy = idx
            if ix < 0 or iy < 0 or ix > max_ix or iy > max_iy:
                return False
            x, y = ix * resolution, iy * resolution
            if (
                x < self.robot_radius
                or x > self.room_w - self.robot_radius
                or y < self.robot_radius
                or y > self.room_h - self.robot_radius
            ):
                return False
            for ox, oy, ow, oh in obstacles:
                if (
                    ox - inflate <= x <= ox + ow + inflate
                    and oy - inflate <= y <= oy + oh + inflate
                ):
                    return False
            return True

        if not free(start) or not free(goal):
            return False

        queue = deque([start])
        visited = {start}
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

        while queue:
            node = queue.popleft()
            if node == goal:
                return True
            for dx, dy in neighbors:
                nxt = (node[0] + dx, node[1] + dy)
                if nxt not in visited and free(nxt):
                    visited.add(nxt)
                    queue.append(nxt)

        return False

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

        # Initial lidar state
        self._prev_lidar  = self._get_lidar()
        self._last_lidar  = self._prev_lidar.copy()

        if self.render_mode == "human" and self._window is None:
            self._init_pygame()

        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1

        v_linear  = float(action[0])
        v_angular = float(action[1])

        # Differential drive kinematics
        self.robot_theta += v_angular * self.dt
        self.robot_theta = (
            (self.robot_theta + math.pi) % (2 * math.pi)
        ) - math.pi

        new_pos = self.robot_pos + np.array([
            v_linear * math.cos(self.robot_theta),
            v_linear * math.sin(self.robot_theta),
        ], dtype=np.float32) * self.dt

        # Collision handling
        collision = self._check_collision(new_pos)

        if not collision:
            self.robot_pos = new_pos

        # Current state measurements
        dist = float(np.linalg.norm(self.goal_pos - self.robot_pos))

        goal_reached = (
            dist < self.robot_radius + 0.2
        )

        curr_lidar = self._get_lidar()

        reward_components = self._empty_reward_components(curr_lidar)

        # Terminal rewards
        if collision:
            reward = self.COLLISION_REWARD
            terminated = True

        elif goal_reached:
            reward = self.GOAL_REWARD
            terminated = True

        else:
            reward, reward_components = self._reward_components(dist, curr_lidar)
            terminated = False

        # Update previous state trackers
        self.prev_dist = dist
        self._prev_lidar = curr_lidar
        self._last_lidar = curr_lidar

        truncated = self.current_step >= self.max_step

        info = {
            "dist_to_goal": dist,
            "collision": collision,
            "goal_reached": goal_reached,
            "steps": self.current_step,
            **reward_components,
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

    def _init_pygame(self, create_window=True):
        pygame.init()
        W = int(self.room_w * self.RENDER_SCALE)
        H = int(self.room_h * self.RENDER_SCALE) + self.HUD_HEIGHT
        if create_window:
            pygame.display.set_caption("DiffDrive")
            self._window = pygame.display.set_mode((W, H))
            self._clock  = pygame.time.Clock()
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
            self._init_pygame(create_window=self.render_mode == "human")
        if self._window is None and self.render_mode == "human":
            self._init_pygame(create_window=True)

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
        if pygame.get_init():
            pygame.quit()
        self._window = None
        self._clock = None
