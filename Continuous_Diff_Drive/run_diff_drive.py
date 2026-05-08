import os
import numpy as np
from diff_drive_env import DiffDriveEnv
from diff_drive_agent import DiffDriveAgent
import torch

# ── Environment config ────────────────────────────────────────────────────────

# A room with a few axis-aligned rectangular obstacles.
# Each obstacle is (x, y, width, height) in metres, origin at bottom-left.
OBSTACLES = [
    (2.0, 4.0, 1.5, 0.3),   # horizontal wall near the middle-left
    (5.0, 2.0, 0.3, 3.0),   # vertical pillar in the centre
    (7.0, 6.0, 1.5, 0.3),   # horizontal shelf upper-right
]

ENV_KWARGS = dict(
    room_size       = (10.0, 10.0),
    obstacles       = None,
    random_obst     = True,
    robot_start     = (1.0, 1.0),
    goal_pos        = (8.5, 8.5),
    max_step        = 1000,
    n_lidar_rays    = 16,
    lidar_max_range = 5.0,
    robot_radius    = 0.3,
    dt              = 0.1,
    render_mode     = "rgb_array",   # required by RecordVideo
)

# ── Hyperparameters ───────────────────────────────────────────────────────────

NUM_EPISODES  = 2_000   # Total training episodes
RECORD_EVERY  = 500     # Save a training video every N episodes
LOG_EVERY     = 100     # Print a progress line every N episodes

ACTOR_LR      = 1e-4
CRITIC_LR     = 1e-3
DISCOUNT      = 0.95    # gamma
TAU           = 0.005   # soft-update rate for target networks
NOISE_STD     = 0.2     # std of Gaussian exploration noise
NOISE_CLIP    = 0.5     # max |noise|
BATCH_SIZE    = 256
BUFFER_SIZE   = 100_000
HIDDEN_DIM    = 128
WARMUP_STEPS  = 1_000   # random actions before learning starts

DEVICE        = "cpu"   # change to "cuda" if a GPU is available

# ── Output directories ────────────────────────────────────────────────────────

os.makedirs("videos/training",   exist_ok=True)
os.makedirs("videos/evaluation", exist_ok=True)
os.makedirs("images",            exist_ok=True)

# ── Build environment & agent ─────────────────────────────────────────────────

env = DiffDriveEnv(**ENV_KWARGS)

agent = DiffDriveAgent(
    env          = env,
    actor_lr     = ACTOR_LR,
    critic_lr    = CRITIC_LR,
    discount     = DISCOUNT,
    tau          = TAU,
    noise_std    = NOISE_STD,
    noise_clip   = NOISE_CLIP,
    batch_size   = BATCH_SIZE,
    buffer_size  = BUFFER_SIZE,
    hidden_dim   = HIDDEN_DIM,
    warmup_steps = WARMUP_STEPS,
    device       = DEVICE,
)

# ── Training ──────────────────────────────────────────────────────────────────

print("=" * 60)
print("  DiffDrive · DDPG Training")
print(f"  Episodes   : {NUM_EPISODES}")
print(f"  Warmup     : {WARMUP_STEPS} steps")
print(f"  Buffer     : {BUFFER_SIZE}")
print(f"  Batch size : {BATCH_SIZE}")
print(f"  Device     : {DEVICE}")
print("=" * 60)

agent.train_recorded(
    num_episodes = NUM_EPISODES,
    video_folder = "videos/training",
    record_every = RECORD_EVERY,
    log_every    = LOG_EVERY,
    add_noise    = True
)

# ── Evaluation ────────────────────────────────────────────────────────────────
agent.eval_recorded(
    video_folder = "videos/evaluation/DiffDriveOrnsteinUhlenbeckNoise",
    name_prefix  = "DiffDriveEval",
    n_episodes   = 3,
    add_noise    = False
)

# Nel tuo main.py o notebook:
agent.plot_critic_heatmap(resolution=60, theta=0.0) # Robot che guarda a "destra"
agent.plot_critic_heatmap(resolution=60, theta=np.pi) # Robot che guarda a "sinistra"
