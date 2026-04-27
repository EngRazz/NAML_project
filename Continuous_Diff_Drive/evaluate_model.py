import os
import numpy as np
from diff_drive_env import DiffDriveEnv
from diff_drive_agent import DiffDriveAgent
import torch
import pygame

# A room with a few axis-aligned rectangular obstacles.
# Each obstacle is (x, y, width, height) in metres, origin at bottom-left.
OBSTACLES = [
    (2.0, 4.0, 3.0, 0.3),   # horizontal wall near the middle-left
    (5.0, 2.0, 0.3, 3.0),   # vertical pillar in the centre
    (7.0, 6.0, 1.5, 0.3),   # horizontal shelf upper-right
]

ENV_KWARGS = dict(
    room_size       = (10.0, 10.0),
    obstacles       = OBSTACLES,
    random_obst     = False,
    robot_start     = (1.0, 1.0),
    goal_pos        = (8.5, 8.5),
    max_step        = 1000,
    n_lidar_rays    = 16,
    lidar_max_range = 5.0,
    robot_radius    = 0.3,
    dt              = 0.1,
    render_mode     = "rgb_array",   # required by RecordVideo
)

env = DiffDriveEnv(**ENV_KWARGS)

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

ddpd_checkpoint_path = "Continuous_Diff_Drive/models/ddpg_checkpoint.pt"
if os.path.exists(ddpd_checkpoint_path):
    print(f"Loading checkpoint from {ddpd_checkpoint_path}")
    checkpoint = torch.load(ddpd_checkpoint_path, map_location=DEVICE)
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.critic.load_state_dict(checkpoint["critic"])
    agent.actor_target.load_state_dict(checkpoint["actor_target"])
    agent.critic_target.load_state_dict(checkpoint["critic_target"])
else:
    print(f"No checkpoint found at {ddpd_checkpoint_path}")
    exit(1)  

N = 3
for i in range(N):
    try:
        os.remove(f"videos/evaluation/DiffDriveOrnsteinUhlenbeckNoise/DiffDriveEval_NeverseenObstacles-episode-{i}.mp4")
    except FileNotFoundError:
        pass

agent.eval_recorded(
    video_folder = "videos/evaluation/DiffDriveOrnsteinUhlenbeckNoise",
    name_prefix  = "DiffDriveEval_NeverseenObstacles",
    n_episodes   = N,
    add_noise    = False
)
pygame.quit()
