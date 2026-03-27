import numpy as np
import gymnasium
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

from grid_envs import GridFlagEnv
from grid_agents import GridFlagAgent, obs_to_key

# ── Environment config ────────────────────────────────────────────────────────

FLAG_CELLS = [
    (1, 2),
    (3, 7),
    (6, 1),
    (7, 8),
    (9, 4),
]

ENV_KWARGS = dict(
    grid_size=(10, 10),
    max_step=100,
    agent_start=(5, 5),
    flag_value=1,
    flag_cells=FLAG_CELLS,
)

gymnasium.register(
    id="GridFlagEnv",
    entry_point="grid_envs:GridFlagEnv",
    kwargs=ENV_KWARGS,
)

# ── Hyperparameters ───────────────────────────────────────────────────────────

NUM_EPISODES    = 5_000   # Total training episodes
RECORD_EVERY    = 500     # Save a video every N training episodes
LEARNING_RATE   = 0.1
INITIAL_EPSILON = 1.0
FINAL_EPSILON   = 0.05
# Decay epsilon linearly so it reaches FINAL_EPSILON by the last episode
EPSILON_DECAY   = (INITIAL_EPSILON - FINAL_EPSILON) / NUM_EPISODES
DISCOUNT_FACTOR = 0.95

# ── Training ──────────────────────────────────────────────────────────────────

# RecordVideo requires render_mode="rgb_array" to capture frames.
# We wrap with:
#   RecordVideo        – saves an .mp4 every RECORD_EVERY episodes
#   RecordEpisodeStatistics – tracks reward, length, and time per episode
# The base env must use render_mode="rgb_array" for RecordVideo to work
train_env = gymnasium.make("GridFlagEnv", render_mode="rgb_array")

agent = GridFlagAgent(
    env=train_env,
    learning_rate=LEARNING_RATE,
    initial_epsilon=INITIAL_EPSILON,
    epsilon_decay=EPSILON_DECAY,
    final_epsilon=FINAL_EPSILON,
    discount_factor=DISCOUNT_FACTOR,
)

agent.train_recorded(
    num_episodes=NUM_EPISODES,
    video_folder="videos/training",
    record_every=RECORD_EVERY,
    log_every=500,
)

agent.plot_training(window=100)

# ── Evaluation ────────────────────────────────────────────────────────────────

# Set epsilon to 0 so the agent acts greedily (pure exploitation, no exploration)
agent.eval_recorded(
    video_folder="videos/evaluation",
    name_prefix="eval",
)
