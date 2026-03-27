import gymnasium

from grid_flag_env import GridFlagEnv
from grid_flag_agent import GridFlagAgent
from gymnasium.utils.env_checker import check_env

FLAG_CELLS = [
    (1, 2),
    (3, 7),
    (6, 1),
    (7, 8),
    (9, 4),
]

# register the environment with Gymnasium
gymnasium.register(
    id="GridFlagEnv-v0",
    entry_point="grid_flag_env:GridFlagEnv",
    kwargs={
        "grid_size": (10, 10),
        "max_step": 100,
        "agent_start": (0, 0),
        "flag_value": 1,
        "flag_cells": FLAG_CELLS,
    },
)

try:
    env = gymnasium.make("GridFlagEnv-v0")
    check_env(env)
    print("Environment passes all checks!")
    
except Exception as e:
    print(f"Environment has issues: {e}")

