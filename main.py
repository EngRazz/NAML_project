from grid_envs import GridFlagEnv
from grid_agents import GridFlagAgent
from gymnasium.utils.env_checker import check_env

try:
    env = GridFlagEnv()
    check_env(env)
    print("Environment passes all checks!")
except Exception as e:
    print(f"Environment has issues: {e}")

FLAG_CELLS = [
    (1, 2),
    (3, 7),
    (6, 1),
    (7, 8),
    (9, 4),
]