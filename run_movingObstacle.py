import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
from gymnasium.utils.env_checker import check_env
from grid_envs2 import GridWorldMovingObstacle
import grid_agent2
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# ── Iperparametri ────────────────────────────────────────────────
learning_rate  = 0.01
n_episodes     = 5000
start_epsilon  = 1.0
epsilon_decay  = start_epsilon / (n_episodes / 2)
final_epsilon  = 0.01
VISUALIZE_EVERY = 100   # registra un episodio ogni N

# ── Check env (sull'env nudo, senza wrapper) ─────────────────────
_check_env = GridWorldMovingObstacle(size=5, max_steps=50)
check_env(_check_env, warn=True)
_check_env.close()
print("Environment passes all checks!")

# ── Env di training (nessun rendering → più veloce) ──────────────
gym.register(
    id="GridWorldMovingObstacle-v0",
    entry_point="grid_envs2:GridWorldMovingObstacle",
    max_episode_steps=500,
)

env = gym.make("GridWorldMovingObstacle-v0", render_mode=None)
env = RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = grid_agent2.GridWorldMovingObstacleAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

# ── Funzione di visualizzazione → salva un .mp4 ──────────────────
def record_episode(episode_num: int):
    """
    Crea un env separato con rgb_array + RecordVideo,
    gira un episodio greedy (epsilon=0) e lo salva.
    """
    rec_env = gym.make("GridWorldMovingObstacle-v0", render_mode="rgb_array")
    rec_env = RecordVideo(
        rec_env,
        video_folder="./videos",
        episode_trigger=lambda _: True,   # registra sempre (è un env usa-e-getta)
        name_prefix=f"ep{episode_num:04d}",
    )

    obs, _ = rec_env.reset()
    done = False
    saved_epsilon = agent.epsilon          # salva epsilon corrente
    agent.epsilon = 0.0                    # greedy pura per la visualizzazione

    while not done:
        action = agent.get_action(obs)
        obs, _, terminated, truncated, _ = rec_env.step(action)
        done = terminated or truncated

    agent.epsilon = saved_epsilon          # ripristina epsilon
    rec_env.close()

# ── Training loop ────────────────────────────────────────────────
def train():
    for episode in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        done = False

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.update(obs, action, reward, terminated, next_obs)
            obs = next_obs
            done = terminated or truncated

        agent.decay_epsilon()

        if episode % VISUALIZE_EVERY == 0:
            record_episode(episode)

# ── Plot risultati ───────────────────────────────────────────────
def get_moving_avgs(arr, window, convolution_mode):
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

def plot_training():
    rolling_length = 50  # era 500 ma hai solo 500 episodi totali
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    axs[0].set_title("Episode rewards")
    axs[0].plot(get_moving_avgs(env.return_queue, rolling_length, "valid"))
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    axs[1].set_title("Episode lengths")
    axs[1].plot(get_moving_avgs(env.length_queue, rolling_length, "valid"))
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    axs[2].set_title("Training Error")
    axs[2].plot(get_moving_avgs(agent.training_error, rolling_length, "same"))
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout()
    plt.savefig("./videos/training_plot.png", dpi=150)
    plt.show()

# ── Entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    train()
    env.close()
    plot_training()