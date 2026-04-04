import gymnasium as gym

from grid_envs2 import GridWorldMovingObstacle
from grid_agent2 import GridWorldMovingObstacleAgent
from gymnasium.utils.env_checker import check_env
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pygame
FLAG_CELLS = [
    (1, 2),
    (3, 7),
    (6, 1),
    (7, 8),
    (9, 4),
]

# register the environment with Gymnasium
gym.register(
    id = "GridWorldMovingObstacle-v0",
    entry_point = "grid_envs2:GridWorldMovingObstacle",
    max_episode_steps = 500, #prevent infinite episodes
)

learning_rate = 0.01
n_episodes = 500
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.01



env = gym.make("GridWorldMovingObstacle-v0", render_mode="human")
check_env(env)
print("Environment passes all checks!")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
agent = GridWorldMovingObstacleAgent(
    env = env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)
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

        # visualizza ogni tanto
        if episode % 50 == 0:
            visualize_episode()

def visualize_episode():
    obs, _ = env.reset()
    done = False

    while not done:
        action = agent.get_action(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        env.render()

def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

def plot_training():
    # Smooth over a 500-episode window
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
    # Episode rewards (win/loss performance)
    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
    env.return_queue,
    rolling_length,
    "valid"
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    # Episode lengths (how many actions per hand)
    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        env.length_queue,
        rolling_length,
        "valid"
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    # Training error (how much we're still learning)
    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent.training_error,
        rolling_length,
        "same"
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout()
    plt.show()


try:
    # 1. Reset dell'ambiente per inizializzare le posizioni
    obs, info = env.reset()
    
    print("Finestra aperta. Premi la 'X' della finestra per chiudere.")
    
    # 2. Loop infinito per mantenere la finestra attiva
    running = True
    while running:
        # --- GESTIONE EVENTI (Fondamentale per non far crashare la finestra) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- LOGICA (opzionale: muovi l'agente a caso per testare) ---
        # action = env.action_space.sample()
        # obs, reward, terminated, truncated, info = env.step(action)
        # if terminated or truncated:
        #     env.reset()

        # --- RENDERING ---
        env.render()
        
    # 3. Pulizia finale
    env.close()

except Exception as e:
    print(f"Errore durante l'esecuzione: {e}")
    import traceback
    traceback.print_exc()
