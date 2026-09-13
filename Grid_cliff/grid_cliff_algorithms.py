"""Tabular cliff experiments; both algorithms share exploration and cutoff handling."""
from pathlib import Path
import gymnasium as gym
import numpy as np
from Grid_cliff.grid_cliff_env import ENV_ID, RandomCliffWalkingEnv
from rl_project.tabular import TabularAgent


def register_cliff_env():
    if ENV_ID not in gym.registry:
        gym.register(id=ENV_ID, entry_point='Grid_cliff.grid_cliff_env:RandomCliffWalkingEnv')


def train(algorithm, episodes=5000, seed=0, output_root=None, record_every=None,
          max_env_steps=None, env=None, plot=True):
    agent = TabularAgent(env if env is not None else RandomCliffWalkingEnv(),
                         learning_rate=.1, discount_factor=.99, initial_epsilon=1.,
                         epsilon_decay=.9/3000, final_epsilon=.1, seed=seed, environment='cliff')
    result = agent.train(episodes, algorithm=algorithm, output_root=output_root,
                         max_env_steps=max_env_steps, record_every=record_every, plot=plot)
    return agent, result


def train_q_learning(**kwargs):
    return train('qlearning', **kwargs)


def train_sarsa(**kwargs):
    return train('sarsa', **kwargs)


def plot_comparison(q_result, sarsa_result, save_path, window=100):
    import matplotlib.pyplot as plt
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for result, label in ((q_result, 'Q-learning'), (sarsa_result, 'SARSA')):
        for ax, key in zip(axes, ('reward', 'success', 'length')):
            values = np.array([row[key] for row in result.episodes], dtype=float)
            means = [values[max(0, i-window+1):i+1].mean() for i in range(len(values))]
            ax.plot(np.arange(1, len(values)+1), means, label=label)
    for ax, title in zip(axes, ('Return', 'Success rate', 'Episode length')):
        ax.set(title=f'{title}: rolling mean (up to {window})', xlabel='Episode')
        ax.grid(alpha=.2)
        ax.legend()
    axes[1].set_ylim(-.05, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
