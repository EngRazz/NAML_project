from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm

class GridWorldMovingObstacleAgent:
    def __init__(
            self,
            env: gym.Env,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay:float,
            final_epsilon: float,
            discount_factor: float = 0.95,
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor

        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.training_error = []
    
    def state_to_key(self, obs: dict[str, np.ndarray]) -> tuple:
        return (
            tuple(obs["agent"]),
            tuple(obs["target"])
        )
    
    def get_action(self, obs: dict[str, np.ndarray]) -> int:
        state = self.state_to_key(obs) #make the observation hashable

        if np.random.random()< self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[state]))
    
    def update(
            self,
            obs: dict[str, np.ndarray],
            action: int,
            reward: float,
            terminated: bool,
            next_obs: dict[str, np.ndarray],
    ):
        state = self.state_to_key(obs)
        next_state = self.state_to_key(next_obs)
        future_q_value = (not terminated) * np.max(self.q_values[next_state])
        target = reward + self.discount_factor * future_q_value
        temporal_difference = target - self.q_values[state][action]
        
        self.q_values[state][action] += self.learning_rate * temporal_difference

        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
    
