from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from grid_envs import GridFlagEnv

class GridFlagAgent:
    def __init__(
        self,
        env: GridFlagEnv,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Choose an action using epsilon-greedy strategy.
        Returns:
            action: 0 (stand) or 1 (hit)
        """
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # With probability (1-epsilon): exploit (best known action)
        else:
            return int(np.argmax(self.q_values[obs]))
        
    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """
        Update Q-value based on experience.
        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[obs][action]

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)
    
    def decay_epsilon(self):
        """
        Reduce exploration rate after each episode.
        """
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def train(self, num_episodes: int):
        """
        Train the agent for a given number of episodes.
        """
        for _ in tqdm(range(num_episodes), desc="Training"):
            obs, _ = self.env.reset()
            terminated = False

            while not terminated:
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                self.update(obs, action, reward, terminated or truncated, next_obs)
                obs = next_obs
            
            self.decay_epsilon()

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
    
