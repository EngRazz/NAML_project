from collections import defaultdict
import numpy as np
from tqdm import tqdm
from grid_flag_env import GridFlagEnv
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
import matplotlib.pyplot as plt

# Helper: convert the env's dict observation into a hashable tuple key for the Q-table.
# The env returns {"agent": [row, col], "flags": [0/1, ...]}.
# It flattens both arrays into one flat tuple so it can be used as a dict key.
def obs_to_key(obs: dict) -> tuple:
    return tuple(obs["agent"]) + tuple(obs["flags"])


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

        # Q-table: maps (state, action) to expected reward.
        # Keys are hashable tuples produced by obs_to_key().
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor  # how much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    def get_action(self, obs: tuple) -> int:
        """
        Choose an action using epsilon-greedy strategy.
        Args:
            obs: Hashable observation key (from obs_to_key).
        Returns:
            action: 0 (stay), 1 (up), 2 (down), 3 (left), or 4 (right)
        """
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # With probability (1-epsilon): exploit (best known action)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple,
    ):
        """
        Update Q-value based on experience.
        Args:
            obs: Hashable key for current state.
            next_obs: Hashable key for next state.
        """
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[obs][action]

        # Update our estimate in the direction of the error
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)
    
    def update_SARSA(
        self,
        obs: tuple,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple,
    ):
        """"
        Update Q-table based on SARSA update
        
        """ 
        next_action = self.get_action(next_obs)
        future_q_value = (not terminated) *  self.q_values[next_obs][next_action] 
        #MAIN difference with Q-learning is that the future Q_value is chosed following the polic
        target = reward + self.discount_factor * future_q_value
        temporal_difference = target - self.q_values[obs][action]
        
        # Update our estimate in the direction of the error
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

    #@ DEPRECATED? Train and train_sarsa aren't referenced anywhere 
    def train(self, num_episodes: int):
        """
        Train the agent for a given number of episodes.
        """
        for _ in tqdm(range(num_episodes), desc="Training"):
            obs_dict, _ = self.env.reset()
            obs = obs_to_key(obs_dict)  # Convert dict obs to hashable key

            terminated = False
            truncated = False

            # BUG FIX: also check truncated — the env signals episode end via
            # truncated (time limit), not terminated (which is always False here).
            while not terminated and not truncated:
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)

                self.update(obs_to_key(next_obs), action, reward, terminated or truncated, next_obs)
                obs = next_obs

            self.decay_epsilon()
    
    def train_SARSA(self, num_episodes: int):
        """"
        Train the agent for a given number of episodes usinf the SARSA algorithm
        
        """
        for _ in tqdm(range(num_episodes), desc="Training"):
            obs_dict, _ = self.env.reset()
            obs = obs_to_key(obs_dict)  # Convert dict obs to hashable key

            terminated = False
            truncated = False

            # BUG FIX: also check truncated — the env signals episode end via
            # truncated (time limit), not terminated (which is always False here).
            while not terminated and not truncated:
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)

                self.update_SARSA(obs_to_key(next_obs), action, reward, terminated or truncated, next_obs)
                obs = next_obs

            self.decay_epsilon()

    def train_recorded(self, num_episodes, video_folder="videos/training",
                    record_every=500, log_every=500):
        env = RecordVideo(
            self.env,
            video_folder=video_folder,
            name_prefix="GridFlagTrain",
            episode_trigger=lambda ep: ep % record_every == 0,
        )
        env = RecordEpisodeStatistics(env, buffer_length=num_episodes)

        # Store stats for plotting
        episode_rewards = []
        episode_lengths = []
        epsilons        = []

        for ep in tqdm(range(num_episodes), desc="Training"):
            obs_dict, _ = env.reset()
            obs = obs_to_key(obs_dict)
            terminated = False
            truncated  = False
            episode_reward = 0

            while not terminated and not truncated:
                action = self.get_action(obs)
                next_obs_dict, reward, terminated, truncated, _ = env.step(action)
                next_obs = obs_to_key(next_obs_dict)
                self.update(obs, action, reward, terminated or truncated, next_obs)
                obs = next_obs
                
                # Record stats
                episode_reward += reward

            self.decay_epsilon()

            # Record stats
            episode_rewards.append(episode_reward)
            episode_lengths.append(list(env.length_queue)[-1])
            epsilons.append(self.epsilon)

            if (ep + 1) % log_every == 0:
                recent = episode_rewards[-100:]
                avg = np.mean(recent)
                print(f"  Episode {ep + 1:>5} | "
                    f"avg reward (last 100): {avg:.2f} | "
                    f"epsilon: {self.epsilon:.3f}")

        env.close()

        rewards  = np.array(episode_rewards)
        lengths  = np.array(episode_lengths)
        episodes = np.arange(1, len(rewards) + 1)

        fig, axes = plt.subplots(2, 1, figsize=(16, 8))
        fig.suptitle("Training Curves", fontsize=14, fontweight="bold")

        axes[0].plot(episodes, rewards, color="steelblue", alpha=0.6, marker="o", markersize=3, label="Episode reward")
        axes[0].axhline(len(self.env.unwrapped.flag_cells) * self.env.unwrapped.flag_value, color="gold",
                        linestyle="--", linewidth=1, label="Max reward")
        
        ar_rewards = list()
        for i in range(100, len(rewards)):
            ar_rewards.append(np.mean(rewards[i-100:i]))
        axes[0].plot(episodes[100:], ar_rewards, color="orange", alpha=0.9, marker="o", markersize=3, label="Mean(100) reward")
        
        axes[0].set_ylabel("Reward")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

        axes[1].plot(episodes, lengths, color="coral", alpha=0.8, marker="o", markersize=3, label="Episode steps")
        axes[1].axhline(min(lengths), color="purple", linestyle="--", linewidth=1, label="Min steps")
        axes[1].axhline(lengths[-1], color="green" if min(lengths) == lengths[-1] else "red", linestyle="--", linewidth=1, label="Last steps")

        ar_lengths = list()
        for i in range(100, len(lengths)):
            ar_lengths.append(np.mean(lengths[i-100:i]))
        axes[1].plot(episodes[100:], ar_lengths, color="purple", alpha=0.9, marker="o", markersize=3, label="Mean(100) steps")

        axes[1].set_ylabel("Steps")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("./images/GridFlag_training_curves.png", dpi=150)
        print("Plot saved to GridFlag_training_curves.png")
        
    def train_recorded_SARSA(self, num_episodes, video_folder="videos/training",
                    record_every=500, log_every=500):
        env = RecordVideo(
            self.env,
            video_folder=video_folder,
            name_prefix="GridFlagTrain",
            episode_trigger=lambda ep: ep % record_every == 0,
        )
        env = RecordEpisodeStatistics(env, buffer_length=num_episodes)

        # Store stats for plotting
        episode_rewards = []
        episode_lengths = []
        epsilons        = []

        for ep in tqdm(range(num_episodes), desc="Training"):
            obs_dict, _ = env.reset()
            obs = obs_to_key(obs_dict)
            terminated = False
            truncated  = False
            episode_reward = 0

            while not terminated and not truncated:
                action = self.get_action(obs)
                next_obs_dict, reward, terminated, truncated, _ = env.step(action)
                next_obs = obs_to_key(next_obs_dict)
                self.update_SARSA(obs, action, reward, terminated or truncated, next_obs)
                obs = next_obs
                
                # Record stats
                episode_reward += reward

            self.decay_epsilon()

            # Record stats
            episode_rewards.append(episode_reward)
            episode_lengths.append(list(env.length_queue)[-1])
            epsilons.append(self.epsilon)

            if (ep + 1) % log_every == 0:
                recent = episode_rewards[-100:]
                avg = np.mean(recent)
                print(f"  Episode {ep + 1:>5} | "
                    f"avg reward (last 100): {avg:.2f} | "
                    f"epsilon: {self.epsilon:.3f}")

        env.close()

        rewards  = np.array(episode_rewards)
        lengths  = np.array(episode_lengths)
        episodes = np.arange(1, len(rewards) + 1)

        fig, axes = plt.subplots(2, 1, figsize=(16, 8))
        fig.suptitle("Training Curves SARSA", fontsize=14, fontweight="bold")

        axes[0].plot(episodes, rewards, color="steelblue", alpha=0.6, marker="o", markersize=3, label="Episode reward")
        axes[0].axhline(len(self.env.unwrapped.flag_cells) * self.env.unwrapped.flag_value, color="gold",
                        linestyle="--", linewidth=1, label="Max reward")
        
        ar_rewards = list()
        for i in range(100, len(rewards)):
            ar_rewards.append(np.mean(rewards[i-100:i]))
        axes[0].plot(episodes[100:], ar_rewards, color="orange", alpha=0.9, marker="o", markersize=3, label="Mean(100) reward")
        
        axes[0].set_ylabel("Reward")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

        axes[1].plot(episodes, lengths, color="coral", alpha=0.8, marker="o", markersize=3, label="Episode steps")
        axes[1].axhline(min(lengths), color="purple", linestyle="--", linewidth=1, label="Min steps")
        axes[1].axhline(lengths[-1], color="green" if min(lengths) == lengths[-1] else "red", linestyle="--", linewidth=1, label="Last steps")

        ar_lengths = list()
        for i in range(100, len(lengths)):
            ar_lengths.append(np.mean(lengths[i-100:i]))
        axes[1].plot(episodes[100:], ar_lengths, color="purple", alpha=0.9, marker="o", markersize=3, label="Mean(100) steps")

        axes[1].set_ylabel("Steps")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("./images/GridFlag_training_curves_SARSA.png", dpi=150)
        print("Plot saved to GridFlag_training_curves_SARSA.png")


    def eval_recorded(
        self,
        video_folder: str = "videos/evaluation",
        name_prefix: str = "eval",
    ):
        """
        Run a single greedy episode and record it.

        Args:
            video_folder: Where to save the .mp4 file.
            name_prefix:  Prefix for the video filename.
        """
        prev_epsilon = self.epsilon
        self.epsilon = 0.0  # pure exploitation

        env = RecordVideo(
            self.env,
            video_folder=video_folder,
            name_prefix=name_prefix,
            episode_trigger=lambda ep: True,
        )
        env = RecordEpisodeStatistics(env)

        obs_dict, _ = env.reset()
        obs = obs_to_key(obs_dict)
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = self.get_action(obs)
            obs_dict, reward, terminated, truncated, info = env.step(action)
            obs = obs_to_key(obs_dict)

        total_reward = list(env.return_queue)[-1]
        total_steps  = list(env.length_queue)[-1]
        print(f"Eval episode: reward = {total_reward:.1f}, steps = {total_steps}")

        env.close()
        self.epsilon = prev_epsilon  # restore epsilon after eval