from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics

from diff_drive_env import DiffDriveEnv
from replay_buffer import ReplayBuffer
from networks import Actor, Critic, OUNoise, init_weights


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_PATH = BASE_DIR / "models" / "ddpg_checkpoint.pt"
DEFAULT_PLOT_PATH = BASE_DIR / "images" / "ddpg_diff_drive_training_curves.png"


def _artifact_path(path):
    path = Path(path)
    return path if path.is_absolute() else BASE_DIR / path


class DiffDriveAgent:
    """
    DDPG (Deep Deterministic Policy Gradient) agent for the DiffDriveEnv.

    DDPG maintains four networks:
        actor         — maps obs → action (the policy we're learning)
        actor_target  — slow-moving copy of actor, used for stable TD targets
        critic        — maps (obs, action) → Q-value
        critic_target — slow-moving copy of critic, used for stable TD targets

    At each step:
        1. Actor selects an action + Ornstein-Uhlenbeck exploration noise
        2. Transition is stored in the replay buffer
        3. A random batch is sampled and used to update critic (minimize Bellman error)
        4. Actor is updated to maximise Q(s, actor(s))
        5. Target networks are soft-updated toward online networks
    """

    def __init__(
        self,
        env: DiffDriveEnv,
        actor_lr:     float = 1e-4,
        critic_lr:    float = 1e-3,
        discount:     float = 0.99,       # gamma
        tau:          float = 0.005,      # soft update rate for target networks
        noise_std:    float = 0.2,        # std of Ornstein-Uhlenbeck exploration noise
        noise_clip:   float = 0.5,        # max absolute value of noise
        batch_size:   int   = 256,
        buffer_size:  int   = 100_000,
        hidden_dim:   int   = 256,
        warmup_steps: int   = 1_000,      # random actions before training starts --> to fill the buffer
        device:       str   = "cpu",
    ):
        self.env        = env
        self.discount   = discount
        self.tau        = tau
        self.noise_clip = noise_clip
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.device     = torch.device(device)

        obs_dim    = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_low  = env.action_space.low
        action_high = env.action_space.high

        self.ou_noise = OUNoise(action_dim, sigma=noise_std)

        # ── Networks ──────────────────────────────────────────────────
        self.actor  = Actor(obs_dim, action_dim, action_low, action_high, hidden_dim).to(self.device)
        self.critic = Critic(obs_dim, action_dim, hidden_dim).to(self.device)

        # Initialise online networks before target networks copy them.
        self.actor.apply(init_weights)
        self.critic.apply(init_weights)

        # Target networks start as exact copies
        self.actor_target  = Actor(obs_dim, action_dim, action_low, action_high, hidden_dim).to(self.device)
        self.critic_target = Critic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict()) #che fa?
        self.critic_target.load_state_dict(self.critic.state_dict())

        # ── Optimisers ────────────────────────────────────────────────
        self.actor_optim  = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # ── Replay buffer ─────────────────────────────────────────────
        self.buffer = ReplayBuffer(obs_dim, action_dim, buffer_size)

        # ── Bookkeeping ───────────────────────────────────────────────
        self.total_steps   = 0
        self.training_info = []   # list of dicts with per-update losses

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    
    def select_action(self, obs: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Choose an action given an observation.

        During training (add_noise=True) OU noise is added to encourage
        exploration. At evaluation time, set add_noise=False for pure exploitation.
        """
        if add_noise and self.total_steps < self.warmup_steps:
            # Warm-up: purely random actions to fill the replay buffer
            return self.env.action_space.sample()

        obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(self.device) #transform the obs in tensor, with unsqueeze change the dimension in (1,obs.shape[0]) since tf use this format and move data to the device with the network
        action = self.actor(obs_t).detach().cpu().numpy()[0] #get answer from network, detach to avoid backprog, cpu to move data in cpu (maybe where moved in other device)and numpy to convert in np.array. [0] to invert unsqueeze 

        if add_noise:
            noise  = np.clip(self.ou_noise.sample(), -self.noise_clip, self.noise_clip)
            action = np.clip(action + noise, self.env.action_space.low, self.env.action_space.high)

        return action.astype(np.float32)

    # ------------------------------------------------------------------
    # Learning update
    # ------------------------------------------------------------------

    def _update(self):
        """Sample a batch from the buffer and update both networks."""
        batch = self.buffer.sample(self.batch_size)

        # Prepare the data of the batch to be used by the tensors
        states      = torch.FloatTensor(batch["states"]).to(self.device)
        actions     = torch.FloatTensor(batch["actions"]).to(self.device)
        rewards     = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones       = torch.FloatTensor(batch["dones"]).to(self.device)

        # -- Compute Bellman target from the Critic Target and Actor Target networks -----------
        with torch.no_grad():
            # Target actor selects the next action
            next_actions = self.actor_target(next_states)
            # Target critic estimates its value
            target_q = self.critic_target(next_states, next_actions)
            # Bellman target: r + γ * Q_target(s', a')  (zero if episode ended)
            target_q = rewards + self.discount * (1.0 - dones) * target_q

        # ── Critic update ─────────────────────────────────────────────
        current_q   = self.critic(states, actions) #predict of current critic
        critic_loss = F.mse_loss(current_q, target_q) #loss of current critic wrt target value

        self.critic_optim.zero_grad() #clean-up the old gradients
        critic_loss.backward() #compute new gradient
        self.critic_optim.step() #compute new weigths

        # ── Actor update ──────────────────────────────────────────────
        # Maximise Q(s, actor(s))  ≡  minimise -Q(s, actor(s))
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ── Soft update of target networks ────────────────────────────
        self._soft_update(self.actor,  self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return float(critic_loss.detach().cpu()), float(actor_loss.detach().cpu())

    def _soft_update(self, online: torch.nn.Module, target: torch.nn.Module):
        """θ_target ← τ * θ_online + (1 - τ) * θ_target"""
        for online_p, target_p in zip(online.parameters(), target.parameters()):
            target_p.data.copy_(self.tau * online_p.data + (1.0 - self.tau) * target_p.data)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_recorded(
        self,
        num_episodes:  int,
        video_folder:  str = "videos/training",
        record_every:  int = 100,
        log_every:     int = 100,
        add_noise:     bool = True,
        name_prefix:   str = "ddpg_diff_drive_training",
        checkpoint_path = DEFAULT_CHECKPOINT_PATH,
        plot_path      = DEFAULT_PLOT_PATH,
    ):
        video_folder = _artifact_path(video_folder)
        video_folder.mkdir(parents=True, exist_ok=True)
        checkpoint_path = _artifact_path(checkpoint_path)
        plot_path = _artifact_path(plot_path)

        env = RecordVideo(
            self.env,
            video_folder=str(video_folder),
            name_prefix=name_prefix,
            episode_trigger=lambda ep: record_every > 0 and ep % record_every == 0,
        )
        env = RecordEpisodeStatistics(env, buffer_length=num_episodes)

        episode_rewards = []
        episode_lengths = []
        critic_losses   = []
        actor_losses    = []

        for ep in tqdm(range(num_episodes), desc="Training"):
            obs, _     = env.reset()
            self.ou_noise.reset()
            done       = False
            ep_reward  = 0
            ep_c_loss  = []
            ep_a_loss  = []

            while not done:
                action = self.select_action(obs, add_noise)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                self.buffer.add(obs, action, reward, next_obs, done)
                obs = next_obs
                ep_reward     += reward
                self.total_steps += 1

                # Start learning only after warm-up and when buffer is ready
                if self.buffer.ready(self.batch_size) and self.total_steps >= self.warmup_steps:
                    c_loss, a_loss = self._update()
                    ep_c_loss.append(c_loss)
                    ep_a_loss.append(a_loss)

            episode_rewards.append(ep_reward)
            episode_lengths.append(list(env.length_queue)[-1])
            if ep_c_loss:
                critic_losses.append(np.mean(ep_c_loss))
                actor_losses.append(np.mean(ep_a_loss))

            if (ep + 1) % log_every == 0:
                avg_r = np.mean(episode_rewards[-log_every:])
                print(f"  Episode {ep+1:>5} | "
                      f"avg reward (last {log_every}): {avg_r:.2f} | "
                      f"buffer: {len(self.buffer):>6} | "
                      f"steps: {self.total_steps}")

        # Save final model
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor" : self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target" : self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict()
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        env.close()
        self._plot(episode_rewards, episode_lengths, critic_losses, actor_losses, plot_path=plot_path)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def eval_recorded(
        self,
        video_folder: str = "videos/evaluation",
        name_prefix:  str = "ddpg_diff_drive_eval_greedy",
        n_episodes:   int = 3,
        add_noise:    bool = False,
    ):
        """Run n_episodes greedy episodes and record them."""
        video_folder = _artifact_path(video_folder)
        video_folder.mkdir(parents=True, exist_ok=True)

        env = RecordVideo(
            self.env,
            video_folder=str(video_folder),
            name_prefix=name_prefix,
            episode_trigger=lambda ep: True,
        )
        env = RecordEpisodeStatistics(env)

        for ep in range(n_episodes):
            obs, _ = env.reset()
            self.ou_noise.reset()
            done   = False

            while not done:
                action = self.select_action(obs, add_noise)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            total_reward = list(env.return_queue)[-1]
            total_steps  = list(env.length_queue)[-1]
            goal = "✓" if info.get("goal_reached") else "✗"
            print(f"  Eval ep {ep+1}: reward = {total_reward:.1f} | "
                  f"steps = {total_steps} | goal reached: {goal}")

        env.close()

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    @staticmethod
    def get_smooth_statistics(values, window: int):
        """Return rolling mean and standard deviation for a 1D sequence."""
        values = np.asarray(values, dtype=np.float32)
        if window <= 0:
            raise ValueError("window must be positive")
        if len(values) < window:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        means = np.array([
            np.mean(values[i - window:i])
            for i in range(window, len(values) + 1)
        ], dtype=np.float32)
        stds = np.array([
            np.std(values[i - window:i])
            for i in range(window, len(values) + 1)
        ], dtype=np.float32)
        return means, stds

    def _plot(self, rewards, lengths, critic_losses, actor_losses, plot_path=DEFAULT_PLOT_PATH):
        plot_path = _artifact_path(plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(16, 8))
        fig.suptitle("DDPG Training Curves", fontsize=14, fontweight="bold")

        episodes = np.arange(1, len(rewards) + 1)

        # Episode reward
        axes[0, 0].plot(episodes, rewards, color="steelblue", alpha=0.5, linewidth=0.8)
        window = 100
        if len(rewards) >= window:
            means, stds = self.get_smooth_statistics(rewards, window)
            x_axis = episodes[window-1:]
            #ma = [np.mean(rewards[i-100:i]) for i in range(100, len(rewards)+1)]
            axes[0, 0].plot(x_axis, means, color="orange", linewidth=1.5, label=f"MA({window})")
            axes[0,0].fill_between(x_axis, means-2*stds, means+2*stds, color="orange", alpha=0.2, label="Confidence (std)")
            axes[0, 0].legend(fontsize=8)
        axes[0, 0].set_title("Episode Reward")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(alpha=0.3)

        # Episode length
        axes[0, 1].plot(episodes, lengths, color="coral", alpha=0.5, linewidth=0.8)
        axes[0, 1].set_title("Episode Length")
        axes[0, 1].set_ylabel("Steps")
        axes[0, 1].grid(alpha=0.3)

        # Critic loss
        c_eps = np.arange(1, len(critic_losses) + 1)
        axes[1, 0].plot(c_eps, critic_losses, color="tomato", alpha=0.7, linewidth=0.8)
        axes[1, 0].set_title("Critic Loss")
        axes[1, 0].set_ylabel("MSE Loss")
        axes[1, 0].grid(alpha=0.3)

        # Actor loss
        axes[1, 1].plot(c_eps, actor_losses, color="mediumseagreen", alpha=0.7, linewidth=0.8)
        axes[1, 1].set_title("Actor Loss")
        axes[1, 1].set_ylabel("-Q value")
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Plot saved to {plot_path}")
