from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics

from diff_drive_env import DiffDriveEnv
from replay_buffer import ReplayBuffer, TorchReplayBuffer
from networks import Actor, Critic, OUNoise, init_weights, GaussianActor, DoubleCritic


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_PATH = BASE_DIR / "models" / "ddpg_checkpoint.pt"
DEFAULT_PLOT_PATH = BASE_DIR / "images" / "ddpg_diff_drive_training_curves.png"
DEFAULT_PLOT_PATH2 = BASE_DIR / "images" / "ddpg_diff_drive_training_curves2.png"
DEFAULT_SAC_CHECKPOINT_PATH = BASE_DIR / "models" / "sac_checkpoint.pt"
DEFAULT_SAC_PLOT_PATH = BASE_DIR / "images" / "sac_diff_drive_training_curves.png"
DEFAULT_SAC_PLOT_PATH2 = BASE_DIR / "images" / "sac_diff_drive_training_curves2.png"
DEFAULT_TD3_CHECKPOINT_PATH = BASE_DIR / "models" / "td3_checkpoint.pt"
DEFAULT_TD3_PLOT_PATH = BASE_DIR / "images" / "td3_diff_drive_training_curves.png"
DEFAULT_TD3_PLOT_PATH2 = BASE_DIR / "images" / "td3_diff_drive_training_curves2.png"


def _artifact_path(path):
    path = Path(path)
    return path if path.is_absolute() else BASE_DIR / path


class DiffDriveDDPGAgent:
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

        info = {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
        }
        self.training_info.append(info)

        return info["critic_loss"], info["actor_loss"]

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
        plot_path2     = DEFAULT_PLOT_PATH2,
    ):
        video_folder = _artifact_path(video_folder)
        video_folder.mkdir(parents=True, exist_ok=True)
        checkpoint_path = _artifact_path(checkpoint_path)
        plot_path = _artifact_path(plot_path)
        plot_path2 = _artifact_path(plot_path2)

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
        episode_goal_distances = []
        episode_successes = []
        info = {}

        for ep in tqdm(range(num_episodes), desc="Training"):
            obs, _     = env.reset()
            self.ou_noise.reset()
            done       = False
            ep_reward  = 0
            ep_c_loss  = []
            ep_a_loss  = []

            while not done:
                action = self.select_action(obs, add_noise)
                next_obs, reward, terminated, truncated, info = env.step(action)
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
            episode_successes.append(1 if info.get("goal_reached", False) else 0)
            episode_goal_distances.append(float(info.get("dist_to_goal", np.nan)))

            if ep_c_loss:
                critic_losses.append(np.mean(ep_c_loss))
                actor_losses.append(np.mean(ep_a_loss))

            if (ep + 1) % log_every == 0:
                avg_reward = np.mean(episode_rewards[-log_every:])
                avg_success = np.mean(episode_successes[-log_every:])
                avg_goal_dist = np.mean(episode_goal_distances[-log_every:])
                avg_actor_loss = np.mean(actor_losses[-log_every:]) if actor_losses else 0.0
                avg_critic_loss = np.mean(critic_losses[-log_every:]) if critic_losses else 0.0
                print(
                    f"Episode {ep+1:>5} | "
                    f"avg reward (last {log_every}): {avg_reward:.2f} | "
                    f"avg success (last {log_every}): {avg_success:.2f} | "
                    f"avg goal dist (last {log_every}): {avg_goal_dist:.2f} | "
                    f"avg actor loss (last {log_every}): {avg_actor_loss:.4f} | "
                    f"avg critic loss (last {log_every}): {avg_critic_loss:.4f}"
                    f"buffer: {len(self.buffer):>6} | "
                    f"total steps: {self.total_steps}"
                )

        self.save_checkpoint(checkpoint_path)
        env.close()
        self.plot(
            episode_rewards,
            episode_lengths,
            critic_losses,
            actor_losses,
            episode_successes,
            episode_goal_distances,
            plot_path=plot_path,
            plot_path2=plot_path2
        )

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
            info   = {}

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
    
    def _plot(self, rewards, lengths, critic_losses, actor_losses,success_rate, ep_goal_dist, plot_path=DEFAULT_PLOT_PATH, plot_path2=DEFAULT_PLOT_PATH2):
        
        plot_path = _artifact_path(plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        
        plot_path2 = _artifact_path(plot_path2)
        plot_path2.parent.mkdir(parents=True, exist_ok=True)
        
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
        
        fig2, axs2 = plt.subplots(1, 2, figsize=(16,8))
        axs2[0].plot(episodes, success_rate)
        axs2[0].set_title("Success rate")
        axs2[0].plot(episodes, ep_goal_dist)
        axs2[0].set_title("Ep_goal_dist")
        plt.tight_layout()
        fig2.savefig(plot_path2, dpi=150)
        plt.close(fig2)
        print(f"Plot 2 saved to {plot_path2}")
        
    
    def plot_critic_heatmap(self, resolution: int = 50, theta: float = 0.0):
        """
        Genera una heatmap del valore Q calcolato dal Critic per ogni (x, y).
        Assume che le prime due dimensioni dell'observation siano x e y.
        """
        # 1. Definiamo i limiti della griglia in base all'ambiente
        # (Adatta questi valori ai limiti reali del tuo DiffDriveEnv)
        x_range = np.linspace(-5, 5, resolution) 
        y_range = np.linspace(-5, 5, resolution)
        grid_x, grid_y = np.meshgrid(x_range, y_range)
        
        q_values = np.zeros((resolution, resolution))
        
        self.critic.eval()
        self.actor.eval()
        
        with torch.no_grad():
            for i in range(resolution):
                for j in range(resolution):
                    # Costruiamo un'osservazione fittizia
                    # Esempio: [x, y, cos(theta), sin(theta), lidar_1, ..., lidar_n]
                    # NOTA: Qui devi replicare l'esatta struttura del tuo vettore 'obs'
                    obs = np.zeros(self.env.observation_space.shape[0])
                    obs[0] = grid_x[i, j]
                    obs[1] = grid_y[i, j]
                    # Se l'orientamento è nelle obs (es. pos 2 e 3)
                    if len(obs) > 3:
                        obs[2] = np.cos(theta)
                        obs[3] = np.sin(theta)
                    
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    
                    # Chiediamo all'Actor cosa farebbe in quel punto
                    action_t = self.actor(obs_t)
                    # Il Critic valuta l'azione dell'Actor
                    q_val = self.critic(obs_t, action_t)
                    
                    q_values[i, j] = q_val.cpu().item()

        # 2. Plotting
        plt.figure(figsize=(8, 6))
        im = plt.imshow(q_values, extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]], 
                        origin='lower', cmap='viridis')
        plt.colorbar(im, label='Valore Q (Stima del premio futuro)')
        plt.title(f"Critic Heatmap (Orientation: {np.degrees(theta)}°)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
        
        self.critic.train()
        self.actor.train()


class DiffDriveSACAgent:
    """
    SAC (Soft Actor-Critic) agent for the shared DiffDriveEnv.

    SAC uses a stochastic Gaussian actor, twin critics, one target twin critic,
    entropy regularization, and an off-policy replay buffer. Training samples
    stochastic actions; evaluation uses the deterministic mean action.
    """

    def __init__(
        self,
        env: DiffDriveEnv,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        batch_size: int = 256,
        buffer_size: int = 100_000,
        hidden_dim: int = 256,
        warmup_steps: int = 1_000,
        device: str = "cpu",
        automatic_entropy_tuning: bool = True,
    ):
        self.env = env
        self.discount = discount
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.device = torch.device(device)
        self.automatic_entropy_tuning = automatic_entropy_tuning

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_low = env.action_space.low
        action_high = env.action_space.high

        self.actor = GaussianActor(
            obs_dim,
            action_dim,
            action_low,
            action_high,
            hidden_dim,
        ).to(self.device)

        self.critic = DoubleCritic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = DoubleCritic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.tensor(
            [np.log(alpha)],
            dtype=torch.float32,
            requires_grad=automatic_entropy_tuning,
            device=self.device,
        )
        self.alpha_optim = (
            torch.optim.Adam([self.log_alpha], lr=alpha_lr)
            if automatic_entropy_tuning
            else None
        )
        self.alpha = self.log_alpha.exp()

        self.buffer = TorchReplayBuffer(
            obs_dim,
            action_dim,
            buffer_size,
            device=self.device,
        )

        self.total_steps = 0
        self.training_info = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select a stochastic training action or deterministic eval action."""
        if self.total_steps < self.warmup_steps and not evaluate:
            return self.env.action_space.sample().astype(np.float32)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if evaluate:
                action = self.actor.act(obs_t)
            else:
                action, _ = self.actor.sample(obs_t)

        return action.detach().cpu().numpy()[0].astype(np.float32)

    # ------------------------------------------------------------------
    # Learning update
    # ------------------------------------------------------------------

    def _update(self):
        """Sample a batch and update SAC critic, actor, alpha, and target critic."""
        batch = self.buffer.sample(self.batch_size)

        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            min_q_target = torch.min(q1_target, q2_target)
            target_q = rewards + (1.0 - dones) * self.discount * (
                min_q_target - self.alpha.detach() * next_log_probs
            )

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        sampled_actions, log_probs = self.actor.sample(states)
        q1_pi, q2_pi = self.critic(states, sampled_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_probs - min_q_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_probs + self.target_entropy).detach()
            ).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)

        self._soft_update(self.critic, self.critic_target)

        info = {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
        }
        self.training_info.append(info)
        return info["critic_loss"], info["actor_loss"], info["alpha_loss"], info["alpha"]

    def _soft_update(self, online: torch.nn.Module, target: torch.nn.Module):
        """theta_target <- tau * theta_online + (1 - tau) * theta_target"""
        for online_p, target_p in zip(online.parameters(), target.parameters()):
            target_p.data.copy_(self.tau * online_p.data + (1.0 - self.tau) * target_p.data)

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def save_checkpoint(self, checkpoint_path=DEFAULT_SAC_CHECKPOINT_PATH):
        checkpoint_path = _artifact_path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha": float(self.alpha.detach().cpu()),
            "alpha_optim": (
                self.alpha_optim.state_dict()
                if self.alpha_optim is not None
                else None
            ),
            "automatic_entropy_tuning": self.automatic_entropy_tuning,
            "total_steps": self.total_steps,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"SAC checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path=DEFAULT_SAC_CHECKPOINT_PATH, load_optimizers: bool = True):
        checkpoint_path = _artifact_path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])

        if load_optimizers:
            if "actor_optim" in checkpoint:
                self.actor_optim.load_state_dict(checkpoint["actor_optim"])
            if "critic_optim" in checkpoint:
                self.critic_optim.load_state_dict(checkpoint["critic_optim"])
            if self.alpha_optim is not None and checkpoint.get("alpha_optim") is not None:
                self.alpha_optim.load_state_dict(checkpoint["alpha_optim"])

        if "log_alpha" in checkpoint:
            self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
            self.alpha = self.log_alpha.exp()
        elif "alpha" in checkpoint:
            alpha_value = torch.tensor([checkpoint["alpha"]], dtype=torch.float32, device=self.device)
            self.log_alpha.data.copy_(alpha_value.log())
            self.alpha = self.log_alpha.exp()

        self.total_steps = int(checkpoint.get("total_steps", self.total_steps))
        print(f"SAC checkpoint loaded from {checkpoint_path}")
        return checkpoint

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_recorded(
        self,
        num_episodes: int,
        video_folder: str = "videos/training_sac",
        record_every: int = 100,
        log_every: int = 100,
        name_prefix: str = "sac_diff_drive_training",
        checkpoint_path=DEFAULT_SAC_CHECKPOINT_PATH,
        plot_path=DEFAULT_SAC_PLOT_PATH,
        plot_path2=DEFAULT_SAC_PLOT_PATH2,
    ):
        video_folder = _artifact_path(video_folder)
        video_folder.mkdir(parents=True, exist_ok=True)
        checkpoint_path = _artifact_path(checkpoint_path)
        plot_path = _artifact_path(plot_path)
        plot_path2 = _artifact_path(plot_path2)

        env = RecordVideo(
            self.env,
            video_folder=str(video_folder),
            name_prefix=name_prefix,
            episode_trigger=lambda ep: record_every > 0 and ep % record_every == 0,
        )
        env = RecordEpisodeStatistics(env, buffer_length=num_episodes)

        episode_rewards = []
        episode_lengths = []
        critic_losses = []
        actor_losses = []
        alpha_losses = []
        alpha_values = []
        episode_successes = []
        episode_goal_distances = []

        for ep in tqdm(range(num_episodes), desc="SAC Training"):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            ep_c_loss = []
            ep_a_loss = []
            ep_alpha_loss = []
            ep_alpha_values = []
            info = {}

            while not done:
                action = self.select_action(obs, evaluate=False)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                self.buffer.add(obs, action, reward, next_obs, done)
                obs = next_obs
                ep_reward += reward
                self.total_steps += 1

                if self.buffer.ready(self.batch_size) and self.total_steps >= self.warmup_steps:
                    critic_loss, actor_loss, alpha_loss, alpha_value = self._update()
                    ep_c_loss.append(critic_loss)
                    ep_a_loss.append(actor_loss)
                    ep_alpha_loss.append(alpha_loss)
                    ep_alpha_values.append(alpha_value)

            episode_rewards.append(ep_reward)
            episode_lengths.append(list(env.length_queue)[-1])
            episode_successes.append(1 if info.get("goal_reached", False) else 0)
            episode_goal_distances.append(float(info.get("dist_to_goal", np.nan)))

            if ep_c_loss:
                critic_losses.append(float(np.mean(ep_c_loss)))
            if ep_a_loss:
                actor_losses.append(float(np.mean(ep_a_loss)))
            if ep_alpha_loss:
                alpha_losses.append(float(np.mean(ep_alpha_loss)))
            if ep_alpha_values:
                alpha_values.append(float(np.mean(ep_alpha_values)))

            if (ep + 1) % log_every == 0:
                avg_reward = np.mean(episode_rewards[-log_every:])
                avg_success = np.mean(episode_successes[-log_every:])
                avg_goal_dist = np.nanmean(episode_goal_distances[-log_every:])
                avg_actor_loss = np.mean(actor_losses[-log_every:]) if actor_losses else 0.0
                avg_critic_loss = np.mean(critic_losses[-log_every:]) if critic_losses else 0.0
                avg_alpha_loss = np.mean(alpha_losses[-log_every:]) if alpha_losses else 0.0
                print(
                    f"Episode {ep+1:>5} | "
                    f"avg reward: {avg_reward:.2f} | "
                    f"success: {avg_success:.2f} | "
                    f"avg dist: {avg_goal_dist:.2f} | "
                    f"actor loss: {avg_actor_loss:.4f} | "
                    f"critic loss: {avg_critic_loss:.4f} | "
                    f"alpha loss: {avg_alpha_loss:.4f} | "
                    f"buffer: {len(self.buffer):>6} | "
                    f"steps: {self.total_steps}"
                )

        self.save_checkpoint(checkpoint_path)
        env.close()
        self._plot(
            episode_rewards,
            episode_lengths,
            critic_losses,
            actor_losses,
            alpha_losses,
            alpha_values,
            episode_successes,
            episode_goal_distances,
            plot_path=plot_path,
            plot_path2=plot_path2,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def eval_recorded(
        self,
        video_folder: str = "videos/evaluation_sac",
        name_prefix: str = "sac_diff_drive_eval",
        n_episodes: int = 3,
    ):
        """Run deterministic SAC episodes and record videos."""
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
            done = False
            info = {}

            while not done:
                action = self.select_action(obs, evaluate=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            total_reward = list(env.return_queue)[-1]
            total_steps = list(env.length_queue)[-1]
            goal = "yes" if info.get("goal_reached") else "no"
            print(
                f"  SAC eval ep {ep+1}: reward = {total_reward:.1f} | "
                f"steps = {total_steps} | goal reached: {goal}"
            )

        env.close()

    def evaluate(self, n_episodes: int = 5):
        """Run deterministic SAC evaluation without video recording."""
        for ep in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action = self.select_action(obs, evaluate=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

            print(f"SAC eval episode {ep+1}: reward = {total_reward:.2f}")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot(
        self,
        rewards,
        lengths,
        critic_losses,
        actor_losses,
        alpha_losses,
        alpha_values,
        successes,
        goal_distances,
        plot_path=DEFAULT_SAC_PLOT_PATH,
        plot_path2=DEFAULT_SAC_PLOT_PATH2,
    ):
        plot_path = _artifact_path(plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_path2 = _artifact_path(plot_path2)
        plot_path2.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(16, 8))
        fig.suptitle("SAC Training Curves", fontsize=14, fontweight="bold")

        episodes = np.arange(1, len(rewards) + 1)
        axes[0, 0].plot(episodes, rewards, color="steelblue", alpha=0.7, linewidth=0.9)
        axes[0, 0].set_title("Episode Reward")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(episodes, lengths, color="coral", alpha=0.7, linewidth=0.9)
        axes[0, 1].set_title("Episode Length")
        axes[0, 1].set_ylabel("Steps")
        axes[0, 1].grid(alpha=0.3)

        loss_x = np.arange(1, len(critic_losses) + 1)
        axes[1, 0].plot(loss_x, critic_losses, color="tomato", alpha=0.8, linewidth=0.9)
        axes[1, 0].set_title("Critic Loss")
        axes[1, 0].set_ylabel("MSE Loss")
        axes[1, 0].grid(alpha=0.3)

        actor_x = np.arange(1, len(actor_losses) + 1)
        axes[1, 1].plot(actor_x, actor_losses, color="mediumseagreen", alpha=0.8, linewidth=0.9, label="Actor")
        alpha_x = np.arange(1, len(alpha_losses) + 1)
        if len(alpha_losses) > 0:
            axes[1, 1].plot(alpha_x, alpha_losses, color="mediumpurple", alpha=0.8, linewidth=0.9, label="Alpha")
            axes[1, 1].legend(fontsize=8)
        axes[1, 1].set_title("Actor / Alpha Loss")
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"SAC plot saved to {plot_path}")

        fig2, axs2 = plt.subplots(1, 3, figsize=(18, 5))
        axs2[0].plot(episodes, successes, color="mediumseagreen", linewidth=0.9)
        axs2[0].set_title("Success")
        axs2[0].set_ylim(-0.05, 1.05)
        axs2[0].grid(alpha=0.3)

        axs2[1].plot(episodes, goal_distances, color="steelblue", linewidth=0.9)
        axs2[1].set_title("Final Distance to Goal")
        axs2[1].grid(alpha=0.3)

        alpha_value_x = np.arange(1, len(alpha_values) + 1)
        axs2[2].plot(alpha_value_x, alpha_values, color="mediumpurple", linewidth=0.9)
        axs2[2].set_title("Entropy Coefficient Alpha")
        axs2[2].grid(alpha=0.3)

        plt.tight_layout()
        fig2.savefig(plot_path2, dpi=150)
        plt.close(fig2)
        print(f"SAC plot 2 saved to {plot_path2}")


class DiffDriveTD3Agent:
    """
    TD3 improves on DDPG with three tricks:
      1. Twin critics            -> reduce Q-value overestimation by taking
                                    min(Q1, Q2) when computing the TD target.
      2. Delayed actor updates   -> update the actor (and its target) every
                                    `policy_delay` critic updates, not every step.
      3. Target policy smoothing -> add small clipped Gaussian noise to the
                                    target actor's action when computing the
                                    target Q-value, so the critic does not
                                    over-fit to narrow Q-value spikes.

    Networks:
        actor         -> deterministic policy (reuses Actor from networks.py)
        actor_target  -> slow copy of actor
        critic        -> twin Q-networks (reuses DoubleCritic from networks.py)
        critic_target -> slow copy of critic
    """

    def __init__(
        self,
        env: DiffDriveEnv,
        actor_lr:          float = 1e-4,
        critic_lr:         float = 1e-3,
        discount:          float = 0.99,   # gamma
        tau:               float = 0.005,  # soft-update rate for target networks
        policy_delay:      int   = 2,      # actor updated every N critic updates
        target_noise_std:  float = 0.2,    # Trick 3: stddev of target smoothing noise
        target_noise_clip: float = 0.5,    # Trick 3: clip noise to +/- this value
        expl_noise_std:    float = 0.1,    # stddev of Gaussian exploration noise
        batch_size:        int   = 256,
        buffer_size:       int   = 100_000,
        hidden_dim:        int   = 256,
        warmup_steps:      int   = 1_000,  # random actions before training starts
        device:            str   = "cpu",
    ):
        self.env               = env
        self.discount          = discount
        self.tau               = tau
        self.policy_delay      = policy_delay
        self.target_noise_std  = target_noise_std
        self.target_noise_clip = target_noise_clip
        self.expl_noise_std    = expl_noise_std
        self.batch_size        = batch_size
        self.warmup_steps      = warmup_steps
        self.device            = torch.device(device)

        obs_dim     = env.observation_space.shape[0]
        action_dim  = env.action_space.shape[0]
        action_low  = env.action_space.low
        action_high = env.action_space.high

        # Cache action bounds as tensors on the right device (used for clipping in _update)
        self.action_low_t  = torch.as_tensor(action_low,  dtype=torch.float32, device=self.device)
        self.action_high_t = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)

        # ── Networks ──────────────────────────────────────────────────
        # Deterministic actor (same architecture as DDPG)
        self.actor        = Actor(obs_dim, action_dim, action_low, action_high, hidden_dim).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim, action_low, action_high, hidden_dim).to(self.device)
        self.actor.apply(init_weights)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Twin critics (same DoubleCritic class SAC uses)
        self.critic        = DoubleCritic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = DoubleCritic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # ── Optimisers ────────────────────────────────────────────────
        self.actor_optim  = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # ── Replay buffer (NumPy, same one DDPG uses) ─────────────────
        self.buffer = ReplayBuffer(obs_dim, action_dim, buffer_size)

        # ── Bookkeeping ───────────────────────────────────────────────
        self.total_steps   = 0   # env steps taken so far
        self.total_updates = 0   # gradient updates done so far (drives policy_delay)
        self.training_info = []  # per-update losses

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Choose an action given an observation.

        During training (add_noise=True) Gaussian noise is added to the actor's
        output for exploration. TD3 uses plain Gaussian noise (not OU like DDPG)
        — simpler, and equally effective in the original paper.

        At evaluation time set add_noise=False for pure exploitation.
        """
        # Warm-up phase: take random actions to fill the buffer with diverse data
        if add_noise and self.total_steps < self.warmup_steps:
            return self.env.action_space.sample().astype(np.float32)

        # Forward pass through the deterministic actor
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy()[0]

        if add_noise:
            # Gaussian exploration noise, then clip to action bounds
            noise  = np.random.normal(0.0, self.expl_noise_std, size=action.shape)
            action = np.clip(
                action + noise,
                self.env.action_space.low,
                self.env.action_space.high,
            )

        return action.astype(np.float32)

    # ------------------------------------------------------------------
    # Learning update
    # ------------------------------------------------------------------

    def _update(self):
        """
        Sample a batch and apply the three TD3 tricks:
          1. Twin critics with clipped double-Q     (min of Q1, Q2 in target)
          2. Delayed actor updates                  (actor only every policy_delay steps)
          3. Target policy smoothing                (clipped Gaussian noise on target action)
        """
        batch = self.buffer.sample(self.batch_size)

        states      = torch.FloatTensor(batch["states"]).to(self.device)
        actions     = torch.FloatTensor(batch["actions"]).to(self.device)
        rewards     = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones       = torch.FloatTensor(batch["dones"]).to(self.device)

        # ── Compute TD target (no gradients) ──────────────────────────
        with torch.no_grad():
            # TRICK 3: target policy smoothing
            #   a' = clip(actor_target(s') + clip(noise, -c, c),  a_low,  a_high)
            noise = torch.randn_like(actions) * self.target_noise_std
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)

            next_actions = self.actor_target(next_states) + noise
            next_actions = torch.max(torch.min(next_actions, self.action_high_t), self.action_low_t)

            # TRICK 1: twin critics, take the MIN of the two Q estimates
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            min_q_target         = torch.min(q1_target, q2_target)

            # Bellman target: r + γ * min_Q(s', a')  (zero if episode ended)
            target_q = rewards + self.discount * (1.0 - dones) * min_q_target

        # ── Critic update (both Q1 and Q2 trained jointly) ────────────
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.total_updates += 1
        actor_loss_value = None

        # ── TRICK 2: Delayed actor + target updates ───────────────────
        if self.total_updates % self.policy_delay == 0:
            # Maximise Q1(s, actor(s))  ≡  minimise -Q1(s, actor(s))
            # We use only Q1 for the actor loss (standard TD3 practice).
            actor_loss = -self.critic.q1(states, self.actor(states)).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # Soft-update BOTH target networks only when the actor is updated
            self._soft_update(self.actor,  self.actor_target)
            self._soft_update(self.critic, self.critic_target)

            actor_loss_value = float(actor_loss.detach().cpu())

        critic_loss_value = float(critic_loss.detach().cpu())

        self.training_info.append({
            "critic_loss": critic_loss_value,
            "actor_loss":  actor_loss_value,   # may be None on non-delayed steps
        })
        return critic_loss_value, actor_loss_value

    def _soft_update(self, online: torch.nn.Module, target: torch.nn.Module):
        """θ_target ← τ * θ_online + (1 - τ) * θ_target"""
        for online_p, target_p in zip(online.parameters(), target.parameters()):
            target_p.data.copy_(self.tau * online_p.data + (1.0 - self.tau) * target_p.data)

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def save_checkpoint(self, checkpoint_path=DEFAULT_TD3_CHECKPOINT_PATH):
        """Persist all networks, optimisers, and training counters to disk."""
        checkpoint_path = _artifact_path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "actor":         self.actor.state_dict(),
            "actor_target":  self.actor_target.state_dict(),
            "critic":        self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optim":   self.actor_optim.state_dict(),
            "critic_optim":  self.critic_optim.state_dict(),
            "total_steps":   self.total_steps,
            "total_updates": self.total_updates,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"TD3 checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path=DEFAULT_TD3_CHECKPOINT_PATH, load_optimizers: bool = True):
        """Restore networks, optimisers, and training counters from disk."""
        checkpoint_path = _artifact_path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])

        if load_optimizers:
            if "actor_optim" in checkpoint:
                self.actor_optim.load_state_dict(checkpoint["actor_optim"])
            if "critic_optim" in checkpoint:
                self.critic_optim.load_state_dict(checkpoint["critic_optim"])

        self.total_steps   = int(checkpoint.get("total_steps",   self.total_steps))
        self.total_updates = int(checkpoint.get("total_updates", self.total_updates))

        print(f"TD3 checkpoint loaded from {checkpoint_path}")
        return checkpoint

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_recorded(
        self,
        num_episodes:    int,
        video_folder:    str  = "videos/training_td3",
        record_every:    int  = 100,
        log_every:       int  = 100,
        name_prefix:     str  = "td3_diff_drive_training",
        checkpoint_path       = DEFAULT_TD3_CHECKPOINT_PATH,
        plot_path             = DEFAULT_TD3_PLOT_PATH,
        plot_path2            = DEFAULT_TD3_PLOT_PATH2,
    ):
        """
        Train TD3 for `num_episodes`, periodically recording videos and
        printing progress. Saves a final checkpoint and two summary plots.
        """
        video_folder    = _artifact_path(video_folder)
        video_folder.mkdir(parents=True, exist_ok=True)
        checkpoint_path = _artifact_path(checkpoint_path)
        plot_path       = _artifact_path(plot_path)
        plot_path2      = _artifact_path(plot_path2)

        env = RecordVideo(
            self.env,
            video_folder=str(video_folder),
            name_prefix=name_prefix,
            episode_trigger=lambda ep: record_every > 0 and ep % record_every == 0,
        )
        env = RecordEpisodeStatistics(env, buffer_length=num_episodes)

        episode_rewards        = []
        episode_lengths        = []
        critic_losses          = []
        actor_losses           = []
        episode_successes      = []
        episode_goal_distances = []

        for ep in tqdm(range(num_episodes), desc="TD3 Training"):
            obs, _    = env.reset()
            done      = False
            ep_reward = 0.0
            ep_c_loss = []
            ep_a_loss = []
            info      = {}

            while not done:
                action = self.select_action(obs, add_noise=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                self.buffer.add(obs, action, reward, next_obs, done)
                obs              = next_obs
                ep_reward       += reward
                self.total_steps += 1

                # Learn once buffer has enough data and warmup is over
                if self.buffer.ready(self.batch_size) and self.total_steps >= self.warmup_steps:
                    c_loss, a_loss = self._update()
                    ep_c_loss.append(c_loss)
                    if a_loss is not None:   # actor only updates every policy_delay steps
                        ep_a_loss.append(a_loss)

            episode_rewards.append(ep_reward)
            episode_lengths.append(list(env.length_queue)[-1])
            episode_successes.append(1 if info.get("goal_reached", False) else 0)
            episode_goal_distances.append(float(info.get("dist_to_goal", np.nan)))

            if ep_c_loss:
                critic_losses.append(float(np.mean(ep_c_loss)))
            if ep_a_loss:
                actor_losses.append(float(np.mean(ep_a_loss)))

            if (ep + 1) % log_every == 0:
                avg_r      = np.mean(episode_rewards[-log_every:])
                avg_succ   = np.mean(episode_successes[-log_every:])
                avg_dist   = np.nanmean(episode_goal_distances[-log_every:])
                avg_c_loss = np.mean(critic_losses[-log_every:]) if critic_losses else 0.0
                avg_a_loss = np.mean(actor_losses[-log_every:])  if actor_losses  else 0.0
                print(
                    f"Episode {ep+1:>5} | "
                    f"avg reward: {avg_r:.2f} | "
                    f"success: {avg_succ:.2f} | "
                    f"avg dist: {avg_dist:.2f} | "
                    f"actor loss: {avg_a_loss:.4f} | "
                    f"critic loss: {avg_c_loss:.4f} | "
                    f"buffer: {len(self.buffer):>6} | "
                    f"steps: {self.total_steps}"
                )

        self.save_checkpoint(checkpoint_path)
        env.close()
        self._plot(
            episode_rewards,
            episode_lengths,
            critic_losses,
            actor_losses,
            episode_successes,
            episode_goal_distances,
            plot_path=plot_path,
            plot_path2=plot_path2,
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    @staticmethod
    def get_smooth_statistics(values, window: int):
        """Rolling mean and std for a 1D sequence."""
        values = np.asarray(values, dtype=np.float32)
        if window <= 0:
            raise ValueError("window must be positive")
        if len(values) < window:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        means = np.array([np.mean(values[i - window:i]) for i in range(window, len(values) + 1)], dtype=np.float32)
        stds  = np.array([np.std(values[i - window:i])  for i in range(window, len(values) + 1)], dtype=np.float32)
        return means, stds

    def _plot(
        self,
        rewards,
        lengths,
        critic_losses,
        actor_losses,
        successes,
        goal_distances,
        plot_path  = DEFAULT_TD3_PLOT_PATH,
        plot_path2 = DEFAULT_TD3_PLOT_PATH2,
    ):
        plot_path  = _artifact_path(plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_path2 = _artifact_path(plot_path2)
        plot_path2.parent.mkdir(parents=True, exist_ok=True)

        # ── Figure 1: rewards / lengths / losses ──────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(16, 8))
        fig.suptitle("TD3 Training Curves", fontsize=14, fontweight="bold")

        episodes = np.arange(1, len(rewards) + 1)

        axes[0, 0].plot(episodes, rewards, color="steelblue", alpha=0.5, linewidth=0.8)
        window = 100
        if len(rewards) >= window:
            means, stds = self.get_smooth_statistics(rewards, window)
            x_axis = episodes[window - 1:]
            axes[0, 0].plot(x_axis, means, color="orange", linewidth=1.5, label=f"MA({window})")
            axes[0, 0].fill_between(x_axis, means - 2 * stds, means + 2 * stds,
                                    color="orange", alpha=0.2, label="Confidence (2 std)")
            axes[0, 0].legend(fontsize=8)
        axes[0, 0].set_title("Episode Reward")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(episodes, lengths, color="coral", alpha=0.7, linewidth=0.9)
        axes[0, 1].set_title("Episode Length")
        axes[0, 1].set_ylabel("Steps")
        axes[0, 1].grid(alpha=0.3)

        c_eps = np.arange(1, len(critic_losses) + 1)
        axes[1, 0].plot(c_eps, critic_losses, color="tomato", alpha=0.8, linewidth=0.9)
        axes[1, 0].set_title("Critic Loss")
        axes[1, 0].set_ylabel("MSE Loss")
        axes[1, 0].grid(alpha=0.3)

        a_eps = np.arange(1, len(actor_losses) + 1)
        axes[1, 1].plot(a_eps, actor_losses, color="mediumseagreen", alpha=0.8, linewidth=0.9)
        axes[1, 1].set_title("Actor Loss")
        axes[1, 1].set_ylabel("-Q value")
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"TD3 plot saved to {plot_path}")

        # ── Figure 2: success rate and final distance to goal ─────────
        fig2, axs2 = plt.subplots(1, 2, figsize=(16, 5))
        axs2[0].plot(episodes, successes, color="mediumseagreen", linewidth=0.9)
        axs2[0].set_title("Success (1 = goal reached)")
        axs2[0].set_ylim(-0.05, 1.05)
        axs2[0].grid(alpha=0.3)

        axs2[1].plot(episodes, goal_distances, color="steelblue", linewidth=0.9)
        axs2[1].set_title("Final Distance to Goal")
        axs2[1].grid(alpha=0.3)

        plt.tight_layout()
        fig2.savefig(plot_path2, dpi=150)
        plt.close(fig2)
        print(f"TD3 plot 2 saved to {plot_path2}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def eval_recorded(
        self,
        video_folder: str = "videos/evaluation_td3",
        name_prefix:  str = "td3_diff_drive_eval",
        n_episodes:   int = 3,
    ):
        """Run n_episodes deterministic (greedy) episodes and record videos."""
        video_folder = _artifact_path(video_folder)
        video_folder.mkdir(parents=True, exist_ok=True)

        env = RecordVideo(
            self.env,
            video_folder=str(video_folder),
            name_prefix=name_prefix,
            episode_trigger=lambda ep: True,   # record every eval episode
        )
        env = RecordEpisodeStatistics(env)

        for ep in range(n_episodes):
            obs, _ = env.reset()
            done   = False
            info   = {}

            while not done:
                action = self.select_action(obs, add_noise=False)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            total_reward = list(env.return_queue)[-1]
            total_steps  = list(env.length_queue)[-1]
            goal = "yes" if info.get("goal_reached") else "no"
            print(
                f"  TD3 eval ep {ep+1}: reward = {total_reward:.1f} | "
                f"steps = {total_steps} | goal reached: {goal}"
            )

        env.close()

    def evaluate(self, n_episodes: int = 5):
        """Run deterministic episodes without recording (quick numeric check)."""
        for ep in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action = self.select_action(obs, add_noise=False)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

            print(f"TD3 eval episode {ep+1}: reward = {total_reward:.2f}")
