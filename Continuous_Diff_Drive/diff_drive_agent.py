from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics

from diff_drive_env import DiffDriveEnv
from replay_buffer import ReplayBuffer
from networks import Actor, Critic, OUNoise, init_weights


class DiffDriveAgent:
    """
    DDPG (Deep Deterministic Policy Gradient) agent for the DiffDriveEnv.

    DDPG maintains four networks:
        actor         — maps obs → action (the policy we're learning)
        actor_target  — slow-moving copy of actor, used for stable TD targets
        critic        — maps (obs, action) → Q-value
        critic_target — slow-moving copy of critic, used for stable TD targets

    At each step:
        1. Actor selects an action + Gaussian exploration noise
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
        noise_std:    float = 0.2,        # std of Gaussian exploration noise
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
        # self.noise_std  = noise_std
        # self.noise_clip = noise_clip
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

        # Target networks start as exact copies
        self.actor_target  = Actor(obs_dim, action_dim, action_low, action_high, hidden_dim).to(self.device)
        self.critic_target = Critic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict()) #che fa?
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Initialise weights
        self.actor.apply(init_weights)
        self.critic.apply(init_weights)

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

        During training (add_noise=True) Gaussian noise is added to encourage
        exploration. At evaluation time, set add_noise=False for pure exploitation.
        """
        if self.total_steps < self.warmup_steps:
            # Warm-up: purely random actions to fill the replay buffer
            return self.env.action_space.sample()

        obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(self.device) #transform the obs in tensor, with unsqueeze change the dimension in (1,obs.shape[0]) since tf use this format and move data to the device with the network
        action = self.actor(obs_t).detach().cpu().numpy()[0] #get answer from network, detach to avoid backprog, cpu to move data in cpu (maybe where moved in other device)and numpy to convert in np.array. [0] to invert unsqueeze 

        if add_noise: 
            # noise  = np.random.normal(0, self.noise_std, size=action.shape)
            # noise  = np.clip(noise, -self.noise_clip, self.noise_clip)
            noise  = self.ou_noise.sample()
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

        return float(critic_loss), float(actor_loss)

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
    ):
        env = RecordVideo(
            self.env,
            video_folder=video_folder,
            name_prefix="DiffDriveTrain",
            episode_trigger=lambda ep: ep % record_every == 0,
        )
        env = RecordEpisodeStatistics(env, buffer_length=num_episodes)

        episode_rewards = []
        episode_lengths = []
        critic_losses   = []
        actor_losses    = []

        for ep in tqdm(range(num_episodes), desc="Training"):
            obs, _     = env.reset()
            done       = False
            ep_reward  = 0
            ep_c_loss  = []
            ep_a_loss  = []

            while not done:
                action = self.select_action(obs, add_noise)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                self.buffer.add(obs, action, reward, next_obs, terminated)
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
        torch.save({
            "actor" : self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target" : self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict()
        }, "Continuous_Diff_Drive/models/ddpg_checkpoint.pt")

        env.close()
        self._plot(episode_rewards, episode_lengths, critic_losses, actor_losses)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def eval_recorded(
        self,
        video_folder: str = "videos/evaluation",
        name_prefix:  str = "eval",
        n_episodes:   int = 3,
        add_noise:    bool = False,
    ):
        """Run n_episodes greedy episodes and record them."""
        env = RecordVideo(
            self.env,
            video_folder=video_folder,
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

    def get_smooth_statistics(self, data, window=100):
        """Restituisce media e deviazione standard mobile."""
        data = np.array(data)
        if len(data) < window:
            return data, np.zeros_like(data)
    
        means = np.convolve(data, np.ones(window)/window, mode='valid')
        # Calcoliamo la deviazione standard mobile
        stds = np.array([np.std(data[i:i+window]) for i in range(len(data) - window + 1)])
        return means, stds
    
    def _plot(self, rewards, lengths, critic_losses, actor_losses):
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
        plt.savefig("./images/DiffDrive_training_curves.png", dpi=150)
        print("Plot saved to DiffDrive_training_curves.png")
    
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