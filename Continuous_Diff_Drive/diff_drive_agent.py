"""DDPG, TD3 and SAC with normalized inputs and shared experiment workflows."""
import numpy as np
import torch
import torch.nn.functional as F
from Continuous_Diff_Drive.diff_drive_env import DiffDriveEnv
from Continuous_Diff_Drive.replay_buffer import ReplayBuffer, TorchReplayBuffer
from Continuous_Diff_Drive.networks import Actor, Critic, OUNoise, init_weights, GaussianActor, DoubleCritic
from rl_project.continuous import ContinuousWorkflow
from rl_project.runtime import frozen, seed_everything


def _normalize_tensor(obs, env, device):
    values = torch.as_tensor(obs, dtype=torch.float32, device=device)
    scale = values.new_tensor([env.lidar_max_range] * env.n_lidar_rays +
                              [float(np.hypot(env.room_w, env.room_h)), float(np.pi)])
    return values / scale


def _clip_grad_norm(module, max_norm):
    if max_norm is not None and max_norm > 0:
        torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm)


class DiffDriveDDPGAgent(ContinuousWorkflow):
    algorithm = "ddpg"

    def __init__(
        self,
        env: DiffDriveEnv,
        actor_lr:     float = 1e-4,
        critic_lr:    float = 1e-3,
        discount:     float = 0.99,
        tau:          float = 0.005, # Soft update rate for target networks
        noise_std:    float = 0.2,
        noise_clip:   float = 0.5,
        batch_size:   int   = 256,
        buffer_size:  int   = 300_000,
        hidden_dim:   int   = 256,
        warmup_steps: int   = 5_000,
        updates_per_step: int = 1,
        max_grad_norm: float = 1.0,
        device:       str   = "cpu",
        seed: int = 0,
    ):
        self.agent_config = {k: v for k, v in locals().copy().items() if k not in ("self", "env", "__class__")}
        seed_everything(seed, env)
        self.total_updates = 0
        self.env        = env
        self.discount   = discount
        self.tau        = tau
        self.noise_clip = noise_clip
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.updates_per_step = updates_per_step
        self.max_grad_norm = max_grad_norm
        self.device     = torch.device(device)

        obs_dim    = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_low  = env.action_space.low
        action_high = env.action_space.high

        self.ou_noise = OUNoise(action_dim, sigma=noise_std)

        self.actor  = Actor(obs_dim, action_dim, action_low, action_high, hidden_dim).to(self.device)
        self.critic = Critic(obs_dim, action_dim, hidden_dim).to(self.device)

        self.actor.apply(init_weights)
        self.critic.apply(init_weights)

        self.actor_target  = Actor(obs_dim, action_dim, action_low, action_high, hidden_dim).to(self.device)
        self.critic_target = Critic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim  = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.buffer = ReplayBuffer(obs_dim, action_dim, buffer_size)

        self.total_steps   = 0

    @torch.inference_mode()
    def select_action(self, obs: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Choose an action given an observation.

        During training (add_noise=True) OU noise is added to encourage
        exploration. At evaluation time, set add_noise=False for pure exploitation.
        """
        if add_noise and self.total_steps < self.warmup_steps:
            return self.env.action_space.sample().astype(np.float32)

        obs_t = _normalize_tensor(obs, self.env, self.device).unsqueeze(0)
        action = self.actor(obs_t).detach().cpu().numpy()[0]

        if add_noise:
            noise  = np.clip(self.ou_noise.sample(), -self.noise_clip, self.noise_clip)
            action = np.clip(action + noise, self.env.action_space.low, self.env.action_space.high)

        return action.astype(np.float32)

    def _update(self):
        """Sample a batch from the buffer and update both networks."""
        batch = self.buffer.sample(self.batch_size)

        states      = _normalize_tensor(batch["states"], self.env, self.device)
        actions     = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards     = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_states = _normalize_tensor(batch["next_states"], self.env, self.device)
        dones       = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + self.discount * (1.0 - dones) * target_q

        current_q   = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        _clip_grad_norm(self.critic, self.max_grad_norm)
        self.critic_optim.step()

        with frozen(self.critic):
            actor_loss = -self.critic(states, self.actor(states)).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            _clip_grad_norm(self.actor, self.max_grad_norm)
            self.actor_optim.step()

        # ── Soft update of target networks ────────────────────────────
        self._soft_update(self.actor,  self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        info = {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
        }

        self.total_updates += 1
        return info["critic_loss"], info["actor_loss"]

    def _soft_update(self, online: torch.nn.Module, target: torch.nn.Module):
        """theta_target <- tau * theta_online + (1 - tau) * theta_target"""
        for online_p, target_p in zip(online.parameters(), target.parameters()):
            target_p.data.copy_(self.tau * online_p.data + (1.0 - self.tau) * target_p.data)


class DiffDriveSACAgent(ContinuousWorkflow):
    algorithm = "sac"

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
        buffer_size: int = 300_000,
        hidden_dim: int = 256,
        warmup_steps: int = 5_000,
        updates_per_step: int = 2,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
        automatic_entropy_tuning: bool = True,
        seed: int = 0,
    ):
        self.agent_config = {k: v for k, v in locals().copy().items() if k not in ("self", "env", "__class__")}
        seed_everything(seed, env)
        self.total_updates = 0
        self.env = env
        self.discount = discount
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.updates_per_step = updates_per_step
        self.max_grad_norm = max_grad_norm
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

    @torch.inference_mode()
    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select a stochastic training action or deterministic eval action."""
        if self.total_steps < self.warmup_steps and not evaluate:
            return self.env.action_space.sample().astype(np.float32)

        obs_t = _normalize_tensor(obs, self.env, self.device).unsqueeze(0)
        with torch.no_grad():
            if evaluate:
                action = self.actor.act(obs_t)
            else:
                action, _ = self.actor.sample(obs_t)

        return action.detach().cpu().numpy()[0].astype(np.float32)

    def _update(self):
        """Sample a batch and update SAC critic, actor, alpha, and target critic."""
        batch = self.buffer.sample(self.batch_size)

        states = _normalize_tensor(batch["states"], self.env, self.device)
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = _normalize_tensor(batch["next_states"], self.env, self.device)
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
        _clip_grad_norm(self.critic, self.max_grad_norm)
        self.critic_optim.step()

        with frozen(self.critic):
            sampled_actions, log_probs = self.actor.sample(states)
            q1_pi, q2_pi = self.critic(states, sampled_actions)
            min_q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.alpha.detach() * log_probs - min_q_pi).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            _clip_grad_norm(self.actor, self.max_grad_norm)
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
        self.total_updates += 1
        return info["critic_loss"], info["actor_loss"], info["alpha_loss"], info["alpha"]

    def _soft_update(self, online: torch.nn.Module, target: torch.nn.Module):
        """theta_target <- tau * theta_online + (1 - tau) * theta_target"""
        for online_p, target_p in zip(online.parameters(), target.parameters()):
            target_p.data.copy_(self.tau * online_p.data + (1.0 - self.tau) * target_p.data)


class DiffDriveTD3Agent(ContinuousWorkflow):
    algorithm = "td3"

    def __init__(
        self,
        env: DiffDriveEnv,
        actor_lr:          float = 1e-4,
        critic_lr:         float = 1e-3,
        discount:          float = 0.99,
        tau:               float = 0.005,
        policy_delay:      int   = 2,
        target_noise_std:  float = 0.2,
        target_noise_clip: float = 0.5,
        expl_noise_std:    float = 0.2,
        expl_noise_clip:   float = 0.5,
        batch_size:        int   = 256,
        buffer_size:       int   = 300_000,
        hidden_dim:        int   = 256,
        warmup_steps:      int   = 5_000,
        updates_per_step:  int   = 1,
        max_grad_norm:     float = 1.0,
        device:            str   = "cpu",
        seed: int = 0,
    ):
        self.agent_config = {k: v for k, v in locals().copy().items() if k not in ("self", "env", "__class__")}
        seed_everything(seed, env)
        self.total_updates = 0
        self.env               = env
        self.discount          = discount
        self.tau               = tau
        self.policy_delay      = policy_delay
        self.target_noise_std  = target_noise_std
        self.target_noise_clip = target_noise_clip
        self.expl_noise_std    = expl_noise_std
        self.expl_noise_clip   = expl_noise_clip
        self.batch_size        = batch_size
        self.warmup_steps      = warmup_steps
        self.updates_per_step  = updates_per_step
        self.max_grad_norm     = max_grad_norm
        self.device            = torch.device(device)

        obs_dim     = env.observation_space.shape[0]
        action_dim  = env.action_space.shape[0]
        action_low  = env.action_space.low
        action_high = env.action_space.high

        self.action_low_t  = torch.as_tensor(action_low,  dtype=torch.float32, device=self.device)
        self.action_high_t = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)

        self.actor        = Actor(obs_dim, action_dim, action_low, action_high, hidden_dim).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim, action_low, action_high, hidden_dim).to(self.device)
        self.actor.apply(init_weights)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic        = DoubleCritic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = DoubleCritic(obs_dim, action_dim, hidden_dim).to(self.device)
        # DoubleCritic defaults to xavier init (SAC's choice). Override with the
        # same small orthogonal init DDPG's critic uses — it keeps initial
        # Q-values near zero, which bootstraps far more stably. (TD3-only: SAC
        # builds its own DoubleCritic instance and is untouched.)
        self.critic.apply(init_weights)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim  = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Temporally correlated OU behavior noise; target smoothing remains Gaussian.
        self.ou_noise = OUNoise(action_dim, sigma=self.expl_noise_std)

        self.buffer = ReplayBuffer(obs_dim, action_dim, buffer_size)

        self.total_steps   = 0
        self.total_updates = 0

    @torch.inference_mode()
    def select_action(self, obs: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Choose an action given an observation.

        During training (add_noise=True) temporally-correlated OU noise is added
        to the actor's output for exploration, in raw action units and clipped to
        +/- expl_noise_clip — identical to the DDPG agent's exploration scheme.

        At evaluation time set add_noise=False for pure exploitation.
        """
        # Warm-up phase: take random actions to fill the buffer with diverse data
        if add_noise and self.total_steps < self.warmup_steps:
            return self.env.action_space.sample().astype(np.float32)

        # Forward pass through the deterministic actor
        obs_t = _normalize_tensor(obs, self.env, self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy()[0]

        if add_noise:
            # OU exploration noise (temporally correlated), added in raw action
            # units and clipped to +/- expl_noise_clip — matches DDPG exactly
            # (no per-dimension half-range scaling). Final action clipped to bounds.
            noise  = np.clip(self.ou_noise.sample(), -self.expl_noise_clip, self.expl_noise_clip)
            action = np.clip(
                action + noise,
                self.env.action_space.low,
                self.env.action_space.high,
            )

        return action.astype(np.float32)

    def _update(self):
        """
        Sample a batch and apply the three TD3 tricks:
          1. Twin critics with clipped double-Q     (min of Q1, Q2 in target)
          2. Delayed actor updates                  (actor only every policy_delay steps)
          3. Target policy smoothing                (clipped Gaussian noise on target action)
        """
        batch = self.buffer.sample(self.batch_size)

        states      = _normalize_tensor(batch["states"], self.env, self.device)
        actions     = torch.FloatTensor(batch["actions"]).to(self.device)
        rewards     = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = _normalize_tensor(batch["next_states"], self.env, self.device)
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
        _clip_grad_norm(self.critic, self.max_grad_norm)
        self.critic_optim.step()

        self.total_updates += 1
        actor_loss_value = None

        # ── TRICK 2: Delayed actor + target updates ───────────────────
        if self.total_updates % self.policy_delay == 0:
            # Maximise Q1(s, actor(s))  ≡  minimise -Q1(s, actor(s))
            # We use only Q1 for the actor loss (standard TD3 practice).
            with frozen(self.critic):
                actor_loss = -self.critic.q1(states, self.actor(states)).mean()

                self.actor_optim.zero_grad()
                actor_loss.backward()
                _clip_grad_norm(self.actor, self.max_grad_norm)
                self.actor_optim.step()

            # Soft-update BOTH target networks only when the actor is updated
            self._soft_update(self.actor,  self.actor_target)
            self._soft_update(self.critic, self.critic_target)

            actor_loss_value = float(actor_loss.detach().cpu())

        critic_loss_value = float(critic_loss.detach().cpu())

        return critic_loss_value, actor_loss_value

    def _soft_update(self, online: torch.nn.Module, target: torch.nn.Module):
        """θ_target ← τ * θ_online + (1 - τ) * θ_target"""
        for online_p, target_p in zip(online.parameters(), target.parameters()):
            target_p.data.copy_(self.tau * online_p.data + (1.0 - self.tau) * target_p.data)
