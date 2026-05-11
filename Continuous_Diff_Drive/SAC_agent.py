from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics

from SAC_env import DiffDriveEnv
from SAC_replay_buffer import ReplayBuffer
from SAC_nn import GaussianActor, DoubleCritic, init_weights


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_PATH = BASE_DIR / "models" / "sac_checkpoint.pt"
DEFAULT_PLOT_PATH = BASE_DIR / "images" / "sac_training_curves.png"
DEFAULT_PLOT_PATH2 = BASE_DIR / "images" / "sac_training_curves2.png"



def _artifact_path(path):
    path = Path(path)
    return path if path.is_absolute() else BASE_DIR / path


class DiffDriveSACAgent:

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
        warmup_steps: int = 1000,
        device: str = "cpu",
        automatic_entropy_tuning: bool = True,
    ):

        self.env = env
        self.discount = discount
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.device = torch.device(device)

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.action_low = torch.FloatTensor(env.action_space.low).to(self.device)
        self.action_high = torch.FloatTensor(env.action_space.high).to(self.device)

        # ============================================================
        # Actor
        # ============================================================

        self.actor = GaussianActor(
            obs_dim,
            action_dim,
            env.action_space.low,
            env.action_space.high,
            hidden_dim,
        ).to(self.device)

        # ============================================================
        # Critics
        # ============================================================

        self.critic = DoubleCritic(
            obs_dim,
            action_dim,
            hidden_dim
        ).to(self.device)

        self.critic_target = DoubleCritic(
            obs_dim,
            action_dim,
            hidden_dim
        ).to(self.device)
        

        self.critic_target.load_state_dict(self.critic.state_dict())

        # ============================================================
        # Optimizers
        # ============================================================

        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr
        )

        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr
        )

        # ============================================================
        # Entropy coefficient α
        # ============================================================

        self.automatic_entropy_tuning = automatic_entropy_tuning

        if automatic_entropy_tuning:

            self.target_entropy = -float(action_dim)

            self.log_alpha = torch.zeros(
                1,
                requires_grad=True,
                device=self.device
            )

            self.alpha_optim = torch.optim.Adam(
                [self.log_alpha],
                lr=alpha_lr
            )

            self.alpha = self.log_alpha.exp()

        else:
            self.alpha = torch.tensor(alpha).to(self.device)

        # ============================================================
        # Replay Buffer
        # ============================================================

        self.buffer = ReplayBuffer(
            obs_dim,
            action_dim,
            buffer_size
        )

        self.total_steps = 0

    # ================================================================
    # Action Selection
    # ================================================================

    def select_action(self, obs, evaluate=False):

        if self.total_steps < self.warmup_steps and not evaluate:
            return self.env.action_space.sample()

        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():

            if evaluate:
                action, _ = self.actor.act(obs_t)
            else:
                action, _ = self.actor.sample(obs_t)

        return action.cpu().numpy()[0].astype(np.float32)

    # ================================================================
    # SAC Update
    # ================================================================

    def _update(self):

        batch = self.buffer.sample(self.batch_size)

        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]
        #states = torch.FloatTensor(batch["states"]).to(self.device)
        #actions = torch.FloatTensor(batch["actions"]).to(self.device)
        #rewards = torch.FloatTensor(batch["rewards"]).unsqueeze(-1).to(self.device)
        #next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        #dones = torch.FloatTensor(batch["dones"]).unsqueeze(-1).to(self.device)

        # ============================================================
        # Critic target
        # ============================================================

        with torch.no_grad():
            #Actor select the next actions
            next_actions, next_log_probs = self.actor.sample(next_states)
            #Get the two Q-value from the two Critic network
            q1_target, q2_target = self.critic_target(
                next_states,
                next_actions
            )
            #find the less optimistic Q value
            min_q_target = torch.min(q1_target, q2_target)
            #Target Q value by bellman target
            target_q = rewards + (1.0 - dones) * self.discount * (min_q_target - self.alpha * next_log_probs)

        # ============================================================
        # Critic update
        # ============================================================

        current_q1, current_q2 = self.critic(states, actions)

        critic_loss = (
            F.mse_loss(current_q1, target_q) +
            F.mse_loss(current_q2, target_q)
        )

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ============================================================
        # Actor update
        # ============================================================

        sampled_actions, log_probs = self.actor.sample(states)

        q1_pi, q2_pi = self.critic(states, sampled_actions)

        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (
            self.alpha * log_probs - min_q_pi
        ).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ============================================================
        # Alpha update
        # ============================================================

        if self.automatic_entropy_tuning:

            alpha_loss = -(
                self.log_alpha * (
                    log_probs + self.target_entropy
                ).detach()
            ).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()

        else:
            alpha_loss = torch.tensor(0.0)

        # ============================================================
        # Soft update critic target
        # ============================================================

        self._soft_update(self.critic, self.critic_target)

        return (
            float(critic_loss.detach().cpu()),
            float(actor_loss.detach().cpu()),
            float(alpha_loss.detach().cpu()),
        )

    # ================================================================
    # Soft update
    # ================================================================

    def _soft_update(self, online, target):

        for online_p, target_p in zip(online.parameters(),target.parameters()):
            target_p.data.copy_(
                self.tau * online_p.data +
                (1.0 - self.tau) * target_p.data
            )

    # ================================================================
    # Training Loop
    # ================================================================

    def train_recorded(
        self,
        num_episodes: int,
        video_folder: str = "videos/training",
        record_every: int = 100,
        log_every: int = 100,
        name_prefix: str = "sac_diff_drive_training",
        checkpoint_path = DEFAULT_CHECKPOINT_PATH,
        plot_path = DEFAULT_PLOT_PATH,
        plot_path2 = DEFAULT_PLOT_PATH2,
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
            episode_trigger=lambda ep: (
                record_every > 0 and ep % record_every == 0
            ),
        )

        env = RecordEpisodeStatistics(
            env,
            buffer_length=num_episodes
        )

        episode_rewards = []
        episode_lengths = []

        critic_losses = []
        actor_losses = []
        alpha_losses = []

        ep_goal_dist = []
        success_rate = []

        for ep in tqdm(range(num_episodes), desc="Training"):

            obs, _ = env.reset()

            done = False
            ep_reward = 0

            ep_c_loss = []
            ep_a_loss = []
            ep_alpha_loss = []

            while not done:

                action = self.select_action(obs)

                next_obs, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated

                self.buffer.add(
                    obs,
                    action,
                    reward,
                    next_obs,
                    done
                )

                obs = next_obs

                ep_reward += reward

                self.total_steps += 1

                # Metrics tracking
                ep_goal_dist.append(next_obs[-2])

                if info.get("goal_reached", False):
                    success_rate.append(1)
                else:
                    success_rate.append(0)

                # Learning
                if self.buffer.ready(self.batch_size) and self.total_steps >= self.warmup_steps:

                    critic_loss, actor_loss, alpha_loss = self._update()
                    ep_c_loss.append(critic_loss)
                    ep_a_loss.append(actor_loss)
                    ep_alpha_loss.append(alpha_loss)

            # Episode stats
            episode_rewards.append(ep_reward)

            episode_lengths.append(
                list(env.length_queue)[-1]
            )

            if ep_c_loss:
                critic_losses.append(np.mean(ep_c_loss))

            if ep_a_loss:
                actor_losses.append(np.mean(ep_a_loss))

            if ep_alpha_loss:
                alpha_losses.append(np.mean(ep_alpha_loss))

            # Logging
            if (ep + 1) % log_every == 0:

                avg_reward = np.mean(episode_rewards[-log_every:])

                avg_goal_dist = np.mean(ep_goal_dist[-log_every:])

                avg_success = np.mean(success_rate[-log_every:])

                avg_actor_loss = (np.mean(actor_losses[-log_every:])  if actor_losses else 0.0)
        

                avg_critic_loss = (np.mean(critic_losses[-log_every:]) if critic_losses else 0.0)

                avg_alpha_loss = (np.mean(alpha_losses[-log_every:]) if alpha_losses else 0.0)

                print(
                    f"Episode {ep+1:>5} | "
                    f"avg reward: {avg_reward:.2f} | "
                    f"avg dist: {avg_goal_dist:.2f} | "
                    f"success: {avg_success:.2f} | "
                    f"actor loss: {avg_actor_loss:.4f} | "
                    f"critic loss: {avg_critic_loss:.4f} | "
                    f"alpha loss: {avg_alpha_loss:.4f} | "
                    f"buffer: {len(self.buffer):>6} | "
                    f"steps: {self.total_steps}"
                )

        # Save checkpoint
        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "target_critic1": self.target_critic1.state_dict(),
                "target_critic2": self.target_critic2.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
            },
            checkpoint_path,
        )

        print(f"Checkpoint saved to {checkpoint_path}")

        env.close()

        # Optional plots
        self._plot_training_curves(
            rewards=episode_rewards,
            actor_losses=actor_losses,
            critic_losses=critic_losses,
            alpha_losses=alpha_losses,
            success_rate=success_rate,
            plot_path=plot_path,
            plot_path2=plot_path2,
        )

    # ================================================================
    # Evaluation
    # ================================================================

    def evaluate(self, n_episodes=5):

        for ep in range(n_episodes):

            obs, _ = self.env.reset()

            done = False
            total_reward = 0

            while not done:

                action = self.select_action(
                    obs,
                    evaluate=True
                )

                obs, reward, terminated, truncated, _ = self.env.step(action)

                done = terminated or truncated

                total_reward += reward

            print(
                f"Eval Episode {ep+1} | "
                f"Reward: {total_reward:.2f}"
            )