"""Training and checkpoint lifecycle shared by DDPG, TD3 and SAC."""
from pathlib import Path
import numpy as np
import torch

from .runtime import (Run, TrainingProgress, clone_env, evaluate_policy, jsonable, outcome, plot_metrics,
                      preserve_rng, recording_env, rollout, seed_everything)

NORMALIZATION = "physical-observation-v1"


class ContinuousWorkflow:
    def greedy_action(self, obs):
        if self.algorithm == "sac":
            return self.select_action(obs, evaluate=True)
        return self.select_action(obs, add_noise=False)

    def training_action(self, obs):
        return self.select_action(obs)

    def checkpoint_metadata(self):
        env = self.env.unwrapped
        return jsonable({"format_version": 2, "algorithm": self.algorithm,
            "preprocessing": NORMALIZATION, "reward_version": env.REWARD_VERSION,
            "cutoff_semantics": "bootstrap", "agent_config": self.agent_config,
            "environment_config": env.config, "observation_shape": env.observation_space.shape,
            "action_low": env.action_space.low, "action_high": env.action_space.high,
            "normalization_scale": [env.lidar_max_range] * env.n_lidar_rays +
                                   [float(np.hypot(env.room_w, env.room_h)), float(np.pi)],
            "dynamics": {"dt": env.dt, "robot_radius": env.robot_radius},
            "source_checkpoint": getattr(self, "source_checkpoint", None)})

    def save_checkpoint(self, checkpoint_path):
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {"metadata": self.checkpoint_metadata(), "total_steps": self.total_steps,
                      "total_updates": self.total_updates,
                      "actor_updates": getattr(self, "actor_updates", 0)}
        for name in ("actor", "critic", "actor_target", "critic_target"):
            if hasattr(self, name):
                checkpoint[name] = getattr(self, name).state_dict()
        if self.algorithm == "sac":
            checkpoint["log_alpha"] = self.log_alpha.detach().cpu()
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)
        return path

    def load_checkpoint(self, checkpoint_path, mode="evaluation", allow_legacy_normalized=False):
        """Load a policy, or initialize a fresh learner from learned weights.

        Optimizers and replay are never resumed. Legacy opt-in is restricted to
        DDPG evaluation and explicitly asserts that its input scaling was verified.
        """
        if mode not in ("evaluation", "warm_start"):
            raise ValueError("mode must be evaluation or warm_start")
        path = Path(checkpoint_path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        metadata = checkpoint.get("metadata")
        if metadata is None:
            if not (allow_legacy_normalized and self.algorithm == "ddpg" and mode == "evaluation"):
                raise ValueError("Checkpoint has no preprocessing metadata; retrain with normalized inputs")
        else:
            expected = self.checkpoint_metadata()
            if metadata.get("agent_config", {}).get("hidden_dim") != self.agent_config["hidden_dim"]:
                raise ValueError("Incompatible checkpoint hidden_dim")
            for key in ("format_version", "algorithm", "preprocessing", "reward_version",
                        "cutoff_semantics", "observation_shape", "action_low", "action_high",
                        "normalization_scale", "dynamics"):
                if metadata.get(key) != expected[key]:
                    raise ValueError(f"Incompatible checkpoint {key}: {metadata.get(key)!r}")
        for name in ("actor", "critic"):
            getattr(self, name).load_state_dict(checkpoint[name])
        for name in ("actor_target", "critic_target"):
            if hasattr(self, name):
                source = checkpoint[name] if mode == "evaluation" and name in checkpoint else checkpoint[name.replace("_target", "")]
                getattr(self, name).load_state_dict(source)
        if mode == "warm_start":
            self.buffer.ptr = self.buffer.size = 0
            self.actor_optim.state.clear()
            self.critic_optim.state.clear()
            self.total_steps = self.total_updates = self.actor_updates = 0
            if hasattr(self, "ou_noise"):
                self.ou_noise.reset()
            if self.algorithm == "sac":
                with torch.no_grad():
                    self.log_alpha.fill_(np.log(self.agent_config["alpha"]))
                if self.alpha_optim is not None:
                    self.alpha_optim.state.clear()
                self.alpha = self.log_alpha.exp()
            seed_everything(self.agent_config["seed"], self.env)
        else:
            self.total_steps = int(checkpoint.get("total_steps", 0))
            self.total_updates = int(checkpoint.get("total_updates", 0))
            self.actor_updates = int(checkpoint.get("actor_updates", 0))
            if self.algorithm == "sac" and "log_alpha" in checkpoint:
                with torch.no_grad():
                    self.log_alpha.copy_(checkpoint["log_alpha"].to(self.device))
                self.alpha = self.log_alpha.exp()
        self.source_checkpoint = {"path": str(path.resolve()), "mode": mode,
                                  "legacy_normalized_assertion": metadata is None}
        return checkpoint

    def train(self, num_episodes=5000, *, seed=None, output_root=None,
              max_env_steps=None, record_every=None, plot=True, show_progress=True, log_every=500):
        if getattr(self, "source_checkpoint", {}).get("mode") == "evaluation":
            raise ValueError("Load with mode='warm_start' before training an evaluation checkpoint")
        if (num_episodes is None and max_env_steps is None
                or num_episodes is not None and num_episodes < 1
                or max_env_steps is not None and max_env_steps < 1):
            raise ValueError("Supply a positive episode or environment-step budget")
        if log_every < 1:
            raise ValueError("log_every must be positive")
        seed = self.agent_config["seed"] if seed is None else seed
        if seed != self.agent_config["seed"]:
            raise ValueError("Construct the agent with the requested seed so initialization is reproducible")
        seed_everything(seed, self.env)
        run = Run("continuous", self.algorithm, seed,
                  {"kind": "training", **self.checkpoint_metadata(),
                   "num_episodes": num_episodes, "max_env_steps": max_env_steps,
                   "record_every": record_every, "log_every": log_every}, output_root)
        self.actor_updates = getattr(self, "actor_updates", 0)
        ep, steps = 0, 0
        progress = TrainingProgress(self.algorithm.upper(), num_episodes, max_env_steps, show_progress, log_every)
        try:
            while (num_episodes is None or ep < num_episodes) and (max_env_steps is None or steps < max_env_steps):
                obs, _ = self.env.reset(seed=seed + ep)
                if hasattr(self, "ou_noise"):
                    self.ou_noise.reset()
                total, length = 0., 0
                episode_start_steps = self.total_steps
                # Constant storage per episode, even at a very large update ratio.
                sums = {k: [0., 0] for k in ("critic_loss", "actor_loss", "alpha_loss", "alpha")}
                while True:
                    action = self.training_action(obs)
                    nxt, reward, terminated, truncated, info = self.env.step(action)
                    steps += 1
                    self.total_steps += 1
                    length += 1
                    total += float(reward)
                    if max_env_steps is not None and steps >= max_env_steps and not terminated:
                        truncated = True
                    self.buffer.add(obs, action, reward, nxt, terminated)
                    obs = nxt
                    if self.buffer.ready(self.batch_size) and self.total_steps >= self.warmup_steps:
                        for _ in range(self.updates_per_step):
                            values = self._update()
                            if values[1] is not None:
                                self.actor_updates += 1
                            for key, value in zip(sums, values):
                                if value is not None:
                                    sums[key][0] += float(value)
                                    sums[key][1] += 1
                    progress.step()
                    if terminated or truncated:
                        break
                row = outcome(ep + 1, seed + ep, total, length, terminated, truncated, info)
                row.update(environment_steps=steps, critic_updates=self.total_updates,
                           actor_updates=self.actor_updates)
                warmup = min(length, max(0, self.warmup_steps - episode_start_steps))
                row.update(random_action_steps=warmup, exploration_mode=(
                    "uniform_warmup" if warmup == length else "mixed_warmup_policy" if warmup else
                    "stochastic_policy" if self.algorithm == "sac" else "ou_noise"))
                row.update({key: value/count if count else None for key, (value, count) in sums.items()})
                if self.algorithm == "sac" and row["alpha"] is None:
                    row["alpha"] = float(self.alpha.detach().cpu())
                run.append(row)
                progress.episode(row)
                if record_every and ep % record_every == 0:
                    with preserve_rng():
                        demo = clone_env(self.env, "rgb_array")
                        try:
                            demo = recording_env(demo, run.path / "videos/training", prefix=f"episode_{ep}")
                            rollout(demo, self.greedy_action, seed + ep)
                        finally:
                            demo.close()
                ep += 1
            self.save_checkpoint(run.path / "checkpoints/model.pt")
            if plot:
                plot_metrics(run.rows, run.path / "plots", self.algorithm.upper(), num_episodes=num_episodes)
        except BaseException:
            self.save_checkpoint(run.path / "checkpoints/interrupted.pt")
            run.finish("failed")
            raise
        finally:
            progress.close()
            self.env.close()
        self.last_result = run.finish()
        return self.last_result

    def evaluate(self, episodes=1, seed=1234, output_root=None, record=False):
        return evaluate_policy(self.env, self.greedy_action, "continuous", self.algorithm,
                               episodes, seed, output_root, record,
                               config={"checkpoint": getattr(self, "source_checkpoint", None)})
