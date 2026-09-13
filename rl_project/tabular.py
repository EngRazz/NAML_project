"""One tabular training loop, with explicit Q-learning and SARSA targets."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np

from .runtime import (Run, TrainingProgress, clone_env, evaluate_policy, jsonable, outcome, plot_metrics,
                      preserve_rng, recording_env, rollout, seed_everything)


class TabularAgent:
    def __init__(self, env, learning_rate=.1, initial_epsilon=1., epsilon_decay=.00019,
                 final_epsilon=.05, discount_factor=.95, seed=0, environment="flag"):
        self.env = env
        self.lr = learning_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.environment = environment
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n, dtype=np.float64))
        self.total_steps = 0
        self.last_result = None
        self.algorithm = "qlearning"
        seed_everything(seed, env)

    def state_to_key(self, obs):
        if isinstance(obs, dict):
            names = (("agent", "flags") if "flags" in obs else
                     ("agent", "target") + (("obstacle",) if "obstacle" in obs else ()))
            return tuple(int(x) for name in names for x in np.asarray(obs[name]).flat)
        return tuple(int(x) for x in np.asarray(obs).reshape(-1))

    def get_action(self, obs):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.env.action_space.n))
        return self.greedy_action(obs)

    def greedy_action(self, obs):
        values = self.q_values.get(self.state_to_key(obs))
        return int(np.argmax(values)) if values is not None else 0

    def update(self, obs, action, reward, terminated, next_obs):
        state, nxt = self.state_to_key(obs), self.state_to_key(next_obs)
        future = 0. if terminated else float(np.max(self.q_values[nxt]))
        error = reward + self.discount_factor * future - self.q_values[state][action]
        self.q_values[state][action] += self.lr * error
        return float(error)

    def update_SARSA(self, obs, action, reward, terminated, next_obs, next_action):
        state, nxt = self.state_to_key(obs), self.state_to_key(next_obs)
        if not terminated and next_action is None:
            raise ValueError("A nonterminal SARSA backup requires the next action")
        future = 0. if terminated else self.q_values[nxt][next_action]
        error = reward + self.discount_factor * future - self.q_values[state][action]
        self.q_values[state][action] += self.lr * error
        return float(error)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"format_version": 1, "environment": self.environment,
                    "algorithm": self.algorithm, "state_encoding": "complete-tabular-v2",
                    "cutoff_semantics": "bootstrap", "environment_config": self.env.unwrapped.config,
                    "action_count": self.env.action_space.n, "total_steps": self.total_steps}
        keys = list(self.q_values)
        np.savez_compressed(path, states=np.array(keys, dtype=np.int64),
                            values=np.array([self.q_values[k] for k in keys]),
                            metadata=json.dumps(jsonable(metadata)))
        return path

    def load_checkpoint(self, path, mode="evaluation"):
        if mode not in ("evaluation", "warm_start"):
            raise ValueError("mode must be evaluation or warm_start")
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            if (metadata.get("state_encoding") != "complete-tabular-v2"
                    or metadata.get("environment") != self.environment
                    or metadata.get("action_count") != self.env.action_space.n):
                raise ValueError("Incompatible tabular checkpoint")
            expected = jsonable(self.env.unwrapped.config)
            actual = metadata["environment_config"].copy()
            for config in (expected, actual):
                config.pop("render_mode", None)
                config.pop("max_step", None)
                config.pop("max_steps", None)
            if actual != expected:
                raise ValueError("Checkpoint task configuration does not match this environment")
            self.q_values.clear()
            for key, values in zip(saved["states"], saved["values"]):
                self.q_values[tuple(int(x) for x in key)] = values.copy()
        self.algorithm = metadata["algorithm"]
        self.load_mode = mode
        self.total_steps = 0 if mode == "warm_start" else metadata["total_steps"]
        self.epsilon = self.initial_epsilon if mode == "warm_start" else 0.
        self.rng = np.random.default_rng(self.seed)
        return metadata

    def train(self, num_episodes=5000, *, algorithm="qlearning", seed=None,
              output_root=None, max_env_steps=None, record_every=None, plot=True, show_progress=True, log_every=500):
        if getattr(self, "load_mode", None) == "evaluation":
            raise ValueError("Load with mode='warm_start' before training an evaluation checkpoint")
        if algorithm not in ("qlearning", "sarsa"):
            raise ValueError("algorithm must be qlearning or sarsa")
        if (num_episodes is None and max_env_steps is None
                or num_episodes is not None and num_episodes < 1
                or max_env_steps is not None and max_env_steps < 1):
            raise ValueError("Supply a positive episode or environment-step budget")
        if log_every < 1:
            raise ValueError("log_every must be positive")
        seed = self.seed if seed is None else seed
        seed_everything(seed, self.env)
        self.rng = np.random.default_rng(seed)
        self.algorithm = algorithm
        run = Run(self.environment, algorithm, seed, {
            "kind": "training", "environment_config": self.env.unwrapped.config,
            "state_encoding": "complete-tabular-v2", "cutoff_semantics": "bootstrap",
            "learning_rate": self.lr, "discount": self.discount_factor,
            "initial_epsilon": self.epsilon, "epsilon_decay": self.epsilon_decay,
            "final_epsilon": self.final_epsilon, "num_episodes": num_episodes,
            "max_env_steps": max_env_steps, "record_every": record_every, "log_every": log_every}, output_root)
        steps, ep = 0, 0
        label = f"{self.environment}: {'SARSA' if algorithm == 'sarsa' else 'Q-learning'}"
        progress = TrainingProgress(label, num_episodes, max_env_steps, show_progress, log_every)
        try:
            while (num_episodes is None or ep < num_episodes) and (max_env_steps is None or steps < max_env_steps):
                obs, _ = self.env.reset(seed=seed + ep)
                action = self.get_action(obs)
                total, length, error_sum = 0., 0, 0.
                epsilon = self.epsilon
                while True:
                    nxt, reward, terminated, truncated, info = self.env.step(action)
                    steps += 1
                    self.total_steps += 1
                    length += 1
                    if max_env_steps is not None and steps >= max_env_steps and not terminated:
                        truncated = True
                    if algorithm == "sarsa":
                        next_action = None if terminated else self.get_action(nxt)
                        error = self.update_SARSA(obs, action, reward, terminated, nxt, next_action)
                    else:
                        error = self.update(obs, action, reward, terminated, nxt)
                        next_action = None
                    total += float(reward)
                    error_sum += abs(error)
                    obs = nxt
                    progress.step()
                    if terminated or truncated:
                        break
                    action = next_action if algorithm == "sarsa" else self.get_action(obs)
                row = outcome(ep + 1, seed + ep, total, length, terminated, truncated, info)
                row.update(epsilon=epsilon, exploration_mode="epsilon_greedy",
                           environment_steps=steps, td_error=error_sum/length)
                run.append(row)
                progress.episode(row)
                self.decay_epsilon()
                if record_every and ep % record_every == 0:
                    with preserve_rng():
                        demo = clone_env(self.env, "rgb_array")
                        try:
                            demo = recording_env(demo, run.path / "videos/training", prefix=f"episode_{ep}")
                            rollout(demo, self.greedy_action, seed + ep)
                        finally:
                            demo.close()
                ep += 1
            self.save_checkpoint(run.path / "checkpoints/model.npz")
            if plot:
                plot_metrics(run.rows, run.path / "plots", f"{self.environment}: {algorithm}", num_episodes=num_episodes)
        except BaseException:
            self.save_checkpoint(run.path / "checkpoints/interrupted.npz")
            run.finish("failed")
            raise
        finally:
            progress.close()
            self.env.close()
        self.last_result = run.finish()
        return self.last_result

    def evaluate(self, episodes=1, seed=1234, output_root=None, record=False):
        return evaluate_policy(self.env, self.greedy_action, self.environment, self.algorithm,
                               episodes, seed, output_root, record)
