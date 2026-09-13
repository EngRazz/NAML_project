"""Portable artifacts, bounded videos and reproducible evaluation."""
from contextlib import contextmanager
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import copy
import csv
import json
import random
import subprocess
import uuid
import warnings
from html import escape
from importlib.metadata import version

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FIELDS = (
    "episode", "seed", "algorithm", "scenario", "reward", "length", "success", "collision",
    "terminated", "truncated", "termination_reason", "final_distance",
    "flags_collected", "epsilon", "exploration_mode", "random_action_steps", "environment_steps", "critic_updates",
    "actor_updates", "critic_loss", "actor_loss", "alpha_loss", "alpha", "td_error",
)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def code_version():
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        return {"commit": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def seed_everything(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if env is not None:
        env.action_space.seed(seed)


@contextmanager
def preserve_rng():
    """Separate demonstration runs must not consume the learner's RNG stream."""
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


@contextmanager
def frozen(module):
    flags = [p.requires_grad for p in module.parameters()]
    try:
        module.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(module.parameters(), flags):
            p.requires_grad_(flag)


def clone_env(env, render_mode=None):
    base = env.unwrapped
    config = copy.deepcopy(base.config)
    config["render_mode"] = render_mode
    return type(base)(**config)


@dataclass
class RunResult:
    run_dir: Path
    episodes: list

    @property
    def artifacts(self):
        return {"run_dir": str(self.run_dir), "metrics": str(self.run_dir / "metrics.csv"),
                "config": str(self.run_dir / "config.json"),
                "files": [str(p) for p in sorted(self.run_dir.rglob("*")) if p.is_file()]}


class Run:
    def __init__(self, environment, algorithm, seed, config, output_root=None):
        root = Path(output_root) if output_root is not None else ARTIFACTS
        if not root.is_absolute():
            root = ROOT / root
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.path = root / environment / algorithm / f"{stamp}_seed{seed}_{uuid.uuid4().hex[:8]}"
        self.path.mkdir(parents=True, exist_ok=False)
        for name in ("checkpoints", "plots", "videos/training", "videos/evaluation"):
            (self.path / name).mkdir(parents=True, exist_ok=True)
        self.config = {"environment": environment, "algorithm": algorithm, "seed": seed,
                       "code_version": code_version(),
                       "dependencies": {name: version(name) for name in ("gymnasium", "numpy", "torch", "matplotlib", "pygame", "moviepy")},
                       **jsonable(config), "status": "running"}
        write_json(self.path / "config.json", self.config)
        self.rows = []
        with (self.path / "metrics.csv").open("w", newline="") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writeheader()

    def append(self, row):
        row = jsonable(row)
        with (self.path / "metrics.csv").open("a", newline="") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writerow(row)
        self.rows.append(row)

    def finish(self, status="completed"):
        self.config["status"] = status
        write_json(self.path / "config.json", self.config)
        return RunResult(self.path, self.rows)


class _NotebookProgressBar(tqdm):
    """One updating notebook output, with persistent block summaries above it."""
    def __init__(self, *args, **kwargs):
        self.summaries = []
        self._display_handle = None
        super().__init__(*args, **kwargs)

    def display(self, msg=None, pos=None):
        from IPython.display import HTML, display
        summaries = "\n\n".join(self.summaries)
        status = str(self) if msg is None else msg
        content = HTML(
            f'<pre style="white-space:pre-wrap">{escape(summaries)}</pre>'
            f'<progress value="{self.n}" max="{self.total or 1}" style="width:100%"></progress>'
            f'<pre style="white-space:pre-wrap">{escape(status)}</pre>'
        )
        if self._display_handle is None:
            self._display_handle = display(content, display_id=True)
        else:
            self._display_handle.update(content)


def _in_notebook():
    try:
        from IPython import get_ipython
        return getattr(get_ipython(), "kernel", None) is not None
    except ImportError:
        return False


def episode_summary(rows):
    """Describe one block of completed episodes using zero-based episode labels."""
    first, last = int(rows[0]["episode"]) - 1, int(rows[-1]["episode"]) - 1
    fields = []
    for key, label, format_spec in (
        ("reward", "mean reward", ".2f"), ("length", "mean length", ".1f"),
        ("success", "success rate", ".1%"), ("final_distance", "mean final distance (m)", ".2f"),
        ("critic_loss", "critic loss", ".4g"), ("actor_loss", "actor loss", ".4g"),
        ("alpha_loss", "temperature loss", ".4g"), ("alpha", "temperature", ".4g"),
        ("epsilon", "mean epsilon", ".3f"), ("td_error", "mean absolute TD error", ".4g"),
    ):
        values = [float(r[key]) for r in rows if r.get(key) is not None and np.isfinite(r[key])]
        if values:
            fields.append(f"{label}={np.mean(values):{format_spec}}")
    return f"Episodes {first}–{last} ({len(rows)} episodes; {last + 1} completed): " + " | ".join(fields)


class TrainingProgress:
    """Live total progress below a summary for every block of completed episodes."""
    def __init__(self, label, num_episodes, max_env_steps, enabled=True, log_every=500):
        if log_every < 1:
            raise ValueError("log_every must be positive")
        self.by_episode = num_episodes is not None
        self.recent = deque(maxlen=100)
        self.block = []
        self.log_every = log_every
        self.enabled = enabled
        self._closed = False
        bar_type = _NotebookProgressBar if enabled and _in_notebook() else tqdm
        self.bar = bar_type(
            total=num_episodes if self.by_episode else max_env_steps,
            desc=f"{label} total", unit="episode" if self.by_episode else "step",
            dynamic_ncols=True, mininterval=0.5, disable=not enabled,
        )

    def step(self):
        if not self.by_episode:
            self.bar.update(1)

    def episode(self, row):
        if not self.enabled:
            return
        self.recent.append(row)
        self.block.append(row)
        status = {
            "reward": f"{row['reward']:.1f}",
            "avg_reward": f"{np.mean([r['reward'] for r in self.recent]):.1f}",
            "success": f"{np.mean([r['success'] for r in self.recent]):.0%}",
        }
        if self.by_episode:
            status["steps"] = row["environment_steps"]
        else:
            status["episodes"] = row["episode"]
        if row.get("epsilon") is not None:
            status["epsilon"] = f"{row['epsilon']:.3f}"
        self.bar.set_postfix(status, refresh=False)
        if self.by_episode:
            self.bar.update(1)
        if len(self.block) == self.log_every:
            self._summarize()

    def _summarize(self):
        if not self.block:
            return
        summary = episode_summary(self.block)
        if isinstance(self.bar, _NotebookProgressBar):
            self.bar.summaries.append(summary)
            self.bar.refresh()
        else:
            self.bar.write(summary, file=self.bar.fp)
        self.block.clear()

    def close(self):
        if self._closed:
            return
        self._summarize()
        self.bar.close()
        self._closed = True


class SmallFrames(gym.Wrapper):
    """Keep native frames up to the width limit; filter larger frames when shrinking."""
    def __init__(self, env, max_width=800):
        super().__init__(env)
        if max_width < 1:
            raise ValueError("max_width must be positive")
        self.max_width = max_width

    def render(self):
        frame = self.env.render()
        if frame is None or frame.shape[1] <= self.max_width:
            return frame
        import pygame
        height = max(1, round(frame.shape[0] * self.max_width / frame.shape[1]))
        # Area filtering includes thin grid lines that pixel skipping can miss.
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        resized = pygame.transform.smoothscale(surface, (self.max_width, height))
        return np.ascontiguousarray(np.transpose(pygame.surfarray.array3d(resized), (1, 0, 2)))


class LimitedRecordVideo(gym.wrappers.RecordVideo):
    """Record one demonstration with an exact filename stem and frame bound."""
    def __init__(self, env, *, max_frames=300, **kwargs):
        self.frame_limit = max_frames
        super().__init__(env, video_length=max_frames, **kwargs)

    def start_recording(self, video_name):
        # Each wrapper records one demonstration, so its local episode ID is redundant.
        destination = Path(self.video_folder) / f"{self.name_prefix}.mp4"
        if destination.exists():
            raise FileExistsError(f"Video already exists: {destination}")
        super().start_recording(self.name_prefix)

    def _capture_frame(self):
        super()._capture_frame()
        if self.recording and len(self.recorded_frames) >= self.frame_limit:
            self.stop_recording()


def recording_env(env, directory, prefix="episode_0", max_frames=300, max_width=800):
    """Record one demonstration as <prefix>.mp4, without a wrapper episode suffix."""
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    if max_width < 1:
        raise ValueError("max_width must be positive")
    Path(directory).mkdir(parents=True, exist_ok=True)
    if (Path(directory) / f"{prefix}.mp4").exists() or any(Path(directory).glob(f"{prefix}-*.mp4")):
        raise FileExistsError(f"Video prefix already exists: {directory}/{prefix}")
    with warnings.catch_warnings():
        # Gymnasium warns even for a pre-created empty directory; the guard
        # above prevents overwriting an actual video with this name or legacy prefix.
        warnings.filterwarnings("ignore", message=".*Overwriting existing videos.*")
        return LimitedRecordVideo(
            SmallFrames(env, max_width=max_width), video_folder=str(directory), name_prefix=prefix,
            episode_trigger=lambda episode: episode == 0,
            max_frames=max_frames, disable_logger=True,
        )


def video_sort_key(path):
    """Sort episode_0, episode_500, episode_1000 numerically in notebook displays."""
    stem = Path(path).stem
    index = stem.removeprefix("episode_")
    return (0, int(index)) if index.isdigit() else (1, stem)


def outcome(episode, seed, reward, length, terminated, truncated, info):
    success = bool(info.get("goal_reached", False))
    collision = bool(info.get("collision", False))
    reason = "success" if success else "collision" if collision else "terminated" if terminated else "cutoff"
    return {"episode": episode, "seed": seed, "reward": reward, "length": length,
            "success": success, "collision": collision, "terminated": bool(terminated),
            "truncated": bool(truncated), "termination_reason": reason,
            "final_distance": info.get("dist_to_goal", info.get("distance")),
            "flags_collected": info.get("flags_collected")}


def rollout(env, action_fn, seed, episode=1):
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    total, length = 0.0, 0
    while True:
        obs, reward, terminated, truncated, info = env.step(action_fn(obs))
        total += float(reward)
        length += 1
        if terminated or truncated:
            return outcome(episode, seed, total, length, terminated, truncated, info)


def evaluate_policy(env, action_fn, environment, algorithm, episodes=1, seed=1234,
                    output_root=None, record=False, max_frames=300, config=None):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    run = Run(environment, algorithm, seed,
              {"kind": "evaluation", "environment_config": env.unwrapped.config,
               "episodes": episodes, **(config or {})}, output_root)
    try:
        with preserve_rng():
            for ep in range(episodes):
                current = clone_env(env, "rgb_array" if record and ep == 0 else None)
                try:
                    if record and ep == 0:
                        current = recording_env(current, run.path / "videos/evaluation", max_frames=max_frames)
                    run.append(rollout(current, action_fn, seed + ep, ep + 1))
                finally:
                    current.close()
    except BaseException:
        run.finish("failed")
        raise
    return run.finish()


def plot_metrics(rows, directory, title, *, num_episodes=None):
    """Episode-aligned diagnostics; bands describe one run, not confidence."""
    import matplotlib.pyplot as plt
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    xs = np.array([r["episode"] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    tabular = any("epsilon" in row for row in rows)
    keys = ("reward", "length", "td_error", "epsilon") if tabular else ("reward", "length", "critic_loss", "actor_loss")
    labels = ("Episode return", "Episode length", "Mean absolute TD error", "Epsilon") if tabular else ("Episode return", "Episode length", "Critic loss", "Actor loss")
    for ax, key, label in zip(axes.flat, keys, labels):
        values = np.array([np.nan if r.get(key) is None else r[key] for r in rows], dtype=float)
        ax.plot(xs, values, alpha=.5, label="Episode value")
        if key in ("reward", "length") and len(values):
            means = np.array([np.mean(values[max(0, i-99):i+1]) for i in range(len(values))])
            ax.plot(xs, means, label="Rolling mean (up to 100 episodes)")
            if key == "reward":
                std = np.array([np.std(values[max(0, i-99):i+1]) for i in range(len(values))])
                ax.fill_between(xs, means-2*std, means+2*std, alpha=.15, label="Rolling mean ± 2 rolling std")
        ax.set(xlabel="Episode", title=label)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(directory / "training_curves.png", dpi=150)
    plt.close(fig)
    plot_outcomes(rows, directory, title, num_episodes=num_episodes)


def plot_outcomes(rows, directory, title, *, num_episodes=None):
    """Readable outcome plots; usable on saved metrics without redrawing training curves."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    xs = np.array([r["episode"] for r in rows])
    diagnostics = []
    if any(r.get("flags_collected") is not None for r in rows):
        diagnostics.append(("flags_collected", "Flags collected"))
    if any(r.get("alpha") is not None for r in rows):
        diagnostics.append(("alpha", "Entropy temperature (SAC)"))
    fig, axes = plt.subplots(1 + len(diagnostics), 1,
                             figsize=(10, 4 * (1 + len(diagnostics))), squeeze=False)
    axes = axes[:, 0]
    success = np.array([r["success"] for r in rows], dtype=float)
    if len(success):
        axes[0].plot(xs, [np.mean(success[max(0, i-99):i+1]) for i in range(len(success))], label="Rolling success rate (up to 100 episodes)")
    axes[0].set_ylim(-.05, 1.05)
    axes[0].set(title="Rolling success rate", ylabel="Successful episodes")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend()
    for ax, (key, label) in zip(axes[1:], diagnostics):
        ax.plot(xs, [np.nan if r.get(key) is None else r[key] for r in rows])
        ax.set_title(label)
    for ax in axes:
        ax.set_xlabel("Episode")
        ax.grid(alpha=.2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(directory / "outcomes.png", dpi=150)
    plt.close(fig)

    if any(r.get("final_distance") is not None for r in rows):
        distance = np.array([np.nan if r.get("final_distance") is None else r["final_distance"] for r in rows], dtype=float)
        means = []
        for i in range(len(distance)):
            window = distance[max(0, i-99):i+1]
            valid = window[np.isfinite(window)]
            means.append(np.mean(valid) if len(valid) else np.nan)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(xs, distance, alpha=.2, linewidth=.7, label="Final distance per episode")
        ax.plot(xs, means, linewidth=2, label="Rolling mean (up to 100 episodes)")
        ax.set(xlabel="Episode", ylabel="Distance to goal (m)",
               title=f"{title}: final goal distance", ylim=(0, 10),
               xlim=(0, max(1, num_episodes or (max(xs) if len(xs) else 1))))
        ax.set_box_aspect(1)
        ax.grid(alpha=.2)
        ax.legend()
        above = np.isfinite(distance) & (distance > 10)
        if np.any(above):
            # Keep the requested axis while marking values outside the visible range.
            ax.scatter(xs[above], np.full(above.sum(), 10.), marker="^", s=15, clip_on=False,
                       label=f"{above.sum()} episodes above 10 m")
            ax.legend()
        fig.tight_layout()
        fig.savefig(directory / "final_goal_distance.png", dpi=150)
        plt.close(fig)
