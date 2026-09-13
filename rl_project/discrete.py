"""Checkpoint selection for flag and cliff notebooks; training stays in tabular.py."""
import json
from pathlib import Path

import numpy as np

from .runtime import ARTIFACTS, ROOT
from .tabular import TabularAgent

ALGORITHMS = ("qlearning", "sarsa")


def validate_algorithms(algorithms):
    """Reject typos and duplicate runs before a notebook starts training."""
    names = tuple(algorithms)
    if not names or len(set(names)) != len(names) or any(a not in ALGORITHMS for a in names):
        raise ValueError("Select 'qlearning', 'sarsa', or both, without duplicates")
    return names


def _selection(environment, algorithm):
    if environment not in ("flag", "cliff"):
        raise ValueError("environment must be 'flag' or 'cliff'")
    validate_algorithms((algorithm,))


def _path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def latest_checkpoint(environment, algorithm, output_root=None):
    """Find the latest completed run for this environment and algorithm only."""
    _selection(environment, algorithm)
    root = ARTIFACTS if output_root is None else _path(output_root)
    for candidate in sorted((root / environment / algorithm).glob("*/checkpoints/model.npz"), reverse=True):
        config_path = candidate.parents[1] / "config.json"
        try:
            config = json.loads(config_path.read_text())
        except (OSError, ValueError):
            continue
        if (isinstance(config, dict)
                and config.get("kind") == "training" and config.get("status") == "completed"
                and config.get("environment") == environment and config.get("algorithm") == algorithm
                and config.get("state_encoding") == "complete-tabular-v2"):
            return candidate
    return None


def load_for_evaluation(environment, algorithm, checkpoint_path, *, seed=1234, env_overrides=None):
    """Restore the saved task and return (agent, status), or an explicit unavailable result."""
    _selection(environment, algorithm)
    if checkpoint_path is None:
        return None, "unavailable: no selected checkpoint"
    path = _path(checkpoint_path)
    if not path.is_file():
        return None, f"unavailable: checkpoint does not exist: {path}"
    env = None
    try:
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
        if (not isinstance(metadata, dict) or metadata.get("format_version") != 1
                or metadata.get("state_encoding") != "complete-tabular-v2"
                or metadata.get("cutoff_semantics") != "bootstrap"
                or metadata.get("environment") != environment
                or metadata.get("algorithm") != algorithm):
            raise ValueError("Checkpoint format, environment or algorithm does not match the selection")
        config = {**metadata["environment_config"], **(env_overrides or {}), "render_mode": None}
        if environment == "flag":
            from Grid_flag.grid_flag_env import GridFlagEnv
            env = GridFlagEnv(**config)
        else:
            from Grid_cliff.grid_cliff_env import RandomCliffWalkingEnv
            env = RandomCliffWalkingEnv(**config)
        agent = TabularAgent(env, seed=seed, environment=environment)
        agent.load_checkpoint(path, mode="evaluation")
        agent.source_checkpoint = {"path": str(path.resolve()), "mode": "evaluation"}
        return agent, "available"
    except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
        if env is not None:
            env.close()
        return None, f"incompatible: {error}"
