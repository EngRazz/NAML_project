"""Explicit training configurations and reproducible shared-scenario comparisons."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch

from Continuous_Diff_Drive.diff_drive_env import DiffDriveEnv
from Continuous_Diff_Drive.diff_drive_agent import DiffDriveDDPGAgent, DiffDriveSACAgent, DiffDriveTD3Agent
from rl_project.runtime import ARTIFACTS, ROOT, Run, jsonable, recording_env, rollout, write_json

AGENTS = {"ddpg": DiffDriveDDPGAgent, "sac": DiffDriveSACAgent, "td3": DiffDriveTD3Agent}
FIXED_OBSTACLES = [(2., 4., 3., .3), (5., 2., .3, 3.), (7., 6., 1.5, .3)]
ENVIRONMENT = dict(room_size=(10., 10.), robot_start=(1., 1.), goal_pos=(8.5, 8.5),
                   max_step=1000, random_obst=True, obstacle_mode="curriculum", render_mode=None)
DEFAULTS = {
    "ddpg": dict(actor_lr=1e-4, critic_lr=1e-3, discount=.995, tau=.001,
                 batch_size=256, buffer_size=300000, hidden_dim=256, warmup_steps=5000, updates_per_step=1),
    "sac": dict(actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, discount=.99, tau=.005,
                batch_size=256, buffer_size=100000, hidden_dim=64, warmup_steps=5000,
                updates_per_step=2, automatic_entropy_tuning=True),
    "td3": dict(actor_lr=1e-4, critic_lr=1e-3, discount=.99, tau=.005, policy_delay=2,
                batch_size=256, buffer_size=300000, hidden_dim=256, warmup_steps=5000, updates_per_step=2),
}
EPISODES = {"ddpg": 5000, "sac": 3000, "td3": 5000}


def create_agent(algorithm, seed=0, env_kwargs=None, agent_kwargs=None, comparison=False):
    config = copy.deepcopy(ENVIRONMENT)
    if algorithm == "sac" and not comparison:
        config["max_step"] = 1200
    config.update(env_kwargs or {})
    params = {**DEFAULTS[algorithm], **(agent_kwargs or {}), "seed": seed}
    return AGENTS[algorithm](DiffDriveEnv(**config), **params)


def latest_checkpoint(algorithm, output_root=None):
    root = Path(output_root) if output_root is not None else ARTIFACTS
    if not root.is_absolute():
        root = ROOT / root
    for candidate in sorted((root / "continuous" / algorithm).glob("*/checkpoints/model.pt"), reverse=True):
        config = candidate.parents[1] / "config.json"
        if config.exists():
            metadata = json.loads(config.read_text())
            if metadata.get("kind") == "training" and metadata.get("status") == "completed":
                return candidate
    return None


def load_for_evaluation(algorithm, checkpoint_path, env_kwargs=None, allow_legacy_normalized=False):
    if checkpoint_path is None or not Path(checkpoint_path).is_file():
        return None, "unavailable: no selected checkpoint"
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    params = data.get("metadata", {}).get("agent_config", DEFAULTS[algorithm]).copy()
    params.update(device="cpu", buffer_size=max(1, params.get("batch_size", 256)))
    seed = params.pop("seed", 0)
    env = {**ENVIRONMENT, "random_obst": False, "obstacles": FIXED_OBSTACLES, **(env_kwargs or {})}
    agent = create_agent(algorithm, seed, env, params, comparison=True)
    try:
        agent.load_checkpoint(checkpoint_path, mode="evaluation", allow_legacy_normalized=allow_legacy_normalized)
    except (ValueError, RuntimeError) as error:
        agent.env.close()
        return None, f"incompatible: {error}"
    return agent, "available"


def build_scenarios(seeds=range(10000, 10050)):
    scenarios = [{"id": "historical-fixed", "seed": 1234,
                  "environment": {**ENVIRONMENT, "random_obst": False, "obstacles": FIXED_OBSTACLES}}]
    env = DiffDriveEnv(**ENVIRONMENT)
    try:
        for seed in seeds:
            env.reset(seed=seed)
            scenarios.append({"id": f"mixture-{seed}", "seed": seed,
                              "environment": {**ENVIRONMENT, "random_obst": False,
                                              "obstacles": copy.deepcopy(env.obstacles)}})
    finally:
        env.close()
    return jsonable(scenarios)


def compare_checkpoints(checkpoints, scenarios=None, output_root=None, record=False):
    scenarios = build_scenarios() if scenarios is None else scenarios
    run = Run("continuous", "comparison", 1234,
              {"kind": "evaluation-comparison", "checkpoints": checkpoints,
               "scenario_count": len(scenarios)}, output_root)
    write_json(run.path / "scenarios.json", scenarios)
    statuses, summary = {}, {}
    try:
        for algorithm in AGENTS:
            path = checkpoints.get(algorithm)
            rows = []
            for index, scenario in enumerate(scenarios):
                agent, status = load_for_evaluation(algorithm, path, scenario["environment"])
                statuses[algorithm] = status
                if agent is None:
                    break
                env = agent.env
                try:
                    if record and index == 0:
                        env.render_mode = "rgb_array"
                        env = recording_env(env, run.path / "videos/evaluation", prefix=algorithm)
                    row = rollout(env, agent.greedy_action, scenario["seed"], index + 1)
                    row.update(algorithm=algorithm, scenario=scenario["id"])
                    run.append(row)
                    rows.append(row)
                finally:
                    env.close()
            if rows:
                summary[algorithm] = {"scenarios": len(rows), "mean_reward": np.mean([r["reward"] for r in rows]),
                                      "mean_length": np.mean([r["length"] for r in rows]),
                                      "success_rate": np.mean([r["success"] for r in rows])}
        write_json(run.path / "summary.json", {"status": statuses, "metrics": summary,
                   "complete_three_agent_comparison": all(statuses.get(a) == "available" for a in AGENTS),
                   "interpretation": "Descriptive results on shared scenarios; no algorithm ranking or independent training-seed confidence interval."})
    except BaseException:
        run.finish("failed")
        raise
    return run.finish()


def run_multiseed_comparison(output_root=None, seeds=range(5), steps=1500000):
    """Expensive: only called by an explicit --train-comparison request."""
    run = Run("continuous", "multiseed", 0,
              {"kind": "multiseed-comparison", "training_seeds": list(seeds),
               "steps_per_agent": steps, "cutoff": 1000, "agent_defaults": DEFAULTS}, output_root)
    scenarios = build_scenarios()
    write_json(run.path / "scenarios.json", scenarios)
    manifest = []
    try:
        for seed in seeds:
            checkpoints = {}
            training = {}
            for algorithm in AGENTS:
                agent = create_agent(algorithm, seed=seed, comparison=True)
                result = agent.train(num_episodes=None, max_env_steps=steps, output_root=output_root)
                checkpoints[algorithm] = str(result.run_dir / "checkpoints/model.pt")
                training[algorithm] = {"run": str(result.run_dir), "environment_steps": agent.total_steps,
                                       "critic_updates": agent.total_updates, "actor_updates": agent.actor_updates}
                del agent
            evaluation = compare_checkpoints(checkpoints, scenarios, output_root)
            manifest.append({"seed": seed, "training": training, "evaluation": str(evaluation.run_dir)})
            write_json(run.path / "runs.json", manifest)
        aggregates = {}
        for algorithm in AGENTS:
            rows = [json.loads((Path(entry["evaluation"]) / "summary.json").read_text())["metrics"][algorithm]
                    for entry in manifest]
            aggregates[algorithm] = {key: {"mean": float(np.mean([r[key] for r in rows])),
                                          "std_across_training_seeds": float(np.std([r[key] for r in rows], ddof=1)) if len(rows)>1 else None}
                                     for key in ("mean_reward", "mean_length", "success_rate")}
        write_json(run.path / "aggregate.json", aggregates)
    except BaseException:
        run.finish("failed")
        raise
    return run.finish()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-comparison", action="store_true", help="Explicitly start 15 long training runs")
    parser.add_argument("--output-root")
    parser.add_argument("--steps", type=int, default=1500000)
    args = parser.parse_args(argv)
    if args.train_comparison:
        print(run_multiseed_comparison(args.output_root, steps=args.steps).artifacts)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
