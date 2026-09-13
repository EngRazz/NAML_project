# Flag collection

The active environment is a 10×10 grid. The agent starts at `(5,5)` and collects fixed flags at `(1,2)`, `(3,7)`, `(6,1)`, `(7,8)` and `(9,4)`.

- Observation: agent row/column plus five remaining-flag bits.
- Actions: `0=stay`, `1=up`, `2=down`, `3=left`, `4=right`.
- Reward: +10 for an uncollected flag, otherwise −0.5.
- Final-flag collection terminates immediately. The 100-step external cutoff truncates only if the task has not terminated.

Q-learning uses a greedy next-state maximum. SARSA selects its next action once, uses it in the update and executes it next. Both bootstrap on truncation and suppress the bootstrap on termination. Both use learning rate 0.1, discount 0.95, 5,000 episodes by default and epsilon declining from 1 to 0.05 over 5,000 episodes. Seed 0 is the default; shortened smoke runs retain the same exploration schedule.

## Notebooks

After [installation](../README.md#install), open [train_grid_flag.ipynb](train_grid_flag.ipynb). Imports work with the kernel in the repository root or `Grid_flag/`.

1. Review the environment, budgets, seed and the separate Q-learning/SARSA parameter dictionaries in the configuration cell. `ALGORITHMS=("qlearning", "sarsa")` runs both independently; use `("sarsa",)` or `("qlearning",)` for just one.
2. Inspect `RUN_TRAINING` before running the cells: `True` starts fresh training; `False` skips it, creates no run and loads no checkpoint. Each fresh training run saves its checkpoint, metrics, plots and optional videos in its own directory. `NUM_EPISODES=5000` is the budget **per algorithm**; changing it leaves the explicit exploration schedule unchanged.
3. Inspect the displayed plots and recordings. `LOG_EVERY=500` controls persistent summaries above the live total bar; `RECORD_EVERY=500` records after episodes 0, 500, etc. Set `RECORD_EVERY=None` to disable training videos.
4. Open [eval_grid_flag.ipynb](eval_grid_flag.ipynb). Set explicit `CHECKPOINT_PATHS`, or leave them `None` to use the latest completed training run for each algorithm. Relative checkpoint paths start at the repository root. Missing or incompatible models are reported; a missing explicit path never selects another checkpoint silently.

Evaluation restores the saved grid, flags, reward and cutoff. `ENV_OVERRIDES` can change the external cutoff, but changes to the learned task are rejected. The greedy evaluation table shows return, steps, goal completion, end reason and flags collected. One episode is the default because this fixed task is deterministic; repeated identical episodes are not independent evidence. Set `RECORD=True` for presentation videos named `episode_0.mp4`, saved in separate evaluation runs with checkpoint provenance.

All parameters remain in the notebook. `config.json` is an automatic snapshot of a run's effective settings. For report export, the training notebook identifies the Q-learning and SARSA figure names; update the report analysis only after reviewing the new curves and saved metrics.

## Command line

From the project root after [installation](../README.md#install):

```bash
python -m Grid_flag.run_grid_flag --episodes 5000 --seed 0
python -m Grid_flag.run_grid_flag --episodes 10 --output-root artifacts/smoke
```

Add `--record` for separate greedy demonstrations every 500 episodes and one greedy evaluation recording. Videos are capped; evaluation length is unaffected. Outputs live under `artifacts/flag/qlearning/` and `artifacts/flag/sarsa/` in unique run directories.

Training recordings use zero-based names: `episode_0.mp4`, `episode_500.mp4`, etc., each recorded after that training episode. A summary of every 500 completed episodes stays above the updating total progress bar. The outcome plot retains rolling success rate and flags collected, with no binary success/collision indicators.

## Python API

```python
from Grid_flag.grid_flag_env import GridFlagEnv
from Grid_flag.grid_flag_agent import GridFlagAgent
from Grid_flag.run_grid_flag import ENV_KWARGS

agent = GridFlagAgent(GridFlagEnv(**ENV_KWARGS), seed=0)
result = agent.train(5000, algorithm="sarsa", output_root="artifacts")
evaluation = agent.evaluate(episodes=1)
print(result.artifacts, evaluation.episodes)
```

`train()` is now the single maintained training API. The old recorded/non-recorded duplicated methods were removed. `update_SARSA(..., next_action)` requires the explicit backup action. Checkpoints are numeric NPZ files with state-encoding metadata; older tables cannot be silently treated as the corrected task.

The 13 September 2026 seed-0 training curves and greedy evaluations are included in the [report](../Report/README.md). Both final policies collect all five flags in 27 steps with return 39.0; both complete every final 250-episode training rollout. The [report figure guide](../Report/figures/README.md) identifies the selected runs and copied figures. These are fixed-layout results from one training seed per algorithm.
