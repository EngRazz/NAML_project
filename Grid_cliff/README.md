# Cliff walking

The active map is a canonical 5×16 grid. Start is `(4,0)`, goal is `(4,15)`, and the intervening fourteen bottom-row cells are cliffs. Observation is `16*row + column`; actions are `0=up`, `1=right`, `2=down`, `3=left`.

Ordinary steps, including goal arrival, give −1. Falling gives −100 and returns the agent to the start **without terminating**. The 200-step external cutoff applies even on a fall. Goal arrival terminates and takes precedence over truncation.

Both algorithms use learning rate 0.1, discount 0.99, 5,000 episodes and seed 0 by default. Epsilon declines from 1 to 0.1 during the first 3,000 episodes and stays at 0.1 thereafter. Both bootstrap on truncation. SARSA carries its sampled backup action into execution; Q-learning backs up the greedy maximum.

With continuing exploration, SARSA accounts for future accidental cliff moves and can favor a longer, safer route. Q-learning can favor the shortest cliff-adjacent greedy route while suffering falls during exploration. This explains the difference in **training return** observed in the selected runs described below; it does not imply a greedy-evaluation advantage for SARSA.

## Notebooks

After [installation](../README.md#install), open [train_grid_cliff.ipynb](train_grid_cliff.ipynb). Imports work with the kernel in the repository root or `Grid_cliff/`.

1. Review the map, cutoff, budgets, seed and separate Q-learning/SARSA parameter dictionaries. `ALGORITHMS=("qlearning", "sarsa")` runs both independently; use `("sarsa",)` or `("qlearning",)` for just one.
2. Inspect `RUN_TRAINING` before running the cells: `True` starts fresh training; `False` skips it and loads no checkpoint. The episode budget applies to each algorithm; changing it does not change the explicit exploration schedule. `LOG_EVERY=500` keeps summaries above the live total bar. `RECORD_EVERY=500` records after episodes 0, 500, etc.; use `None` to disable training videos.
3. Inspect individual curves and the joint return/success/length chart. The comparison is created only when both algorithms finish in this execution. Its unique `artifacts/cliff/comparison/` run records both source run paths and `COMPARISON_WINDOW` (100 by default).
4. Open [eval_grid_cliff.ipynb](eval_grid_cliff.ipynb). Set explicit `CHECKPOINT_PATHS`, or leave them `None` to use the latest completed training run per algorithm. Relative paths start at the repository root. Missing or incompatible checkpoints are reported, and missing explicit paths never trigger fallback to another model.

Evaluation restores each checkpoint's map and cutoff. `ENV_OVERRIDES` can change the external cutoff, but changes to the learned task are rejected. The table shows return, steps, goal completion and end reason: a successful cliff route has a negative return. One greedy episode is the default. With `layout="random"`, `map_seed` determines the saved fixed map; changing the evaluation seed does not create a new map. Inspect the restored settings before comparing checkpoints from different experiments.

Set `RECORD=True` in evaluation to save `episode_0.mp4` for each available algorithm in separate evaluation directories. The training notebook identifies the comparison image for report export. All parameters remain editable in the notebooks; `config.json` is the automatic snapshot saved with each run.

## Command line

From the project root after [installation](../README.md#install):

```bash
python -m Grid_cliff.run_grid_cliff --episodes 5000 --seed 0
```

Add `--record` for optional demonstrations every 500 episodes and one recorded greedy evaluation. The deterministic fixed map is evaluated once per algorithm; duplicate identical episodes are not independent observations.

New cliff recordings keep the native 800×250 resolution, preserving square 50×50-pixel cells and all grid lines. Restart the notebook kernel before recording with updated code; older 320-pixel videos require a new evaluation recording from the saved checkpoint.

Training recordings use zero-based names: `episode_0.mp4`, `episode_500.mp4`, etc., each recorded after that training episode. A summary of every 500 completed episodes stays above the updating total progress bar. The outcome plot shows rolling success rate without binary success/collision indicators.

Training returns a trained `TabularAgent` and `RunResult` through `train_q_learning()` or `train_sarsa()`. Run results contain episode dictionaries and artifact paths. The runner saves the joint return/success/length figure in a unique `artifacts/cliff/comparison/` run; individual runs contain numerical tables, CSV metrics and plots.

See [shared workflows](../rl_project/README.md) for checkpoint and output details. The [report](../Report/README.md) includes the 13 September 2026 seed-0 training comparison and saved greedy evaluations. SARSA has fewer final-block training falls (2 versus Q-learning's 102 over 250 episodes), but takes a longer greedy route (23 versus 17 steps). The [figure inventory](../Report/figures/README.md) records the source runs and distinguishes these training and evaluation results.
