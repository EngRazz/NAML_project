# Shared experiment workflows

`runtime.py` supplies unique artifact directories, incremental CSV logging, seeding, structured results, plot generation and bounded video recording. `tabular.py` supplies the common Q-learning/SARSA loop. `discrete.py` supplies checkpoint selection and environment restoration for the [flag](../Grid_flag/README.md#notebooks) and [cliff](../Grid_cliff/README.md#notebooks) notebook pairs. `continuous.py` supplies the checkpoint and training lifecycle shared by the neural agents; their algorithm-specific equations remain in the continuous section.

## APIs

- `agent.train(num_episodes, seed=..., output_root=..., max_env_steps=..., record_every=..., plot=True, show_progress=True, log_every=500)` returns `RunResult`. For tabular agents, pass `algorithm="qlearning"` or `"sarsa"`. Use `num_episodes=None` with a positive step budget to train by samples alone. Set the continuous seed when constructing the agent as well, so initialization matches.
- `agent.evaluate(episodes=1, seed=1234, output_root=..., record=False)` returns `RunResult` with episode records and saved paths. It runs a separate environment and does not alter the learner's RNG stream.
- `RunResult.episodes` contains structured dictionaries; `RunResult.artifacts` lists output paths.
- `save_checkpoint(path)` and `load_checkpoint(path, mode="evaluation"|"warm_start")` are explicit. Table checkpoints use NPZ; neural checkpoints use PyTorch tensors and primitive metadata. No exact resume is claimed.

Relative output roots are anchored at the repository root. Every run creates its own `artifacts/<environment>/<algorithm>/<UTC>_seed<seed>_<suffix>/` directory. A run starts with status `running`; completion is recorded only after its outputs succeed. Exceptions retain incremental logs and an `interrupted` checkpoint where possible. Checkpoints are saved before plotting, so optional figure errors do not discard learned weights.

`metrics.csv` includes episode and seed, return, length, task success, collision, terminal/truncation flags, termination reason, and applicable fields such as final distance, flags collected, epsilon, absolute TD error, losses, entropy temperature and update counts. Empty fields mean not applicable or no update yet. Configuration metadata records the initialization settings and code version, including whether the checkout was dirty.

`config.json` is an automatically saved snapshot, not an input file. Training notebooks expose their environment and agent parameters directly, independently for each algorithm. The snapshot records the effective values for later interpretation and checkpoint selection, even if the notebook is subsequently edited.

The discrete evaluation helper selects only completed new-format training runs for the requested environment and algorithm. Explicit checkpoint paths take precedence, and relative paths resolve from the repository root. Loading restores the checkpoint's task configuration and validates its algorithm, environment and state encoding. External cutoff overrides are allowed; incompatible task changes are rejected. Evaluation notebooks save checkpoint provenance alongside results and report missing models explicitly.

## Live training status

Training shows a progress bar by default in notebooks and terminals, including the completed episode count (for example, `100/500`), elapsed time and ETA. The latest episode reward and the average reward/success rate over up to 100 recent episodes appear alongside it. Step-only training budgets display completed environment steps instead. Set `show_progress=False` to disable the display; CSV logging remains enabled.

Every `log_every=500` completed episodes, a permanent summary appears above the live total bar. It reports block means for reward, length, final goal distance and available losses, plus the block success rate; tabular summaries include epsilon and TD error. The final partial block is summarized when training ends. Episode labels in summaries start at zero (the first block is 0–499); the total bar and CSV `episode` column count completed episodes from 1. The notebook display uses native HTML and requires no `ipywidgets` installation.

If you updated the code while a notebook was running, its existing training call keeps the previously loaded implementation. Restart the kernel before the next run to load the new progress display; interrupting an active run is not necessary.

## Video and randomness

Training videos are named `episode_0.mp4`, `episode_500.mp4`, `episode_1000.mp4`, and so on when `record_every=500`. The filename uses the zero-based training episode index: episode 0 is the first training episode, and its demonstration is recorded after that episode's updates. Thus video index 500 corresponds to completed-episode count 501 in the CSV. Every new run follows this convention. Standalone evaluation records `episode_0.mp4`; shared comparisons use algorithm names to distinguish their videos. `video_sort_key` orders videos numerically in notebooks.

Separate greedy demonstrations preserve the learner's Python, NumPy and PyTorch RNG state. Recordings keep the native resolution up to 800 pixels wide; larger frames are shrunk with area filtering while preserving the aspect ratio. Cliff videos therefore retain their native 800×250 image and square 50×50 cells. The earlier 320-pixel pixel-skipping method could erase thin grid lines. Existing evaluation videos can be regenerated from a checkpoint to benefit from the fix; restart the notebook kernel and rerun evaluation with `RECORD=True`. No retraining is needed. The low-level `recording_env(..., max_width=800)` option can set a different width limit.

`LimitedRecordVideo` still caps each recording at 300 frames, including the reset frame. Stopping recording never stops the measured rollout. Wrappers and environments are closed in `finally` blocks.

Seeds cover Python, NumPy, PyTorch, environment resets and action spaces. Use the same seed, device and dependency versions when comparing repeated runs; bitwise agreement across different devices or dependency versions is not guaranteed.

## Plots

`training_curves.png` retains reward, episode length, and the relevant learning diagnostics. `outcomes.png` shows rolling success rate, plus flags collected or SAC temperature when applicable. Binary success and collision indicators are omitted; their values remain in the CSV. Continuous runs also save `final_goal_distance.png`: a square plot with a 0–10 m vertical axis, faint per-episode values, and a rolling mean over up to 100 episodes. Markers at the top identify values above the displayed range. Its horizontal axis spans the configured episode budget (0–5000 for 5,000 episodes), or the observed count for step-only runs. `plot_outcomes(rows, directory, title, num_episodes=...)` can redraw these outcome plots from saved metrics without changing `training_curves.png` or retraining.
