# Reinforcement learning in discrete and continuous environments

A NAML project comparing tabular **Q-learning and SARSA** in flag collection and cliff walking, and **DDPG, TD3 and SAC** for a differential-drive robot navigating with LiDAR.

The maintained implementation includes corrected episode boundaries, normalized neural-network inputs, reproducible run metadata, bounded optional videos, and shared training/evaluation utilities. The report includes completed corrected-code runs for both grid algorithms in both environments and for DDPG, TD3 and SAC, with images and analysis tied to their saved metrics. Grid results include saved greedy evaluations. The controlled comparison across independent training seeds remains pending.

## Project guide

| Section | Contents |
|---|---|
| [Flag collection](Grid_flag/README.md) | Five flags on a 10×10 grid; Q-learning and SARSA. |
| [Cliff walking](Grid_cliff/README.md) | Canonical 5×16 cliff task and the on-policy/off-policy comparison. |
| [Continuous navigation](Continuous_Diff_Drive/README.md) | Robot environment, DDPG/TD3/SAC, training and evaluation notebooks. |
| [Report](Report/README.md) | Revised LaTeX report, portable figures and result provenance. |
| [Shared utilities](rl_project/README.md) | Training results, artifact layout, seeding and video limits. |
| [Historical archive](archive/README.md) | Original notebook outputs, retained before the fixes. |

## Install

Use **Python 3.11**. From the repository root:

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements pin the direct dependency versions used for local validation. On Windows, activate with `venv\Scripts\activate`. For notebooks, also install `requirements-notebooks.txt`, launch `python -m jupyterlab`, and select the project environment as the kernel. The standard-library `typing` module needs no separate installation.

## Run

Commands run from the repository root. Scripts use module imports and do not start training when imported.

```bash
python -m Grid_flag.run_grid_flag --help
python -m Grid_cliff.run_grid_cliff --help
```

For a small discrete run:

```bash
python -m Grid_flag.run_grid_flag --episodes 10 --seed 0
```

The default episode count is 5,000. Add `--record` for optional greedy demonstration videos.

Each environment also has separate training and evaluation notebooks:

| Environment | Training | Evaluation |
|---|---|---|
| Flag collection | [train_grid_flag.ipynb](Grid_flag/train_grid_flag.ipynb) | [eval_grid_flag.ipynb](Grid_flag/eval_grid_flag.ipynb) |
| Cliff walking | [train_grid_cliff.ipynb](Grid_cliff/train_grid_cliff.ipynb) | [eval_grid_cliff.ipynb](Grid_cliff/eval_grid_cliff.ipynb) |
| Continuous navigation | Three `train_*.ipynb` notebooks | Three `eval_*.ipynb` notebooks; see the [continuous guide](Continuous_Diff_Drive/README.md) |

Inspect `RUN_TRAINING` in the discrete notebooks before running all cells; it is currently enabled in the saved notebooks. Set it to `False` to skip training. `ALGORITHMS` selects Q-learning, SARSA, or both, with an editable parameter dictionary and separate outputs for each. Evaluation restores the selected checkpoint's environment settings, shows task completion and return, and optionally records with `RECORD=True`. Imports work from the repository root or the notebook's directory.

Continuous notebooks likewise expose their environment and agent parameters directly. Inspect the configuration and `RUN_TRAINING` before running all cells; the DDPG notebook currently has training enabled. No notebook silently loads old checkpoints. See the [continuous guide](Continuous_Diff_Drive/README.md) for the optional five-seed comparison.

## Outputs and checkpoint changes

New generated files live in:

```text
artifacts/<environment>/<algorithm>/<UTC timestamp>_seed<seed>_<unique suffix>/
  config.json
  metrics.csv
  checkpoints/model.pt       # continuous; model.npz for tables
  plots/training_curves.png
  plots/outcomes.png
  plots/final_goal_distance.png  # continuous runs
  videos/training/episode_0.mp4  # then episode_500.mp4, etc., if enabled
  videos/evaluation/episode_0.mp4
```

Directories are created before use. Separate runs and algorithms cannot overwrite one another. Artifacts are ignored by Git; explicitly export reviewed figures to `Report/figures/`. Old checkpoints and media remain in their existing locations and are not migrated or deleted automatically.

`config.json` automatically records the settings used for that run. Set parameters in the notebook; DDPG, TD3 and SAC have independent configurations. Training retains a summary every 500 completed episodes above one updating total progress bar. Video filenames and summary labels start at episode 0; the CSV and progress bar count completed episodes from 1. Outcome plots show rolling success rate without binary success/collision indicators, plus a separate square 0–10 m goal-distance plot for continuous runs.

**Retrain SAC with normalized inputs.** Its old checkpoint has no compatible preprocessing metadata. New checkpoints support evaluation or a warm start, which retains network weights but resets replay, optimizers, exploration, temperature and counters. They do not provide exact training resumption. The verified historical DDPG checkpoint can be evaluated only through explicit legacy-normalized opt-in.

## Modeling and interpreting results

- Time limits are external sampling cutoffs: `terminated` or `truncated` ends a rollout, but only `terminated` removes the value bootstrap.
- The robot has no extra timeout penalty. Goal and collision rewards remain +100 and −100.
- SARSA executes the action used in its backup. Persistent cliff exploration can favor a safer route; this expected training-return effect is distinct from greedy evaluation.
- Success means completing the task, even when shaped return is negative. A rolling standard-deviation band is not a confidence interval across independent training runs.
- Historical notebook outputs are archived. The report distinguishes current grid training and greedy evaluation, current continuous training runs, and the historical robot fixed-map benchmark.

## Checking runs and the report

Automated test and notebook/report validation scripts are not included in the current checkout. For a quick workflow check, use a short discrete run as shown above and inspect its saved metrics and plots. Short runs check execution; they do not establish learning performance.

For report compilation and layout verification, upload `main_new.tex` and `figures/` to Overleaf as described in the [report guide](Report/README.md).
