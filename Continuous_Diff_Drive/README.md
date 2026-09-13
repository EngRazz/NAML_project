# Continuous robot navigation

A circular differential-drive robot navigates a room with static rectangular obstacles using DDPG, TD3 or SAC. See the [root README](../README.md#install) for installation.

## Environment

The default experimental room is 10×10 m, with start `(1,1)` and goal `(8.5,8.5)`. Observations have 18 values: 16 body-relative LiDAR distances, Euclidean goal distance, and relative goal angle. The obstacle map and global robot pose are not observed; these observations need not uniquely identify the simulator state.

Actions are linear velocity in `[-0.5,1.0]` m/s and angular velocity in `[-π,π]` rad/s. With timestep 0.1 s, heading is updated before position. Radius is 0.3 m; the goal is reached at distance below `radius + 0.2` m. Collision terminates with −100; goal arrival without collision terminates with +100.

Other transitions use the active shaped reward:

- Progress weight 10 when the goal-direction LiDAR ray is clear at 0.8 m, otherwise 1.
- Danger penalty −0.5 times the **squared** normalized deficit below 0.5 m minimum clearance.
- Front danger penalty −0.5 times the **squared** normalized deficit below 1 m front clearance.
- Orientation term `0.3 * (1 - abs(relative_angle)/π)`.
- Step cost −0.5 and angular-speed cost `−0.02*abs(angular_velocity)`.
- Danger-reduction shaping is disabled (coefficient zero).

The episode limit is an external cutoff. It sets `truncated`, retains the bootstrap and **adds no timeout penalty**. True goal/collision termination takes precedence. This corrected reward is versioned `navigation-cutoff-v2`; old training figures have not been recalculated to match it.

The mode called `curriculum` samples a fixed mixture: 40% perturbed detour maps, 40% walls with gaps, and 20% random blocks. Clearance and discretized path checks filter proposals; it is not progressively increasing difficulty. Evaluation maps drawn from these same families are not an out-of-distribution benchmark.

## Algorithms and individual-run defaults

| Setting | DDPG | TD3 | SAC |
|---|---:|---:|---:|
| Episodes | 5,000 | 5,000 | 3,000 |
| Cutoff | 1,000 | 1,000 | 1,200 |
| Hidden width, two layers | 256 | 256 | 64 |
| Actor learning rate | 0.0001 | 0.0001 | 0.0003 |
| Critic learning rate | 0.001 | 0.001 | 0.0003 |
| Discount | 0.995 | 0.99 | 0.99 |
| Target interpolation τ | 0.001 | 0.005 | 0.005 |
| Replay capacity | 300,000 | 300,000 | 100,000 |
| Critic updates per sampled environment step after warmup | 1 | 2 | 2 |
| Actor update interval, in critic updates | 1 | 2 | 1 |

All use batch size 256, 5,000 random-action warmup steps and actor/critic gradient clipping at norm 1. Inputs are normalized consistently during action selection and replay updates: LiDAR/range, distance/room diagonal, bearing/π. Raw observations are stored in replay; tensor normalization happens on the learner's device.

DDPG uses one critic, a deterministic actor and target networks. TD3 uses twin critics, minimum target values, clipped Gaussian target-policy smoothing (standard deviation 0.2, clipping 0.5), and delayed actor/target updates. Both use OU behavior noise with standard deviation 0.2, clipped to 0.5 in physical action units. SAC uses a tanh-squashed Gaussian policy, twin critics, and automatic temperature tuning (initial alpha 0.2, learning rate 0.0003, target entropy −2).

Action selection does not track gradients. During actor optimization the critic is frozen, while derivatives through its action input remain available. Learning logs store episode summaries rather than every update indefinitely. These mechanisms do not establish convergence of neural training.

## Notebook workflow

1. Open `train_ddpg.ipynb`, `train_td3.ipynb` or `train_sac.ipynb` in the project kernel. Each notebook lists its environment settings and all agent constructor parameters explicitly in `ENV_KWARGS` and `AGENT_KWARGS`. Edit these independently for each algorithm; the notebooks do not import training defaults from `experiments.py`. The optional comparison runner keeps its own documented defaults there.
2. Set `RUN_TRAINING=True` to start. Leave `WARM_START_CHECKPOINT=None` for fresh training. All notebooks otherwise skip training without loading old models.
3. Keep `LOG_EVERY=500` for a permanent summary of each 500-episode block above the updating total progress bar. Set `RECORD_EVERY=500` for separate bounded greedy demonstrations named `episode_0.mp4`, `episode_500.mp4`, etc.; `None` disables recording. The DDPG notebook currently has recording enabled. Inspect the returned run directory: metrics, checkpoint and plots are portable. `config.json` is an automatic record of the effective settings, not a file you need to edit before training.
4. Open the corresponding `eval_*.ipynb`. It selects the latest **completed new run**, or accepts an explicit `CHECKPOINT_PATH`. Fixed-map evaluation runs once; random-layout evaluation defaults to ten seeded episodes.
5. Use `compare_ddpg_sac_td3.ipynb` for a shared serialized scenario set. Missing/incompatible models are listed as unavailable and an incomplete comparison is clearly marked. No implicit training occurs.

Notebooks work from this directory or the repository root. Original outputs and the completed DDPG notebook before the display changes are preserved in [the archive](../archive/README.md); updated notebooks start with cleared outputs. A video index is zero-based and names the training episode after which its greedy demonstration runs. CSV episode values and the overall progress bar count completed episodes from 1. No saved historical statistic should be interpreted as a result from the current cells.

## Checkpoints

The old SAC checkpoint is incompatible with current normalized inputs; delete it yourself when ready and retrain. Existing checkpoint files are not altered by these fixes.

New checkpoints record architecture, environment configuration, observation scaling, action bounds, reward version and source provenance. Use `agent.load_checkpoint(path, mode="evaluation")` or `mode="warm_start"`. Warm starts retain actor/critic weights, copy online weights to targets, and reset replay, optimizer state, OU noise, temperature and counters. They do not resume the old run.

Metadata-free checkpoints are rejected by default. The audited old DDPG checkpoint can be evaluated with `allow_legacy_normalized=True` after explicitly asserting that its scaling matches; this exception cannot be used for SAC or for warm starts. Reward/representation-incompatible new checkpoints are rejected. The TD3 model corresponding to historical notebook outputs is absent locally.

## Optional controlled comparison

```bash
python -m Continuous_Diff_Drive.experiments --help
python -m Continuous_Diff_Drive.experiments --train-comparison
```

The second command explicitly launches **15 long runs**: seeds 0–4 for each algorithm, 1.5 million environment steps per run, and a common 1,000-step cutoff. It evaluates the historical fixed map plus 50 mixture layouts generated with seeds 10000–10049. Every agent uses the same saved geometries. `--steps` and `--output-root` allow explicit overrides.

The run manifest records actual environment steps, critic updates and actor updates. Aggregate files report mean and standard deviation across training seeds, separately from variation between scenarios. Different widths and optimization budgets remain documented; equal sample budgets do not make all other settings identical. This full comparison was not executed during the fixes.

## Result interpretation

Success and collision frequencies, final distance, return and length describe navigation. Actor/critic losses are optimization diagnostics. Plots retain the reward band labelled rolling mean ± two rolling standard deviations within one run. `outcomes.png` shows rolling success rate and, for SAC, entropy temperature. Binary success and collision indicators are omitted. `final_goal_distance.png` gives distance a separate square plot, with a 0–10 m vertical axis, a horizontal axis spanning the training budget, faint individual values and a clear rolling mean; values above 10 m are marked at the upper edge. Collision and success flags remain in `metrics.csv`. Optional videos retain native resolution up to 800 pixels wide, with filtered resizing for larger frames, and are capped at 300 frames; numerical evaluation continues until the actual episode boundary.
