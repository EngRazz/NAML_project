# Report figure inventory

Upload this complete directory together with `main.tex` to Overleaf. The revised report includes **fourteen current training figures** from completed runs of the corrected implementation: five discrete and nine continuous. Four older DDPG/TD3 images remain preserved separately under their original names; they are no longer used as the report's continuous training figures.

## Selected continuous runs

The figures use the latest completed training run for each algorithm. TD3's latest run follows hyperparameter tuning; these selections therefore do not constitute a comparison with equal tuning effort. DDPG and SAC were trained on 10 September 2026 and TD3 on 13 September 2026. All three use seed 0, physical observation normalization (`physical-observation-v1`), the corrected reward (`navigation-cutoff-v2`), external cutoffs that retain bootstrapping, and no warm-start checkpoint. Their stored Git metadata marks the working tree as modified, so the commit hash alone is not a complete source snapshot.

| Algorithm | Source run under `artifacts/continuous/` | Completed episodes |
|---|---|---:|
| DDPG | [ddpg/20260910T154140101411Z_seed0_3d2d5e39](../../artifacts/continuous/ddpg/20260910T154140101411Z_seed0_3d2d5e39) | 5,000 |
| TD3 | [td3/20260913T102310246485Z_seed0_5cf3e17f](../../artifacts/continuous/td3/20260913T102310246485Z_seed0_5cf3e17f) | 5,000 |
| SAC | [sac/20260910T181357962640Z_seed0_88a0e82c](../../artifacts/continuous/sac/20260910T181357962640Z_seed0_88a0e82c) | 3,000 |

## Included continuous figures

Every report image below is an exact copy of the corresponding PNG under its selected run's `plots/` directory. Original training artifacts and historical report figures were not overwritten.

| Report filename | Contents | Source filename |
|---|---|---|
| [ddpg_training_curves.png](ddpg_training_curves.png) | Return, episode length, critic and actor losses | `training_curves.png` |
| [ddpg_outcomes.png](ddpg_outcomes.png) | Rolling success rate | `outcomes.png` |
| [ddpg_final_goal_distance.png](ddpg_final_goal_distance.png) | Final goal distance and rolling mean; square 0–10 m plot | `final_goal_distance.png` |
| [td3_training_curves.png](td3_training_curves.png) | Return, episode length, critic and actor losses | `training_curves.png` |
| [td3_outcomes.png](td3_outcomes.png) | Rolling success rate | `outcomes.png` |
| [td3_final_goal_distance.png](td3_final_goal_distance.png) | Final goal distance and rolling mean; square 0–10 m plot | `final_goal_distance.png` |
| [sac_training_curves.png](sac_training_curves.png) | Return, episode length, critic and actor losses | `training_curves.png` |
| [sac_outcomes.png](sac_outcomes.png) | Rolling success rate and episode-averaged temperature | `outcomes.png` |
| [sac_final_goal_distance.png](sac_final_goal_distance.png) | Final goal distance and rolling mean; square 0–10 m plot | `final_goal_distance.png` |

## Numerical interpretation

- Training outcomes include exploration and changing policies. They are not greedy evaluations of only the final checkpoint.
- Plot smoothing uses the preceding **up to 100 episodes**. All three reward bands are rolling mean ± two rolling population standard deviations within one run, not confidence intervals across training seeds.
- Success plots show the rolling fraction of episodes that reach the goal. Binary success and collision indicators are omitted. `outcomes.png` contains no goal-distance panel; distance is in the separate figure.
- Losses are averaged over the updates performed in each episode. Missing early losses mean no update occurred. SAC temperature is an episode average over updates, or its current value before learning begins; it is not a directly measured entropy curve.
- Distances include both successes and failures. The 10 m upper axis is a display limit, not the maximum possible distance. Top markers represent 25 DDPG, 44 TD3 and 15 SAC episodes above 10 m. Neither means nor reported statistics clip those values.
- CSV and plot episode numbers count completed episodes from 1. Recording filenames use zero-based indices; `episode_500.mp4` follows completed-episode count 501.

The report's numerical summaries below use the **last 250 episodes**, so they differ from the 100-episode endpoints drawn in the figures. They were recomputed directly from each selected run's `metrics.csv`.

| Algorithm | Last episode | Mean return | Success | Mean length | Mean distance (m) | Collisions | Cutoffs |
|---|---:|---:|---:|---:|---:|---:|---:|
| DDPG | 5,000 | 118.49 | 85.2% | 169.04 | 1.29 | 28 / 250 | 9 / 250 |
| TD3 | 5,000 | 128.22 | 88.8% | 194.29 | 1.00 | 13 / 250 | 15 / 250 |
| SAC | 3,000 | 106.71 | 83.2% | 214.08 | 1.31 | 26 / 250 | 16 / 250 |

The final 100-episode success rates drawn in the figures are 82% for DDPG, 89% for TD3 and 79% for SAC. Tuned TD3's final 100-episode mean return is 130.54, length 192.68 and distance 0.91 m. Its final 250-episode cutoff fraction is 6.0%. The run improves substantially but is not monotonic: the window ending at episode 3,000 reaches 93.6% success and return 141.79, whereas episodes 4,501–4,750 fall to 78.8% success before recovery. Critic loss remains nonzero; its maximum episode average is 142.35 at episode 40, rather than the earlier run's extreme mid-training spike.

### TD3 tuning and current evaluation

Relative to [the initial TD3 run](../../artifacts/continuous/td3/20260910T205530372576Z_seed0_2e48e740), the saved agent settings differ only in `critic_lr` (`1e-3` → `3e-4`) and `updates_per_step` (2 → 1). The task, seed, 5,000-episode budget and other agent settings were retained, and training restarted from scratch. Final-block success increased from 53.2% to 88.8%; the joint change does not isolate either parameter's contribution. The earlier run's plots and metrics remain in their original artifact directory.

The selected TD3 run collected **1,301,110 training transitions**, with **1,296,111 critic updates** and **648,055 actor updates**. With `policy_delay=2`, the actor and targets now update once per two environment steps after warmup. Different episode lengths and update ratios prevent treating equal episode budgets as equal optimization budgets.

Both current evaluation configurations below explicitly identify the selected tuned TD3 checkpoint, use its saved dynamics/reward, and disable exploration noise. The fixed-map episode uses seed 1234; the sampled-layout run uses seeds 1234–1243. Means include the failed episode.

| Evaluation run under `artifacts/continuous/td3/` | Successes | Mean return | Mean steps | Mean final distance |
|---|---:|---:|---:|---:|
| [20260913T131252423944Z_seed1234_be0e9c73](../../artifacts/continuous/td3/20260913T131252423944Z_seed1234_be0e9c73) — fixed map | 1/1 | 167.09 | 127 | 0.432 m |
| [20260913T131254673271Z_seed1234_dd1e83d8](../../artifacts/continuous/td3/20260913T131254673271Z_seed1234_dd1e83d8) — sampled layouts | 9/10 | 135.17 | 206.5 | 1.04 m |

The sampled check has no collisions and one 1,000-step cutoff, ending 6.36 m from the goal. These small-sample greedy results support the improvement while retaining a clear limitation; they are not the historical three-agent benchmark or the full controlled comparison.

## Historical figures preserved but not displayed as current results

| Preserved file | Original provenance |
|---|---|
| `ddpg_diff_drive_training_curves.png` | Original `Continuous_Diff_Drive/images/ddpg_diff_drive_training_curves.png`. |
| `ddpg_diff_drive_training_curves2.png` | Original `Continuous_Diff_Drive/images/ddpg_diff_drive_training_curves2.png`. |
| `td3_diff_drive_training_curves.png` | `archive/notebooks/Continuous_Diff_Drive/train_td3.ipynb`, cell 15, output 0. |
| `td3_diff_drive_training_curves2.png` | Same archived notebook, cell 15, output 1. |

Archived notebook cell/output indices are zero-based. Their content hashes remain in `archive/notebooks/manifest.json`. The historical three-agent fixed-map benchmark in the report still comes from the archived comparison notebook; it does not evaluate the new checkpoints. Current DDPG and tuned TD3 evaluation logs are identified separately in the report.

## Selected discrete runs

The four runs below completed on **13 September 2026**, each with 5,000 episodes and seed 0. They are the latest completed training runs for their environment and algorithm, selected by timestamp rather than return. Saved metadata identifies `complete-tabular-v2`, external cutoffs with bootstrapping, learning rate 0.1, and a modified working tree. The flag runs use discount 0.95 and a 100-step cutoff; the cliff runs use discount 0.99, a 200-step cutoff and the canonical map. The notebooks' configuration cells agree with these saved settings.

| Environment | Algorithm | Source run under `artifacts/` | Training transitions / tabular updates |
|---|---|---|---:|
| Flag | Q-learning | [flag/qlearning/20260913T085943978379Z_seed0_85399cae](../../artifacts/flag/qlearning/20260913T085943978379Z_seed0_85399cae) | 338,791 |
| Flag | SARSA | [flag/sarsa/20260913T085951982917Z_seed0_d34cfb8b](../../artifacts/flag/sarsa/20260913T085951982917Z_seed0_d34cfb8b) | 348,260 |
| Cliff | Q-learning | [cliff/qlearning/20260913T084122019817Z_seed0_8fb8d21b](../../artifacts/cliff/qlearning/20260913T084122019817Z_seed0_8fb8d21b) | 357,234 |
| Cliff | SARSA | [cliff/sarsa/20260913T084132538934Z_seed0_52f083d2](../../artifacts/cliff/sarsa/20260913T084132538934Z_seed0_52f083d2) | 333,769 |

The joint cliff image comes from [cliff/comparison/20260913T084141139368Z_seed0_c520a674](../../artifacts/cliff/comparison/20260913T084141139368Z_seed0_c520a674). Its completed configuration names precisely the two cliff training runs above and a 100-episode smoothing window. Its own CSV has no episode rows; comparison statistics come from the two training CSVs.

| Included report image | Source within the selected run |
|---|---|
| [GridFlag_training_curves.png](GridFlag_training_curves.png) | Flag Q-learning: `plots/training_curves.png` |
| [GridFlag_training_curves_SARSA.png](GridFlag_training_curves_SARSA.png) | Flag SARSA: `plots/training_curves.png` |
| [GridFlag_outcomes.png](GridFlag_outcomes.png) | Flag Q-learning: `plots/outcomes.png` |
| [GridFlag_outcomes_SARSA.png](GridFlag_outcomes_SARSA.png) | Flag SARSA: `plots/outcomes.png` |
| [cliff_qlearning_vs_sarsa.png](cliff_qlearning_vs_sarsa.png) | Cliff comparison: `plots/cliff_qlearning_vs_sarsa.png` |

All five images are exact copies. The individual cliff training and outcome images remain in the original runs; the report uses their joint comparison to avoid repeating the return, success and length curves.

### Discrete numerical interpretation

The training figures show epsilon-greedy behavior, not greedy evaluation. Flag reward bands show rolling mean ± two rolling population standard deviations; such a band can extend beyond feasible returns and is not a confidence interval. Outcome upper panels show rolling completion fractions; the flag count below is per episode. The cliff comparison contains rolling means without bands. Plot windows contain up to 100 episodes; the following table instead summarizes **episodes 4,751–5,000**, recomputed directly from each training CSV.

| Environment | Algorithm | Mean return | Mean length | Completion | Total cliff falls |
|---|---|---:|---:|---:|---:|
| Flag | Q-learning | 37.628 | 29.744 | 250/250 | — |
| Flag | SARSA | 37.890 | 29.220 | 250/250 | — |
| Cliff | Q-learning | -62.876 | 22.484 | 250/250 | 102 |
| Cliff | SARSA | -26.912 | 26.120 | 250/250 | 2 |

- Both flag agents collect five flags in every final-block episode. In episodes 2,251–2,500, Q-learning completed 228/250 episodes (91.2%), versus SARSA's 157/250 (62.8%). This supports earlier reliable collection for Q-learning in this seed, not a universal learning-speed claim.
- The flag return identity `G = 10.5*k - 0.5*T` holds for every saved row. Final mean absolute TD errors are 0.000459 for Q-learning and 0.612689 for SARSA. SARSA's sampled backup under changing exploration can remain variable despite successful collection; TD errors do not independently measure navigation quality or establish convergence.
- Cliff fall counts are derived per episode as `F = (-G - T)/99`, and are nonnegative integers in every record. They are nonterminal falls, not the CSV's robot collision indicator. In the final block, Q-learning's 102 falls occur across 71 episodes; SARSA's two falls occur in two episodes. Mean falls per episode are 0.408 and 0.008.
- Both cliff agents complete all final 2,000 episodes while epsilon remains 0.1. Across that entire block, mean returns are -61.009 (Q-learning) and -28.862 (SARSA), with 781 and 59 falls respectively. The observed training distinction extends beyond the last 250 episodes.
- Final **100-episode** plotted endpoints: flag Q-learning return 37.790 / length 29.420; flag SARSA 38.005 / 28.990; cliff Q-learning -65.220 / 22.650; cliff SARSA -25.960 / 25.960. Completion is 100% for all four endpoints. These are not the same windows as the table above.

### Saved discrete greedy evaluations

Each evaluation below completed on 13 September 2026 with seed 1234 and one episode, using the corresponding training checkpoint above in evaluation mode. Configuration records verify the checkpoint association and restore the saved task without overrides. These are fixed-task checks, not independent scenario samples or new training runs.

| Environment | Algorithm | Evaluation run under `artifacts/` | Return | Steps | Task completed |
|---|---|---|---:|---:|---|
| Flag | Q-learning | [flag/qlearning/20260913T090053526784Z_seed1234_368e4e46](../../artifacts/flag/qlearning/20260913T090053526784Z_seed1234_368e4e46) | 39.0 | 27 | Yes: five flags |
| Flag | SARSA | [flag/sarsa/20260913T090053591520Z_seed1234_fb76d5f6](../../artifacts/flag/sarsa/20260913T090053591520Z_seed1234_fb76d5f6) | 39.0 | 27 | Yes: five flags |
| Cliff | Q-learning | [cliff/qlearning/20260913T085847992512Z_seed1234_2c6a67cd](../../artifacts/cliff/qlearning/20260913T085847992512Z_seed1234_2c6a67cd) | -17.0 | 17 | Yes: no falls |
| Cliff | SARSA | [cliff/sarsa/20260913T085848075281Z_seed1234_e473fa3c](../../artifacts/cliff/sarsa/20260913T085848075281Z_seed1234_e473fa3c) | -23.0 | 23 | Yes: no falls |

For verification, each checkpoint was replayed greedily in memory, without updates or new saved experiments; return, length and completion reproduced its existing evaluation row. The cliff Q-learning route is `(4,0) → (3,0) → (3,15) → (4,15)`, traversing each intervening cell. SARSA follows `(4,0) → (0,0) → (0,15) → (4,15)`. This verifies the reported route geometry as well as the different lengths. SARSA's training-return advantage under exploration therefore coexists with a lower return in greedy evaluation.

The two flag routes visit flags in order `(3,7), (7,8), (9,4), (6,1), (1,2)`, with different intermediate cells but the same 27 steps. Enumerating all 120 flag orders and summing Manhattan distances from `(5,5)` gives a minimum of 27. Because the grid has no internal obstacles, the evaluated routes attain that minimum transition count. This is a geometric check for this task, not an asymptotic learning guarantee or a generalization result.

All previously missing discrete figure slots are populated. The report's fallback boxes remain for accidental file omissions during upload; no discrete analysis is still marked pending.

## SHA-256 verification

These hashes identify the exact copied PNGs and the numerical inputs used for the analysis. Metrics and configuration records remain with their source runs; they are not required to compile the report.

| File | SHA-256 |
|---|---|
| `ddpg_training_curves.png` | `64d9902b69dae1bbe83b7ca98581660a959e859faf8ab8bb5a84f8e91ef03208` |
| `ddpg_outcomes.png` | `34915b618e6431f76b3e050a4e4b3027e9db3f4de2312231772a6f522eb44cdc` |
| `ddpg_final_goal_distance.png` | `eb0d7ffcdd693abe7bd577570208fbff999da939fde84be671d87981c553141a` |
| `sac_training_curves.png` | `1574c00c4f83446ceb3b1974cd3f887cf54804b74919c87f5a4478db6696fe83` |
| `sac_outcomes.png` | `8baacf35c46bcdb2f0d85ef7c642bdcb63f0c753d97394fe404e59301779f27e` |
| `sac_final_goal_distance.png` | `345f89c914b0d54e83163d1248049d78a163118bd7680f2c64d7cf52dd150693` |
| `td3_training_curves.png` | `c31125157f477e71dd970bc6174f1598d25f14470ea1af68001f06115117cbba` |
| `td3_outcomes.png` | `f446288561f39d162e8f74b42f60eb52b26fda07fe313eed884015dfec7c9462` |
| `td3_final_goal_distance.png` | `99c0495eac07d49991e1a6c473160f0c4a5d248029928f559041b4789bdada81` |
| DDPG selected run: `metrics.csv` | `d7809f364156a436e689f6308eaee7b778250496bc784cb09f0b4ff4576bfe88` |
| DDPG selected run: `config.json` | `6450ddd0843d711d9bb5cb36e2fe7320e01d05310cde75e5c231d6b182245bcc` |
| TD3 selected run: `metrics.csv` | `f515a95ecd13ee6c36587679fc8c975328c36ca63e0dd682171df2e275bf1f3f` |
| TD3 selected run: `config.json` | `e052bf509b7e4b0aa81852e83eef91750409d6ea2d3b651c16d3f6c2bb2e8595` |
| TD3 current fixed-map evaluation: `metrics.csv` | `8ca95eb2584f94e004d32ff7013134481fb6c63aebbc0e5ed308df9b63f8ea01` |
| TD3 current fixed-map evaluation: `config.json` | `21ab6484db67f392ab276eb491fddae0134472b9090711b94f664712e87e2aa8` |
| TD3 current sampled-layout evaluation: `metrics.csv` | `ad6ae055b416fef0174a79eaf21850f4cdbf767dd7d284f5e1e24b2ec86bfd60` |
| TD3 current sampled-layout evaluation: `config.json` | `fa394a86639d117e6065d71932bfc82eead192389cf3f3bf60ff8029f2f5dfae` |
| SAC selected run: `metrics.csv` | `929c9d6eeb5af5ee726af8943d48319e6137c44e4e7373f322fb9390db1c1368` |
| SAC selected run: `config.json` | `918ebf28f66c7ca9214bc290f646d332e26a9c755459b241b6a6f44195037bac` |

## Discrete SHA-256 verification

These hashes identify the five copied images, the saved numerical inputs, and the checkpoints replayed to verify route geometry. The exact source run directories appear above.

| File | SHA-256 |
|---|---|
| `GridFlag_training_curves.png` | `805328ce5b3bf750a87ff6f842757183e686ebc79f0772d2817810e2f1ec8588` |
| `GridFlag_training_curves_SARSA.png` | `16404b7fad7a7a91c869162451947585178d0e732e44e0fea062475e038a9dc0` |
| `GridFlag_outcomes.png` | `2a0bdc6c0df3c0cf0c35d3fc5503a6585867d3ffa6fb538c77df6b007772969c` |
| `GridFlag_outcomes_SARSA.png` | `41d78348fff062c411c044ba31743e09e0626d04289b28dbed1cb30eef84e7ca` |
| `cliff_qlearning_vs_sarsa.png` | `59a4b7fd1d947fc25585698ec3487ebf18e47850aca5218f7bff9cb3d3eb1f25` |
| `flag/qlearning training: metrics.csv` | `7de93171a7352dd8944ed719da545a37983d55af61a4850ca6b415e495d18a2b` |
| `flag/qlearning training: config.json` | `4f44ca2b0c80020a704f7d9d392e9d5537bbbbd07d2c6c81408373cbb3e01960` |
| `flag/qlearning evaluation: metrics.csv` | `62ecb1493f555b6757c3180fe107c5b0a9bf57aa7c6ec33407059099842b0732` |
| `flag/qlearning evaluation: config.json` | `17989aa8360cf54870da85f04097ac1495042d61761d340aed01285dcdbca6b7` |
| `flag/qlearning training: checkpoints/model.npz` | `8e946f1c88ab105c89a8c87cc17356e64a65cf13a89893c613f6018a0b73aca7` |
| `flag/sarsa training: metrics.csv` | `a86fa1f6e4a987d92d6780986f62d654007031f7c7983215968aa71550cd52b5` |
| `flag/sarsa training: config.json` | `c2c0d795b8716c9375e8651573612ff0f9f8deb6aa506f0f7afa543f03506c85` |
| `flag/sarsa evaluation: metrics.csv` | `62ecb1493f555b6757c3180fe107c5b0a9bf57aa7c6ec33407059099842b0732` |
| `flag/sarsa evaluation: config.json` | `65273ce2da0d2eb6483216c39bc6f9d104afff718b73bd8746e809aef355b971` |
| `flag/sarsa training: checkpoints/model.npz` | `3e7591bccda11b425b5cbda3aa4a8436d13633f928b47aadf3d82dc37a30bc98` |
| `cliff/qlearning training: metrics.csv` | `a9418c4e1f71bc01e8c9248e46aa35dca821f831956636861cfdcec9eb69d23a` |
| `cliff/qlearning training: config.json` | `824433b8163989c37985d239d451076cb5518bbdaf0f214181b61ebdb64793ec` |
| `cliff/qlearning evaluation: metrics.csv` | `de0884fd1f15c7347766698e932e569854609d96d49405d758c032832f8072d5` |
| `cliff/qlearning evaluation: config.json` | `632bf094dc0dc4123b138d5f30e49e214d6331a23474908e8b4353e8ffd4b5ff` |
| `cliff/qlearning training: checkpoints/model.npz` | `1e1f37cdbcf9dc43e864d5eb5cc7b3ded83b79c3f7a04068ee0a582afb6d69dc` |
| `cliff/sarsa training: metrics.csv` | `6a6bfbc4b931d66d0af4191193ff69a3c6c496302e9b285eb4f1827870ed77d0` |
| `cliff/sarsa training: config.json` | `f4b0d9f6f4245f9dc2e05d4894ec49841354302e202b769915db8c7217476cf4` |
| `cliff/sarsa evaluation: metrics.csv` | `a92bfa5b9736d33140d31d83c84eef482cf5ff58b73b3e03511f937c372ee678` |
| `cliff/sarsa evaluation: config.json` | `25ed60a8295d2e12c6570f22034f28e5997954757e74dbc30253572444aa05fd` |
| `cliff/sarsa training: checkpoints/model.npz` | `95bf012c5b07cb4f7bb1fb683389f49121594f14a111cef51bdf856ad335b7cb` |
| `cliff comparison: config.json` | `aa1674881413850224a0a64171cc7b4b0fef621dcacbbf947c3a71e724b9f3e0` |
