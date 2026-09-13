# Report

`main.tex` is the current English report. The report covers flag collection, cliff walking and continuous robot navigation, with Q-learning, SARSA, DDPG, TD3 and SAC.

## Build

Upload `main.tex` and the complete `figures/` folder to Overleaf. Select `main.tex` as the main document and use pdfLaTeX. The bibliography is internal; there is no separate BibTeX file. With a local installation, run pdfLaTeX twice from this directory. `NAMLproject.pdf` is an existing compiled copy; recompile the updated source to refresh it.

No local LaTeX engine is available for compilation and page-layout verification. Static checks cover figure paths, labels, references, environment/bracket balance and the numerical claims tied to the saved metrics. Compile in Overleaf and check the resulting page layout.

## Current experimental evidence

The report includes fourteen current training images: five from flag/cliff runs completed on 13 September 2026, six from DDPG and SAC runs of 10 September 2026, and three from the tuned TD3 run of 13 September 2026. The four discrete runs, DDPG and TD3 each completed 5,000 episodes; SAC completed 3,000. All use seed 0 and the corrected implementation. The [figure inventory](figures/README.md) records exact training/evaluation run IDs, image mappings, source hashes and numerical interpretation.

Flag Q-learning and SARSA both complete all final 250 training episodes and collect every flag in 27 steps with return 39.0 during saved greedy evaluation. Q-learning reaches reliable collection earlier in the selected run. The report includes both algorithms' training curves and outcome plots, and checks returns against the exact flag reward formula. The 27-step greedy routes match the minimum Manhattan tour length for the saved fixed layout.

The cliff comparison demonstrates the exploration tradeoff: over the final 250 episodes, SARSA averages -26.91 return with two falls in total, while Q-learning averages -62.88 with 102 falls. Both reach the goal in every episode of that block. Greedy evaluation reverses the return ordering: Q-learning takes 17 steps along the cliff edge, versus SARSA's 23 steps via the top row. Saved checkpoint replays reproduce the evaluation records and establish these routes without learning updates. The report distinguishes the training advantage under exploration from performance with exploration disabled.

Training analysis follows the pictures and their saved episode metrics. All three continuous runs show substantial learning with remaining failures. TD3 now achieves 88.8% final-block training success, mean return 128.22 and mean final distance 1.00 m. Its concise tuning note records a fresh run with critic learning rate reduced from `1e-3` to `3e-4` and updates per environment step from two to one. The report distinguishes 100-episode plotted rolling curves from 250-episode numerical summaries, and records the new actual environment-step and actor/critic-update budgets.

The square goal-distance figures use 0–10 m axes, with markers for values above that display range. Success figures contain rolling rates, not binary episode indicators; the SAC outcome figure also contains its logged temperature. All PNGs are copied without alteration from the selected run artifacts.

Current DDPG and TD3 evaluation records are discussed separately from training. The tuned TD3 checkpoint succeeds on the fixed map in 127 steps with return 167.09, and on nine of ten sampled layouts, with one cutoff and no collisions. The old three-agent shared-map benchmark remains explicitly historical. Different training budgets, hyperparameters and tuning effort, a single selected training seed per algorithm and the small evaluation sample constrain comparisons.

## Preserved and missing material

The original four DDPG/TD3 report images remain under their historical names. They are preserved as provenance and are not displayed as current training evidence. The original notebook archive and all run checkpoints, metrics and source images are unchanged.

All discrete figure slots are populated, and the introduction, discussion and conclusions reflect the measured grid results and the revised TD3 evidence. The figure macro retains visible fallbacks for accidental missing files during upload. The full controlled comparison across five independent training seeds and 51 shared evaluation scenarios remains pending. The fixed-grid evaluations and single training seeds do not establish generalization or a statistical algorithm ranking.
