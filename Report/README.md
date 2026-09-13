# Report

`main_new.tex` is the revised English report. `main.tex` is the original draft and remains unchanged. The report covers flag collection, cliff walking and continuous robot navigation, with Q-learning, SARSA, DDPG, TD3 and SAC.

## Build

Upload `main_new.tex` and the complete `figures/` folder to Overleaf. Select `main_new.tex` as the main document and use pdfLaTeX. The bibliography is internal; there is no separate BibTeX file. With a local installation, run pdfLaTeX twice from this directory.

No local LaTeX engine is available for compilation and page-layout verification. Static checks cover figure paths, labels, references, environment/bracket balance and the numerical claims tied to the saved metrics. Compile in Overleaf and check the resulting page layout.

## Current experimental evidence

The report includes fourteen current training images: five from flag/cliff runs completed on 13 September 2026, and nine from DDPG, TD3 and SAC runs of 10 September 2026. The four discrete runs, DDPG and TD3 each completed 5,000 episodes; SAC completed 3,000. All use seed 0 and the corrected implementation. The [figure inventory](figures/README.md) records exact training/evaluation run IDs, image mappings, source hashes and numerical interpretation.

Flag Q-learning and SARSA both complete all final 250 training episodes and collect every flag in 27 steps with return 39.0 during saved greedy evaluation. Q-learning reaches reliable collection earlier in the selected run. The report includes both algorithms' training curves and outcome plots, and checks returns against the exact flag reward formula. The 27-step greedy routes match the minimum Manhattan tour length for the saved fixed layout.

The cliff comparison demonstrates the exploration tradeoff: over the final 250 episodes, SARSA averages -26.91 return with two falls in total, while Q-learning averages -62.88 with 102 falls. Both reach the goal in every episode of that block. Greedy evaluation reverses the return ordering: Q-learning takes 17 steps along the cliff edge, versus SARSA's 23 steps via the top row. Saved checkpoint replays reproduce the evaluation records and establish these routes without learning updates. The report distinguishes the training advantage under exploration from performance with exploration disabled.

Training analysis was rewritten against the pictures and their saved episode metrics. DDPG and SAC show substantial learning with remaining failures. The selected TD3 run has persistent cutoffs, a large critic-loss spike and markedly lower training success than its historical predecessor. The report distinguishes 100-episode plotted rolling curves from 250-episode numerical summaries, and records actual environment-step and actor/critic-update budgets.

The square goal-distance figures use 0–10 m axes, with markers for values above that display range. Success figures contain rolling rates, not binary episode indicators; the SAC outcome figure also contains its logged temperature. All PNGs are copied without alteration from the selected run artifacts.

Two current DDPG evaluation records are discussed separately. The old three-agent shared-map benchmark remains explicitly historical and is not treated as a new comparison of the selected models. Different training budgets and hyperparameters, a single selected seed per algorithm and missing common evaluation constrain algorithm comparisons.

## Preserved and missing material

The original four DDPG/TD3 report images remain under their historical names. They are preserved as provenance and are not displayed as current training evidence. The original notebook archive and all run checkpoints, metrics and source images are unchanged.

All discrete figure slots are now populated, and the introduction, discussion and conclusions reflect the measured grid results. The figure macro retains visible fallbacks for accidental missing files during upload. The optional comparison across five independent training seeds and common-scenario evaluation of the current three continuous checkpoints remain pending. The fixed-grid evaluations and single training seeds do not establish generalization or a statistical algorithm ranking.
