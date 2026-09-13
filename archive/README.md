# Historical notebook archive

`notebooks/` preserves the notebooks exactly as they were before the implementation fixes, with their original section directory names and saved outputs. `notebooks/manifest.json` lists original paths, archive paths and SHA-256 hashes. The empty legacy `testreward.ipynb` is preserved here only as a provenance record.

| Archived section | Current status |
|---|---|
| `notebooks/Continuous_Diff_Drive/` | Historical outputs; maintained notebooks are in the active `Continuous_Diff_Drive/` section. |
| `notebooks/Grid_flag_2/` | Historical moving-obstacle and tutorial notebooks; their active section has been removed. |

These files are **historical evidence, not supported execution entry points**. They may contain old reward definitions, stale imports, incompatible checkpoint assumptions, incorrect metrics or recorded errors. Use the [maintained continuous notebooks](../Continuous_Diff_Drive/README.md#notebook-workflow) for new robot experiments. The archived prototype notebooks depend on code that is no longer included in the active project.

The original DDPG/TD3 report images remain preserved under their historical filenames. The LaTeX report now displays figures from selected completed corrected-code DDPG, TD3 and SAC runs; its retained historical three-agent benchmark is labelled separately. That benchmark must not be interpreted as a fresh evaluation of the new checkpoints. In particular, old SAC outputs do not validate the old checkpoint with current normalized preprocessing.

No archived notebook was rerun or reformatted. Old Python implementations were removed after reference checks; their tracked history remains available through Git.

## Notebook display revision

[notebook_display_20260910T143650Z](notebook_display_20260910T143650Z/README.md) preserves the three training notebooks and the completed DDPG run's plots immediately before the approved presentation changes. Unlike the original pre-fix archive, this includes a completed post-fix DDPG experiment. Its manifest records exact-copy hashes. The active run's outcome plots were redrawn from its saved metrics, and its video names were migrated to zero-based `episode_<index>.mp4` names; checkpoints and metrics were preserved.
