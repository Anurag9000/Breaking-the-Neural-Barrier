# Tabular MLP Experiment Handoff

This repo snapshot is meant to be resumed from other machines without guessing.
Keep the archived result trees in Git and resume each depth-specific run root from disk.

## Current active task set

The active tabular suite is:

- `representation`
- `autoencoding`
- `generation`
- `denoising`
- `anomaly`
- `simulation`
- `prediction`

The removed tasks are not part of the active sweep:

- `inverse`
- `control`
- `selfsupervised`

## Current artifact roots to keep

Do not delete these if you want the published repo state to remain reproducible:

- `MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu`
- `MLPS/tabular/shared/dae_dnn/results/goliath_w2d_anomaly_onward_gpu`
- `MLPS/tabular/shared/dae_dnn/results/stl_ablation_all_tasks_d3plus_w64plus`
- `MLPS/tabular/shared/dae_dnn/results/analysis/loss_vs_params_linearx_logy`

The plots in `analysis/` are derived from the corresponding run roots and archived CSV/JSON summaries.

Important limitation:

- CSV, JSON, TXT, MD, and PNG artifacts are tracked in Git when staged.
- PyTorch checkpoints (`*.pt`, `*.pth`, `*.ckpt`) are still ignored by the repo rules.
- That means a fresh clone can recover the run plan, logs, metrics, plots, and candidate metadata from GitHub, but **exact optimizer-state resume requires the checkpoint files to be copied separately** from the machine that produced them or archived in another artifact store.

## How the width-only sweep is organized

For the current long sweep, each depth has its own resumable run root:

- `.../${task}_d1`
- `.../${task}_d2`
- ...
- `.../${task}_d10`

The unit of resumption is the per-depth directory, not the whole task wave.

If a machine dies or a job is interrupted, rerun the exact same command with the same `--run-root`.
The candidate metadata and `phase_progress.csv` will be reused. If the matching checkpoint files are present on disk, the optimizer state and best state resume exactly; otherwise the run restarts from the best recoverable metadata state.

## Suggested execution split

To parallelize across laptops without overlapping work, split the depth range into waves:

- wave 1: depths `1 2 3 4 5`
- wave 2: depths `6 7 8`
- wave 3: depths `9 10`

Tasks stay sequential within a machine unless you intentionally split them further.

Recommended launch pattern:

```bash
cd /home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier
git checkout main
git pull origin main

tasks=(representation autoencoding generation denoising anomaly simulation prediction)
depths=(1 2 3 4 5)

for task in "${tasks[@]}"; do
  for depth in "${depths[@]}"; do
    .venv/bin/python MLPS/tabular/shared/dae_dnn/run_goliath_staged_width_only.py \
      --data-dir ./data \
      --results-dir MLPS/tabular/shared/dae_dnn/results \
      --run-root MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu/${task}_d${depth} \
      --tasks "$task" \
      --stl-depth "$depth" \
      --alt-start-width 1 \
      --patience 10 \
      --width-expansion-patience 10 \
      --num-workers 0 &
  done
  wait
done
```

Use the same shape for the other waves by swapping the depth list:

- `depths=(6 7 8)`
- `depths=(9 10)`

If you want strict single-process execution per depth, remove the trailing `&` and `wait`.

## Push / pull consolidation protocol

Use this order when moving results between laptops:

1. Finish a batch or a depth wave.
2. Commit and push the new run-root contents and analysis artifacts.
3. On the next laptop, `git pull origin main` before starting any new run.
4. Reuse the same per-depth `--run-root` if a partially completed depth needs to continue.
5. Do not rename the per-depth root mid-run.
6. Copy the checkpoint files too if you need an exact cross-machine resume.

That keeps the experiment graph unified on GitHub while allowing independent machines to continue from the same state.

## Plot regeneration notes

The saved comparison figures are derived artifacts and can be regenerated from the tracked summaries and run roots.

Useful current plot roots:

- `analysis/loss_vs_params_linearx_logy/w2d/...`
- `analysis/loss_vs_params_linearx_logy/width_only/...`

The `representation` width-only plot labels STL architectures directly in the figure.

## Resume rule of thumb

If you are unsure whether to rerun or resume, use this rule:

- same `--run-root` and same depth = resume
- new `--run-root` = fresh run

That is the only distinction that matters for the saved metadata path.
For exact optimizer-state resume, the checkpoint files must exist on the target machine as well.
