# Tabular MLP Experiment Handoff

This repo snapshot is meant to be resumed from other machines without guessing.
Keep the archived result trees in Git and resume each depth-specific run root from disk.

## Current active task set

The active tabular suite is:

- `classification`
- `autoencoding`
- `generation`
- `denoising`
- `anomaly`
- `simulation`
- `prediction`

For the `generation` task on laptop 2, the only dataset that must be present before the run is `Covertype`.
If you want to warm the shared benchmark cache anyway, use:

```bash
python3 scripts/prefetch_dae_dnn_datasets.py --all
```

The removed tasks are not part of the active sweep:

- `inverse`
- `control`
- `selfsupervised`

### Laptop 2 generation resume state

Current resume target on this machine:

- task: `generation`
- remaining depths: `1 2 3 4 5 6 7 8 9 10`
- concurrency: `3`

The launcher now follows the documented waves:

- wave 1: `1 2 3 4`
- wave 2: `5 6 7 8`
- wave 3: `9 10`

Depths `1 2 3 4` are already preserved in the result tree.

## Current artifact roots to keep

Do not delete these if you want the published repo state to remain reproducible:

- `MLPS/tabular/shared/dae_dnn/results/goliath_active_suite_width_only_gpu`
- `MLPS/tabular/shared/dae_dnn/results/goliath_w2d_anomaly_onward_gpu`
- `MLPS/tabular/shared/dae_dnn/results/stl_ablation_all_tasks_d3plus_w64plus`
- `MLPS/tabular/shared/dae_dnn/results/analysis/loss_vs_params_linearx_logy`

The plots in `analysis/` are derived from the corresponding run roots and archived CSV/JSON summaries.

Important limitation:

- CSV, JSON, TXT, MD, PNG, and forced-tracked checkpoint artifacts are preserved in this handoff snapshot when staged.
- The current result tree includes PyTorch checkpoints (`*.pt`, `*.pth`, `*.ckpt`) for the preserved runs so a fresh clone of this commit can resume the saved state without a separate artifact store.

## How the width-only sweep is organized

For the current long sweep, each depth has its own resumable run root:

- `.../${task}_d1`
- `.../${task}_d2`
- ...
- `.../${task}_d10`

The unit of resumption is the per-depth directory, not the whole task wave.

If a machine dies or a job is interrupted, rerun the exact same command with the same `--run-root`.
The candidate metadata and `phase_progress.csv` will be reused. If the matching checkpoint files are present on disk, the optimizer state and best state resume exactly; otherwise the run restarts from the best recoverable metadata state.

The staged tabular runner also checkpoints at batch boundaries inside each epoch.
That means a restart from the same `--run-root` picks up the next unprocessed batch in the current epoch when the last checkpoint survived.
To keep that deterministic, do not change the command, task list, or `--stl-depth` while resuming.

## Suggested execution split

For the current generation resume on this machine, use the launcher in `scripts/run_generation_suite.sh` with concurrency `4`.
It runs the three depth waves above and wraps every depth in the watchdog.

Recommended launch pattern:

```bash
cd /home/anurag/Projects/Breaking-the-Neural-Barrier
git checkout main
git pull origin main
./scripts/run_generation_suite.sh 2>&1 | tee generation_run.log
```

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

The `classification` width-only plot labels STL architectures directly in the figure.

## Resume rule of thumb

If you are unsure whether to rerun or resume, use this rule:

- same `--run-root` and same depth = resume
- new `--run-root` = fresh run

That is the only distinction that matters for the saved metadata path.
For exact optimizer-state resume, the checkpoint files must exist on the target machine as well.
