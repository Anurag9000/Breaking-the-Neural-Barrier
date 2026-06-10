# Tabular DAE/DNN Methodology and Handoff

This document records the exact code paths and workflow used for the tabular
MLP experiments in this repo. Use it as the reference when running the same
study on another machine.

## Canonical task names

Current task names:

- `classification`
- `autoencoding`
- `denoising`
- `anomaly`
- `simulation`
- `prediction`

## Exact code paths to use

Task definitions and dataset mapping:

- `MLPS/tabular/shared/dae_dnn/tasks.py`

STL sweep orchestration:

- `MLPS/tabular/shared/dae_dnn/run_stl_ablation.py`
- `MLPS/tabular/shared/dae_dnn/run_stl_ablation_parallel.py`
- `MLPS/tabular/shared/dae_dnn/run_stl_small_grid.py`

ADP width-to-depth orchestration:

- `MLPS/tabular/shared/dae_dnn/run_task.py`
- `MLPS/tabular/shared/dae_dnn/adp_search.py`
- `utils/adp_contract.py`
- `MLPS/tabular/shared/dae_dnn/run_adp_w2d_suite_parallel.py`

Historical ADP W2D archives already restored into the repo:

- `MLPS/tabular/shared/dae_dnn/results/archive/goliath_w2d_anomaly_onward_gpu`
- `MLPS/tabular/shared/dae_dnn/results/archive/goliath_w2d_staged_current`

The staged legacy lineage is exposed in the repo-visible archive root as
`classification` for that task lineage.

## Width-to-depth rule

For width-to-depth runs, a depth expansion is not treated as a terminal step.
After the new layer is added, the controller continues widening neuron by
neuron until the model is uniform again, and only then does the patience
counter resume. That prevents the run from quitting while the post-depth
warmup is still in progress.

## Small historical STL archive

This is separate from the massive STL ablation TODO.

The lightweight historical STL archive at `results/archive/classification_trial1`
is a one-off fixed-grid summary, not a repeat suite. It covers the following
tasks:

- `classification`
- `autoencoding`
- `generation`
- `denoising`
- `anomaly`

The archived STL ablation grid is sparse and fixed:

- depths: `3`, `4`, `6`, `8`, `10`
- widths: `64`, `96`, `128`, `160`, `192`, `224`, `256`
- repeats: none

It does not include dedicated `simulation` or `prediction` STL roots. Treat
those two tasks as absent from the historical STL material until new results
are generated.

Same `--run-root` means resume. Keep this separate from the massive all-task
STL sweep.

## Small STL follow-up on slave laptops

If you want to run the missing `simulation` and `prediction` tasks as a
separate, lightweight follow-up on a slave machine, use the dedicated
no-repeat grid runner and keep it out of the massive all-task STL ablation
tree.

Use this command:

```bash
cd /home/anurag-basistha/Projects/Untapped/Breaking-the-Neural-Barrier

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python MLPS/tabular/shared/dae_dnn/run_stl_small_grid.py \
  --data-dir ./data \
  --results-dir MLPS/tabular/shared/dae_dnn/results \
  --run-root MLPS/tabular/shared/dae_dnn/results/stl/small_grid/simulation_prediction_v1 \
  --source-run-root MLPS/tabular/shared/dae_dnn/results/archive/classification_trial1 \
  --tasks simulation prediction \
  --depths 3 4 6 8 10 \
  --widths 64 96 128 160 192 224 256 \
  --num-workers 0 \
  --patience 10 \
  --batch-size 9312 \
  --max-epochs 100000000
```

This follow-up is separate from the massive all-task STL TODO and should not
be mixed with the live repeat-5 ADP run. It is one run per `(task, depth,
width)` candidate, with no repeats.

## ADP repeat schedule

The current ADP width-to-depth suite uses five repeats for:

- `classification`
- `autoencoding`
- `generation`
- `denoising`
- `anomaly`

It uses six repeats for:

- `simulation`
- `prediction`

This is intentional and reflected in the suite launcher. Keep that split when
restarting the suite or comparing results across tasks.

Legacy and analysis helpers:

- `MLPS/tabular/shared/dae_dnn/run_goliath.py`
- `MLPS/tabular/shared/dae_dnn/generate_loss_vs_params_plots.py`
- `MLPS/tabular/shared/dae_dnn/generate_width_only_depth_sweep_plots.py`
- `MLPS/tabular/shared/dae_dnn/generate_w2d_trajectory_loglog_plots.py`
- `MLPS/tabular/shared/dae_dnn/recover_trial1_w2d_history_from_git.py`

## Current training style

The repo uses the same broad method pattern across the tabular sweeps:

1. build the task from `tasks.py`
2. launch a runner entry point
3. write run metadata, logs, CSVs, and phase/task state into the run root
4. restore from the saved checkpoint/state files when resuming
5. keep the repeat-level results separate from the derived analysis outputs

The current result roots are:

- STL: `MLPS/tabular/shared/dae_dnn/results/stl/ablation/<suite_name>/`
- ADP: `MLPS/tabular/shared/dae_dnn/results/adp/w2d/<suite_name>/`
- archive: `MLPS/tabular/shared/dae_dnn/results/archive/<legacy_suite>/`

## Resume rule

- same `--run-root` means resume
- new `--run-root` means fresh run
- do not change task order, candidate grid, or repeat count while resuming
- keep the local `main` branch authoritative

## Export rule for other laptops

Non-master laptops should export only generated experiment data:

- `results/`
- `runs/`
- `logs/`
- `metrics/`
- `plots/`
- `checkpoints/`
- `tensorboard/`
- CSV and JSON outputs
- generated figures

Do not merge code changes from those machines into canonical `main`.

## Recommended refresh commands

Refresh the inventory after new results land:

```bash
./.venv/bin/python scripts/update_experiment_inventory.py
```

## Result organization

The canonical result layout is documented in:

- `MLPS/tabular/shared/dae_dnn/results/catalog/`
- `MLPS/tabular/shared/dae_dnn/results/README.md`
- `docs/tabular_dae_dnn/README.md`
