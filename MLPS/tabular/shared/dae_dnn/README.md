# DNN STL + ADP (DAE/DNN)

This folder provides plain-MLP baselines for the 7 active non-vision tasks in
`DAE/DNN/tasks.py`. Each task runs in:

- STL mode: fixed architecture
- ADP mode: adaptive width/depth search

The benchmarks are real public datasets, with the main task families centered
on:

- `Covertype`
- `YearPredictionMSD`
- `California Housing`

Supported ADP phases for active runs:
- `ae_alt_width`
- `ae_alt_depth`
- `ae_width_to_depth`
- `ae_depth_to_width`

Files
- `DEFAULT_TASKS.md`: default task/dataset mapping
- `mlp.py`: plain MLP backbone
- `tasks.py`: dataset builders + task registry
- `adp_search.py`: ADP search (width/depth expansions)
- `run_task.py`: run one task (STL or ADP)
- `run_all.py`: run STL + all ADP modes for all tasks
- `run_goliath.py`: sequential STL + ADP experiment runner with resumable
  checkpoints
- `run_search_suite.py`: baseline-only benchmark suite for grid search,
  random search, Bayesian HPO, and greedy NAS-style growth; it can compare
  against a completed goliath reference run, but it does not run ADP variants

`run_goliath.py` now runs only the four supported ADP phases:
- `ae_alt_width`
- `ae_alt_depth`
- `ae_width_to_depth`
- `ae_depth_to_width`

All ADP phases start from a fixed `2x2` seed and preserve the best checkpoint
found during search, not just the last epoch. After each ADP phase, the runner
automatically trains an STL refit on that ADP-discovered architecture and logs
the ADP-vs-STL comparison for the task. A standalone STL baseline is optional
if you explicitly include `stl` in `--phases`.

At the end of a full goliath run, the runner writes:
- `final_report.json`
- `final_report.md`

These summarize, for each task and each ADP phase, the best architecture found,
the paired STL refit loss on that same architecture, and the overall task
winner.

Tasks and default benchmark mappings
- prediction: YearPredictionMSD
- representation: Covertype
- autoencoding: Covertype
- generation: Covertype
- denoising: Covertype
- anomaly: Covertype
- simulation: California Housing

Run one task (STL, fixed architecture)
```bash
python DAE/DNN/run_task.py --task classification --mode stl --hidden 50 50 --data-dir ./data --results-dir DAE/DNN/results
```

Run one task (ADP, width then depth)
```bash
python DAE/DNN/run_task.py --task classification --mode adp --adp-mode width_to_depth --hidden 50 50 --ex-k 1 --max-depth 10 --patience 10 --data-dir ./data --results-dir DAE/DNN/results
```

Run one task (ADP, alternating width-first)
```bash
python DAE/DNN/run_task.py --task classification --mode adp --adp-mode alt_width --hidden 50 50 --ex-k 1 --patience 10 --data-dir ./data --results-dir DAE/DNN/results
```

Run one task (ADP, alternating depth-first)
```bash
python DAE/DNN/run_task.py --task classification --mode adp --adp-mode alt_depth --hidden 50 50 --ex-k 1 --patience 10 --data-dir ./data --results-dir DAE/DNN/results
```

Run one task (ADP, depth then width)
```bash
python DAE/DNN/run_task.py --task classification --mode adp --adp-mode depth_to_width --hidden 50 50 --ex-k 1 --patience 10 --data-dir ./data --results-dir DAE/DNN/results
```

Run all tasks (STL + ADP modes)
```bash
python DAE/DNN/run_all.py --data-dir ./data --results-dir DAE/DNN/results --hidden 50 50
```

Linux CUDA setup
```bash
bash scripts/setup_cuda_venv.sh
source .venv/bin/activate
```
Then launch the sequential experiment:
```bash
python DAE/DNN/run_goliath.py --tasks all --data-dir ./data --results-dir DAE/DNN/results --stl-width 128 --stl-depth 2 --alt-start-width 2 --alt-start-depth 2 --patience 5 --seed 0
```
By default, `run_goliath.py` runs the four supported ADP phases first and then
an STL refit on each ADP-discovered architecture. Include `stl` in `--phases`
only if you also want a standalone baseline STL run.

To run the broader benchmark-suite comparison:
```bash
python DAE/DNN/run_search_suite.py --tasks all --data-dir ./data --results-dir DAE/DNN/results --reference-run-root DAE/DNN/results/goliath_<timestamp> --batch-size 32768 --candidate-budget 0 --seed 0
```
This evaluates grid search, random search, Bayesian HPO, and greedy NAS-style
growth by default, then refits STL on the best architecture found by each
method. Use `run_goliath.py` for ADP/STL comparisons; `run_search_suite.py`
never reruns ADP.

Common flags
- `--hidden`: starting widths (length = starting depth)
- `--ex-k`: width expansion step
- `--max-width`, `--max-depth`, `--max-neurons`: hard caps
- `--patience`: early stopping for a single run
- `--trials-width`, `--trials-depth`: expansion patience
- `--max-epochs`: cap for each single-shot training
- `--seed`, `--batch-size`, `--num-workers`
- Default batch size is `32768`; the adaptive controller shrinks it automatically if VRAM pressure crosses the threshold.

Where results go
- Per-run folder:
  `DAE/DNN/results/<task>_<mode>_<adp_mode>_d<d>_w<w>_exk<k>_<timestamp>/`
- Files:
  `training_log.txt`, `training_stats.csv`, `val_loss_vs_step.png`,
  `loss_vs_neurons_best.png`

`run_goliath.py` adds a deeper hierarchy:
- `results/goliath_<timestamp>/<task>/<phase>/cand_###_d##_w##/`
- Each candidate dir stores `metadata.json`, `candidate_state.json`,
  `checkpoint_last.pt`, `checkpoint_best.pt`, `training_log.txt`,
  `training_stats.csv`
- Phase roots store `search_state.json`, `phase_summary.json`, and
  `phase_progress.csv`
- The run root also stores `final_report.json` and `final_report.md`, which
  summarize the best ADP architecture per variant, the paired STL refit loss
  on the same architecture, and the overall winner per task.
