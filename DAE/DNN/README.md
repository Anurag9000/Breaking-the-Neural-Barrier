# DNN STL + ADP (DAE/DNN)

This folder provides plain-MLP baselines for the 15 non-vision tasks in
`DAE/DNN/tasks.py`. Each task runs in:

- STL mode: fixed architecture
- ADP mode: adaptive width/depth search

The benchmarks are real public datasets, with the main task families centered
on:

- `Covertype`
- `YearPredictionMSD`
- `California Housing`

Files
- `DEFAULT_TASKS.md`: default task/dataset mapping
- `mlp.py`: plain MLP backbone
- `tasks.py`: dataset builders + task registry
- `adp_search.py`: ADP search (width/depth expansions)
- `run_task.py`: run one task (STL or ADP)
- `run_all.py`: run STL + all ADP modes for all tasks
- `run_goliath.py`: sequential STL + ADP experiment runner with resumable
  checkpoints

Tasks and default benchmark mappings
- prediction: YearPredictionMSD
- representation: Covertype
- autoencoding: Covertype
- generation: Covertype
- denoising: Covertype
- anomaly: Covertype
- inverse: California Housing
- control: California Housing
- clustering: Covertype
- compression: Covertype
- ranking: YearPredictionMSD
- multimodal: Covertype + parity scalar
- selfsupervised: Covertype feature permutation prediction
- simulation: California Housing
- misc: California Housing residual regression

Run one task (STL, fixed architecture)
```bash
python DAE/DNN/run_task.py --task classification --mode stl --hidden 50 50 --data-dir ./data --results-dir DAE/DNN/results
```

Run one task (ADP, width only)
```bash
python DAE/DNN/run_task.py --task classification --mode adp --adp-mode width_only --hidden 50 50 --ex-k 1 --patience 10 --data-dir ./data --results-dir DAE/DNN/results
```

Run one task (ADP, depth only)
```bash
python DAE/DNN/run_task.py --task classification --mode adp --adp-mode depth_only --hidden 50 50 --max-depth 10 --patience 10 --data-dir ./data --results-dir DAE/DNN/results
```

Run one task (ADP, width then depth)
```bash
python DAE/DNN/run_task.py --task classification --mode adp --adp-mode width_to_depth --hidden 50 50 --ex-k 1 --max-depth 10 --patience 10 --data-dir ./data --results-dir DAE/DNN/results
```

Run one task (ADP, depth then width)
```bash
python DAE/DNN/run_task.py --task classification --mode adp --adp-mode depth_to_width --hidden 50 50 --ex-k 1 --max-depth 10 --patience 10 --data-dir ./data --results-dir DAE/DNN/results
```

Run one task (ADP, alternating)
```bash
python DAE/DNN/run_task.py --task classification --mode adp --adp-mode alt_width --hidden 50 50 --ex-k 1 --patience 10 --data-dir ./data --results-dir DAE/DNN/results
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

Common flags
- `--hidden`: starting widths (length = starting depth)
- `--ex-k`: width expansion step
- `--max-width`, `--max-depth`, `--max-neurons`: hard caps
- `--patience`: early stopping for a single run
- `--trials-width`, `--trials-depth`: expansion patience
- `--max-epochs`: cap for each single-shot training
- `--seed`, `--batch-size`, `--num-workers`

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
