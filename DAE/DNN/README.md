# DNN STL + ADP (DAE/DNN)

This folder provides plain-MLP baselines for each task in `not_accomplished.md`,
with STL (fixed architecture) and 6 ADP modes.

Files
- `DEFAULT_TASKS.md`: default dataset/task mapping
- `mlp.py`: plain MLP backbone
- `tasks.py`: dataset builders + task registry
- `adp_search.py`: ADP search (width/depth expansions)
- `run_task.py`: run one task (STL or ADP)
- `run_all.py`: run STL + all 6 ADP modes for all tasks
- `run_goliath.py`: sequential STL + AE experiment runner with resumable checkpoints

Tasks and default datasets
- prediction: synthetic regression (20-d -> 1-d)
- classification: MNIST
- representation: MNIST (embedding + kNN metric)
- autoencoding: MNIST (x -> x)
- generation: MNIST (noise -> image)
- denoising: MNIST (noisy -> clean)
- anomaly: MNIST (train 0-4, test 5-9)
- sequence: synthetic sine wave (window=20 -> next value)
- inverse: synthetic linear inverse (y -> x)
- control: synthetic LQR (state -> action)
- clustering: MNIST (embedding + k-means NMI)
- compression: MNIST (autoencode + compression ratio)
- ranking: synthetic score regression (pairwise metric)
- multimodal: MNIST + parity scalar
- selfsupervised: MNIST rotation prediction
- simulation: synthetic linear dynamics
- edge: MNIST (ADP max width capped at 32)
- misc: synthetic residual regression

Run one task (STL, fixed architecture)
```
python DAE/DNN/run_task.py --task classification --mode stl --hidden 50 50 --data-dir .\data --results-dir DAE/DNN/results
```

Run one task (ADP, width only)
```
python DAE/DNN/run_task.py --task classification --mode adp --adp-mode width_only --hidden 50 50 --ex-k 1 --patience 10 --data-dir .\data --results-dir DAE/DNN/results
```

Run one task (ADP, depth only)
```
python DAE/DNN/run_task.py --task classification --mode adp --adp-mode depth_only --hidden 50 50 --max-depth 10 --patience 10 --data-dir .\data --results-dir DAE/DNN/results
```

Run one task (ADP, width then depth)
```
python DAE/DNN/run_task.py --task classification --mode adp --adp-mode width_to_depth --hidden 50 50 --ex-k 1 --max-depth 10 --patience 10 --data-dir .\data --results-dir DAE/DNN/results
```

Run one task (ADP, depth then width)
```
python DAE/DNN/run_task.py --task classification --mode adp --adp-mode depth_to_width --hidden 50 50 --ex-k 1 --max-depth 10 --patience 10 --data-dir .\data --results-dir DAE/DNN/results
```

Run one task (ADP, alternating)
```
python DAE/DNN/run_task.py --task classification --mode adp --adp-mode alt_width --hidden 50 50 --ex-k 1 --patience 10 --data-dir .\data --results-dir DAE/DNN/results
```

Run all tasks (STL + 6 ADP modes)
```
python DAE/DNN/run_all.py --data-dir .\data --results-dir DAE/DNN/results --hidden 50 50
```

Linux CUDA setup
```
bash scripts/setup_cuda_venv.sh
source .venv/bin/activate
```
Then launch the full sequential experiment:
```
python DAE/DNN/run_goliath.py --tasks all --data-dir ./data --results-dir DAE/DNN/results --stl-width 128 --stl-depth 2 --alt-start-width 2 --alt-start-depth 2 --patience 5 --seed 0
```

Common flags
- `--hidden`: starting widths (length = starting depth)
- `--ex-k`: width expansion step
- `--max-width`, `--max-depth`, `--max-neurons`: hard caps
- `--patience`: early-stopping patience per single run
- `--trials-width`, `--trials-depth`: expansion patience (<=0 means infinite)
- `--max-epochs`: cap for each single-shot training
- `--seed`, `--batch-size`, `--num-workers`

Where results go
- Per run folder: `DAE/DNN/results/<task>_<mode>_<adp_mode>_d<d>_w<w>_exk<k>_<timestamp>/`
- Files: `training_log.txt`, `training_stats.csv`, `val_loss_vs_step.png`, `loss_vs_neurons_best.png`

`run_goliath.py` adds a deeper hierarchy:
- `results/goliath_<timestamp>/<task>/<phase>/cand_###_d##_w##/`
- Each candidate dir stores `metadata.json`, `candidate_state.json`, `checkpoint_last.pt`, `checkpoint_best.pt`, `training_log.txt`, `training_stats.csv`
- Phase roots store `search_state.json`, `phase_summary.json`, and `phase_progress.csv`
