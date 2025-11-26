# ADP Diffusion SSL Runners

Scripts:
- `run_adp_diff_ssl_width_only.py`
- `run_adp_diff_ssl_depth_only.py`
- `run_adp_diff_ssl_width.py` (width→depth)
- `run_adp_diff_ssl_depth.py` (depth→width)
- `run_adp_diff_ssl_alt_width.py` (alternate, width-first)
- `run_adp_diff_ssl_alt_depth.py` (alternate, depth-first)

Common flags:
```
--data-root ./data --batch-size 128 --init-width 64 --init-depth 3 --pool-idx 0 2
--max-epochs 20 --ex-k 16 --max-depth 32 --max-width 1024 --max-neurons 1200000
```

Example:
```
python Diffusion/Self-Supervised/Runs/adp/run_adp_diff_ssl_width_only.py --download --max-epochs 5
```

Results: checkpoints and logs can be stored under `runs/diff_ssl/<policy>/` (configure via your own wrapper or by editing the script to save `state_dict` after training).
