## ADP expansion algorithms (Dyn_DNN4OPF reference)

Source files studied (external repo `D:\OPF_ADP\DNN\Dyn_DNN4OPF\models`):
- `adp_den_depth_only.py`
- `adp_den_width_only.py`
- `adp_depth.py` (depth-first, then width sweeps)
- `adp_den_expandtillplateuing.py` (width-then-depth until plateau)
- `adp_den_alt_depth_4_head.py` (alternate depth/width, depth first; multi-head variant)
- `adp_den_alt_width_1_head.py` (alternate width/depth, width first; single-head)

Below is a distilled, line-by-line behavioral summary (ignoring multi-head specifics as requested).

### Common building blocks
- Base class: `DEN` (sets up `layers`/`hidden_layers`, `head`, task metadata).
- Resizers:
  - `_resize_linear(old, new_out, new_in=None)`: create new `Linear` with overlapping weight/bias copied.
  - `_resize_head(old, new_in)`: resize only the input dim of the head.
- Snapshot/restore:
  - `snapshot_state()` → `{state_dict, hidden_sizes}`.
  - `restore_state(snap)`: rebuild stack if depth changed; resize layers to target sizes; reload state.
- Bounded outputs: optional `BoundedAct(bounds_low, bounds_high, mask)`; else `Identity`.
- Optimizer/scheduler: rebuilt after any structural change via `get_optimizer_scheduler(..., lr=self.lr, **SCHEDULER_PARAMS)`.
- Metrics:
  - Primary: validation MSE.
  - Physics/constraints: `power_balance_residuals`, `mean_constraint_violation`; combined `I"P+I"Q` used to pick global-best snapshot.
- Early stopping: patience counter resets on improvement (`val < best_val - delta_eps`), decrements otherwise; plateau triggers expansion logic.
- Capacity guards: `max_width`, `max_depth`, `max_neurons` checked before expanding.
- Failure counters: separate for depth vs. width expansions; if failures hit `trials_*`, rollback to last accepted snapshot and stop expanding that dimension.

### `adp_den_depth_only.py` (depth-only expansion)
- Inner loop: train with early stopping on val MSE; track `best_snapshot` (by MSE) and `best_csum` (I"P+I"Q).
- Global best: track snapshot with lowest I"P+I"Q across all phases.
- Plateau handling:
  - On first plateau, set `exp_accept_snapshot = best_snapshot`, `exp_accept_val = best_val_mse`.
  - Propose depth expansion: append one hidden layer with width = last hidden width; resize head input accordingly.
  - Train enlarged model; accept if `best_val_mse < pre_exp_val_mse - delta`; else increment `depth_failures`.
  - If `depth_failures` < `trials_depth`: try another expansion; else rollback to pre-expansion snapshot and stop.
- Stop when patience exhausted on failures, capacity guard hit, or max global epochs reached.
- Final restore: snapshot with lowest I"P+I"Q.

### `adp_den_width_only.py` (width-only expansion)
- Same control flow as depth-only; only difference is the expansion step:
  - `_widen_all_hidden(step=ex_k)`: add `ex_k` neurons to every hidden layer; propagate new fan-in to downstream layers and head.
  - Acceptance criterion identical: `best_val_mse < pre_exp_val_mse - delta`.
- Uses `trials_depth` as the width-failure patience knob (same naming as depth-only file).

### `adp_depth.py` (depth-first, then width sweeps)
- Exposes `hidden_layers = layers` from `DEN`.
- Hyperparameters: `delta, patience, ex_k, max_neurons, max_width, max_depth, trials_depth, trials_width`.
- Globals: `global_epoch` to enforce a total budget across all calls to `train_early_stop`.
- Utilities:
  - `expand_depth(model)`: append one square hidden layer (width = last hidden out_features) before head.
  - `expand_width(model, inc)`: add `inc` neurons to every hidden layer; fix fan-in of successors/head.
  - `train_early_stop(model, train_loader, val_loader, patience, delta, max_epochs)`: trains with early stopping; returns best val; restores best weights.
- Search policy:
  1. Depth phase: while `depth_failures < trials_depth` and depth < `max_depth`:
     - Snapshot + evaluate val via `train_early_stop`.
     - If improved over baseline by `delta`: accept (reset depth_failures); else depth_failures++.
     - On each acceptance, consider width sweeps (below).
  2. Width phase (for current depth): while `width_failures < trials_width` and width < `max_width`/`max_neurons`:
     - Expand width by `ex_k`, re-train via `train_early_stop`.
     - Accept if improved by `delta`, else width_failures++ and rollback.
  3. Stop when both dimensions exhaust patience or guards.
- Final restore: best snapshot by combined constraints (I"P+I"Q).

### `adp_den_expandtillplateuing.py` (width-then-depth until plateau)
- Variant of width-first then depth (sequential, not alternating):
  - Run width-only expansion loop with patience; when width trials exhausted or capacity hit, switch to depth-only loop.
  - Acceptance rule: improvement by `delta`; rollback on repeated failures.
  - Uses same early stopping inner loop and physics-based global best.

### `adp_den_alt_depth_4_head.py` (alternate depth → width → depth…; multi-head)
- Alternating policy (depth first):
  - Maintain `pw`, `pd` (patience for width/depth).
  - Loop: expand depth, train, accept if improvement; else `pd--`. If `pd` reaches 0, stop or switch to width.
  - Then expand width, train, accept if improvement; else `pw--`. If `pw` reaches 0, stop or switch back.
  - Continues until both patience counters exhausted or capacity limits hit.
- Multi-head specifics (ignored per instruction): heads likely resized alongside hidden stack; otherwise logic mirrors width/depth alternation.

### `adp_den_alt_width_1_head.py` (alternate width → depth → width…; single-head)
- Same as above but starts with width expansion (`pw`), then depth (`pd`), alternating until patience runs out or capacity guards trigger.

### Key knobs and behaviors to preserve if porting
- `delta`: minimum improvement in val MSE to accept an expansion.
- `patience` (inner): early stopping within a fixed architecture.
- `trials_depth`, `trials_width`: how many failed expansions before rollback/stop.
- `ex_k`: width increment per layer (width-only or width steps in mixed policies).
- Capacity guards: `max_width`, `max_depth`, `max_neurons`.
- Snapshot discipline: always snapshot pre-expansion; rollback on failure streak; track a separate global-best by physics metric (I"P+I"Q).
- Optimizer/scheduler reset after any structural change.
- Optional bounded output layer should be preserved if bounds are provided.

Use this as a reference when recreating ADP variants; mirror the control-flow, acceptance tests, rollback, and resizing mechanics shown above.
