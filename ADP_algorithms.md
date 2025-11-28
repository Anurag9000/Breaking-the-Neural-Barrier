Got it, so **failed expansions should still march forward** (deeper/wider) as long as patience is not exhausted.

* You **still track a global best** (by val loss).
* You **do NOT rollback** on each failed expansion.
* You **only rollback to global best at the end of that search context** (1D search, inner loop, phase, outer loop) when you stop.

Below is the **fully modified spec** with that logic baked in everywhere.

---

## 0. Common Infrastructure (unchanged idea)

```python
# Globals / hyperparams (conceptual)
max_depth
max_width
max_neurons

patience_es        # early stopping patience
patience_width_exp # width expansion patience
patience_depth_exp # depth expansion patience

delta_width        # threshold for width improvement
delta_depth        # threshold for depth improvement
ex_k_width         # width increment per expansion
```

### 0.2 train_with_early_stopping (unchanged)

```python
def train_with_early_stopping(model):
    best_val = float('inf')
    best_state = None
    best_phys = None

    es_counter = 0

    for epoch in range(max_epochs):
        train_one_epoch(model, train_data)
        val_loss, phys_metric = evaluate(model, val_data)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy_state(model)
            best_phys = phys_metric
            es_counter = 0
        else:
            es_counter += 1

        if es_counter >= patience_es:
            break

    model.load_state_dict(best_state)
    return best_val, best_state, best_phys
```

### 0.3 Structural ops (unchanged semantics)

```python
def expand_width(model, ex_k_width):
    # Increase hidden width by ex_k_width for all hidden layers.
    # Adjust downstream layers and head input, init new neurons.
    pass

def expand_depth(model, ex_k_depth):
    # Add ex_k_depth new hidden layers with in/out = model.width.
    # Adjust head input.
    pass

def snapshot_arch_and_state(model):
    arch = {
        "depth": model.depth,
        "width": model.width,
        # plus any other structural info you need
    }
    state = copy_state(model)
    return {"arch": arch, "state": state}

def restore_arch_and_state(model, snap):
    rebuild_model_arch(model, snap["arch"])
    model.load_state_dict(snap["state"])
```

---

## 1. Depth-Only ADP (ADP_DEPTH_ONLY)

> **New behavior**:
>
> * Start from current depth.
> * On **improvement**, update global best and reset `depth_failure_count`.
> * On **no improvement**, increment `depth_failure_count` but **do not rollback**.
> * Keep expanding deeper as long as `depth_failure_count < patience_depth_exp` and `depth < max_depth`.
> * At the **end**, restore the global best architecture once.

```python
def adp_depth_only(model):
    # Initial training at starting depth
    best_val, best_state, best_phys = train_with_early_stopping(model)
    best_depth = model.depth
    best_snap = snapshot_arch_and_state(model)

    depth_failure_count = 0

    while depth_failure_count < patience_depth_exp and model.depth < max_depth:
        # Always expand from current architecture, no rollback on fail
        expand_depth(model, ex_k_depth=1)

        val, state, phys = train_with_early_stopping(model)

        if val < best_val - delta_depth:
            # Improvement: update global best and reset failure streak
            best_val = val
            best_state = state
            best_depth = model.depth
            best_snap = snapshot_arch_and_state(model)
            depth_failure_count = 0
        else:
            # No improvement: keep this deeper arch as base and count failure
            depth_failure_count += 1

    # After depth search, restore best architecture
    restore_arch_and_state(model, best_snap)
    return model, best_val, best_depth
```

---

## 2. Width-Only ADP (ADP_WIDTH_ONLY)

> Same idea, but along width.

```python
def adp_width_only(model):
    best_val, best_state, best_phys = train_with_early_stopping(model)
    best_width = model.width
    best_snap = snapshot_arch_and_state(model)

    width_failure_count = 0

    while (width_failure_count < patience_width_exp
           and model.width < max_width
           and model.width < max_neurons):

        # Expand from current width, no rollback on fail
        expand_width(model, ex_k_width)

        val, state, phys = train_with_early_stopping(model)

        if val < best_val - delta_width:
            # Improvement
            best_val = val
            best_state = state
            best_width = model.width
            best_snap = snapshot_arch_and_state(model)
            width_failure_count = 0
        else:
            # No improvement, but keep going forward
            width_failure_count += 1

    # Restore global best width architecture
    restore_arch_and_state(model, best_snap)
    return model, best_val, best_width
```

---

## 3. Depth-Outer / Width-Inner (ADP_DEPTH_OUTER_WIDTH_INNER)

### 3.1 Inner: optimize_width_at_fixed_depth

> **Inner rule** is the same:
>
> * At **fixed depth**, keep widening from current width.
> * No rollback per failed step, only at the end revert to best at this depth.

```python
def optimize_width_at_fixed_depth(model):
    best_val, best_state, best_phys = train_with_early_stopping(model)
    best_width = model.width
    best_snap = snapshot_arch_and_state(model)

    width_failure_count = 0

    while (width_failure_count < patience_width_exp
           and model.width < max_width
           and model.width < max_neurons):

        # Always expand from current width
        expand_width(model, ex_k_width)

        val, state, phys = train_with_early_stopping(model)

        if val < best_val - delta_width:
            best_val = val
            best_state = state
            best_width = model.width
            best_snap = snapshot_arch_and_state(model)
            width_failure_count = 0
        else:
            width_failure_count += 1
            # do NOT rollback; continue from this (possibly worse) width

    # After inner search: ensure model is set to best width at this depth
    restore_arch_and_state(model, best_snap)
    return best_val, best_state, best_width
```

---

### 3.2 Outer: depth expansions with expansion patience

> **New outer rule**:
>
> * Start from some depth with width optimized by inner.
> * For each outer step: depth+1 → run inner width search.
> * If that depth’s best (after inner) beats global best by `delta_depth`, accept it as new global best.
> * If not, **do not rollback**; keep that deeper architecture as the base, increment `depth_failure_count`.
> * Keep increasing depth further (5→6→7…) even if 5 was worse than 4, until `depth_failure_count` hits `patience_depth_exp`.
> * At the end, rollback once to global best architecture.

```python
def adp_depth_outer_width_inner(model):
    # First, optimise width at starting depth
    base_val, base_state, base_width = optimize_width_at_fixed_depth(model)

    best_val = base_val
    best_state = base_state
    best_depth = model.depth
    best_width = base_width
    best_snap = snapshot_arch_and_state(model)

    depth_failure_count = 0

    while depth_failure_count < patience_depth_exp and model.depth < max_depth:
        # Outer: always expand depth from current architecture
        expand_depth(model, ex_k_depth=1)

        # Inner: re-optimise width at this new depth
        val_d, state_d, width_d = optimize_width_at_fixed_depth(model)
        # NOTE: optimize_width_at_fixed_depth leaves model at that depth, best width_d

        if val_d < best_val - delta_depth:
            # Global improvement
            best_val = val_d
            best_state = state_d
            best_depth = model.depth
            best_width = width_d
            best_snap = snapshot_arch_and_state(model)
            depth_failure_count = 0
        else:
            # No global improvement: keep deeper model as base, increase failure streak
            depth_failure_count += 1

    # After depth-outer search, restore best global architecture
    restore_arch_and_state(model, best_snap)
    return model, best_val, (best_depth, best_width)
```

---

## 4. Width-Outer / Depth-Inner (ADP_WIDTH_OUTER_DEPTH_INNER)

### 4.1 Inner: optimize depth at fixed width

> Same “no rollback per failure” rule, but along depth at fixed width.

```python
def optimize_depth_at_fixed_width(model):
    best_val, best_state, best_phys = train_with_early_stopping(model)
    best_depth = model.depth
    best_snap = snapshot_arch_and_state(model)

    depth_failure_count = 0

    while depth_failure_count < patience_depth_exp and model.depth < max_depth:
        # Expand from current depth
        expand_depth(model, ex_k_depth=1)

        val, state, phys = train_with_early_stopping(model)

        if val < best_val - delta_depth:
            best_val = val
            best_state = state
            best_depth = model.depth
            best_snap = snapshot_arch_and_state(model)
            depth_failure_count = 0
        else:
            depth_failure_count += 1
            # No rollback; keep going deeper

    # Ensure model is at best depth for this width
    restore_arch_and_state(model, best_snap)
    return best_val, best_state, best_depth
```

---

### 4.2 Outer: width expansions with expansion patience

> Symmetric to 3.2 but swapping roles.
>
> * Expand width from current architecture.
> * Run inner depth search at that width.
> * If the result beats global best by `delta_width`, accept as global best.
> * Otherwise, keep the wider architecture and increment `width_failure_count`.
> * Stop when width patience exhausted or width hits caps.
> * Finally rollback to global best only once.

```python
def adp_width_outer_depth_inner(model):
    # First, optimise depth at starting width
    base_val, base_state, base_depth = optimize_depth_at_fixed_width(model)

    best_val = base_val
    best_state = base_state
    best_width = model.width
    best_depth = base_depth
    best_snap = snapshot_arch_and_state(model)

    width_failure_count = 0

    while (width_failure_count < patience_width_exp
           and model.width < max_width
           and model.width < max_neurons):

        # Outer: always widen from current width
        expand_width(model, ex_k_width)

        # Inner: re-optimise depth at this new width
        val_w, state_w, depth_w = optimize_depth_at_fixed_width(model)
        # optimize_depth_at_fixed_width leaves model at that width, best depth_w

        if val_w < best_val - delta_width:
            # Global improvement
            best_val = val_w
            best_state = state_w
            best_width = model.width
            best_depth = depth_w
            best_snap = snapshot_arch_and_state(model)
            width_failure_count = 0
        else:
            # No global improvement; keep wider model as base
            width_failure_count += 1

    # After width-outer search, restore global best
    restore_arch_and_state(model, best_snap)
    return model, best_val, (best_depth, best_width)
```

---

## 5. ALT_DEPTH – Alternating Phases, Depth First

> **Key change** inside each phase:
>
> * Depth-phase: keep marching to deeper depths even if some are worse than best, until `depth_failure_count` hits `patience_depth_exp` or `max_depth`.
> * Width-phase: same for widths.
> * **No per-expansion rollback**; only after the phase ends do we restore to global best (so the next phase always starts from the global best arch).
> * Global best (`best_snap`) is updated only on real improvements.

```python
def adp_alt_depth(model):
    # Initial training to get global baseline
    best_val, best_state, best_phys = train_with_early_stopping(model)
    best_depth = model.depth
    best_width = model.width
    best_snap = snapshot_arch_and_state(model)

    # Ensure model is set to global best before phases
    restore_arch_and_state(model, best_snap)

    depth_saturated = False
    width_saturated = False
    mode = 'depth'  # start with depth-phase

    while not (depth_saturated and width_saturated):
        improved_in_phase = False

        if mode == 'depth':
            depth_failure_count = 0

            while depth_failure_count < patience_depth_exp and model.depth < max_depth:
                # Expand deeper from current architecture
                expand_depth(model, ex_k_depth=1)

                val, state, phys = train_with_early_stopping(model)

                if val < best_val - delta_depth:
                    # Global improvement by depth move
                    best_val = val
                    best_state = state
                    best_depth = model.depth
                    best_width = model.width  # unchanged, but record
                    best_snap = snapshot_arch_and_state(model)
                    improved_in_phase = True
                    depth_failure_count = 0
                else:
                    depth_failure_count += 1
                    # No rollback; keep going deeper

            if not improved_in_phase:
                depth_saturated = True

            # End of depth-phase: revert model to global best for next phase
            restore_arch_and_state(model, best_snap)
            mode = 'width'

        else:  # mode == 'width'
            width_failure_count = 0

            while (width_failure_count < patience_width_exp
                   and model.width < max_width
                   and model.width < max_neurons):

                # Expand width from current architecture
                expand_width(model, ex_k_width)

                val, state, phys = train_with_early_stopping(model)

                if val < best_val - delta_width:
                    # Global improvement by width move
                    best_val = val
                    best_state = state
                    best_width = model.width
                    best_depth = model.depth
                    best_snap = snapshot_arch_and_state(model)
                    improved_in_phase = True
                    width_failure_count = 0
                else:
                    width_failure_count += 1
                    # No rollback; keep widening

            if not improved_in_phase:
                width_saturated = True

            # End of width-phase: revert model to global best
            restore_arch_and_state(model, best_snap)
            mode = 'depth'

        # optional: global time/epoch/budget break here

    # Final: ensure model is at global best
    restore_arch_and_state(model, best_snap)
    return model, best_val, (best_depth, best_width)
```

---

## 6. ALT_WIDTH – Alternating Phases, Width First

> Same as above, but starting with the width-phase.

```python
def adp_alt_width(model):
    # Initial global baseline
    best_val, best_state, best_phys = train_with_early_stopping(model)
    best_depth = model.depth
    best_width = model.width
    best_snap = snapshot_arch_and_state(model)

    restore_arch_and_state(model, best_snap)

    depth_saturated = False
    width_saturated = False
    mode = 'width'  # start with width-phase

    while not (depth_saturated and width_saturated):
        improved_in_phase = False

        if mode == 'width':
            width_failure_count = 0

            while (width_failure_count < patience_width_exp
                   and model.width < max_width
                   and model.width < max_neurons):

                # Expand width from current architecture
                expand_width(model, ex_k_width)

                val, state, phys = train_with_early_stopping(model)

                if val < best_val - delta_width:
                    best_val = val
                    best_state = state
                    best_width = model.width
                    best_depth = model.depth
                    best_snap = snapshot_arch_and_state(model)
                    improved_in_phase = True
                    width_failure_count = 0
                else:
                    width_failure_count += 1
                    # No rollback; keep widening

            if not improved_in_phase:
                width_saturated = True

            restore_arch_and_state(model, best_snap)
            mode = 'depth'

        else:  # mode == 'depth'
            depth_failure_count = 0

            while depth_failure_count < patience_depth_exp and model.depth < max_depth:
                # Expand depth from current architecture
                expand_depth(model, ex_k_depth=1)

                val, state, phys = train_with_early_stopping(model)

                if val < best_val - delta_depth:
                    best_val = val
                    best_state = state
                    best_depth = model.depth
                    best_width = model.width
                    best_snap = snapshot_arch_and_state(model)
                    improved_in_phase = True
                    depth_failure_count = 0
                else:
                    depth_failure_count += 1
                    # No rollback; keep going deeper

            if not improved_in_phase:
                depth_saturated = True

            restore_arch_and_state(model, best_snap)
            mode = 'width'

        # optional global budget

    restore_arch_and_state(model, best_snap)
    return model, best_val, (best_depth, best_width)
