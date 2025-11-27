ADP Expansion Algorithms – Clean, Final Spec

This document defines the intended behavior of your ADP expansion algorithms, ignoring number of heads and physics-details except where relevant for selection.

We distinguish three kinds of patience:

Early-Stopping Patience (patience_es) – for training a fixed architecture.

Width Expansion Patience (patience_width_exp) – max consecutive width expansion attempts allowed without improvement.

Depth Expansion Patience (patience_depth_exp) – max consecutive depth expansion attempts allowed without improvement.

Expansion patience exists everywhere:

In simple 1D searches (width-only, depth-only).

In inner loops (optimize width at fixed depth, optimize depth at fixed width).

In outer loops (depth outer, width outer).

In ALT (phase-based) depth/width searches.

0. Common Infrastructure
0.1. Model

We assume a standard MLP-like model:

Input → hidden layers → head (output).

Track:

model.depth: number of hidden layers.

model.width: representative width per hidden layer (e.g., all layers same width).

Constraints:

max_depth

max_width

max_neurons (if needed, same as max_width or absolute cap on any layer).

0.2. Training

train_with_early_stopping(model):

Trains the current architecture on train/val.

Uses inner early-stopping patience patience_es:

If val_loss improves: update best_val, save best_state.

Else: increment ES counter.

Stop when ES counter reaches patience_es or max epochs.

Returns:

best_val: best validation MSE (or other target metric).

best_state: state_dict at that best validation.

phys_metric: an optional physics/constraint metric (e.g., I"P + I"Q).

0.3. Structural Ops

expand_width(model, ex_k_width)

Increase hidden width by ex_k_width (per layer).

Adjust downstream layers and head input.

Initialize new neurons; copy overlapping weights.

expand_depth(model, ex_k_depth)

Add ex_k_depth new hidden layers (usually 1 at a time).

Each new layer uses model.width as in/out dimension.

Adjust head input dimension.

snapshot_arch_and_state(model)

Returns an object containing architecture (depth, widths) and state_dict.

restore_arch_and_state(model, snapshot)

Rebuilds architecture to match snapshot; loads state_dict.

0.4. Patience Parameters

We use:

patience_es – early stopping (inner training).

patience_width_exp – expansion-patience for width.

patience_depth_exp – expansion-patience for depth.

Core rule for expansion patience (applies everywhere):

For a given context (e.g., “optimize width at this depth”), if an expansion (width or depth) does not produce an improvement >= delta_*, we increment the respective failure counter.
Once that counter reaches its patience (patience_width_exp or patience_depth_exp), we stop making expansions in that direction for that context and treat that dimension as “saturated” in that context.

We also have thresholds:

delta_width: min improvement in val to accept a width expansion.

delta_depth: min improvement in val to accept a depth expansion.

1. Depth-Only ADP (ADP_DEPTH_ONLY)

Pure 1D search in depth, with depth expansion patience.

Behavior

Width is fixed.

Start with current depth.

Try depth expansions (adding layers 1-by-1).

Each proposed expansion is trained via ES; if it improves by at least delta_depth, accept and reset depth_failure_count.

If it fails, rollback and increment depth_failure_count.

When depth_failure_count >= patience_depth_exp, stop and return the best depth.

Algorithm
def adp_depth_only(model):
    # Initial training
    best_val, best_state, best_phys = train_with_early_stopping(model)
    best_depth = model.depth
    depth_failure_count = 0

    while depth_failure_count < patience_depth_exp and model.depth < max_depth:
        snap = snapshot_arch_and_state(model)

        # Propose: expand depth by 1
        expand_depth(model, ex_k_depth=1)

        val, state, phys = train_with_early_stopping(model)

        if val < best_val - delta_depth:
            # Accept
            best_val   = val
            best_state = state
            best_depth = model.depth
            depth_failure_count = 0
        else:
            # Reject
            depth_failure_count += 1
            restore_arch_and_state(model, snap)

    model.load_state_dict(best_state)
    return model, best_val, best_depth

2. Width-Only ADP (ADP_WIDTH_ONLY)

Pure 1D search in width, with width expansion patience.

Behavior

Depth is fixed.

Start with current width.

Try widening increments (ex_k_width).

Each proposed width expansion is trained via ES; if it improves by at least delta_width, accept and reset width_failure_count.

Otherwise rollback and increment width_failure_count.

When width_failure_count >= patience_width_exp, stop and return best width.

Algorithm
def adp_width_only(model):
    best_val, best_state, best_phys = train_with_early_stopping(model)
    best_width = model.width
    width_failure_count = 0

    while (width_failure_count < patience_width_exp
           and model.width < max_width
           and model.width < max_neurons):

        snap = snapshot_arch_and_state(model)

        # Propose: widen
        expand_width(model, ex_k_width)

        val, state, phys = train_with_early_stopping(model)

        if val < best_val - delta_width:
            # Accept
            best_val   = val
            best_state = state
            best_width = model.width
            width_failure_count = 0
        else:
            # Reject
            width_failure_count += 1
            restore_arch_and_state(model, snap)

    model.load_state_dict(best_state)
    return model, best_val, best_width

3. Depth-Outer / Width-Inner (ADP_DEPTH_OUTER_WIDTH_INNER)

2D search with:

Inner width loop: find best width at a fixed depth, using width-expansion patience.

Outer depth loop: try depth expansions (one step at a time), each time re-running the inner width optimizer, with depth-expansion patience.

3.1. Inner: optimize width at fixed depth

This is basically adp_width_only, but used as an inner procedure.

def optimize_width_at_fixed_depth(model):
    best_val, best_state, best_phys = train_with_early_stopping(model)
    best_width = model.width
    width_failure_count = 0

    while (width_failure_count < patience_width_exp
           and model.width < max_width
           and model.width < max_neurons):

        snap = snapshot_arch_and_state(model)
        expand_width(model, ex_k_width)

        val, state, phys = train_with_early_stopping(model)

        if val < best_val - delta_width:
            best_val   = val
            best_state = state
            best_width = model.width
            width_failure_count = 0
        else:
            width_failure_count += 1
            restore_arch_and_state(model, snap)

    # Make sure model is at best width
    restore_arch_and_state(model, snapshot_arch_and_state(model))
    model.load_state_dict(best_state)
    return best_val, best_state, best_width


(You can simplify snapshot usage in real code.)

3.2. Outer: depth expansions with expansion patience
def adp_depth_outer_width_inner(model):
    # First, optimise width at starting depth
    base_val, base_state, base_width = optimize_width_at_fixed_depth(model)

    best_val   = base_val
    best_state = base_state
    best_depth = model.depth
    best_width = base_width

    depth_failure_count = 0

    while depth_failure_count < patience_depth_exp and model.depth < max_depth:
        # Snapshot global best architecture
        saved_snap  = snapshot_arch_and_state(model)
        saved_val   = best_val
        saved_depth = best_depth
        saved_width = best_width

        # Propose: expand depth once
        expand_depth(model, ex_k_depth=1)

        # Re-run inner width search at this new depth
        val_d, state_d, width_d = optimize_width_at_fixed_depth(model)

        if val_d < best_val - delta_depth:
            # Accept whole (depth, width) move
            best_val   = val_d
            best_state = state_d
            best_depth = model.depth
            best_width = width_d
            depth_failure_count = 0
        else:
            # Reject this depth move
            depth_failure_count += 1
            restore_arch_and_state(model, saved_snap)
            best_val   = saved_val
            best_depth = saved_depth
            best_width = saved_width

    model.load_state_dict(best_state)
    return model, best_val, (best_depth, best_width)


Here:

patience_width_exp controls how long we search widths for each fixed depth.

patience_depth_exp controls how many bad depth steps we tolerate before stopping the outer search.

4. Width-Outer / Depth-Inner (ADP_WIDTH_OUTER_DEPTH_INNER)

Mirror of 3, with roles swapped:

Inner loop: depth-only expansion with depth-expansion patience.

Outer loop: width expansions, each followed by inner-depth search, with width-expansion patience.

4.1. Inner: optimize depth at fixed width
def optimize_depth_at_fixed_width(model):
    best_val, best_state, best_phys = train_with_early_stopping(model)
    best_depth = model.depth
    depth_failure_count = 0

    while depth_failure_count < patience_depth_exp and model.depth < max_depth:
        snap = snapshot_arch_and_state(model)
        expand_depth(model, ex_k_depth=1)

        val, state, phys = train_with_early_stopping(model)

        if val < best_val - delta_depth:
            best_val   = val
            best_state = state
            best_depth = model.depth
            depth_failure_count = 0
        else:
            depth_failure_count += 1
            restore_arch_and_state(model, snap)

    model.load_state_dict(best_state)
    return best_val, best_state, best_depth

4.2. Outer: width expansions with expansion patience
def adp_width_outer_depth_inner(model):
    base_val, base_state, base_depth = optimize_depth_at_fixed_width(model)

    best_val   = base_val
    best_state = base_state
    best_width = model.width
    best_depth = base_depth

    width_failure_count = 0

    while (width_failure_count < patience_width_exp
           and model.width < max_width
           and model.width < max_neurons):

        saved_snap  = snapshot_arch_and_state(model)
        saved_val   = best_val
        saved_width = best_width
        saved_depth = best_depth

        # Propose: widen once
        expand_width(model, ex_k_width)

        # Inner: re-optimise depth at this new width
        val_w, state_w, depth_w = optimize_depth_at_fixed_width(model)

        if val_w < best_val - delta_width:
            best_val   = val_w
            best_state = state_w
            best_width = model.width
            best_depth = depth_w
            width_failure_count = 0
        else:
            width_failure_count += 1
            restore_arch_and_state(model, saved_snap)
            best_val   = saved_val
            best_width = saved_width
            best_depth = saved_depth

    model.load_state_dict(best_state)
    return model, best_val, (best_depth, best_width)

5. ALT_DEPTH – Alternating Phases, Depth First

Phase-based alternation, starting with depth-phase:

Depth phase: run a depth-only expansion loop with its own patience_depth_exp.

Width phase: then run a width-only expansion loop with its own patience_width_exp.

Repeat: depth phase → width phase → depth phase → …

Stop when:

depth phase yields no improvement (saturated), and

width phase yields no improvement (saturated),
or some global budget is reached.

In each phase, expansions also use expansion patience (i.e., even inside the depth-phase, each bad expansion increments depth_failure_count until patience_depth_exp).

Algorithm
def adp_alt_depth(model):
    best_val, best_state, best_phys = train_with_early_stopping(model)
    best_depth = model.depth
    best_width = model.width
    model.load_state_dict(best_state)

    depth_saturated = False
    width_saturated = False
    mode = 'depth'  # start with depth-phase

    while not (depth_saturated and width_saturated):
        improved_in_phase = False

        if mode == 'depth':
            depth_failure_count = 0

            while depth_failure_count < patience_depth_exp and model.depth < max_depth:
                snap = snapshot_arch_and_state(model)
                expand_depth(model, ex_k_depth=1)

                val, state, phys = train_with_early_stopping(model)

                if val < best_val - delta_depth:
                    best_val   = val
                    best_state = state
                    best_depth = model.depth
                    improved_in_phase = True
                    depth_failure_count = 0
                else:
                    depth_failure_count += 1
                    restore_arch_and_state(model, snap)

            if not improved_in_phase:
                depth_saturated = True

            model.load_state_dict(best_state)
            mode = 'width'

        else:  # mode == 'width'
            width_failure_count = 0

            while (width_failure_count < patience_width_exp
                   and model.width < max_width
                   and model.width < max_neurons):

                snap = snapshot_arch_and_state(model)
                expand_width(model, ex_k_width)

                val, state, phys = train_with_early_stopping(model)

                if val < best_val - delta_width:
                    best_val   = val
                    best_state = state
                    best_width = model.width
                    improved_in_phase = True
                    width_failure_count = 0
                else:
                    width_failure_count += 1
                    restore_arch_and_state(model, snap)

            if not improved_in_phase:
                width_saturated = True

            model.load_state_dict(best_state)
            mode = 'depth'

        # optional: global epoch/time break

    model.load_state_dict(best_state)
    return model, best_val, (best_depth, best_width)

6. ALT_WIDTH – Alternating Phases, Width First

Same as ALT_DEPTH, but start with width-phase instead of depth-phase.

Algorithm
def adp_alt_width(model):
    best_val, best_state, best_phys = train_with_early_stopping(model)
    best_depth = model.depth
    best_width = model.width
    model.load_state_dict(best_state)

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

                snap = snapshot_arch_and_state(model)
                expand_width(model, ex_k_width)

                val, state, phys = train_with_early_stopping(model)

                if val < best_val - delta_width:
                    best_val   = val
                    best_state = state
                    best_width = model.width
                    improved_in_phase = True
                    width_failure_count = 0
                else:
                    width_failure_count += 1
                    restore_arch_and_state(model, snap)

            if not improved_in_phase:
                width_saturated = True

            model.load_state_dict(best_state)
            mode = 'depth'

        else:  # mode == 'depth'
            depth_failure_count = 0

            while depth_failure_count < patience_depth_exp and model.depth < max_depth:
                snap = snapshot_arch_and_state(model)
                expand_depth(model, ex_k_depth=1)

                val, state, phys = train_with_early_stopping(model)

                if val < best_val - delta_depth:
                    best_val   = val
                    best_state = state
                    best_depth = model.depth
                    improved_in_phase = True
                    depth_failure_count = 0
                else:
                    depth_failure_count += 1
                    restore_arch_and_state(model, snap)

            if not improved_in_phase:
                depth_saturated = True

            model.load_state_dict(best_state)
            mode = 'width'

        # optional global budget

    model.load_state_dict(best_state)
    return model, best_val, (best_depth, best_width)
